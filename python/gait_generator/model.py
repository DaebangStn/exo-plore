from python.muscle import RayEnvManager
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from python.gait_generator.util import *
from torch import Tensor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


class MuscleNN(nn.Module):
    def __init__(self, related_dofs, num_dofs, num_muscles):
        super(MuscleNN, self).__init__()
        self.related_dofs = related_dofs
        self.num_dofs = num_dofs
        self.num_muscles = num_muscles

        num_h1 = 256
        num_h2 = 256
        num_h3 = 256

        self.fc = nn.Sequential(
            nn.Linear(related_dofs + num_dofs, num_h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h1, num_h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h2, num_h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h3, num_muscles),
        )

        self.std_muscle_tau = torch.ones(self.related_dofs) * 200.0
        self.std_tau = torch.ones(self.num_dofs) * 200.0
        self.fc.apply(weights_init)
        self.device = 'cpu'
        
    def forward(self, vecA: torch.Tensor, tau_desired: torch.Tensor):
        prev_device = vecA.device
        vecA = vecA.to(self.device)
        tau_desired = tau_desired.to(self.device)
        vecA = vecA / self.std_muscle_tau
        tau_desired = tau_desired / self.std_tau
        out = torch.relu(torch.tanh(self.fc.forward(torch.cat([vecA, tau_desired], dim=-1))))
        return out.to(prev_device)

    def forward_wo_relu(self, vecA: torch.Tensor, tau_desired: torch.Tensor):
        prev_device = vecA.device
        vecA = vecA.to(self.device)
        tau_desired = tau_desired.to(self.device)
        vecA = vecA / self.std_muscle_tau
        tau_desired = tau_desired / self.std_tau
        out = self.fc.forward(torch.cat([vecA, tau_desired], dim=-1))
        return out.to(prev_device)

    def no_grad_forward(self, vecA: torch.Tensor, tau_desired: torch.Tensor):
        vecA = torch.tensor(vecA, device=self.device)
        tau_desired = torch.tensor(tau_desired, device=self.device)
        with torch.no_grad():
            vecA = vecA / self.std_muscle_tau
            tau_desired = tau_desired / self.std_tau
            out = torch.relu(torch.tanh(self.fc.forward(torch.cat([vecA, tau_desired], dim=-1))))
            return out

    def load(self, path):
        print('load muscle nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save muscle nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def to(self, device):
        self.device = device
        self.std_tau = self.std_tau.to(device)
        self.std_muscle_tau = self.std_muscle_tau.to(device)
        self.fc.to(device)
        return self


class SimulationNN(nn.Module):
    def __init__(self, num_states: int, num_exo_states: int, num_actions: int, hparam: dict):
        if 'net_size' in hparam:
            dim_hidden_p = hparam['net_size']['hidden_p']
            dim_hidden_v = hparam['net_size']['hidden_v']
            dim_hidden_mod = hparam['net_size']['hidden_mod']
        else:
            dim_hidden_p = 512
            dim_hidden_v = 512
            dim_hidden_mod = 512

        # Parse modulation flags and scales
        modulation_cfg = hparam['modulation']
        self.net1_enable = modulation_cfg['net1']['enable']
        self.net1_scale = modulation_cfg['net1']['scale']
        self.net1_zero = modulation_cfg['net1']['zero_init']
        self.net2_enable = modulation_cfg['net2']['enable']
        self.net2_scale = modulation_cfg['net2']['scale']
        self.net2_zero = modulation_cfg['net2']['zero_init']
        self.net3_enable = modulation_cfg['net3']['enable']
        self.net3_scale = modulation_cfg['net3']['scale']
        self.net3_zero = modulation_cfg['net3']['zero_init']
        self.net4_enable = modulation_cfg['net4']['enable']
        self.net4_scale = modulation_cfg['net4']['scale']
        self.net4_zero = modulation_cfg['net4']['zero_init']
        self.net4_mask_ubody = modulation_cfg['net4']['mask_ubody']
        
        self._mod_full_state = modulation_cfg.get('full_state', False)
        self._mod_latent = modulation_cfg.get('latent', True)
        self._mod_restore = modulation_cfg.get('restore', False)
        
        value_fn_cfg = hparam.get('value_fn', {})
        self._vf_restore = value_fn_cfg.get('restore', False)

        self._num_exo_states = num_exo_states
        self._is_modulated = False
        self._is_frozen = False
        self._gain = None
        self._modulation_mu: List[float] = [0.0]

        super(SimulationNN, self).__init__()
        self.half()

        self.p_fc1 = nn.Linear(num_states, dim_hidden_p)
        self.p_fc2 = nn.Linear(dim_hidden_p, dim_hidden_p)
        self.p_fc3 = nn.Linear(dim_hidden_p, dim_hidden_p)
        self.p_fc4 = nn.Linear(dim_hidden_p, num_actions)
        torch.nn.init.xavier_uniform_(self.p_fc1.weight)
        torch.nn.init.xavier_uniform_(self.p_fc2.weight)
        torch.nn.init.xavier_uniform_(self.p_fc3.weight)
        torch.nn.init.xavier_uniform_(self.p_fc4.weight)
        self.p_fc1.bias.data.zero_()
        self.p_fc2.bias.data.zero_()
        self.p_fc3.bias.data.zero_()
        self.p_fc4.bias.data.zero_()

        self.v_fc1 = nn.Linear(num_states, dim_hidden_v)
        self.v_fc2 = nn.Linear(dim_hidden_v, dim_hidden_v)
        self.v_fc3 = nn.Linear(dim_hidden_v, dim_hidden_v)
        self.v_fc4 = nn.Linear(dim_hidden_v,1)
        self.reset_value_weight()

        self.mod_fc1 = nn.Linear(num_states, dim_hidden_mod)
        if not self._mod_latent:
            mod_input_dim = dim_hidden_mod
        else:
            if self._mod_full_state:
                mod_input_dim = dim_hidden_mod + num_states
            else:
                mod_input_dim = dim_hidden_mod + num_exo_states
        self.mod_fc2 = nn.Linear(mod_input_dim, dim_hidden_mod)
        self.mod_fc3 = nn.Linear(mod_input_dim, dim_hidden_mod)
        self.mod_fc4 = nn.Linear(mod_input_dim, num_actions)
        self.reset_mod_weight()

        init_log_std = hparam.get('logstd', 1.0) * torch.ones(num_actions)
        init_log_std[18:] *= 0.5 ## For Upper Body
        self.log_std = nn.Parameter(init_log_std)
        if not hparam.get('learn_logstd', False):
            self.log_std.requires_grad = False

        if torch.cuda.is_available():
            self.cuda()
        self.device = next(self.parameters()).device

        self.main_mu = torch.jit.Attribute(torch.zeros(num_actions, device=self.device), Tensor)
        self.mod1_mu = torch.jit.Attribute(torch.zeros(dim_hidden_mod, device=self.device), Tensor)
        self.mod2_mu = torch.jit.Attribute(torch.zeros(dim_hidden_mod, device=self.device), Tensor)
        self.mod3_mu = torch.jit.Attribute(torch.zeros(dim_hidden_mod, device=self.device), Tensor)
        self.mod4_mu = torch.jit.Attribute(torch.zeros(num_actions, device=self.device), Tensor)

    def reset_value_weight(self):
        torch.nn.init.xavier_uniform_(self.v_fc1.weight)
        torch.nn.init.xavier_uniform_(self.v_fc2.weight)
        torch.nn.init.xavier_uniform_(self.v_fc3.weight)
        torch.nn.init.xavier_uniform_(self.v_fc4.weight)
        self.v_fc1.bias.data.zero_()
        self.v_fc2.bias.data.zero_()
        self.v_fc3.bias.data.zero_()
        self.v_fc4.bias.data.zero_()

    def reset_mod_weight(self):
        torch.nn.init.xavier_uniform_(self.mod_fc1.weight)
        torch.nn.init.xavier_uniform_(self.mod_fc2.weight)
        torch.nn.init.xavier_uniform_(self.mod_fc3.weight)
        torch.nn.init.xavier_uniform_(self.mod_fc4.weight)

        scale = 0.1
        if self.net1_zero:
            self.mod_fc1.weight.data.mul_(scale)
        if self.net2_zero:
            self.mod_fc2.weight.data.mul_(scale)
        if self.net3_zero:
            self.mod_fc3.weight.data.mul_(scale)
        if self.net4_zero:
            self.mod_fc4.weight.data.mul_(scale)
        self.mod_fc1.bias.data.zero_()
        self.mod_fc2.bias.data.zero_()
        self.mod_fc3.bias.data.zero_()
        self.mod_fc4.bias.data.zero_()
        for name, param in self.named_parameters():
            if name.startswith('mod_fc'):
                param.requires_grad = False

    def set_modulation(self):
        self._is_modulated = True

    def freeze(self):
        self._is_frozen = True
        self.freeze_base_actor()
        self.freeze_critic()
        self.freeze_modulation()
    
    def freeze_base_actor(self):
        for name, param in self.named_parameters():
            if name.startswith('p_fc'):
                param.requires_grad = False
    
    def freeze_critic(self):
        for name, param in self.named_parameters():
            if name.startswith('v_fc'):
                param.requires_grad = False
    
    def freeze_modulation(self):
        for name, param in self.named_parameters():
            if name.startswith('mod_fc'):
                param.requires_grad = False
                
    def defrost_modulation(self):
        for name, param in self.named_parameters():
            if name.startswith('mod_fc'):
                param.requires_grad = True
    
    def defrost_base_actor(self):
        for name, param in self.named_parameters():
            if name.startswith('p_fc'):
                param.requires_grad = True
                
    def print_mod_params(self):
        # Calculate statistics
        mean_w = self.mod_fc1.weight.mean().item()
        std_w = self.mod_fc1.weight.std().item()
        min_w = self.mod_fc1.weight.min().item()
        max_w = self.mod_fc1.weight.max().item()
        
        # Print without formatting specifiers
        print(f"mod_fc1 weight stats: mean={mean_w}, std={std_w}, min={min_w}, max={max_w}")
        
        mean_b = self.mod_fc1.bias.mean().item()
        std_b = self.mod_fc1.bias.std().item()
        min_b = self.mod_fc1.bias.min().item()
        max_b = self.mod_fc1.bias.max().item()
        
        print(f"mod_fc1 bias stats: mean={mean_b}, std={std_b}, min={min_b}, max={max_b}")
        
        # Print just a small sample of actual values
        print("mod_fc1 weight sample:", self.mod_fc1.weight[0, :5])
        print("mod_fc1 bias sample:", self.mod_fc1.bias[:5])
    
    def actor(self, x: torch.Tensor, record: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x_device = x.device
        x = x.to(self.device)
        # self.print_mod_params()
        
        if record:
            self.mod1_mu = torch.zeros(self.mod1_mu.shape, device=self.device)
            self.mod2_mu = torch.zeros(self.mod2_mu.shape, device=self.device)
            self.mod3_mu = torch.zeros(self.mod3_mu.shape, device=self.device)
            self.mod4_mu = torch.zeros(self.mod4_mu.shape, device=self.device)
            self.main_mu = torch.zeros(self.main_mu.shape, device=self.device)

        self._modulation_mu.clear()
    
        if self._is_modulated:
            base_state = x.clone()
            base_state[:, -self._num_exo_states:] = 0
        else:
            base_state = x
        
        if self._is_modulated and self._mod_latent:
            # gain = torch.mean(torch.abs(x[:, -int(self._num_exo_states / 2):]), dim=-1).unsqueeze(1)

            out1 = torch.relu(self.p_fc1(base_state))
            if self.net1_enable:
                out_mod = torch.relu(self.mod_fc1(x))
                out1 = out1 + self.net1_scale * out_mod
                with torch.no_grad():
                    self._modulation_mu.append(out_mod.abs().mean().item())
                    if record:
                        self.mod1_mu = out_mod.clone().squeeze()

            out2 = torch.relu(self.p_fc2(out1))
            if self.net2_enable:
                if self._mod_full_state:
                    mod_state = torch.cat([x, out1], dim=1)
                else:
                    mod_state = torch.cat([x[:, -self._num_exo_states:], out1], dim=1)
                out_mod = torch.relu(self.mod_fc2(mod_state))
                out2 = out2 + self.net2_scale * out_mod
                with torch.no_grad():
                    self._modulation_mu.append(out_mod.abs().mean().item())
                    if record:
                        self.mod2_mu = out_mod.clone().squeeze()

            out3 = torch.relu(self.p_fc3(out2))
            if self.net3_enable:
                if self._mod_full_state:
                    mod_state = torch.cat([x, out2], dim=1)
                else:
                    mod_state = torch.cat([x[:, -self._num_exo_states:], out2], dim=1)
                out_mod = torch.relu(self.mod_fc3(mod_state))
                out3 = out3 + self.net3_scale * out_mod
                with torch.no_grad():
                    self._modulation_mu.append(out_mod.abs().mean().item())
                    if record:
                        self.mod3_mu = out_mod.clone().squeeze()

            out = self.p_fc4(out3)
            if self.net4_enable:
                if self._mod_full_state:
                    mod_state = torch.cat([x, out3], dim=1)
                else:
                    mod_state = torch.cat([x[:, -self._num_exo_states:], out3], dim=1)
                out_mod = torch.relu(self.mod_fc4(mod_state))
                if self.net4_mask_ubody:
                    mask = torch.ones_like(out_mod)
                    mask[:, 18:] = 0
                    out_mod = out_mod * mask
                with torch.no_grad():
                    self._modulation_mu.append(out_mod.abs().mean().item())
                    if record:
                        self.main_mu = out.clone().squeeze()
                        self.mod4_mu = out_mod.clone().squeeze()
                out = out + self.net4_scale * out_mod

        else:
            out = torch.relu(self.p_fc1(base_state))
            out = torch.relu(self.p_fc2(out))
            out = torch.relu(self.p_fc3(out))
            out = self.p_fc4(out)
            
            if self._is_modulated:
                mod_out = x
                if self.net1_enable:
                    mod_out = torch.relu(self.mod_fc1(mod_out))
                if self.net2_enable:
                    mod_out = torch.relu(self.mod_fc2(mod_out))
                if self.net3_enable:
                    mod_out = torch.relu(self.mod_fc3(mod_out))
                if self.net4_enable:
                    mod_out = self.mod_fc4(mod_out)
                    if self.net4_scale > 0:
                        mod_out = self.net4_scale * torch.tanh(mod_out)
                    if self.net4_mask_ubody:
                        mask = torch.ones_like(mod_out)
                        mask[:, 18:] = 0
                        mod_out = mod_out * mask
                    if record:
                        self.main_mu = out.clone().squeeze()
                        self.mod4_mu = mod_out.clone().squeeze()
                out = out + mod_out
                with torch.no_grad():
                    self._modulation_mu.append(mod_out.abs().mean().item())

        out = out.to(x_device)
        logstd = self.log_std.to(x_device)
        return out, logstd

    def critic(self, x: torch.Tensor) -> torch.Tensor:
        x_device = x.device
        x = x.to(self.device)
        out = torch.relu(self.v_fc1(x))
        out = torch.relu(self.v_fc2(out))
        out = torch.relu(self.v_fc3(out))
        out = self.v_fc4(out)
        out = out.to(x_device)
        return out

    def soft_load_state_dict(self, _state_dict):
        current_state_dict = self.state_dict()
        for k in _state_dict:
            current_state_dict[k] = _state_dict[k]
        self.load_state_dict(current_state_dict)

    def load(self, path):
        print('load simulation nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save simulation nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def get_action(self, s):
        ts = torch.tensor(s)
        p_mu, _ = self.actor(ts)
        return p_mu.cpu().detach().numpy()

    def get_value(self, s):
        ts = torch.tensor(s)
        v = self.critic(ts)
        return v.cpu().detach().numpy()
    
    @property
    def logstd(self) -> float:
        return self.log_std.cpu().detach().mean().item()

    @property
    def gain(self):
        return self._gain
    
    @property
    def is_frozen(self):
        return self._is_frozen

    @property
    def modulation_mu(self) -> List[float]:
        return self._modulation_mu


def append_to_list(target_list, value: Union[float, torch.Tensor, list, None]):
    if value is None:
        target_list.append(0.0)
    elif isinstance(value, torch.Tensor):
        target_list.append(value.abs().mean().item())
    elif isinstance(value, float):
        target_list.append(abs(value))
    elif isinstance(value, int):
        target_list.append(abs(value))
    elif isinstance(value, list):  # Must be List[float]
        if len(value) > 0:
            data = torch.tensor(value)
            target_list.append(data.abs().mean().item())
        else:
            target_list.append(0.0)
    else:
        print(f"[Warn] Unexpected value type({type(value)})")


class PDRayModel(TorchModelV2, SimulationNN):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        SimulationNN.__init__(self, np.prod(obs_space.shape), kwargs["exo_state_dim"], np.prod(action_space.shape), kwargs['hparam'])
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, {}, "PDRayModel")

        if kwargs['value_function'] is not None:
            self.soft_load_state_dict(kwargs['value_function'])
        self._restore_done = False
        self._value = None
        self._main_action_magnitude = []
        self._modulation_action_magnitude = []
        self._sigma_magnitude = []
        self._gain_magnitude = []

    def get_value(self, obs):
        with torch.no_grad():
            v = self.critic(obs)
        return v

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        x = obs.reshape(obs.shape[0], -1)
        self._value = self.critic(x)
        action_mu, action_logstd = self.actor(x)

        # Track magnitudes only when not in training mode
        if not torch.is_grad_enabled():
            self._track_magnitudes(action_mu, action_logstd, self.gain, self.modulation_mu)

        action_logstd = action_logstd.expand_as(action_mu)
        action_tensor = torch.cat([action_mu, action_logstd], dim=1)
        return action_tensor, state

    def _track_magnitudes(self, action_mu, action_logstd, gain, modulation_mu):
        """Helper method to track various magnitudes during evaluation"""
        append_to_list(self._main_action_magnitude, action_mu)
        append_to_list(self._modulation_action_magnitude, modulation_mu)
        append_to_list(self._sigma_magnitude, action_logstd.exp())
        append_to_list(self._gain_magnitude, gain)

    def get_action_magnitude(self) -> Dict[str, float]:
        main_action_magnitude = 0
        modulation_action_magnitude = 0
        sigma_action_magnitude = 0
        gain_magnitude = 0
        if len(self._main_action_magnitude) > 0:
            main_action_magnitude = np.mean(self._main_action_magnitude)
            modulation_action_magnitude = np.mean(self._modulation_action_magnitude)
            sigma_action_magnitude = np.mean(self._sigma_magnitude)
            gain_magnitude = np.mean(self._gain_magnitude)
            self._main_action_magnitude.clear()
            self._modulation_action_magnitude.clear()
            self._sigma_magnitude.clear()
            self._gain_magnitude.clear()
        return {"main": main_action_magnitude, "modulation": modulation_action_magnitude, "sigma": sigma_action_magnitude, "gain": gain_magnitude}

    # Since this method called every epoch, we have to branch the first time it is called
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        if not self._restore_done:
            # Always filter out log_std
            filtered_state_dict = {
                name: param for name, param in state_dict.items() 
                if name != 'log_std'
            }
            
            # Filter modulation weights if not restoring
            if not self._mod_restore:
                filtered_state_dict = {
                    name: param for name, param in filtered_state_dict.items() 
                    if not name.startswith('mod_fc')
                }
                self.reset_mod_weight()
                
            if not self._vf_restore:
                filtered_state_dict = {
                    name: param for name, param in filtered_state_dict.items() 
                    if not name.startswith('v_fc')
                }
                self.reset_value_weight()

            # Load filtered state dict
            out = super().load_state_dict(filtered_state_dict, False, assign)

            # Verify keys in a generalized way
            expected_missing = {'log_std'}  # log_std is always expected to be missing
            if not self._mod_restore:
                expected_missing.update(k for k in self.state_dict().keys() if k.startswith('mod_fc'))
            if not self._vf_restore:
                expected_missing.update(k for k in self.state_dict().keys() if k.startswith('v_fc'))
            
            actual_missing = set(out.missing_keys)
            unexpected_missing = actual_missing - expected_missing
            assert not unexpected_missing, f"Found unexpected missing keys: {unexpected_missing}"
            assert not out.unexpected_keys, f"Found unexpected keys: {out.unexpected_keys}"
        else:
            out = super().load_state_dict(state_dict, strict, assign)
        return out

    def set_restore_done(self):
        self._restore_done = True

    def value_function(self):
        return self._value.squeeze(1)


class NoFilter_TorchScript(nn.Module):
    def __init__(self):
        super(NoFilter_TorchScript, self).__init__()

    def forward(self, x):
        return x


class MeanStdFilter_TorchScript(nn.Module):
    def __init__(self, filter: ray.rllib.utils.filter.MeanStdFilter):
        super(MeanStdFilter_TorchScript, self).__init__()
        self.mean = nn.Parameter(
            torch.from_numpy(filter.rs.mean.astype(np.float32)).reshape(1, -1), requires_grad=False)
        self.std = nn.Parameter(
            torch.from_numpy(filter.rs.std.astype(np.float32)).reshape(1, -1), requires_grad=False)

    def forward(self, x):
        x = x - self.mean
        x = x / (self.std + 1e-8)
        return x


class PDInterferenceModel(SimulationNN):
    def __init__(self, num_states: int, num_exo_states: int, num_actions: int, metadata: dict, policy_state, filter_state, device):
        """
        :param num_states:
        :param num_exo_states:
        :param num_actions:
        :param metadata:
        :param policy_state:
        :param filter_state:
        :param device:
        """
        SimulationNN.__init__(self, num_states, num_exo_states, num_actions, metadata['learning']['hparam'])
        self.to(device).eval()
        self.load_state_dict(policy_state)
        self.filter = filter_state

        if isinstance(self.filter, ray.rllib.utils.filter.MeanStdFilter):
            self.filter_ts = MeanStdFilter_TorchScript(filter_state).to(device)
        else:
            self.filter_ts = NoFilter_TorchScript()
        self._is_modulated = False

    def forward(self, obs: torch.Tensor, use_modulation: bool = True):
        self._is_modulated = use_modulation
        with torch.no_grad():
            obs = self.filter_ts(obs).reshape(1, -1)
            mu, _ = self.actor(obs, record=True)
            return mu.squeeze()

    def state_dict(self):
        return {"weight": super().state_dict(), "filter": self.filter}

    def load_state_dict(self, state_dict, strict=True):
        filtered_state_dict = state_dict
        # filtered_state_dict = {
        #     name: param for name, param in state_dict.items() 
        #     if not name.startswith('mod_fc')
        # }
         
        adjusted_state_dict = {}
        for key, value in filtered_state_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.tensor(value)
            adjusted_state_dict[key] = value
        
        out = super().load_state_dict(adjusted_state_dict, strict=False)
        if strict:
            expected_missing = [name for name in state_dict.keys() if name.startswith('mod_fc')]
            missing_keys = set(out.missing_keys) - set(expected_missing)
            unexpected_keys = set(out.unexpected_keys)
            assert not unexpected_keys, f"Found unexpected keys: {unexpected_keys}"
            assert not missing_keys, f"Found missing keys: {missing_keys}"
        return out


class SelectiveUnpickler(Unpickler):
    def __init__(self, file):
        super().__init__(file)
        self._allowed_classes = {
            ('builtins', 'dict'),
            ('builtins', 'list'),
            ('builtins', 'str'),
            ('builtins', 'int'),
            ('builtins', 'float'),
            ('builtins', 'bool'),
            ('torch', 'Tensor'),
            ('numpy', 'ndarray'),
            ('numpy.core.numeric', '_frombuffer'),
            ('numpy', 'dtype'),
            ('ray.rllib.utils.filter', 'NoFilter'),
        }

    def find_class(self, module, name):
        if (module, name) in self._allowed_classes:
            return super().find_class(module, name)
        else:
            # print(f"Class {module}.{name} is not allowed to be unpickled.")
            return Dummy


def load_from_checkpoint(ckpt_path:str, metadata_path: str, device="cpu"):
    env = RayEnvManager(metadata_path)
    state = dill.load(open(ckpt_path, "rb"))
    with open(metadata_path, "r") as f:
        metadata = yaml.load(f, IgnoreUndefinedLoader)

    # Suppressing unpickling error for ray 1.8.0, but it cannot unpickle the modulation model
    # worker_state = SelectiveUnpickler(BytesIO(state['worker'])).load()
    worker_state = dill.load(BytesIO(state['worker']))
    model_weight = worker_state["state"]['default_policy']['weights']
    filter_state = worker_state["filters"]['default_policy']
    device = torch.device(device)

    pd_net = PDInterferenceModel(env.GetNumState(), len(env.GetStateExo()), env.GetNumAction(), metadata, model_weight, filter_state, device).cpu()
    # For debugging model parameters
    # pprint.pprint({name: param.flatten()[:5].data for name, param in pd_net.named_parameters() if name.startswith("nn.p_")})
    muscle_net = None
    if env.GetUseMuscle() and env.GetNumMuscles():
        if "model" in state["muscle"]:
            muscle_state = convert_to_torch_tensor(state["muscle"]['model'])
        else:
            muscle_state = convert_to_torch_tensor(state["muscle"])
        muscle_net = MuscleNN(env.GetRelatedDof(), env.GetMcnDof(), env.GetNumMuscles()).to(device)
        muscle_net.load_state_dict(muscle_state)
        muscle_net = muscle_net.cpu()
    return pd_net, muscle_net

def metadata_from_checkpoint(checkpoint_path) -> Union[str, dict]:
    state = dill.load(open(checkpoint_path, 'rb'))
    return state["metadata"]
