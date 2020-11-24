from python.gait_generator.model import MuscleNN
from python.gait_generator.util import *


class MuscleLearner:
    def __init__(self, env_config: dict, hparam: dict, device):
        self.device = device
        self._num_epochs = hparam['epoch']
        self._batch_size = hparam['batch_size']
        co_act_indices = env_config['co_act_indices']  # type: List[List[int]]
        self._co_act_num = len(co_act_indices)
        self._co_act_crit = hparam.get('co_act_crit', 0.1)
        if self._co_act_num > 0:
            max_co_act_size = max([len(group) for group in co_act_indices])
            config_tensor = torch.zeros((self._co_act_num, max_co_act_size), dtype=torch.long)
            config_mask = torch.zeros((self._co_act_num, max_co_act_size), dtype=torch.bool)

            for i, group in enumerate(co_act_indices):
                group_size = len(group)
                config_tensor[i, :group_size] = torch.tensor(group, dtype=torch.long)
                config_mask[i, :group_size] = 1
                for i, group in enumerate(co_act_indices):
                    group_size = len(group)
                    config_tensor[i, :group_size] = torch.tensor(group, dtype=torch.long)
                    config_mask[i, :group_size] = True  # Valid indices

            config_tensor = config_tensor.unsqueeze(0).expand(self._batch_size, -1, -1)
            config_mask = config_mask.unsqueeze(0).expand(self._batch_size, -1, -1)
            count = config_mask.sum(dim=2).clamp(min=1).float()  # [self._batch_size, self._co_act_num]
            self.config_tensor = config_tensor.to(self.device)
            self.config_mask = config_mask.to(self.device)
            self.count = count.to(self.device)

        self.model = MuscleNN(env_config['num_muscle_dofs'], env_config['num_mcn_dof'], env_config['num_muscles']
                              ).to(self.device).train()

        hparam['lr_start'] = float(hparam['lr_start'])
        hparam['lr_end'] = float(hparam['lr_end'])
        self._w_regularize = hparam['loss']['w_regularize']
        self._w_regression = hparam['loss']['w_regression']
        self._w_co_act = hparam['loss']['w_co_act']

        self.optimizer = optim.Adam(self.model.parameters(), lr=hparam['lr_start'])
        def lr_lambda(epoch):
            if epoch < hparam['start_epoch']:
                return 1.0
            elif epoch < hparam['end_epoch']:
                return (1 - (epoch - hparam['start_epoch']) / (hparam['end_epoch'] - hparam['start_epoch']) *
                        (1 - hparam['lr_end'] / hparam['lr_start']))
            else:
                return hparam['lr_end'] / hparam['lr_start']
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = torch.cuda.amp.GradScaler()

    def get_state_dict(self) -> Dict:
        return {
            'model': self.get_model_weights(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict()
        }

    def set_state_dict(self, state_dict: Dict) -> None:
        self.set_model_weights(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.scaler.load_state_dict(state_dict['scaler'])

    def set_model_weights(self, weights) -> None:
        weights = convert_to_torch_tensor(weights, device=self.device)
        self.model.load_state_dict(weights)

    def get_optimizer_weights(self) -> Dict:
        return self.optimizer.state_dict()

    def set_optimizer_weights(self, weights) -> None:
        self.optimizer.load_state_dict(weights)

    def get_model_weights(self, device=None) -> Dict:
        if device:
            return {k: v.to(device) for k, v in self.model.state_dict().items()}
        else:
            return self.model.state_dict()

    def is_training(self):
        return self.optimizer.param_groups[0]['lr'] > 0

    def learn(self, muscle_tuples: list) -> Dict:
        start_time = time.perf_counter()
        num_samples = len(muscle_tuples[0])

        JtA_all = torch.tensor(muscle_tuples[0], device=self.device)
        JtA_reduced_all = torch.tensor(muscle_tuples[1], device=self.device)
        tau_active_all = torch.tensor(muscle_tuples[2], device=self.device)

        converting_time = (time.perf_counter() - start_time) * 1000
        start_time = time.perf_counter()
        tot_loss = 0.0
        tot_loss_regularize = 0.0
        tot_loss_co_act = 0.0
        tot_loss_regression = 0.0
        tot_over_count = 0.0

        for epoch in range(self._num_epochs):
            idx_all = torch.randperm(num_samples, device=self.device)
            epoch_loss = 0.0
            epoch_loss_regularize = 0.0
            epoch_loss_co_act = 0.0
            epoch_loss_regression = 0.0
            epoch_over_count = 0.0
            for i in range(0, num_samples - self._batch_size + 1, self._batch_size):
                batch_indices = idx_all[i: i + self._batch_size]

                JtA = JtA_all[batch_indices]
                JtA_reduced = JtA_reduced_all[batch_indices]
                tau_active = tau_active_all[batch_indices]

                if self.is_training():
                    with torch.cuda.amp.autocast():
                        loss_reg_wo_relu = torch.tensor(0.0, device=self.device)

                        # activation_wo_relu = self.model.forward_wo_relu(JtA, (tau_des - b)).unsqueeze(2)
                        # activation = torch.relu(torch.tanh(activation_wo_relu))
                        # loss_reg_wo_relu = activation_wo_relu.pow(2).mean()

                        activation = self.model.forward(JtA_reduced, tau_active).unsqueeze(2)
                        tau = torch.bmm(JtA, activation).squeeze(-1)
                        loss_regularize = activation.pow(2).mean()
                        loss_regression = (((tau - tau_active) / 100.0).pow(2)).mean()

                        loss_co_act = torch.tensor(0.0, device=self.device)
                        over_threshold = torch.tensor(0.0, device=self.device)
                        if self._co_act_num > 0:
                            activation = activation.reshape(self._batch_size, 1, -1).expand(-1, self._co_act_num, -1)  # [batch, num_groups, max_group_size]
                            activation_group = torch.gather(activation, 2, self.config_tensor)
                            mean_activation = (activation_group * self.config_mask).sum(dim=2) / self.count  # [batch, num_groups]
                            deviation = activation_group - mean_activation.unsqueeze(2)  # [batch, num_groups, max_group_size]
                            abs_deviation = torch.abs(deviation)
                            over_threshold = abs_deviation > self._co_act_crit
                            loss_co_act = (abs_deviation * over_threshold.float() * self.config_mask).pow(2).sum() / self._co_act_num
                    loss = self._w_regularize * (loss_regularize + loss_reg_wo_relu) + \
                           self._w_co_act * loss_co_act + \
                           self._w_regression * loss_regression

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    with torch.no_grad():
                        activation = self.model.forward(JtA_reduced, tau_active).unsqueeze(2)
                        tau = torch.bmm(JtA, activation).squeeze(-1)
                        loss_regularize = activation.pow(2).mean()
                        loss_regression = (((tau - tau_active) / 100.0).pow(2)).mean()
                        loss_co_act = torch.tensor(0.0, device=self.device)
                        over_threshold = torch.tensor(0.0, device=self.device)
                    if self._co_act_num > 0:
                        activation = activation.reshape(self._batch_size, 1, -1).expand(-1, self._co_act_num, -1)
                        activation_group = torch.gather(activation, 2, self.config_tensor)  # [64, 62, 5]
                        mean_activation = (activation_group * self.config_mask).sum(dim=2) / self.count  # [64, 62]
                        deviation = activation_group - mean_activation.unsqueeze(2)  # [64, 62, 5]
                        abs_deviation = torch.abs(deviation)  # [64, 62, 5]
                        over_threshold = abs_deviation > self._co_act_crit  # [batch, num_groups, max_group_size]
                        loss_co_act = (abs_deviation * over_threshold.float() * self.config_mask).pow(2).sum() / self._co_act_num
                    loss = self._w_regularize * loss_regularize + \
                           self._w_co_act * loss_co_act + \
                           self._w_regression * loss_regression

                epoch_loss += loss.item()
                epoch_loss_regularize += loss_regularize.item()
                epoch_loss_co_act += loss_co_act.item()
                epoch_loss_regression += loss_regression.item()
                epoch_over_count += over_threshold.sum().item()
            tot_loss += epoch_loss / (num_samples / self._batch_size)
            tot_loss_regularize += epoch_loss_regularize / (num_samples / self._batch_size)
            tot_loss_co_act += epoch_loss_co_act / (num_samples / self._batch_size)
            tot_loss_regression += epoch_loss_regression / (num_samples / self._batch_size)
            tot_over_count += epoch_over_count / num_samples

        self.scheduler.step()
        return {
            'loss': {
                "total": tot_loss / self._num_epochs,
                "regularize": tot_loss_regularize / self._num_epochs,
                "co_act": tot_loss_co_act / self._num_epochs,
                "regression": tot_loss_regression / self._num_epochs,
                "over_count": tot_over_count / self._num_epochs
            },
            'lr_muscle': self.optimizer.param_groups[0]['lr'],
            'time': {
                'converting_time_ms': converting_time,
                'learning_time_ms': (time.perf_counter() - start_time) * 1000
            }
        }
