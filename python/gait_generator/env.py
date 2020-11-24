from python import muscle
from python.gait_generator.model import MuscleNN
# from gym.utils import seeding
import copy
from python.gait_generator.util import *


# class GymEnvWrapper(gym.Env):
#     @override
#     def seed(self, seed=None):
#         # This method overrides the default method becuase Ray calls the method but it prints too noisy warning
#         self._np_random, seed = seeding.np_random(seed)
#         return [seed]

class MyEnv(gym.Env):
    def __init__(self, metadata_path: str, seed: int = None):
        # if seed is None:
            # seed = randint(0, 1000)
        # seed_all(seed)

        metadata_path = PROJECT_ROOT / metadata_path
        self._muscle_tuple_bound = 480  # User Parameter
        self._curriculum_level = 0
        self._tuple_accum = 0
        self._current_params = {}

        self._env = muscle.RayEnvManager(str(metadata_path))
        self.num_state = self._env.GetNumState()
        self.pd_dof = self._env.GetPdDof()
        self.num_exo_state = len(self._env.GetStateExo())
        self.num_action = self._env.GetNumAction()

        with open(metadata_path, "r") as f:
            self.metadata = yaml.safe_load(f)

        env = self.metadata['environment']
        sim = env['simulation']
        self.num_simulation_Hz = sim['simulation_hz']
        self.num_control_Hz = sim['control_hz']
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

        self._hparam_cfg = self.metadata['learning']['hparam']
        self._param_cfg = self.metadata['learning']['param']

        self._reward_tuples = {}
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._use_muscle = self._env.GetUseMuscle()
        if self._use_muscle:
            self.num_muscles = self._env.GetNumMuscles()
            self.num_muscle_dofs = self._env.GetRelatedDof()
            self.num_mcn_dof = self._env.GetMcnDof()
            self.muscle_model = MuscleNN(self.num_muscle_dofs, self.num_mcn_dof, self.num_muscles).to(self.device)
            self.muscle_model.eval()
            self._muscle_tuples = []

        # For Ray Env
        large_value = 100000
        self.observation_space = gym.spaces.Box(
            low=np.full((self.num_state,), -large_value, dtype=np.float32),
            high=np.full((self.num_state,), large_value, dtype=np.float32),
            dtype=np.float32
        )
        large_value = 30
        self.action_space = gym.spaces.Box(
            low=np.full((self.num_action,), -large_value, dtype=np.float32),
            high=np.full((self.num_action,), large_value, dtype=np.float32),
            dtype=np.float32
        )
        self.stats = {}

        # Initialize parameter bin tracker if adaptive sampling is enabled
        self._adaptive_sampling = self.metadata['learning'].get('adaptive_sampling', {}).get('enable', False)
        if self._adaptive_sampling:
            self._adaptive_weights = {}
            self._episode_stats = []
            self._current_episode = {
                "params": None,
                "length": 0,
                "reward": 0
            }
        
        self.update_param()
        

    def _sample_param(self) -> Dict[str, float]:
        """Sample a parameter from the configuration."""
        sampled_result = {}        
        # Then sample remaining parameters using regular methods
        for key, cfg in self._param_cfg.items():
            # Skip parameters already sampled via adaptive sampling
            if key in sampled_result:
                continue
            
            if "value" in cfg.keys():
                sampled_result[key] = cfg["value"]
                continue
            
            cfg = cfg["sample"]
            sample_type = cfg["type"]
            
            if sample_type == "diagonal_adaptive":
                if key in self._adaptive_weights:
                    sampling_weights = (self._adaptive_weights[key]).flatten()
                else:
                    sampling_weights = np.array([1.0])
                num_bins = len(sampling_weights)
                bin_idx = np.random.choice(num_bins, p=sampling_weights)
                
                sample_range = cfg["value"]
                if not isinstance(sample_range[0], list):
                    sample_range = [sample_range]
                first_param_min, first_param_max = sample_range[0]
                first_param_bin_start = first_param_min + bin_idx * (first_param_max - first_param_min) / num_bins
                first_param_bin_end = first_param_min + (bin_idx + 1) * (first_param_max - first_param_min) / num_bins
                first_param = np.random.uniform(first_param_bin_start, first_param_bin_end)
                sampled_result[key] = first_param
                
                if "paired_to" in cfg:
                    ratio = (first_param - first_param_min) / (first_param_max - first_param_min)
                    for jdx, paired_key in enumerate(cfg["paired_to"]):
                        paired_range = sample_range[jdx + 1]
                        paired_min, paired_max = paired_range
                        sampled_result[paired_key] = paired_min + ratio * (paired_max - paired_min)

            elif sample_type == "uniform_adaptive":
                param_names = [key]
                if "paired_to" in cfg:
                    param_names.extend(cfg["paired_to"])

                if key in self._adaptive_weights:
                    sampling_weights = (self._adaptive_weights[key])
                    bin_counts = sampling_weights.shape
                else:
                    bin_dims = len(param_names)
                    bin_counts = (1,) * bin_dims
                    sampling_weights = np.ones(1).reshape(bin_counts)
                flat_weights = sampling_weights.flatten()
                flat_idx = np.random.choice(len(flat_weights), p=flat_weights)
                bin_idx = np.unravel_index(flat_idx, bin_counts)

                print(f"weights: {sampling_weights}, bin_idx: {bin_idx} bin_counts: {bin_counts}")

                sample_range = cfg["value"]
                if not isinstance(sample_range[0], list):
                    sample_range = [sample_range]
                for i, param in enumerate(param_names):
                    param_min, param_max = sample_range[i]
                    bin_start = param_min + bin_idx[i] * (param_max - param_min) / bin_counts[i]
                    bin_end = param_min + (bin_idx[i] + 1) * (param_max - param_min) / bin_counts[i]
                    sampled_result[param] = np.random.uniform(bin_start, bin_end)
                    print(f"param: {param}, bin_start: {bin_start}, bin_end: {bin_end}, sampled_result: {sampled_result[param]}")

            elif sample_type == "uniform":
                first, second = cfg["value"]
                sampled_result[key] = uniform(first, second)

            elif sample_type == "uniform_plus_point": # (single value with weight) U (uniform interval)
                use_value = np.random.random() < cfg['weight']
                if use_value:
                    sampled_result[key] = cfg['value']
                else:
                    first, second = cfg["interval"]
                    sampled_result[key] = uniform(first, second)
            
            elif sample_type == "uniform_after":
                first, second = cfg["value"]
                step = cfg["step"]
                if self._curriculum_level >= step:
                    sampled_result[key] = uniform(first, second)
                else:
                    if 'default' in cfg:
                        sampled_result[key] = cfg['default']
                    else:
                        sampled_result[key] = first
            
            elif sample_type == "uniform_step":
                first, second = cfg["value"]
                first_step, second_step = cfg["step"]
                if self._curriculum_level < first_step:
                    sampled_result[key] = uniform(0, first)
                elif first_step <= self._curriculum_level < second_step:
                    bound = (self._curriculum_level - first_step) / (second_step - first_step) * (second - first) + first
                    sampled_result[key] = uniform(0, bound)
                else:
                    sampled_result[key] = uniform(0, second)
            
            elif sample_type == "uniform_propagate":
                first_step, second_step = cfg["step"]
                max_val1, max_val2 = max(cfg["value"]), min(cfg["value"])
                if self._curriculum_level < first_step:
                    sampled_result[key] = np.random.uniform(0, max_val1)
                elif first_step <= self._curriculum_level < second_step:
                    bound = (self._curriculum_level - first_step) / (second_step - first_step) * (max_val2 - max_val1) + max_val1
                    sampled_result[key] = np.random.uniform(0, bound)
                else:
                    sampled_result[key] = np.random.uniform(0, max_val2)
            
            elif sample_type == "uniform_propagate_from_max":
                first_step, second_step = cfg["step"]
                max_val, min_val = max(cfg["value"]), min(cfg["value"])
                if self._curriculum_level < first_step:
                    sampled_result[key] = max_val
                elif first_step <= self._curriculum_level < second_step:
                    bound = (self._curriculum_level - first_step) / (second_step - first_step) * (min_val - max_val) + max_val
                    sampled_result[key] = uniform(bound, max_val)
                else:
                    sampled_result[key] = uniform(min_val, max_val)
            
            elif sample_type == "triangle_propagate_from_max":
                first_step, second_step = cfg["step"]
                max_val, min_val = max(cfg["value"]), min(cfg["value"])
                if self._curriculum_level < first_step:
                    sampled_result[key] = max_val
                elif first_step <= self._curriculum_level < second_step:
                    bound = (self._curriculum_level - first_step) / (second_step - first_step) * (min_val - max_val) + max_val
                else:
                    bound = min_val

                """
                    y = -x + b, x-intercept = ratio * (max_val - bound) + max_val   
                    x in [bound, max_val], sample ratio proportional to y
                """
                x_int = cfg["ratio"] * (max_val - bound) + max_val
                b = x_int
                def y(x: Union[np.array, float]) -> Union[np.array, float]:
                    return - x + b
                x_vals = np.linspace(bound, max_val, 100)
                y_vals = y(x_vals)
                if np.any(y_vals < 0):
                    print(f"Warning: negative values in the pdf. config = {cfg}")
                y_vals = np.maximum(y_vals, 0)  # Clip negative values due to the fp error
                norm_factor = np.trapz(y_vals, x=x_vals)
                if norm_factor == 0:
                    print("Warning: normalization_factor = 0, falling back to uniform sampling.")
                    sampled_result[key] = np.random.uniform(bound, max_val)
                pdf_vals = y_vals / norm_factor
                cdf_vals = np.cumsum(pdf_vals)
                cdf_vals /= cdf_vals[-1]  # Ensure last value is 1.0
                random_values = np.random.rand()
                sampled_result[key] = np.interp(random_values, cdf_vals, x_vals)
            
            elif sample_type == "normal_edge":
                first, second = cfg["value"]
                max_val, min_val = max(first, second), min(first, second)
                sigma = gauss(0, (max_val - min_val) / 4)
                if sigma > 0:
                    sampled_result[key] = min(min_val + sigma, max_val)
                else:
                    sampled_result[key] = max(max_val + sigma, min_val)
            
            elif sample_type == "diffuse":
                values = cfg["value"]
                steps = cfg["step"]
                assert isinstance(values, list) and isinstance(steps, list), "Both 'value' and 'step' should be lists."
                assert len(values) == 2 and len(steps) == 2, "'value' and 'step' lists must contain exactly 2 elements."
                assert all(earlier <= later for earlier, later in zip(steps, steps[1:])), "'step' list must be sorted in ascending order."
                max_val, min_val = max(values), min(values)
                mean = (max_val + min_val) / 2
                if self._curriculum_level < steps[0]:
                    sampled_result[key] = mean
                elif self._curriculum_level >= steps[1]:
                    sampled_result[key] = uniform(min_val, max_val)
                else:
                    ratio = (self._curriculum_level - steps[0]) / (steps[1] - steps[0])
                    sigma = (max_val - min_val) * ratio
                    sampled_result[key] = np.clip(gauss(mean, sigma), min_val, max_val)
            
            elif sample_type == "diffuse_mix":
                values = cfg["value"]
                steps = cfg["step"]
                assert isinstance(values, list) and isinstance(steps, list), "Both 'value' and 'step' should be lists."
                assert len(values) == 2 and len(steps) == 2, "'value' and 'step' lists must contain exactly 2 elements."
                assert all(earlier <= later for earlier, later in zip(steps, steps[1:])), "'step' list must be sorted in ascending order."

                max_val, min_val = max(values), min(values)
                mean = (max_val + min_val) / 2
                if self._curriculum_level < steps[0]:
                    sampled_result[key] = mean
                elif self._curriculum_level >= steps[1]:
                    sampled_result[key] = uniform(min_val, max_val)
                else:
                    ratio = (self._curriculum_level - steps[0]) / (steps[1] - steps[0])
                    sigma = (max_val - min_val) / 2 * ratio
                    sample_from_extreme = None
                    sampled_sigma = gauss(0, sigma)
                    if sampled_sigma > 0:
                        sample_from_extreme = min(min_val + sampled_sigma, max_val)
                    else:
                        sample_from_extreme = max(max_val + sampled_sigma, min_val)
                    sampled = np.random.choice([gauss(mean, sigma), sample_from_extreme])
                    sampled_result[key] = np.clip(sampled, min_val, max_val)
            
            elif sample_type == "diffuse_and_reverse":
                values = cfg["value"]
                steps = cfg["step"]
                assert isinstance(values, list) and isinstance(steps, list), "Both 'value' and 'step' should be lists."
                assert len(values) == 2 and len(steps) == 3, "'value' and 'step' lists must contain exactly 2 and 3 elements."
                assert all(earlier <= later for earlier, later in zip(steps, steps[1:])), "'step' list must be sorted in ascending order."

                max_val, min_val = max(values), min(values)
                mean = (max_val + min_val) / 2
                if self._curriculum_level < steps[0]:
                    sampled_result[key] = mean
                elif self._curriculum_level <= steps[1]:  # diffusing
                    ratio = (self._curriculum_level - steps[0]) / (steps[1] - steps[0])
                    sigma = (max_val - min_val) / 2 * ratio
                    sampled_result[key] = np.clip(gauss(mean, sigma), min_val, max_val)
                elif self._curriculum_level <= steps[2]:  # reversing
                    ratio = (self._curriculum_level - steps[1]) / (steps[2] - steps[1])
                    sigma = (max_val - min_val) / 2 * ratio
                    sigma = gauss(0, sigma)
                    if sigma > 0:
                        sampled_result[key] = min(min_val + sigma, max_val)
                    else:
                        sampled_result[key] = max(max_val + sigma, min_val)
                else:
                    sampled_result[key] = uniform(min_val, max_val)
            
            elif sample_type == "categorical":
                values = list(cfg["value"])
                if "start" in cfg.keys():
                    start = cfg["start"]
                else:
                    start = 0
                if self._curriculum_level < start:
                    sampled_result[key] = values[0]
                sampled_result[key] = np.random.choice(values)
            
            elif sample_type == "linear":
                values = cfg["value"]
                steps = cfg["step"]
                assert isinstance(values, list) and isinstance(steps, list), "Both 'value' and 'step' should be lists."
                assert len(values) == len(steps), "'value' and 'step' lists must be of the same length."
                assert all(earlier <= later for earlier, later in zip(steps, steps[1:])), "'step' list must be sorted in ascending order."

                # Handle the case where curr_lev is before the first step
                if self._curriculum_level < steps[0]:
                    sampled_result[key] = values[0]
                elif self._curriculum_level >= steps[-1]:
                    sampled_result[key] = values[-1]
                else:
                    # Iterate through steps to find the correct interval
                    for i in range(1, len(steps)):
                        if steps[i-1] <= self._curriculum_level < steps[i]:
                            # Perform linear interpolation between values[i-1] and values[i]
                            ratio = (self._curriculum_level - steps[i-1]) / (steps[i] - steps[i-1])
                            interpolated_value = values[i-1] + ratio * (values[i] - values[i-1])
                            sampled_result[key] = interpolated_value
            
            elif sample_type == "categorical_paired":
                if "start" in cfg.keys():
                    start = cfg["start"]
                else:
                    start = 0
                if self._curriculum_level < start:
                    paired_values = cfg["default"]
                else:
                    paired_values = random.choice(cfg["value"])
                assert len(cfg["paired_to"]) + 1 == len(paired_values), f"paired Parameter {key} has {len(cfg['paired_to']) + 1} values but {len(paired_values)} values are given."
                sampled_result[key] = paired_values[0]
                for jdx, paired_key in enumerate(cfg["paired_to"]):
                    sampled_result[paired_key] = paired_values[jdx + 1]
            
            elif sample_type == "diagonal_paired":
                paired_values = []
                if "start" in cfg.keys():
                    start = cfg["start"]
                else:
                    start = 0
                if self._curriculum_level < start:
                    paired_values = cfg["default"]
                else:
                    ratio = np.random.rand()
                    for ranges in cfg['value']:
                        paired_values.append(ranges[0] + ratio * (ranges[1] - ranges[0]))
                assert len(cfg["paired_to"]) + 1 == len(paired_values), f"paired Parameter {key} has {len(cfg['paired_to']) + 1} values but {len(paired_values)} values are given."
                sampled_result[key] = paired_values[0]
                for jdx, paired_key in enumerate(cfg["paired_to"]):
                    sampled_result[paired_key] = paired_values[jdx + 1]
            
            elif sample_type == "diagonal_margin_paired":
                paired_values = []
                if "start" in cfg.keys():
                    start = cfg["start"]
                else:
                    start = 0
                if self._curriculum_level < start:
                    paired_values = cfg["default"]
                else:
                    # Get the diagonal ratio (position along the diagonal)
                    diagonal_ratio = np.random.rand()
                    
                    # Get the margin size (default. to 0.2 if not specified)
                    margin = cfg.get("margin", 0.2)
                    
                    # For each parameter range, calculate the value with both ratios
                    for ranges in cfg['value']:
                        # Calculate the diagonal point
                        diagonal_value = ranges[0] + diagonal_ratio * (ranges[1] - ranges[0])
                        
                        # Sample a random distance from the diagonal (-margin to +margin)
                        # This is a separate random value from the diagonal position
                        margin_offset = (np.random.rand() * 2 - 1) * margin
                        final_value = np.clip(diagonal_value + margin_offset, ranges[0], ranges[1])
                        
                        paired_values.append(final_value)
                
                assert len(cfg["paired_to"]) + 1 == len(paired_values), f"paired Parameter {key} has {len(cfg['paired_to']) + 1} values but {len(paired_values)} values are given."
                sampled_result[key] = paired_values[0]
                for jdx, paired_key in enumerate(cfg["paired_to"]):
                    sampled_result[paired_key] = paired_values[jdx + 1]
            
            elif sample_type == "paired":
                assert key in sampled_result.keys(), f"paired Parameter {key} is not sampled yet."
            
            else:
                print(f"Unsupported sampling type {cfg['type']}")
                sampled_result[key] = 0
        return sampled_result
    
    def update_param(self):
        self._current_params = self._sample_param()
        self._env.SetParam(self._current_params)

    def reset(self):
        # Record the previous episode if adaptive sampling is enabled
        if not self._use_muscle or self._tuple_accum > self._muscle_tuple_bound:
            self.update_param()
            self._tuple_accum = 0
        if self._adaptive_sampling and self._current_episode["length"] > 0:
            self._episode_stats.append(self._current_episode.copy())
            self._current_episode = {
                "params": self._current_params,
                "length": 0,
                "reward": 0
            }
            
        self._env.Reset()
        obs = self._env.GetState()
        return obs

    def step(self, action: np.ndarray):
        if self._use_muscle:
            self._tuple_accum += 1
        self._env.SetAction(action)
        mts_for_supervise_learning = None

        if self._use_muscle:
            rand_idx = choice(range(self.num_simulation_per_control))
            for i in range(self.num_simulation_per_control):
                mts = self._env.GetMuscleTuple()

                with torch.no_grad():
                    mts = [mts.JtA, mts.JtA_reduced, mts.tau_active]
                    mts = [mts[0].astype(np.float32), mts[1].astype(np.float32), mts[2].astype(np.float32)]
                    JtA_nan = np.isnan(np.sum(mts[0]))
                    JtA_reduced_nan = np.isnan(np.sum(mts[1]))
                    tau_active_nan = np.isnan(np.sum(mts[2]))
                    if JtA_nan or JtA_reduced_nan or tau_active_nan:
                        print(f"[ERROR] nan in muscle tuple: JtA_nan={JtA_nan}, JtA_reduced_nan={JtA_reduced_nan}, tau_active_nan={tau_active_nan}, tuple_accum={self._tuple_accum}")
                        break
                    if i == rand_idx:
                        # mts_for_supervise_learning = copy.deepcopy(mts)
                        mts_for_supervise_learning = mts
                    activations = self.muscle_model.no_grad_forward(mts[1], mts[2]).cpu().numpy()
                self._env.SetActivationLevels(activations)
                self._env.Step()
        else:
            self._env.StepsAtOnce()

        if self._use_muscle:
            self._muscle_tuples.append(mts_for_supervise_learning)

        obs = self._env.GetState()
        end = self._env.IsEndOfEpisode()
        if end == 1:
            done = True
            terminated = True
            truncated = False
        elif end == 2:
            done = True
            terminated = False
            truncated = True
        else:
            done = False
            terminated = False
            truncated = False
        reward = self._env.GetReward()
        self._append_reward_map()
        
        # Update episode tracking for adaptive sampling
        if self._adaptive_sampling:
            self._current_episode["length"] += 1
            self._current_episode["reward"] += reward
        
        info = {
            "terminated": terminated,
            "truncated": truncated
        }
        return obs, reward, done, info

    def load_muscle_model_weights(self, weights):
        assert self._use_muscle, "Requested muscle model weights but muscle is not used"
        if type(weights) == dict:
            self.muscle_model.load_state_dict(weights)
        else:
            weights = convert_to_torch_tensor(ray.get(weights))
            self.muscle_model.load_state_dict(weights)

    def get_muscle_tuples(self):
        assert self._use_muscle, "Requested muscle tuples but muscle is not used"
        mt = np.array(self._muscle_tuples, dtype=object)
        self._muscle_tuples.clear()
        if mt.ndim == 2:
            expected_columns = 3
            if mt.shape[1] < expected_columns:
                raise ValueError(f"Expected mt to have at least {expected_columns} columns, but got {mt.shape[1]}")
            res = [np.stack(mt[:, 0]), np.stack(mt[:, 1]), np.stack(mt[:, 2])]
        elif mt.ndim == 1:
            expected_elements = 3
            if len(mt) == expected_elements:
                res = [np.array(mt[0]), np.array(mt[1]), np.array(mt[2])]
            elif len(mt) % expected_elements == 0:
                mt_reshaped = mt.reshape(-1, expected_elements)
                res = [np.stack(mt_reshaped[:, 0]), np.stack(mt_reshaped[:, 1]), np.stack(mt_reshaped[:, 2])]
            else:   
                raise ValueError(f"1D mt array length {len(mt)} is not compatible with expected elements per tuple ({expected_elements})")
        else:
            raise ValueError(f"Unsupported mt array dimensionality: {mt.ndim}")
        return res

    def clear_muscle_tuples(self):
        assert self._use_muscle, "Requested muscle tuples but muscle is not used"
        self._muscle_tuples.clear()

    def get_reward_tuples(self) -> Dict[str, float]:
        rs = {
            key: np.mean(self._reward_tuples[key], dtype=object) if self._reward_tuples[key] else 0
            for key in self._reward_tuples
        }
        self._reward_tuples.clear()
        return rs

    def set_curriculum_level(self, level):
        self._curriculum_level = level

    def get_curriculum_level(self):
        return self._curriculum_level

    def _append_reward_map(self):
        for key, val in self._env.GetRewardMap().items():
            self._reward_tuples.setdefault(key, []).append(val)

    @property
    def use_muscle(self):
        return self._use_muscle

    @property
    def get_exo_state(self):
        return self._env.GetStateExo()

    @property
    def hparam_cfg(self):
        return self._hparam_cfg

    def get_episode_stats(self):
        """Get episode statistics for adaptive sampling"""
        if not self._adaptive_sampling:
            print("Warning: get_episode_stats called but adaptive sampling is not enabled")
            return []
        
        stats = self._episode_stats.copy()
        self._episode_stats.clear()
        return stats

    def set_adaptive_weights(self, weights: Dict[str, np.ndarray]):
        """Set sampling weights for adaptive sampling
        
        Args:
            weights: Dictionary of parameter name -> sampling weights
        """
        if not self._adaptive_sampling:
            print("Warning: set_adaptive_weights called but adaptive sampling is not enabled")
            return
        
        self._adaptive_weights = weights.copy()
