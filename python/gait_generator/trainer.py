from python.gait_generator.policy import CustomPPOTorchPolicy
from python.gait_generator.muscle_learner import MuscleLearner
from python.gait_generator.adaptive_sampling import ParameterBinUniform, ParameterBinDiagonal
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.sac import SAC
from ray.rllib.algorithms.sac.sac_torch_policy import SACTorchPolicy
from python.gait_generator.util import *
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.policy import Policy, PolicyID
from typing import Dict, List

class MuscleMixin:
    def setup(self, config):
        super().setup(config)
        self.use_muscle = config['env_config']['use_muscle']
        if self.use_muscle:
            hparam = self.metadata['learning']['hparam']
            hparam['muscle_learner']['batch_size'] = self.config['sgd_minibatch_size']
            config["env_config"]["co_act_indices"] = self.parse_co_activation_config(hparam['muscle_learner']['co_act_idx'])
            self.muscle_learner = MuscleLearner(config["env_config"], hparam['muscle_learner'], get_device())
            self._freeze_muscle = hparam['muscle_learner'].get('freeze', False)

    def step(self):
        # learn pd network
        result = super().step()
        if self.use_muscle:
            self._muscle_learner_step(result)
        return result

    @staticmethod
    def parse_co_activation_config(path: str) -> List[List[int]]:
        co_act_idx_path = PROJECT_ROOT / path
        co_act_indices = []
        if co_act_idx_path.exists():
            print(f"Loading co-activation indices from {co_act_idx_path}")
            with open(co_act_idx_path, 'r') as f:
                co_act_indices = yaml.safe_load(f)
            co_act_indices = [v for k, v in co_act_indices.items() if len(v) > 1]
        return co_act_indices

    def _muscle_learner_step(self, result: dict):
        start = time.perf_counter()
        if not self._freeze_muscle:
            muscle_tuples = self.workers.foreach_worker(
                lambda _worker: _worker.foreach_env(lambda _env: _env.get_muscle_tuples())
            )
            total_muscle_tuples = [[],[],[]]
            for worker in muscle_tuples:
                for env in worker:
                    assert (len(total_muscle_tuples) == len(env))
                    for i in range(len(env)):
                        total_muscle_tuples[i].append(env[i])

            res = []
            for tuples in total_muscle_tuples:
                res.append(np.concatenate(tuples, axis=0))

            loading_time = (time.perf_counter() - start) * 1000
            stats = self.muscle_learner.learn(res)

            distribute_time = time.perf_counter()
            if self.muscle_learner.is_training():
                model_weights = ray.put(self.muscle_learner.get_model_weights(device=torch.device("cpu")))
                self.workers.foreach_worker(lambda _worker: [
                    _worker.foreach_env(lambda _env: _env.load_muscle_model_weights(model_weights))
                ])
            distribute_time = (time.perf_counter() - distribute_time) * 1000
        else:
            self.workers.foreach_worker(lambda _worker: _worker.foreach_env(lambda _env: _env.clear_muscle_tuples()))
            loading_time = (time.perf_counter() - start) * 1000
            stats = {}
            distribute_time = 0
        total_time = (time.perf_counter() - start) * 1000
        result['timers']['muscle'] = stats.pop('time', {})
        result['timers']['muscle']['distribute_time_ms'] = distribute_time
        result['timers']['muscle']['loading_time_ms'] = loading_time
        result['timers']['muscle']['total_ms'] = total_time
        result['info']['learner']['default_policy']['custom_metrics']['muscle_loss'] = stats.pop('loss', {})
        result['info']['learner']['default_policy']['custom_metrics'].update(stats)

    def __getstate__(self):
        state = super().__getstate__()
        if self.use_muscle:
            state["muscle"] = self.muscle_learner.get_state_dict()
            state["muscle_optimizer"] = self.muscle_learner.get_optimizer_weights()
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.use_muscle:
            self.muscle_learner.set_state_dict(state["muscle"])
            if "muscle_optimizer" in state.keys():
                print("Restoring muscle optimizer weights")
                self.muscle_learner.set_optimizer_weights(state["muscle_optimizer"])
            model_weights = self.muscle_learner.get_model_weights(device=torch.device("cpu"))
            self.workers.foreach_worker(lambda worker: [
                worker.foreach_env(lambda env: env.load_muscle_model_weights(model_weights))
            ])

class MetadataMixin:
    def setup(self, config):
        # seed_all(config["seed"])
        self.metadata = config.pop("metadata")
        super().setup(config)

    def step(self):
        result = super().step()
        result['epoch'] = self.iteration
        return result

    def __getstate__(self):
        state = super().__getstate__()
        state["metadata"] = self.metadata
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.callbacks.on_train_result(algorithm=self, result=None)
        if self.metadata['learning'].get('reset_vf_on_restore', False):
            print(f"===================> Resetting value function weight")
            self.workers.foreach_worker(lambda worker: [
                worker.foreach_policy(lambda policy, _: policy.model.reset_value_weight())
            ])
        if self.metadata['learning']['hparam']['modulation'].get('restore', False):
            print(f"===================> Ignore modulation weights")

        # mark the restore done
        self.workers.foreach_worker(lambda worker: [
            worker.foreach_policy(lambda policy, _: policy.model.set_restore_done())
        ])


class CKPTMixin:
    def setup(self, config):
        assert hasattr(self, "metadata"), "metadata is not found in the trainer. Parse metadata in the Algo setup method."
        ckpt_path = config.pop("ckpt_path", None)
        metadata_path = Path(self.logdir) / "metadata.yaml"
        if ckpt_path is not None:
            self.metadata['loaded_from'] = str(ckpt_path)
        with open(metadata_path, "w") as f:
            yaml.dump(self.metadata, f)
        super().setup(config)

    def save_checkpoint(self, _path):
        checkpoint_dir = Path(_path)
        trial_timestamp = Path(self.logdir).name
        with open(checkpoint_dir / f"checkpoint-{self.iteration}_{trial_timestamp}", 'wb') as f:
            pickle.dump(self.__getstate__(), f)
        return _path

    def save_max_checkpoint(self):
        trial_timestamp = Path(self.logdir).name
        name = Path(self.logdir).parent.name
        with open(Path(self._logdir) / f"max_checkpoint_{name}_{trial_timestamp}", 'wb') as f:
            pickle.dump(self.__getstate__(), f)

    def load_checkpoint(self, ckpt_path: Path):
        ckpt_path = PROJECT_ROOT / ckpt_path
        if not ckpt_path.exists():
            raise RuntimeError(f"Checkpoint: {ckpt_path} does not exist!")
        ckpt_path = list(ckpt_path.glob("checkpoint-*"))[-1]
        super().load_checkpoint(ckpt_path)


class AdaptiveSamplingCallback(DefaultCallbacks):
    """Callback to manage adaptive sampling across all workers"""
    
    def _distribute_adaptive_params(self, algorithm):
        sampled_weights = {}
        for key, bin in algorithm.bin_tracker.items():
            weights = bin.sample()
            if weights is not None:
                sampled_weights[key] = weights
        
        algorithm.workers.foreach_worker(lambda worker: 
            worker.foreach_env(lambda env: 
                env.set_adaptive_weights(sampled_weights)
            )
        )
    
    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs):
        # Initialize adaptive sampling if enabled
        algorithm.use_adaptive_sampling = algorithm.metadata['learning'].get('adaptive_sampling', {}).get('enable', False)
        if algorithm.use_adaptive_sampling:
            algorithm.adaptive_sampling_cfg = algorithm.metadata['learning']['adaptive_sampling']
            # Initialize tracking variables
            algorithm.last_bin_update = 0
            algorithm.bin_update_frequency = algorithm.adaptive_sampling_cfg.get('update_frequency', 10)
            algorithm.bin_tracker = {}
            
            for key, cfg in algorithm.metadata['learning']['param'].items():
                if "sample" in cfg and cfg["sample"]["type"].endswith("_adaptive"):
                    if cfg["sample"]["type"] == "uniform_adaptive":
                        algorithm.bin_tracker[key] = ParameterBinUniform(key, cfg["sample"], algorithm.adaptive_sampling_cfg)
                    elif cfg["sample"]["type"] == "diagonal_adaptive":
                        algorithm.bin_tracker[key] = ParameterBinDiagonal(key, cfg["sample"], algorithm.adaptive_sampling_cfg)
                    elif cfg["sample"]["type"] == "pared_adaptive":
                        pass
                    else:
                        assert False, f"Unknown adaptive sampling type: {cfg['sample']['type']}"
                        
            self._distribute_adaptive_params(algorithm)
            print(f"Initialized adaptive sampling for parameters: {list(algorithm.bin_tracker.keys())}")
        else:
            print("Adaptive sampling is not enabled")
            for key, cfg in algorithm.metadata['learning']['param'].items():
                if "sample" in cfg and cfg["sample"]["type"].endswith("_adaptive"):
                    assert False, f"Parameter {key} is using adaptive sampling ({cfg['sample']['type']}), but adaptive sampling is not enabled"

    def on_train_result(self, *, algorithm: "Algorithm", result: dict, **kwargs):
        # Check if adaptive sampling is enabled
        if not algorithm.use_adaptive_sampling:
            return
        
        # Only update bins periodically to reduce overhead
        if algorithm.iteration - algorithm.last_bin_update < algorithm.bin_update_frequency:
            return
        
        algorithm.last_bin_update = algorithm.iteration
        
        # Collect episode statistics from all workers
        episode_stats = algorithm.workers.foreach_worker(lambda worker: 
            worker.foreach_env(lambda env: env.get_episode_stats())
        )
        
        # Process episode statistics and update bin weights
        for worker_stats in episode_stats:
            for env_stats in worker_stats:
                for episode in env_stats:
                    params = episode["params"]
                    length = episode["length"]
                    if params is not None:
                        for _, bin in algorithm.bin_tracker.items():
                            bin.accumulate(params, length)
        
        self._distribute_adaptive_params(algorithm)
                
        # Add statistics to result (flattened structure for better TensorBoard visualization)
        if result is not None:
            result_stats = {}
            for key, bin in algorithm.bin_tracker.items():
                stats = bin.get_statistics()
                result_stats[key] = stats
            result['sampler_results']['custom_metrics']['adaptive_sampling'] = result_stats


class MASSCallback(AdaptiveSamplingCallback):
    """Main callback that inherits adaptive sampling functionality"""
    
    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs):
        # Call parent's init first to set up adaptive sampling
        super().on_algorithm_init(algorithm=algorithm, **kwargs)
        
        # Then do MASS-specific initialization
        hparam = algorithm.metadata['learning']['hparam']
        if hparam.get('normalize_reward', False):
            algorithm.normalize_reward = True
            print(f"===================> Setting reward normalization")
            algorithm.workers.foreach_worker(lambda worker: [
                worker.foreach_policy(lambda policy, _: policy.set_reward_normalization())
            ])
        else:
            algorithm.normalize_reward = False
            
        algorithm.modulation_cfg = hparam['modulation']
        algorithm.value_fn_cfg = hparam['value_fn']
        algorithm.freeze_pd_from = hparam.get('freeze_pd_from', -1)
        algorithm.pretrain_vf = algorithm.value_fn_cfg.get('pretrain_epoch', -1)
        algorithm.is_modulated = False
        algorithm.is_frozen = False
        algorithm.max_reward = 0

    def on_train_result(self, *, algorithm: "Algorithm", result: dict, **kwargs):
        if algorithm.iteration % 10 == 0:
            curriculum_level = algorithm.iteration // 10
            algorithm.workers.foreach_worker(lambda worker: [
                worker.foreach_env(lambda env: env.set_curriculum_level(curriculum_level))
            ])
            if not algorithm.is_modulated and curriculum_level >= algorithm.modulation_cfg['start']:
                print(f"===================> Setting modulation")
                algorithm.workers.foreach_worker(lambda worker: [
                    worker.foreach_policy(lambda policy, _: [
                        policy.model.set_modulation()
                    ])
                ])
                if algorithm.iteration >= algorithm.pretrain_vf:
                    print(f"===================> Defrosting modulation")
                    algorithm.workers.foreach_worker(lambda worker: [
                        worker.foreach_policy(lambda policy, _: [
                            policy.model.freeze_base_actor(),
                            policy.model.defrost_modulation()
                        ])
                    ])
                algorithm.is_modulated = True
            if algorithm.pretrain_vf >= 0 and algorithm.iteration >= algorithm.pretrain_vf:
                algorithm.pretrain_vf = -1
                if algorithm.is_modulated:
                    print(f"===================> Defrosting modulation")
                    algorithm.workers.foreach_worker(lambda worker: [
                        worker.foreach_policy(lambda policy, _: policy.model.defrost_modulation())
                    ])
                else:
                    print(f"===================> Defrosting PD")
                    algorithm.workers.foreach_worker(lambda worker: [
                        worker.foreach_policy(lambda policy, _: policy.model.defrost_base_actor())
                    ])

        buffer = algorithm.workers.foreach_worker(lambda worker: {
            "reward": worker.foreach_env(lambda env: env.get_reward_tuples()),
            "action": worker.foreach_policy(lambda policy, _: policy.model.get_action_magnitude()), 
            "reward_filter_state": worker.foreach_policy(lambda policy, _: policy.reward_filter_state),
            "logstd": worker.foreach_policy(lambda policy, _: policy.model.logstd)
        })
        action_magnitude, rewards, reward_filter_state, logstd = [], [], [], []
        for d in buffer:
            action_magnitude.extend(d["action"])
            rewards.extend(d["reward"])
            if algorithm.normalize_reward:
                reward_filter_state.extend(d["reward_filter_state"])
            logstd.extend(d["logstd"])

        if result is not None:
            result['sampler_results']['custom_metrics']['action'] = mean_dict_list(action_magnitude)
            result['sampler_results']['custom_metrics']['reward'] = mean_dict_list(rewards)
            result['sampler_results']['custom_metrics']['logstd'] = np.mean(logstd)
            if algorithm.normalize_reward:
                result['sampler_results']['custom_metrics']['reward_filter_state'] = mean_dict_list(reward_filter_state)
            result["sampler_results"].pop("hist_stats")
            reward = result['episode_reward_mean']
            if algorithm.max_reward < reward:
                algorithm.max_reward = reward
                if hasattr(algorithm, 'save_max_checkpoint'):
                    algorithm.save_max_checkpoint()
                else:
                    print(f"Since the trainer does not have save_max_checkpoint method, the checkpoint for the reward {reward} is not saved.")

        # Call parent's on_train_result to handle adaptive sampling
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)


class MASSAlgo(MetadataMixin, CKPTMixin, MuscleMixin, PPO):
    @classmethod
    def get_default_config(cls):
        config = super().get_default_config()
        config["gp_lambda"] = 0.0
        return config
    
    def get_default_policy_class(self, config):
        return CustomPPOTorchPolicy
        
    def __setstate__(self, state):
        if self.pretrain_vf >= 0:
            print(f"===================> Pretrain value function until iteration {self.pretrain_vf}")
            self.pretrain_vf += self.iteration
            self.workers.foreach_worker(lambda worker: [
                worker.foreach_policy(lambda policy, _: [
                    policy.model.freeze_base_actor(),
                    policy.model.freeze_modulation()
                ])
            ])
        super().__setstate__(state)
        self.workers.foreach_worker(lambda worker: [
            worker.foreach_env(lambda env: env.update_param())
        ])

class MASSAlgoSAC(MetadataMixin, CKPTMixin, MuscleMixin, SAC):
    def get_default_policy_class(self, config):
        return SACTorchPolicy
