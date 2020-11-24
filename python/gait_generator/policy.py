from ray.rllib.algorithms.ppo.ppo_torch_policy import *
from ray.rllib.evaluation.postprocessing import *
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.filter import MeanStdFilter
import torch
from typing import Dict
import time

class MaxUpdateKLMixin:
    def update_kl(self, sampled_kl):
        if sampled_kl is None:
            return self.kl_coeff

        super().update_kl(sampled_kl)
        if self.kl_coeff > self._max_kl_coeff:
            self.kl_coeff = self._max_kl_coeff
            print(f"KL Coefficient is too high. Setting to {self._max_kl_coeff}")
        return self.kl_coeff


def custom_compute_gae_for_sample_batch(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    """Adds GAE (generalized advantage estimations) to a trajectory.

    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.

    Args:
        policy (Policy): The Policy used to generate the trajectory
            (`sample_batch`)
        sample_batch (SampleBatch): The SampleBatch to postprocess.
        other_agent_batches (Optional[Dict[PolicyID, SampleBatch]]): Optional
            dict of AgentIDs mapping to other agents' trajectory data (from the
            same episode). NOTE: The other agents use the same policy.
        episode (Optional[MultiAgentEpisode]): Optional multi-agent episode
            object in which the agents operated.

    Returns:
        SampleBatch: The postprocessed, modified SampleBatch (or a new one).
    """
    info = sample_batch[SampleBatch.INFOS][-1]
    if isinstance(info, dict):
        terminated = info.get("terminated", False)
    else:
        terminated = False

    # Trajectory is actually complete -> last r=0.0.
    if terminated:
        last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        # Input dict is provided to us automatically via the Model's
        # requirements. It's a single-timestep (last one in trajectory)
        # input_dict.
        # Create an input dict according to the Model's requirements.
        input_dict = sample_batch.get_single_step_input_dict(
            policy.model.view_requirements, index="last"
        )
        last_r = policy._value(**input_dict)

    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config.get("use_critic", True),
    )

    return batch


class CustomPPOTorchPolicy(PPOTorchPolicy):
    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        self._max_kl_coeff = 10
        self._do_normalize = False
        self.reward_filter = None

    def set_reward_normalization(self):
        self._do_normalize = True
        self.reward_filter = MeanStdFilter(
            shape=(),  # scalar shape for rewards
            demean=True,  # subtract mean
            destd=True,  # divide by std
            clip=10.0  # clip normalized values to [-10, 10]
        )
        
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        with torch.no_grad():
            return custom_compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )
   
    @property
    def do_normalize(self):
        return self._do_normalize

    @property
    def reward_filter_state(self) -> Dict[str, float]:
        if not self._do_normalize:
            return {"mean": 0, "std": 0}
        return {"mean": self.reward_filter.rs.mean, "std": self.reward_filter.rs.std}
    
    def normalize_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        if self._do_normalize:
            return self.reward_filter(rewards)
        else:
            return rewards
    
    def stats_fn(self, train_batch):
        stats = super().stats_fn(train_batch)
        stats["gp_loss"] = float(self.model.tower_stats.get("mean_gp_loss", 0.0))
        # stats["gp_loss_time"] = float(self.model.tower_stats.get("gp_loss_time", 0.0))
        # stats["total_loss_time"] = float(self.model.tower_stats.get("total_loss_time", 0.0))
        return stats
    
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """
        
        # Start total timer
        # total_start = time.perf_counter()

        logits, _ = model(train_batch)
        curr_action_dist = dist_class(logits, model)
        reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        curr_action_logp = curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        logp_ratio = torch.exp(curr_action_logp - train_batch[SampleBatch.ACTION_LOGP])
        
        # GP loss timing
        # gp_loss_time = 0.0
        # if True:
        if self.config["gp_lambda"] > 0:
            obs_batch = train_batch[SampleBatch.OBS]
            obs_batch.requires_grad = True
            # gp_start = time.perf_counter()
            _logits, _ = model(train_batch)
            _curr_action_dist = dist_class(_logits, model)
            _curr_action_logp = _curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            grad_log_prob = torch.autograd.grad(_curr_action_logp.sum(), obs_batch, create_graph=True)[0]
            gp_loss = torch.sum(torch.square(grad_log_prob), dim=-1).mean()
            # gp_loss_time = time.perf_counter() - gp_start
        else:
            gp_loss = torch.tensor(0.0, device=logp_ratio.device)
        
        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = 0
            vf_loss_clipped = mean_vf_loss = 0.0

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        total_loss += self.config.get("kl_coeff", 0.0) * mean_kl_loss
        total_loss += self.config.get("gp_lambda", 0.0) * gp_loss

        # End total timer
        # total_time = time.perf_counter() - total_start
        # print(f"Total loss computation time: {total_time:.6f} seconds")
        # print(f"GP loss computation time: {gp_loss_time:.6f} seconds")

        # Optionally, store in model.tower_stats for logging
        # model.tower_stats["gp_loss_time"] = gp_loss_time
        # model.tower_stats["total_loss_time"] = total_time

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        model.tower_stats["mean_gp_loss"] = gp_loss
        return total_loss


    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
        if hasattr(self.model, 'is_frozen') and self.model.is_frozen:
            return {
                "learner_stats": {
                    "cur_lr": 0.0,
                    "total_loss": 0.0,
                    "policy_loss": 0.0,
                    "vf_loss": 0.0,
                    "vf_explained_var": 0.0,
                    "kl": 0.0,
                    "entropy": 0.0,
                    "entropy_coeff": 0.0
                },
                "model": {},
                "custom_metrics": {},
                "num_agent_steps_trained": 0
            }
        else:
            return super(CustomPPOTorchPolicy, self).learn_on_loaded_batch(offset, buffer_index)

