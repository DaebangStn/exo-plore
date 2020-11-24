import copy

CONFIG = dict()

common_config = {
    "env": "MyEnv",
    "env_config": {},
    "framework": "torch",
    "extra_python_environs_for_driver": {},
    "extra_python_environs_for_worker": {},
    "evaluation_config": {},
    "normalize_actions": False,
    "clip_actions": True,
    "batch_mode": "truncate_episodes",
    "observation_filter": "NoFilter",
    "horizon": 10000,
    "grad_clip": None,
    # Device Configuration
    "create_env_on_driver": False,
    "num_cpus_for_driver": 0,
    "num_gpus": 1,
    "num_envs_per_worker": 2,
    "num_cpus_per_worker": 1,
}

CONFIG["ppo"] = copy.deepcopy(common_config)
CONFIG["ppo"].update(
    {
        "model": {
            "custom_model": "MyModel",
            "custom_model_config": {
                "value_function": None,
            },
        },
        "use_critic": True,
        "use_gae": True,
        "shuffle_sequences": True,
        "lambda": 0.99,
        "gamma": 0.99,
        "num_sgd_iter": 4,
        "lr": 5e-5,
        "lr_schedule": [
            [0, 5e-5],
            [1e8, 2e-5],
        ],
        "entropy_coeff": 0.0,
        "entropy_coeff_schedule": None,
        "clip_param": 0.2,
        "vf_clip_param": 100.0,
        "kl_coeff": 0.01,
        "kl_target": 0.01,
    }
)

CONFIG["sac"] = copy.deepcopy(common_config)
CONFIG["sac"].update(
    {
        "policy_model_config": {
            "custom_model": "MyModel",
            "custom_model_config": {
                "value_function": None,
            },
        },
        "twin_q": True,
        "q_model_config": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
        },
        "tau": 0.005,
        "initial_alpha": 1.0,
        "target_entropy": "auto",
        "n_step": 1,
        "replay_buffer_config": {
            "_enable_replay_buffer_api": True,
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": int(1e6),
            # How many steps of the model to sample before learning starts.
            "learning_starts": 1500,
            # If True prioritized replay buffer will be used.
            "prioritized_replay": False,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
            # Whether to compute priorities already on the remote worker side.
            "worker_side_prioritization": False,
        },
        "store_buffer_in_checkpoints": False,
        "optimization": {
            "actor_learning_rate": 1e-4,
            "critic_learning_rate": 1e-4,
            "entropy_learning_rate": 1e-4,
        },
        "train_batch_size": 256,
    }
)

# Large Set (For Cluster)
CONFIG["ppo_large"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_large"].update(
    {
        "train_batch_size": 2048 * 8,
        "sgd_minibatch_size": 512 * 2,
    }
)

# Medium Set (For a node or a PC)
CONFIG["ppo_medium"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_medium"].update(
    {
        "train_batch_size": 32768,
        "sgd_minibatch_size": 1024,
    }
)

# Small Set (For PC)
CONFIG["ppo_small"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_small"].update(
    {
        "train_batch_size": 512,
        "sgd_minibatch_size": 64,
    }
)

# Mini Set (For Test)
CONFIG["mini"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["mini"].update(
    {
        "train_batch_size": 4,
        "sgd_minibatch_size": 2,
        "num_envs_per_worker": 1,
        # "log_level": "DEBUG",
    }
)
CONFIG["mini_sac"] = copy.deepcopy(CONFIG["sac"])
CONFIG["mini_sac"].update(
    {
        "train_batch_size": 4,
        "sgd_minibatch_size": 2,
        "num_envs_per_worker": 1,
        "log_level": "DEBUG",
    }
)

# ===============================Training Configuration For Various Devices=========================================

# Large Set
CONFIG["large_n8"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["large_n8"]["num_workers"] = 128 * 8

CONFIG["large_n4"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["large_n4"]["num_workers"] = 128 * 4

CONFIG["large_n1"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["large_n1"]["num_workers"] = 128

CONFIG["large_pc"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["large_pc"]["num_workers"] = 16

# Medium Set
CONFIG["medium_n8"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["medium_n8"]["num_workers"] = 128 * 8

CONFIG["medium_n4"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["medium_n4"]["num_workers"] = 128 * 4

CONFIG["medium_n3"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["medium_n3"]["num_workers"] = 128 * 3

CONFIG["medium_n2"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["medium_n2"]["num_workers"] = 128 * 2

CONFIG["medium_n1"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["medium_n1"]["num_workers"] = 128

CONFIG["a6000"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["a6000"]["num_workers"] = 96

CONFIG["a6000_half"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["a6000_half"]["num_workers"] = 48

CONFIG["imo"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["imo"]["num_workers"] = 96

CONFIG["medium_pc"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["medium_pc"]["num_workers"] = 32

# Small Set
CONFIG["small_n1"] = copy.deepcopy(CONFIG["ppo_small"])
CONFIG["small_n1"]["num_workers"] = 128 * 1

CONFIG["small_pc1"] = copy.deepcopy(CONFIG["ppo_small"])
CONFIG["small_pc1"]["num_workers"] = 1
CONFIG["small_pc4"] = copy.deepcopy(CONFIG["ppo_small"])
CONFIG["small_pc4"]["num_workers"] = 4
CONFIG["small_pc"] = copy.deepcopy(CONFIG["ppo_small"])
CONFIG["small_pc"]["num_workers"] = 32

# Small Set
CONFIG["mini"]["num_workers"] = 1
CONFIG["mini_sac"]["num_workers"] = 1
