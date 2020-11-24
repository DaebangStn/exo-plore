from python.gait_generator.env import MyEnv
from python.gait_generator.trainer import MASSCallback, SeedManager, MASSAlgo, MASSAlgoSAC
from python.gait_generator.model import PDRayModel
from python.gait_generator.config import CONFIG
from python.gait_generator.util import *

from ray import tune
from ray.tune import CLIReporter
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog


if __name__ == "__main__":
    print(f"Head node on {socket.gethostname()}")
    args = build_args()
    ray.init(address="auto")

    name = args.name
    if name is None:
        try:
            name = Path(args.metadata).stem
        except Exception as e:
            print("Warning, Metadata path is not provided, using 'default' as a ray trial name")
            print(e)
            name = "default"

    config = CONFIG[args.option]
    checkpoint_path = None
    if args.checkpoint is not None:
        config["ckpt_path"] = args.checkpoint
        print(f"Checkpoint from: {checkpoint_path}")
        """
        When restore from ray, PermissionError: [Errno 13] Permission denied: '/tmp/_ray_lockfiles/f03d9e98561ece52e8f8d5327e5a69e5.lock'
        This remedy tends to bypass the lock error.
        """

    print(f"Metadata from: {args.metadata}")
    with open(args.metadata, "r") as f:
        metadata = yaml.safe_load(f)
    metadata['seed'] = args.seed
    # config['seed'] = args.seed
    config["metadata"] = metadata
    seed_manager = SeedManager.remote(args.seed)

    config.update(metadata['learning'].get("override_ppo_config", {}))
    hparam = metadata['learning']['hparam']
    print("Overridden PPO config:")
    print(json.dumps(config, indent=1))
    observation_filter = hparam.get("observation_filter", None)
    if observation_filter is not None:
        config['observation_filter'] = observation_filter

    with MyEnv(args.metadata) as env:
        print("========== Environment ==========")
        print("PD Model: ")
        print("--------------------------------")
        print(f"State dimension     {env.num_state}")
        print(f"Action dimension    {env.num_action}")
        print(f"PD DoFs             {env.pd_dof}")

        config["env_config"]["num_state"] = env.num_state
        config["env_config"]["num_action"] = env.num_action
        config["env_config"]["use_muscle"] = env.use_muscle
        if env.use_muscle:
            config["env_config"]["num_muscles"] = env.num_muscles
            config["env_config"]["num_muscle_dofs"] = env.num_muscle_dofs
            config["env_config"]["num_mcn_dof"] = env.num_mcn_dof
            print("\nMuscle Information")
            print("--------------------------------")
            print(f"Number of muscles      {env.num_muscles}")
            print(f"Muscle Related DoFs    {env.num_muscle_dofs}")
            print(f"MCN DoFs               {env.num_mcn_dof}")
            
        if "model" in config.keys():
            config["model"]["custom_model_config"]["hparam"] = env.hparam_cfg
            config["model"]["custom_model_config"]["exo_state_dim"] = len(env.get_exo_state)
        elif "policy_model_config" in config.keys():
            config["policy_model_config"]["custom_model_config"]["hparam"] = env.hparam_cfg
            config["policy_model_config"]["custom_model_config"]["exo_state_dim"] = len(env.get_exo_state)
        else:
            raise RuntimeError("Model config not found in the config file.")
    print("")
    config["rollout_fragment_length"] = int(config["train_batch_size"] / (config["num_workers"] * config["num_envs_per_worker"]))

    register_env("MyEnv",lambda _: MyEnv(args.metadata,ray.get(seed_manager.get_seed.remote())))
    ModelCatalog.register_custom_model("MyModel", PDRayModel)
    config["callbacks"] = MASSCallback
    algo = MASSAlgoSAC if args.option.endswith("sac") else MASSAlgo

    tune.run(
        algo,
        name=name,
        config=config,
        trial_dirname_creator=(lambda trial: timestamp()),
        local_dir="ray_results",
        checkpoint_freq=metadata['learning'].get("ckpt_freq", 200),
        stop={ "epoch": metadata['learning'].get("train_epoch", 100002) if args.epoch is None else args.epoch},
        progress_reporter=CLIReporter(max_report_frequency=500),
        restore=args.checkpoint,
    )

    ray.shutdown()
