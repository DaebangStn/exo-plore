from python.surrogate.nn.data import GaitData
from python.surrogate.nn.trainer import Trainer as NNTrainer
from python.surrogate.log import MemoLogger
from python.surrogate.util import PROJECT_ROOT
from python.surrogate.util import *
from python.surrogate.nn.module import Regression
import yaml


CONFIG_PATHS = [
    "data/surrogate/debug.yaml",
    # "data/surrogate/regression_soft05.yaml",
    # "data/surrogate/regression_soft.yaml",
    # "data/surrogate/regression_soft2.yaml",
    # "data/surrogate/regression_mid.yaml",
    # "data/surrogate/regression_mid_mse.yaml",
    # "data/surrogate/regression_hard.yaml",
    # "data/surrogate/selected_regression_soft.yaml"
]

def launch_experiment(exp_name):
    for config_path in CONFIG_PATHS:
        print(f"\n============== Training experiment: {exp_name} with {config_path} ==============")
        
        wait_time = 0
        while not os.path.exists(config_path):
            if wait_time > 60:
                raise FileNotFoundError(f"File does not exist: '{config_path}'")
            print(f"Waiting for config file: {config_path}")
            time.sleep(2)
            wait_time += 2

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        config["data"]["name"] = exp_name
        config["model"]["name"] = exp_name
        
        logger = MemoLogger(name=exp_name, model_type="nn", memo="")
        exp_dir = PROJECT_ROOT / "experiment_data/simulation" / exp_name
        ckpt_dir = exp_dir / f"nn_{logger.version}"
        os.makedirs(ckpt_dir, exist_ok=True)
        shutil.copyfile(config_path, ckpt_dir / "config.yaml")
        
        model_config = config["model"]
        trainer_config = config["trainer"]
        data_config = config["data"]

        # Separate trainer specific args from model args if they are mixed
        model_arg_names = Regression.__init__.__code__.co_varnames
        trainer_specific_args = {k: v for k, v in model_config.items() if k not in model_arg_names}
        for k in trainer_specific_args:
            del model_config[k]
        
        # Pass trainer specific args to trainer
        for k, v in trainer_specific_args.items():
            if k not in trainer_config:
                trainer_config[k] = v

        model_config['in_lbl'] = data_config['in_lbl']
        model_config['out_lbl'] = data_config['out_lbl']
        model_config['transform'] = data_config['transform']

        model = Regression(**model_config)
        datamodule = GaitData(**data_config)
        
        if 'logger' in trainer_config:
            del trainer_config['logger']

        trainer = NNTrainer(
            model=model,
            data=datamodule,
            logger=logger,
            ckpt_dir=ckpt_dir,
            use_tensor_ds=data_config['tensor_ds'],
            **trainer_config
        )
        
        trainer.train()
        print(f"============== Training experiment: {exp_name} with {config_path} completed ==============")
    
    print(f"============== All training experiments for {exp_name} completed ==============")
