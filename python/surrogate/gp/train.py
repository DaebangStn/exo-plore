from python.surrogate.data import GaitDataModule
from python.surrogate.gp.module import GPRegression, SVGPModel
from python.surrogate.gp.trainer import GPTrainer
from python.surrogate.util import *
import gpytorch
import tempfile


EXP_NAMES = [
]

CONFIG_PATH = "data/surrogate/gp_regression.yaml"

def launch_experiment(exp_idx, total_exp_num):
    exp_name = EXP_NAMES[exp_idx] if EXP_NAMES else "default_exp"
    print(f"\n============== Training experiment: {exp_name} ==============")
    
    # Wait for config file to be available
    config_path = CONFIG_PATH
    wait_time = 0
    while not os.path.exists(config_path):
        if wait_time > 60:  # Wait up to 60 seconds (adjust as needed)
            raise FileNotFoundError(f"File does not exist: '{config_path}'")
        print(f"Waiting for config file: {config_path}")
        time.sleep(2)
        wait_time += 2

    # Copy config to a unique tempfile and parse from there
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='r+') as tmp:
        local_config_path = tmp.name
        shutil.copyfile(config_path, local_config_path)

        # Load config
        with open(local_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update experiment specific configs
        print("config: ", config)
        config["data"]["name"] = exp_name
        if config["model"] is None:
            config["model"] = {}
        config["model"]["name"] = exp_name
        
        # Prepare data
        datamodule = GaitDataModule(**config["data"])
        datamodule.prepare_data()
        datamodule.setup('fit')
        train_dataloader = datamodule.train_dataloader()
        
        # Extract all data from the DataLoader for GP training
        train_x = []
        train_y = []
        for batch in train_dataloader:
            x, y = batch
            train_x.append(x)
            train_y.append(y)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_x = torch.cat(train_x, dim=0).to(device)
        train_y = torch.cat(train_y, dim=0).squeeze(-1).to(device)

        # Initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model_config = config["model"].copy()
        model_config.pop("lr", None) # Remove lr as it's for the trainer

        gp_type = config["model"].get("gp_type", "simple")

        if gp_type == "svgp":
            num_inducing_points = config["model"].get("num_inducing_points", 1000)
            # Randomly select inducing points from train_x
            # Ensure num_inducing_points does not exceed the number of training points
            num_inducing_points = min(num_inducing_points, train_x.shape[0])
            # Randomly select indices for inducing points
            inducing_points_indices = torch.randperm(train_x.shape[0])[:num_inducing_points]
            inducing_points = train_x[inducing_points_indices]

            model = SVGPModel(
                name=exp_name,
                in_lbl=config["data"]["in_lbl"],
                out_lbl=config["data"]["out_lbl"],
                transform=config["data"]["transform"],
                tensor_ds=config["data"].get("tensor_ds", False),
                inducing_points=inducing_points,
                likelihood=likelihood,
                gp_type=gp_type,
                device=device
            )
        else:
            model = GPRegression(
                name=exp_name,
                in_lbl=config["data"]["in_lbl"],
                out_lbl=config["data"]["out_lbl"],
                transform=config["data"]["transform"],
                tensor_ds=config["data"].get("tensor_ds", False),
                train_x=train_x,
                train_y=train_y,
                likelihood=likelihood,
                gp_type=gp_type
            )
        
        # Set a version for logging and checkpointing
        version_str = f"v{get_current_version(exp_name) + 1}"
        model.ckpt_ver = version_str
        
        # Prepare trainer config
        trainer_config = {
            "lr": config["trainer"]["lr"],
            "epochs": config["trainer"]["max_epochs"],
            "exp_name": exp_name,
            "version": version_str,
            "log_period": config["trainer"]["log_every_n_steps"],
            "save_every_n_epochs": config["trainer"]["save_every_n_epochs"]
        }

        # Create and run trainer
        trainer = GPTrainer(model, likelihood, datamodule, trainer_config)
        trainer.train()
        
        print(f"============== Training experiment: {exp_name} completed ==============")

def main():
    total_exp_num = len(EXP_NAMES) if EXP_NAMES else 1
    failed_exp = []
    
    for exp_idx in range(total_exp_num):
        launch_experiment(exp_idx, total_exp_num)
        # try:
            # launch_experiment(exp_idx, total_exp_num)
        # except Exception as e:
            # exp_name = EXP_NAMES[exp_idx] if EXP_NAMES else "default_exp"
            # print(f"Experiment {exp_name} failed: {e}")
            # failed_exp.append(exp_name)

    if len(failed_exp) > 0:
        print(f"Failed experiments:")
        for exp_name in failed_exp:
            print(f"'{exp_name}',")

if __name__ == "__main__":
    main()
