import gpytorch
from python.surrogate.preprocess import gt_stat_from_exp_name
from python.surrogate.util import *
from python.surrogate.util import complete_ckpt_path
from linear_operator import settings as linops


class GPRegression(gpytorch.models.ExactGP):
    def __init__(self, name: str, in_lbl: List[str], out_lbl: List[str],
                 transform: Union[str, dict] = 'mean_per_cycle', tensor_ds: bool = False,
                 train_x: torch.Tensor = None, train_y: torch.Tensor = None, likelihood: gpytorch.likelihoods.GaussianLikelihood = None,
                 gp_type: str = 'simple'):
        super().__init__(train_x, train_y, likelihood)
        self._exp_name = name
        self._ckpt_ver = None
        self._input_col = in_lbl
        self._output_col = out_lbl
        self._transform = transform
        self._use_tensor_ds = tensor_ds
        self._gp_type = gp_type
        self.save_hyperparameters()
        self._device = None
        self._gt_stat = gt_stat_from_exp_name(name, in_lbl, out_lbl, transform, is_ckpt=False)
        self.register_buffer('_gt_mean_tensor', None)
        self.register_buffer('_gt_std_tensor', None)
        self._build_gt_stat_tensor()

        self.mean_module = gpytorch.means.ConstantMean()
        if self._gp_type == 'simple':
            print(f"[GP] Simple GP with {len(in_lbl)} input dimensions")
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=len(in_lbl)))
        elif self._gp_type == 'kissgp':
            print(f"[GP] KISSGP with {len(in_lbl)} input dimensions")
            self.covar_module = None
        else:
            raise ValueError(f"Unknown gp_type: {self._gp_type}")
        self.set_train_data(train_x, train_y, strict=False)

    def set_train_data(self, train_x, train_y, strict: bool = True):
        self.train_inputs = (train_x,)
        self.train_targets = train_y
        if self._gp_type == 'kissgp' and self.covar_module is None:
            input_dim = train_x.shape[-1]
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim).to(train_x.device)
            base_grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
            grid_size = []
            for col_name in self._input_col:
                gs = np.clip(base_grid_size, 40, 512)
                if col_name == 'Mass':
                    gs = 10
                grid_size.append(gs)
            grid_bound = [(torch.min(train_x[:, i]).item(), torch.max(train_x[:, i]).item()) for i in range(input_dim)]
            grid_bound = [(torch.tensor(b[0], device=train_x.device), torch.tensor(b[1], device=train_x.device)) for b in grid_bound]
            print(f"[KISSGP] field | grid_size | grid_bound")
            for i in range(input_dim):
                print(f"{self._input_col[i]} | {grid_size[i]} | {grid_bound[i][0]:.2f} - {grid_bound[i][1]:.2f}")
            ski_kernel = gpytorch.kernels.GridInterpolationKernel(
                base_kernel=base_kernel,
                grid_size=grid_size,
                grid_bounds=grid_bound,
                num_dims=input_dim
            ).to(train_x.device)
            self.covar_module = gpytorch.kernels.ScaleKernel(ski_kernel)

        if hasattr(self, 'added_loss_terms'):
            if strict:
                self.set_added_loss_terms(self.added_loss_terms)
        if hasattr(self, 'forward_args'):
            if strict:
                self.set_forward_args(self.forward_args)

    def forward(self, x):
        with linops.max_preconditioner_size(0):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def normalize(self, x: Union[Dict[str, float], Lf, Df]) -> Union[Dict[str, float], Lf, Df]:
        is_dict = isinstance(x, dict)
        if is_dict:
            x = Lf([x])
        out = normalize_from_stat(x, self._gt_stat)
        if is_dict:
            out = out.collect().to_dicts()[0]
        return out

    def normalize_tensor(self, x: torch.Tensor, labels: List[str]) -> torch.Tensor:
        assert len(labels) == x.shape[-1], f"Labels mismatch: len({labels}) != {x.shape[-1]}"
        cols = self.cols
        indices = [cols.index(lbl) for lbl in labels]
        return normalize_from_stat_tensor(x, self._gt_mean_tensor[indices], self._gt_std_tensor[indices])

    def denormalize(self, x: Union[Dict[str, float], Lf]) -> Union[Dict[str, float], Lf]:
        is_dict = isinstance(x, dict)
        if is_dict:
            x = Lf([x])
        out = denormalize(x, self._gt_stat)
        if is_dict:
            out = out.collect().to_dicts()[0]
        return out

    def denormalize_tensor(self, x: torch.Tensor, labels: List[str]) -> torch.Tensor:
        assert len(labels) == x.shape[1], f"Labels mismatch: {labels} != {self.cols}"
        cols = self.cols
        indices = [cols.index(lbl) for lbl in labels]
        return denormalize_tensor(x, self._gt_mean_tensor[indices], self._gt_std_tensor[indices])

    def _build_gt_stat_tensor(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mean_values = []
        std_values = []
        for col in self.cols:
            mean_values.append(self._gt_stat.filter(pl.col("describe") == "mean")[col].item())
            std_values.append(self._gt_stat.filter(pl.col("describe") == "std")[col].item())
        self._gt_mean_tensor = torch.tensor(mean_values, dtype=torch.float32).to(device)
        self._gt_std_tensor = torch.tensor(std_values, dtype=torch.float32).to(device)

    @property
    def input_col(self) -> List[str]:
        return self._input_col

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def output_col(self) -> List[str]:
        return self._output_col

    @property
    def cols(self) -> List[str]:
        return self._input_col + self._output_col

    @property
    def exp_name(self) -> str:
        return self._exp_name

    @property
    def ckpt_ver(self) -> str:
        return self._ckpt_ver
    
    @property
    def transform(self) -> str:
        return self._transform
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def gp_type(self) -> str:
        return self._gp_type
    
    @property
    def outputscale(self) -> float:
        return self.covar_module.outputscale.item()
    
    @property
    def lengthscale(self) -> float:
        if self._gp_type == 'kissgp':
            return self.covar_module.base_kernel.base_kernel.lengthscale.mean().item()
        else:
            return self.covar_module.base_kernel.lengthscale.mean().item()
        
    @property
    def lengthscales(self) -> List[float]:
        if self._gp_type == 'kissgp':
            return self.covar_module.base_kernel.base_kernel.lengthscale
        else:
            return self.covar_module.base_kernel.lengthscale

    @ckpt_ver.setter
    def ckpt_ver(self, ver: str):
        if self._ckpt_ver is not None:
            raise ValueError(f"Checkpoint version already set: {self._ckpt_ver}")
        self._ckpt_ver = ver

    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        self._device = args[0] if args else kwargs.get('device', None)
        if self.mean_module is not None:
            self.mean_module.to(*args, **kwargs)
        if self.covar_module is not None:
            self.covar_module.to(*args, **kwargs)
        return model

    def disable_param_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def enable_param_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def save_hyperparameters(self):
        # This is a simplified version for now.
        # In a real scenario, you might want to save specific hyperparameters.
        # For now, we'll just store the ones passed to __init__.
        self._hparams = {
            'name': self._exp_name,
            'in_lbl': self._input_col,
            'out_lbl': self._output_col,
            'transform': self._transform,
            'tensor_ds': self._use_tensor_ds,
            'gp_type': self._gp_type,
        }

    @property
    def hparams(self):
        return self._hparams

class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, name: str, in_lbl: List[str], out_lbl: List[str],
                 transform: Union[str, dict] = 'mean_per_cycle', tensor_ds: bool = False,
                 inducing_points: torch.Tensor = None, likelihood: gpytorch.likelihoods.GaussianLikelihood = None,
                 gp_type: str = 'svgp', device: torch.device = None):
        
        if inducing_points is None:
            raise ValueError("Inducing points must be provided for SVGPModel.")

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-2))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution
        )
        super().__init__(variational_strategy)

        self._exp_name = name
        self._ckpt_ver = None
        self._input_col = in_lbl
        self._output_col = out_lbl
        self._transform = transform
        self._use_tensor_ds = tensor_ds
        self._gp_type = gp_type
        self._device = device
        self.save_hyperparameters()

        self._gt_stat = gt_stat_from_exp_name(name, in_lbl, out_lbl, transform, is_ckpt=False)
        self.register_buffer('_gt_mean_tensor', None)
        self.register_buffer('_gt_std_tensor', None)
        self._build_gt_stat_tensor()

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=len(in_lbl)))

    def set_train_data(self, train_x, train_y, strict: bool = True):
        # For SVGP, train_x and train_y are not directly set on the model like ExactGP
        # Instead, the inducing points are part of the model's parameters.
        # This method might be used to update inducing points or other related parameters
        # if the training data changes, but for now, it can be a no-op or raise an error
        # if it's not expected to be called.
        # Given the context, it's likely that the trainer expects this method to exist.
        # We can make it a no-op for now, or add logic to update inducing points if needed.
        pass

    def forward(self, x):
        with linops.max_preconditioner_size(0):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def normalize(self, x: Union[Dict[str, float], Lf, Df]) -> Union[Dict[str, float], Lf, Df]:
        is_dict = isinstance(x, dict)
        if is_dict:
            x = Lf([x])
        out = normalize_from_stat(x, self._gt_stat)
        if is_dict:
            out = out.collect().to_dicts()[0]
        return out

    def normalize_tensor(self, x: torch.Tensor, labels: List[str]) -> torch.Tensor:
        assert len(labels) == x.shape[-1], f"Labels mismatch: len({labels}) != {x.shape[-1]}"
        cols = self.cols
        indices = [cols.index(lbl) for lbl in labels]
        return normalize_from_stat_tensor(x, self._gt_mean_tensor[indices], self._gt_std_tensor[indices])

    def denormalize(self, x: Union[Dict[str, float], Lf]) -> Union[Dict[str, float], Lf]:
        is_dict = isinstance(x, dict)
        if is_dict:
            x = Lf([x])
        out = denormalize(x, self._gt_stat)
        if is_dict:
            out = out.collect().to_dicts()[0]
        return out

    def denormalize_tensor(self, x: torch.Tensor, labels: List[str]) -> torch.Tensor:
        assert len(labels) == x.shape[1], f"Labels mismatch: {labels} != {self.cols}"
        cols = self.cols
        indices = [cols.index(lbl) for lbl in labels]
        return denormalize_tensor(x, self._gt_mean_tensor[indices], self._gt_std_tensor[indices])

    def _build_gt_stat_tensor(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mean_values = []
        std_values = []
        for col in self.cols:
            mean_values.append(self._gt_stat.filter(pl.col("describe") == "mean")[col].item())
            std_values.append(self._gt_stat.filter(pl.col("describe") == "std")[col].item())
        self._gt_mean_tensor = torch.tensor(mean_values, dtype=torch.float32).to(device)
        self._gt_std_tensor = torch.tensor(std_values, dtype=torch.float32).to(device)

    @property
    def input_col(self) -> List[str]:
        return self._input_col

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def output_col(self) -> List[str]:
        return self._output_col

    @property
    def cols(self) -> List[str]:
        return self._input_col + self._output_col

    @property
    def exp_name(self) -> str:
        return self._exp_name

    @property
    def ckpt_ver(self) -> str:
        return self._ckpt_ver
    
    @property
    def transform(self) -> str:
        return self._transform

    @ckpt_ver.setter
    def ckpt_ver(self, ver: str):
        if self._ckpt_ver is not None:
            raise ValueError(f"Checkpoint version already set: {self._ckpt_ver}")
        self._ckpt_ver = ver

    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        self._device = args[0] if args else kwargs.get('device', None)
        if self.mean_module is not None:
            self.mean_module.to(*args, **kwargs)
        if self.covar_module is not None:
            self.covar_module.to(*args, **kwargs)
        return model

    def disable_param_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def enable_param_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def save_hyperparameters(self):
        # This is a simplified version for now.
        # In a real scenario, you might want to save specific hyperparameters.
        # For now, we'll just store the ones passed to __init__.
        self._hparams = {
            'name': self._exp_name,
            'in_lbl': self._input_col,
            'out_lbl': self._output_col,
            'transform': self._transform,
            'tensor_ds': self._use_tensor_ds,
            'gp_type': self._gp_type,
        }
        
    @property
    def lengthscale(self) -> float:
        return self.covar_module.base_kernel.lengthscale.mean().item()
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def gp_type(self) -> str:
        return self._gp_type
    
    @property
    def outputscale(self) -> float:
        return self.covar_module.outputscale.item()
    
    @property
    def hparams(self):
        return self._hparams

def load_from_ckpt(ckpt: str, epoch: int = -1) -> Union['SVGPModel', 'GPRegression']:
    """
    Load a GP model from checkpoint.
    
    Args:
        ckpt: Path to checkpoint file or experiment name/directory
        epoch: Specific epoch to load (-1 for latest)
    
    Returns:
        Loaded GP model in evaluation mode
    """
    # Get the full checkpoint path
    full_ckpt_path = complete_ckpt_path(ckpt, epoch)
    
    # Load the checkpoint dictionary
    checkpoint = torch.load(full_ckpt_path, map_location='cpu', weights_only=True)
    
    # Extract model and likelihood state_dicts
    model_state_dict = checkpoint['model_state_dict']
    likelihood_state_dict = checkpoint['likelihood_state_dict']
    
    # Extract model hyperparameters
    hparams = checkpoint['hparams']
    
    # Extract train_x and train_y from checkpoint
    train_x = checkpoint['train_x']
    train_y = checkpoint['train_y']
    
    # Create likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.load_state_dict(likelihood_state_dict)

    # Create the appropriate model based on gp_type
    if hparams['gp_type'] == 'svgp':
        # For SVGP, we need to extract the inducing points from the state dict
        inducing_points = model_state_dict['variational_strategy.inducing_points']
        model = SVGPModel(
            name=hparams['name'],
            in_lbl=hparams['in_lbl'],
            out_lbl=hparams['out_lbl'],
            transform=hparams['transform'],
            tensor_ds=hparams['tensor_ds'],
            inducing_points=inducing_points,
            likelihood=likelihood,
            gp_type=hparams.get('gp_type')
        )
    else:
        # For standard GP regression
        model = GPRegression(
            name=hparams['name'],
            in_lbl=hparams['in_lbl'],
            out_lbl=hparams['out_lbl'],
            transform=hparams['transform'],
            tensor_ds=hparams['tensor_ds'],
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            gp_type=hparams.get('gp_type', 'simple')  # Default to 'simple' for backward compatibility
        )
    
    # Load the model state
    model.load_state_dict(model_state_dict)
    
    # Set checkpoint version for tracking
    model.ckpt_ver = full_ckpt_path.name
    
    # Set device and dtype for compatibility
    model.dtype = torch.float32
    
    # Move to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    likelihood = likelihood.to(device)
    
    # Set to evaluation mode
    model.eval()
    likelihood.eval()
    
    print(f"Successfully loaded GP model from {full_ckpt_path}")
    print(f"Model type: {hparams['gp_type']}")
    print(f"Input features: {hparams['in_lbl']}")
    print(f"Output features: {hparams['out_lbl']}")
    print(f"Device: {device}")
    
    return model