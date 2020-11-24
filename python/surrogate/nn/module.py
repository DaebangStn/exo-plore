from python.surrogate.util import complete_ckpt_path
from python.surrogate.preprocess import gt_stat_from_exp_name, gt_from_exp_name
from python.surrogate.util import *


COMPUTE_GP = True
BUFFER_LOG = True


class ResidualMLP(nn.Module):
    def __init__(self, layers: List[int], input_dim: int, output_dim: int, activation: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, layers[0])
        self.input_activation = activation
        
        # Hidden layers with residual connections
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))
            
            # Add skip connection if dimensions match, otherwise use a projection
            if layers[i-1] == layers[i]:
                self.skip_connections.append(nn.Identity())
            else:
                self.skip_connections.append(nn.Linear(layers[i-1], layers[i]))
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1], output_dim)
        
        # Activation for hidden layers - create a new instance to avoid sharing
        self.activation = activation.__class__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layer
        x = self.input_activation(self.input_layer(x))
        
        # Hidden layers with residual connections
        for i, (layer, skip) in enumerate(zip(self.layers, self.skip_connections)):
            residual = skip(x)
            x = layer(x)
            if i < len(self.layers) - 1:  # Don't apply activation to last hidden layer
                x = self.activation(x)
            x = x + residual  # Add residual connection
        
        # Output layer
        x = self.output_layer(x)
        return x


class Regression(nn.Module):
    def __init__(self, name: str, in_lbl: List[str], out_lbl: List[str], layers: List[int],
                 transform: Union[str, dict] = 'mean_per_cycle', act_params: Dict[str, Any] = {}, residual: bool = False, **kwargs):
        super().__init__()
        # gt_from_exp_name(name, in_lbl, out_lbl, transform, is_ckpt=False)[0].collect()
        
        self._exp_name = name
        self._ckpt_ver = None
        self._layers = layers
        self._input_col = in_lbl
        self._output_col = out_lbl
        self._transform = transform
        self._residual = residual
        self._gt_stat = gt_stat_from_exp_name(name, in_lbl, out_lbl, transform, is_ckpt=False)
        self._gt_mean_tensor = None
        self._gt_std_tensor = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._build_gt_stat_tensor()
        
        self._act_params = act_params
        # self._activation = nn.SiLU(**act_params)
        self._activation = nn.GELU(**act_params)
        
        if self._residual:
            self._mlp = self._build_residual_mlp(layers, len(in_lbl), len(out_lbl))
        else:
            self._mlp = self._build_sequential_mlp(layers, len(in_lbl), len(out_lbl))
        for w in self._mlp.parameters():
            if len(w.shape) > 1:
                nn.init.xavier_uniform_(w)
        self.to(self._device)

    def _build_sequential_mlp(self, layers: List[int], input_dim: int, output_dim: int) -> nn.Sequential:
        """Build the original sequential MLP"""
        mlp = nn.Sequential()
        for i, layer in enumerate(layers):
            if i == 0:
                mlp.add_module("input", nn.Linear(input_dim, layer))
                mlp.add_module("act0", self._activation)
            else:
                mlp.add_module(f"hidden{i}", nn.Linear(layers[i - 1], layer))
                mlp.add_module(f"act{i}", self._activation if i < len(layers) - 1 else nn.Tanh())
        mlp.add_module("output", nn.Linear(layers[-1], output_dim))
        return mlp

    def _build_residual_mlp(self, layers: List[int], input_dim: int, output_dim: int) -> nn.Module:
        """Build MLP with residual connections"""
        return ResidualMLP(layers, input_dim, output_dim, self._activation)

    def forward(self, x: Union[torch.Tensor, Df]) -> Union[torch.Tensor, Df]:
        if isinstance(x, Df):
            assert set(x.columns) == set(self._input_col), f"Input columns mismatch: {x.columns} != {self._input_col}"
            x = self.normalize(x).to_tensor()
            y = self._mlp(x)
            y = tensor_to_df(y, self._output_col)
            y = self.denormalize(y)
            return y
        else:
            return self._mlp(x)

    def normalize(self, x: Union[Dict[str, float], Lf, Df]) -> Union[Dict[str, float], Lf, Df]:
        is_dict = isinstance(x, dict)
        if is_dict:
            x = Lf([x])
        out = normalize_from_stat(x, self._gt_stat)
        if is_dict:
            out = out.collect().to_dicts()[0]
        return out

    def normalize_tensor(self, x: torch.Tensor, labels: List[str]) -> torch.Tensor:
        # assert len(labels) == x.shape[-1], f"Labels mismatch: len({labels}) != {x.shape[-1]}"
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
        mean_values = []
        std_values = []
        for col in self.cols:
            mean_values.append(self._gt_stat.filter(pl.col("describe") == "mean")[col].item())
            std_values.append(self._gt_stat.filter(pl.col("describe") == "std")[col].item())
        self._gt_mean_tensor = torch.tensor(mean_values, dtype=torch.float32)
        self._gt_std_tensor = torch.tensor(std_values, dtype=torch.float32)

    def state_dict(self):
        model_weight = self._mlp.state_dict()
        model_config = {
            "name": self._exp_name,
            "in_lbl": self._input_col,
            "out_lbl": self._output_col,
            "layers": self._layers,
            "transform": self._transform,
            "act_params": self._act_params,
            "residual": self._residual,
        }
        return {
            "model_weight": model_weight,
            "model_config": model_config,
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._mlp.load_state_dict(state_dict)

    @property
    def input_col(self) -> List[str]:
        return self._input_col

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
    def device(self) -> str:
        return self._device
    
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
        self._device = next(model.parameters()).device
        self._gt_mean_tensor = self._gt_mean_tensor.to(self._device)
        self._gt_std_tensor = self._gt_std_tensor.to(self._device)
        return model

    def disable_param_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def enable_param_grad(self):
        for param in self.parameters():
            param.requires_grad = True

def load_from_ckpt(ckpt_path: str, epoch: int = -1):
    ckpt_full_path = complete_ckpt_path(ckpt_path, epoch)
    state_dict = torch.load(ckpt_full_path)
    model = Regression(**state_dict["model_config"])
    model.load_state_dict(state_dict["model_weight"])
    return model