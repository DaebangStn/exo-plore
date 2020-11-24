import torch
from tqdm import tqdm
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, GridInterpolationKernel
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
import plotext as pltxt


class SimpleGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim: int=None):
        super(SimpleGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        
        if input_dim is None:
            kernel = RBFKernel()
        else:
            kernel = RBFKernel(ard_num_dims=input_dim)
        self.covar_module = ScaleKernel(kernel)

    def set_train_data(self, train_x, train_y, strict: bool = True):
        self.train_inputs = (train_x,)
        self.train_targets = train_y
        if strict:
            self.set_added_loss_terms(self.added_loss_terms)
            self.set_forward_args(self.forward_args)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def optimize(self, num_steps: int = 100, lr: float = 0.1):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        losses = []
        
        for _ in tqdm(range(num_steps), desc="Optimizing GP", ncols=80):
            optimizer.zero_grad()
            output = self(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        pltxt.clear_figure()
        pltxt.theme("pro")
        pltxt.plot(losses, marker="dot")
        pltxt.title("Loss Convergence")
        pltxt.xlabel("Iteration")
        pltxt.ylabel("Loss")
        pltxt.plotsize(80, 15)
        pltxt.show()
        self.eval()
        self.likelihood.eval()
        


class KISSGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(KISSGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        
        input_dim = train_x.shape[-1]
        base_kernel = RBFKernel(ard_num_dims=input_dim)
        
        grid_size = int(len(train_x) ** (1 / input_dim)) * 10
        grid_bound = [(torch.min(train_x[:, i]), torch.max(train_x[:, i])) for i in range(input_dim)]
        ski_kernel = GridInterpolationKernel(
            base_kernel=base_kernel,
            grid_size=grid_size,
            grid_bounds=grid_bound,
            num_dims=input_dim
        )
        self.covar_module = ScaleKernel(ski_kernel)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
