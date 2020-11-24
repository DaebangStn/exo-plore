import torch
import gpytorch
import plotext as pltxt
from python.plot.util import *
from python.surrogate.gp.gp import SimpleGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
        
        
def main():
    x = torch.linspace(0, 1, 100)
    train_y = torch.sin(x * (2 * torch.pi)) + torch.randn(x.size()) * 0.2
    
    likelihood = GaussianLikelihood()
    model = SimpleGP(x, train_y, likelihood)
    model.optimize(num_steps=100, lr=0.1)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x))
        mean = pred.mean
        lower, upper = pred.confidence_region()
    
        
    axes = set_axes(plt, 1, use_3d=False)
    ax = axes[0]
    set_plot(plt, screen_idx=0, fullscreen=False)
    ax.plot(x, train_y, '-')
    ax.plot(x, mean, '-')
    ax.fill_between(x, lower, upper, alpha=0.2)
    plt.show()


if __name__ == "__main__":
    main()