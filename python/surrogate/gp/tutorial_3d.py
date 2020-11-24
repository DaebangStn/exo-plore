import torch
import gpytorch
import plotext as pltxt
from python.plot.util import *
from python.surrogate.gp.gp import SimpleGP, KISSGP
from python.surrogate.gp.interpolate import KNNInterpolator
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan


NUM_TRAIN = 4
NUM_TEST = 100
RANGE = (0, 5)
DEVICE = 'cpu'

def main():
    xs = torch.linspace(RANGE[0], RANGE[1], NUM_TRAIN).to(DEVICE)
    xs_test = torch.linspace(RANGE[0], RANGE[1], NUM_TEST).to(DEVICE)
    ys = torch.linspace(RANGE[0], RANGE[1], NUM_TRAIN).to(DEVICE)
    ys_test = torch.linspace(RANGE[0], RANGE[1], NUM_TEST).to(DEVICE)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    grid_x_test, grid_y_test = torch.meshgrid(xs_test, ys_test, indexing="ij")
    train_x = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    test_x = torch.stack([grid_x_test.reshape(-1), grid_y_test.reshape(-1)], dim=-1)
    true_z = (train_x[:, 0] ** 2 + 4 * train_x[:, 1] ** 2)
    train_z = true_z + 1 * torch.randn(train_x.size(0)).to(DEVICE)
    
    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-7))
    likelihood.noise = 1e-6
    likelihood.noise_covar.raw_noise.requires_grad_(False)
    # model = SimpleGP(train_x, train_z, likelihood, input_dim=2)
    model = KISSGP(train_x, train_z, likelihood)
    
    model = model.to(DEVICE).train()
    likelihood = likelihood.to(DEVICE).train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    losses = []
    
    tq = tqdm(range(500), desc="Training", ncols=80)
    for _ in tq:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_z)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        tq.set_postfix(loss=losses[-1])
    
    pltxt.plot(losses)
    pltxt.plotsize(80, 15)
    pltxt.show()
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
        mean = pred.mean.reshape(NUM_TEST, NUM_TEST)
        lower, upper = pred.confidence_region()
        
        pred_mse = model(train_x)
        pred_mse_mean = pred_mse.mean
        mse = (pred_mse_mean - train_z) ** 2
        mse = mse.mean().cpu().numpy()
        
    axes = set_axes(plt, 2, use_3d=True)
    ax = axes[0]
    ax.plot_surface(grid_x_test.cpu().numpy(), grid_y_test.cpu().numpy(), mean.cpu().numpy(), color='r', label='gp', alpha=0.2)
    ax.scatter(train_x[:, 0].cpu().numpy(), train_x[:, 1].cpu().numpy(), train_z.cpu().numpy(), label='data')
    ax.set_title(f"MSE: {mse.item():.4f}")
    print(f"MSE: {mse.item():.4f}")
    
    ax = axes[1]
    model = KNNInterpolator(train_x, train_z)
    pred = model(test_x).reshape(NUM_TEST, NUM_TEST)
    
    ax.plot_surface(grid_x_test.cpu().numpy(), grid_y_test.cpu().numpy(), pred.cpu().numpy(), color='r', label='knn', alpha=0.2)
    ax.scatter(train_x[:, 0].cpu().numpy(), train_x[:, 1].cpu().numpy(), train_z.cpu().numpy(), label='data')
    mse = (model(train_x) - true_z) ** 2
    mse = mse.mean().cpu().numpy()
    ax.set_title(f"MSE: {mse.item():.4f}")
    print(f"MSE: {mse.item():.4f}")
    
    set_plot(plt, screen_idx=0, fullscreen=False)
    plt.show()


if __name__ == "__main__":
    main()