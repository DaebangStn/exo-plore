import torch
import gpytorch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os

from linear_operator import settings as linops
from python.surrogate.gp.module import GPRegression, SVGPModel
from python.surrogate.data import GaitDataModule
from python.surrogate.util import PROJECT_ROOT
from typing import Union

class GPTrainer:
    def __init__(self, model: Union[GPRegression, SVGPModel], likelihood: gpytorch.likelihoods.GaussianLikelihood, 
                 train_data: GaitDataModule, config: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.train_data = train_data
        self.config = config
        
        # AMP configuration
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            print(f"[GP] Using Automatic Mixed Precision (AMP) for training")
        else:
            self.scaler = None
            print(f"[GP] Using full precision training")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        if self.model._gp_type == 'svgp':
            self.mll = gpytorch.mlls.VariationalMarginalLogLikelihood(self.likelihood, self.model, num_data=len(self.train_data.train_dataloader().dataset)).to(self.device)
        else:
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(self.device)
        
        self.log_dir = PROJECT_ROOT / "logs" / self.config['exp_name'] / f"gp_{self.config['version']}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.ckpt_dir = PROJECT_ROOT / "experiment_data/simulation" / self.config['exp_name'] / f"gp_{self.config['version']}"
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(self):
        train_x = self.train_data.train_dataloader().dataset.inputs.to(self.device)
        train_y = self.train_data.train_dataloader().dataset.targets.to(self.device).squeeze(-1)
        
        # Update model's train_x and train_y
        # self.model.set_train_data(train_x, train_y, strict=False)

        epochs = self.config['epochs']
        log_period = self.config.get('log_period', 50)
        save_every_n_epochs = self.config.get('save_every_n_epochs', 5)

        self.model.train()
        self.likelihood.train()
        for i in tqdm(range(epochs), desc="Training GP", ncols=80):
            self.optimizer.zero_grad()

            if self.use_amp:
                # Use autocast for forward pass with AMP
                with autocast():
                    output = self.model(train_x)
                loss = -self.mll(output, train_y).sum()
                
                # Scale the loss and perform backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard full precision training
                output = self.model(train_x)
                loss = -self.mll(output, train_y).sum()
                loss.backward()
                self.optimizer.step()

            if i % log_period == 0:
                self.writer.add_scalar('Loss/train', loss.item(), i)
                # for name, param in self.model.named_parameters():
                #     self.writer.add_histogram(f'Parameters/{name}', param, i)
                self.writer.add_histogram('Lengthscales', self.model.lengthscales, i)
                self.writer.add_scalar('Lengthscale', self.model.lengthscale, i)
                self.writer.add_scalar('Outputscale', self.model.outputscale, i)
                self.writer.add_scalar('Noise', self.likelihood.noise.item(), i)
                
                # Log AMP-specific metrics if using AMP
                if self.use_amp:
                    self.writer.add_scalar('AMP/scale', self.scaler.get_scale(), i)

            if i % save_every_n_epochs == 0 or i == epochs - 1:
                ckpt_path = self.ckpt_dir / f'epoch_{i}.ckpt'
                checkpoint_data = {
                    'epoch': i,
                    'model_state_dict': self.model.state_dict(),
                    'likelihood_state_dict': self.likelihood.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_x': train_x,
                    'train_y': train_y,
                    'hparams': self.model.hparams,
                }
                
                # Save AMP scaler state if using AMP
                if self.use_amp:
                    checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
                    checkpoint_data['use_amp'] = True
                else:
                    checkpoint_data['use_amp'] = False
                
                torch.save(checkpoint_data, ckpt_path)
                
        self.writer.close()
        print(f"Training finished. Checkpoints saved to {self.ckpt_dir}")


def check_device(mod: torch.nn.Module):
    for name, param in mod.named_parameters():
        if param.device.type != 'cuda':
            print(f'{name}: {param.device}')
    
    for name, param in mod.named_buffers():
        if param.device.type != 'cuda':
            print(f'{name}: {param.device}')
