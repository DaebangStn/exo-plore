from python.surrogate.nn.module import Regression
from python.surrogate.nn.data import GaitData
from python.surrogate.log import MemoLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from python.surrogate.util import *


TRAIN_WITH_DS = True

class Trainer:
    def __init__(self, model: Regression, data: GaitData, logger: MemoLogger, ckpt_dir: Path,
                 lr: float = 1e-3, min_lr: float = 1e-4, 
                 l1_weight: float = 0.0, l2_weight: float = 0.0, 
                 grad_weight: float = 0.0, recon_weight: float = 1.0, 
                 recon_delta: float = 0.5, recon_type: str = 'mse', recon_start_epoch: int = 0,
                 log_period: int = 50, max_epochs: int = 1000, use_tensor_ds: bool = False, **kwargs):
        self.model = model
        self.data = data
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.saving_period = 1000
        self.total_loss = 0
        self.recon_delta = recon_delta
        self.recon_type = recon_type
        self.recon_start_epoch = recon_start_epoch
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=100,
            threshold=1e-3,
            threshold_mode='rel',
            min_lr=min_lr,
        )

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.grad_weight = grad_weight
        self.recon_weight = recon_weight
        self.log_period = log_period
        self.epochs = max_epochs + 1
        self.use_tensor_ds = use_tensor_ds
        self.ckpt_dir = ckpt_dir

    def train(self):
        for epoch in tqdm(range(self.epochs), desc="Training", ncols=80):
            self.train_epoch(epoch)
            self.validate(epoch)
            if epoch % self.log_period == 0:
                self.log_learning_rate(epoch)
            if epoch % self.saving_period == 0:
                self.save_checkpoint(epoch)
                
    def save_checkpoint(self, epoch):
        filename = f"{self.ckpt_dir}/ep_{epoch:03d}.ckpt"
        torch.save(self.model.state_dict(), filename)
        self.logger.log_metrics({"checkpoint_saved": epoch}, step=epoch)      
                
    def log_learning_rate(self, epoch):
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.log_metrics({
            "lr": current_lr
        }, step=epoch)
        
    @staticmethod
    def cauchy_loss(input, target, delta=1.0):
        delta *= 1.2
        r = input - target
        loss = 0.5 * delta**2 * torch.log1p((r / delta)**2)
        return loss.mean()

    @staticmethod
    def welsch_loss(input, target, delta=1.0):
        delta *= 1.8
        r = input - target
        loss = 0.5 * delta**2 * (1 - torch.exp(-(r / delta)**2))
        return loss.mean()

    @staticmethod
    def geman_mcclure_loss(input, target, delta=1.0):
        delta *= 1.2
        r = input - target
        loss = 0.5 * r**2 / (1 + (r / delta)**2)
        return loss.mean()

    @staticmethod
    def tukey_biweight_loss(input, target, delta=1.0):
        delta *= 3.5
        r = input - target
        mask = r.abs() <= delta
        loss = torch.zeros_like(r)
        z = (r / delta)**2
        loss[mask] = (delta**2 / 6) * (1 - (1 - z[mask])**3)
        loss[~mask] = delta**2 / 6
        return loss.mean()

    def compute_grad(self, epoch, x, y, mask=None):
        self.optimizer.zero_grad()

        l1_reg = torch.zeros(1, device=self.device)
        l2_reg = torch.zeros(1, device=self.device)
        
        for w in self.model.parameters():
            l1_reg += torch.mean(torch.abs(w))
            l2_reg += torch.mean(torch.square(w))

        x.requires_grad = True
        y_hat = self.model(x)
        
        if self.use_tensor_ds:
            y_hat = y_hat[mask]
            y = y[mask]
        
        grad_input = torch.autograd.grad(y_hat, x, grad_outputs=torch.ones_like(y_hat), create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_input = torch.clamp(grad_input, min=-1e3, max=1e3)
        gp_loss = torch.sum(torch.square(grad_input), dim=-1).mean()

        if self.recon_type == 'mse':
            regression = nn.functional.mse_loss(y_hat, y) 
        elif self.recon_type == 'huber':
            regression = nn.functional.huber_loss(y_hat, y, delta=self.recon_delta)
        elif self.recon_type == 'geman':
            if epoch < self.recon_start_epoch:
                regression = nn.functional.huber_loss(y_hat, y, delta=self.recon_delta)
            else:
                regression = self.geman_mcclure_loss(y_hat, y, delta=self.recon_delta)
        elif self.recon_type == 'cauchy':
            regression = self.cauchy_loss(y_hat, y, delta=self.recon_delta)
        elif self.recon_type == 'welsch':
            if epoch < self.recon_start_epoch:
                regression = nn.functional.huber_loss(y_hat, y, delta=self.recon_delta)
            else:
                regression = self.welsch_loss(y_hat, y, delta=self.recon_delta)
        elif self.recon_type == 'tukey':
            if epoch < self.recon_start_epoch:
                regression = nn.functional.huber_loss(y_hat, y, delta=self.recon_delta)
            else:
                regression = self.tukey_biweight_loss(y_hat, y, delta=self.recon_delta)
        else:
            raise ValueError(f"Invalid regression type: {self.recon_type}")
        # # Consistency loss with σ=0.05, λ=0.5
        # noise = torch.randn_like(x) * 0.01
        # x_noisy = x + noise
        # y_hat_noisy = self.model(x_noisy)
        
        # if self.use_tensor_ds:
        #     y_hat_noisy = y_hat_noisy[mask]
        
        # consistency_loss = nn.functional.mse_loss(y_hat, y_hat_noisy)
        # regression = regression + 0.5 * consistency_loss

        loss = self.recon_weight * regression + self.l1_weight * l1_reg + self.l2_weight * l2_reg + self.grad_weight * gp_loss
        
        loss.backward()
        self.optimizer.step()
        self.total_loss += loss.item()

        if epoch % self.log_period == 0:
            self.logger.log_metrics({
                "loss/regression": regression.item(),
                "loss/l1": l1_reg.item(),
                "loss/l2": l2_reg.item(),
                "loss/grad": gp_loss.item(),
                "loss/train": loss.item(),
                "loss": loss.item()
            }, step=epoch)

        avg_loss = self.total_loss / len(self.data.train_dataloader())
        self.scheduler.step(avg_loss)

    def train_epoch(self, epoch):
        self.model.train()
        self.total_loss = 0
        
        if TRAIN_WITH_DS:
            x = self.data.input_tensor
            y = self.data.target_tensor
            self.compute_grad(epoch, x, y)
            
        else:
            for _, batch in enumerate(self.data.train_dataloader()):
                if self.use_tensor_ds:
                    x, y, mask = batch
                else:
                    x, y = batch
                    mask = None
                self.compute_grad(epoch, x, y, mask)

    def validate(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in self.data.val_dataloader():
                if self.use_tensor_ds:
                    x, y, mask = batch
                    x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                else:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    mask = None

                y_hat = self.model(x)
                if self.use_tensor_ds:
                    y_hat = y_hat[mask]
                    y = y[mask]
                mse = nn.functional.mse_loss(y_hat, y)
                total_val_loss += mse.item()

        avg_val_loss = total_val_loss / len(self.data.val_dataloader())
        if epoch % self.log_period == 0:
            self.logger.log_metrics({"loss/val": avg_val_loss}, step=epoch)
