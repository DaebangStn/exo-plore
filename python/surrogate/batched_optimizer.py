import torch
import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Union
from python.surrogate.preprocess import gt_from_exp_name
from python.surrogate.util import *
import plotext as pltxt
from tqdm import trange
try:
    import gpytorch
except ImportError:
    gpytorch = None

class Optimizer:
    def __init__(
        self,
        ckpt: str,
        epoch: int = -1,
        maximize: bool = False,
        reg_input: Dict[str, float] = None,
        reg_batch: float = 0,
        # reg_batch: float = 0.05,
        # reg_batch: float = 2,
        # reg_batch: float = 5,
        reg_batch_threshold: float = 0.0,
        # reg_batch_threshold: float = 0.001,
    ):
        if "gp" in ckpt:
            from python.surrogate.gp.module import load_from_ckpt
            self._model = load_from_ckpt(ckpt, epoch).eval()
        else:
            from python.surrogate.nn.module import load_from_ckpt
            self._model = load_from_ckpt(ckpt, epoch).eval()

        in_cols = self._model.input_col

        self._device = self._model.device
        self._maximize = maximize

        # regularisation weights
        if reg_input and len(reg_input) > 0:
            assert all(k in self._model.input_col for k in reg_input), \
                f"Regularize keys must be in input columns, got {list(reg_input)}"
            self._reg_indices = torch.tensor([in_cols.index(k) for k in reg_input.keys()], device=self._device)
            self._reg_weights = torch.tensor(list(reg_input.values()), device=self._device)
        else:
            self._reg_indices = None
            self._reg_weights = None
        self._reg_batch = reg_batch
        self._reg_batch_threshold = reg_batch_threshold
        # parameter bounds from ground-truth data
        gt, stat = gt_from_exp_name(
            self._model.exp_name,
            self._model.input_col,
            self._model.output_col,
            self._model.transform,
            is_ckpt=False
        )
        self._param = {}
        self._param_min = {}
        self._param_max = {}
        for col in self._model.input_col:
            mn = gt.select(pl.col(col).min()).collect().item()
            mx = gt.select(pl.col(col).max()).collect().item()
            self._param[col] = (mn, mx)
            mean = stat.filter(pl.col("describe") == "mean")[col].item() 
            std = stat.filter(pl.col("describe") == "std")[col].item()
            self._param_min[col] = mn * std + mean
            self._param_max[col] = mx * std + mean
        
        self._low = torch.tensor([self._param[f][0] for f in in_cols], device=self._device)
        self._high = torch.tensor([self._param[f][1] for f in in_cols], device=self._device)

    def run(
        self,
        opt_field: str,
        fixed_fields: torch.Tensor,
        *,
        lr: float = 5e-2,
        scheduler: bool = True,
        trial_size: int = 256,
        max_iter: int = 500,
        reg_1st_order: bool = True,
        reg_2nd_order: bool = False,
        reg_window: bool = False
    ) -> List[Dict[str, float]]:
        """
        Adam-based batched optimisation over self._trial random starts.

        Args
        ----
        field_name   : name of the model output to optimize
        fixed_fields : (N, n_fields) or (n_fields,) tensor of denormalized inputs; NaN for free dims.
                       If 2D, performs N separate optimizations.
        lr           : Adam learning rate
        trial_size   : number of optimizations to run in parallel
        max_iter     : number of Adam steps
        reg_1st_order: whether to use first-order regularization
        reg_2nd_order: whether to use second-order regularization
        reg_window   : whether to use windowed regularization
        Returns
        -------
        results      : list of dicts, where each dict is field→denormalized best input & objective value
        """
        in_cols = self._model.input_col
        out_cols = self._model.output_col
        n = len(in_cols)

        is_batched_problem = fixed_fields.dim() == 2
        if is_batched_problem:
            problem_batch_size = fixed_fields.shape[0]
            fixed_fields_sample = fixed_fields[0]
        else:
            problem_batch_size = 1
            fixed_fields_sample = fixed_fields
            fixed_fields = fixed_fields.unsqueeze(0) # make it (1, n)
        fixed_fields = self.model.normalize_tensor(fixed_fields, in_cols)

        # identify target idx, fixed vs free indices
        obj_idx = out_cols.index(opt_field)
        fixed_idx = torch.nonzero(~torch.isnan(fixed_fields_sample)).reshape(-1).tolist()
        free_idx  = [i for i in range(n) if i not in fixed_idx]

        # quick eval if nothing to optimize
        if not free_idx:
            with torch.no_grad():
                y = self._model(
                    fixed_fields.to(self._device, self._model.dtype)
                )[:, obj_idx] # y is (N,)

            # Denormalize and return a list of results
            results = []
            for i in range(problem_batch_size):
                x_dict = {f: fixed_fields[i, j].item() for j, f in enumerate(in_cols)}
                best_full = self.model.denormalize(x_dict)
                val_norm = y[i].item()
                val = self.model.denormalize({opt_field: val_norm})[opt_field]
                best_full[opt_field] = val
                results.append(best_full)
            return results

        # bounds in normalized space
        low_free  = self._low[free_idx]
        high_free = self._high[free_idx]

        # batched raw parameters
        B = trial_size
        raw = torch.randn(problem_batch_size, B, len(free_idx), device=self._device, requires_grad=True)
        optim = torch.optim.Adam([raw], lr=lr)
        if scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='min', factor=0.5, patience=500, cooldown=100, verbose=True, 
                threshold=5, threshold_mode='abs', min_lr=1e-5
                )
        else:
            scheduler = None
            
        # expand fixed_fields to batch for trials
        fixed_b = fixed_fields.to(self._device).unsqueeze(1).expand(-1, B, -1) # (N, B, n)
        
        pbar = trange(max_iter, desc="Optimising", unit="it", ncols=100)
        loss_history = []
        reg_loss_history = []
        lr_history = []
        for ep in pbar:
            optim.zero_grad()

            # re-param + assemble
            x_free = low_free + (high_free - low_free) * torch.sigmoid(raw)
            x_full = fixed_b.clone()
            x_full[:, :, free_idx] = x_free

            # model forward
            model_input = x_full.reshape(problem_batch_size * B, n)
            if gpytorch is not None:
                with gpytorch.settings.fast_pred_var(), \
                    gpytorch.settings.max_root_decomposition_size(30):
                    y = self._model(model_input)[:, obj_idx].reshape(problem_batch_size, B)
            else:
                y = self._model(model_input)[:, obj_idx].reshape(problem_batch_size, B)
            loss = (-y if self._maximize else y).sum()

            # vectorized regularization
            if self._reg_indices is not None:
                reg_terms = (x_full[:, :, self._reg_indices] * self._reg_weights).sum()
                loss += reg_terms
            
            if self._reg_batch > 0.0:
                reg_loss = 0
                # compute the difference between the current and previous batch
                if reg_1st_order:
                    x_prev = x_full[:-1, ...]
                    x_curr = x_full[1:, ...]
                    x_diff = x_curr - x_prev
                    reg_loss += x_diff.square().sum()
                if reg_2nd_order:
                    x_prev2 = x_full[:-2, ...]
                    x_prev = x_full[1:-1, ...]
                    x_curr = x_full[2:, ...]
                    # x_diff1 = x_prev - x_prev2
                    # x_diff2 = x_curr - x_prev
                    x_diff = x_curr - 2 * x_prev + x_prev2
                    reg_loss += x_diff.square().sum()
                if reg_window:
                    x_prev5 = x_full[5:-5, ...]
                    x_curr = x_full[10:, ...]
                    x_diff = abs(x_curr - x_prev5)
                    reg_loss += x_diff.sum() * 0.1
                reg_loss = reg_loss * self._reg_batch
                loss += reg_loss
                
            # loss
            loss_value = loss.item()
            if ep % 100 == 0:
                loss_history.append(loss_value)
                if self._reg_batch > 0.0:
                    reg_loss_history.append(reg_loss.item())
                if scheduler:
                    lr_history.append(optim.param_groups[0]['lr'])
            loss.backward()
            optim.step()
            
            if scheduler:
                scheduler.step(loss_value)
                # print(f"[debug] best={scheduler.best} last={scheduler.last_epoch}")

            # update the bar's postfix
            pbar.set_postfix(loss=f"{loss_value:.2f}", lr=f"{optim.param_groups[0]['lr']:.6f}")

        
        loss_min = min(loss_history)
        
        # Plot loss convergence
        pltxt.clear_figure()
        pltxt.theme("pro")
        pltxt.plot(loss_history, marker="dot", color="blue")
        pltxt.title("Loss Convergence")
        pltxt.xlabel("Iteration")
        pltxt.ylabel("Loss")
        if loss_min < 0.0:
            pltxt.ylim(loss_min, loss_min/2)
        else:
            pltxt.ylim(loss_min, 2 * loss_min)
        pltxt.xticks([0, max_iter // 4, max_iter // 2, 3 * max_iter // 4, max_iter])
        pltxt.plotsize(80, 15)
        pltxt.show()   
        
        # Plot learning rate if scheduler is used
        if scheduler:
            pltxt.clear_figure()
            pltxt.theme("pro")
            pltxt.plot(lr_history, marker="dot", color="red")
            pltxt.title("Learning Rate Schedule")
            pltxt.xlabel("Iteration")
            pltxt.ylabel("Learning Rate")
            pltxt.xticks([0, max_iter // 4, max_iter // 2, 3 * max_iter // 4, max_iter])
            pltxt.plotsize(80, 15)
            pltxt.show()
            
        # Plot regularization loss
        if self._reg_batch > 0.0:
            pltxt.clear_figure()
            pltxt.theme("pro")
            pltxt.plot(reg_loss_history, marker="dot", color="green")
            pltxt.title("Regularization Loss")
            pltxt.xlabel("Iteration")
            pltxt.ylabel("Reg Loss")
            reg_loss_min = min(reg_loss_history)
            pltxt.ylim(reg_loss_min, reg_loss_min * 2)
            pltxt.xticks([0, max_iter // 4, max_iter // 2, 3 * max_iter // 4, max_iter])
            pltxt.plotsize(80, 15)
            pltxt.show()

        # select best after training
        with torch.no_grad():
            x_free = low_free + (high_free - low_free) * torch.sigmoid(raw)
            x_full = fixed_b.clone()
            x_full[:, :, free_idx] = x_free
            
            model_input = x_full.reshape(problem_batch_size * B, n)
            y = self._model(model_input)[:, obj_idx]
            y = y.reshape(problem_batch_size, B)

            # vectorized regularization
            if self._reg_indices is not None:
                reg_terms = (x_full[:, :, self._reg_indices] * self._reg_weights).sum(dim=-1)
                y = y + reg_terms
            
            if self._maximize:
                best_y, best_idx_in_trial = torch.max(y, dim=1)
            else:
                best_y, best_idx_in_trial = torch.min(y, dim=1)

            idx_for_gather = best_idx_in_trial.reshape(problem_batch_size, 1, 1).expand(-1, -1, n)
            best_vec = torch.gather(x_full, 1, idx_for_gather).squeeze(1)

        # denormalize and return
        results = []
        for i in range(problem_batch_size):
            x_dict = {f: best_vec[i, j].item() for j, f in enumerate(in_cols)}
            denormalized_x = self.model.denormalize(x_dict)
            
            best_val_norm = best_y[i].item()
            denormalized_y = self.model.denormalize({opt_field: best_val_norm})
            denormalized_x[opt_field] = denormalized_y[opt_field]
            results.append(denormalized_x)

        return results
    
    def run_with_constraint(
        self,
        opt_field: str,
        const_field: str,
        const_values: torch.Tensor,
        *,
        const_reg: float = 1000.0,
        lr: float = 5e-2,
        scheduler: bool = True,
        trial_size: int = 256,
        max_iter: int = 20000,
        cherry_pick: bool = False
    ) -> List[Dict[str, float]]:
        """
        Optimize the objective function with a constraint.

        Args:
            opt_field (str): name of the model output to optimize
            const_field (str): name of the model output to constraint
            const_values (torch.Tensor): (N) or (0) tensor of raw constraint values; will be normalized internally
            lr (float, optional): Adam learning rate
            trial_size (int, optional): number of optimizations to run in parallel
            max_iter (int, optional): number of Adam steps
            cherry_pick (bool, optional): whether to cherry pick the best solution from the batch
        Returns:
            List[Dict[str, float]]: list of dicts, where each dict is field→denormalized best input & objective value
        """
        in_cols = self._model.input_col
        out_cols = self._model.output_col

        is_batched_problem = const_values.dim() == 1
        if is_batched_problem:
            problem_batch_size = const_values.shape[0]
        else:
            problem_batch_size = 1
            
        obj_idx = out_cols.index(opt_field)
        const_idx = out_cols.index(const_field)
        
        B = trial_size
        raw = torch.randn(problem_batch_size, B, len(in_cols), device=self._device, requires_grad=True)
        optim = torch.optim.Adam([raw], lr=lr)
        if scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='min', factor=0.5, patience=500, cooldown=100, verbose=True, 
                threshold=1e-3, threshold_mode='rel', min_lr=1e-5
                )
        else:
            scheduler = None
        
        pbar = trange(max_iter, desc="Optimising", unit="it", ncols=100)
        loss_history = []
        const_loss_history = []
        lr_history = []
        
        for ep in pbar:
            optim.zero_grad()
            
            x = self._low + (self._high - self._low) * torch.sigmoid(raw)
            y = self._model(x)
            # loss = 0
            loss = (-y[:, obj_idx] if self._maximize else y[:, obj_idx]).sum()
            
            if not cherry_pick and self._reg_batch > 0.0:
                x_prev = x[:-1, ...]
                x_curr = x[1:, ...]
                x_diff = x_curr - x_prev
                x_diff_mask = torch.abs(x_diff) > self._reg_batch_threshold
                x_masked_diff = (x_diff * x_diff_mask).abs().sum()
                loss += x_masked_diff * self._reg_batch
            
            const_loss_value = 0.0
            if const_reg > 0.0:
                const_loss = (y[:, :, const_idx] - const_values.unsqueeze(1)).abs().sum()
                loss += const_loss * const_reg
                const_loss_value = const_loss.item()

            # loss
            loss_value = loss.item()
            if ep % 20 == 0:
                loss_history.append(loss_value)
                const_loss_history.append(const_loss_value)
                lr_history.append(optim.param_groups[0]['lr'])
            loss.backward()
            optim.step()
            if scheduler:
                scheduler.step(loss_value)
            # Enhanced progress bar with constraint loss
            postfix = {"loss": f"{loss_value:.2f}", "lr": f"{optim.param_groups[0]['lr']:.6f}"}
            if const_reg > 0.0:
                postfix["const_loss"] = f"{const_loss_value:.2f}"
            pbar.set_postfix(**postfix)
        
        loss_min = min(loss_history)
        pltxt.clear_figure()
        pltxt.theme("pro")
        pltxt.plot(loss_history, marker="dot", color="blue")

        # Plot constraint loss if it exists
        if const_loss_history and max(const_loss_history) > 0:
            # Normalize constraint loss to same scale as main loss for visibility
            const_max = max(const_loss_history)
            loss_max = max(loss_history)
            if const_max > 0 and loss_max > 0:
                const_loss_norm = [v * (loss_max / const_max) for v in const_loss_history]
                pltxt.plot(const_loss_norm, marker="dot", color="green")

        if scheduler:
            lr_norm = [v * (max(loss_history) / max(lr_history)) for v in lr_history]
            pltxt.plot(lr_norm, marker="dot", color="red")

        pltxt.title("Loss Convergence")
        pltxt.xlabel("Iteration")
        pltxt.ylabel("Loss")
        pltxt.ylim(loss_min, 2 * abs(loss_min))
        pltxt.xticks([0, max_iter // 4, max_iter // 2, 3 * max_iter // 4, max_iter])
        pltxt.plotsize(80, 15)
        pltxt.show()

        # Enhanced legend with constraint loss
        legend_parts = ["Loss(blue)"]
        if const_loss_history and max(const_loss_history) > 0:
            legend_parts.append("Constraint Loss(green)")
        if scheduler:
            legend_parts.append("Learning Rate(red)")
        print(f"Legend: {', '.join(legend_parts)}")

        # Save loss histories for debugging/analysis
        self.last_loss_history = loss_history
        self.last_const_loss_history = const_loss_history
        self.last_lr_history = lr_history

        # select best after training
        with torch.no_grad():
            x = self._low + (self._high - self._low) * torch.sigmoid(raw)
            y = self._model(x)
            loss = (-y[:, :, obj_idx] if self._maximize else y[:, :, obj_idx])
            
            if cherry_pick:
                
                if const_reg > 0.0:
                    const_loss = (y[:, :, const_idx] - const_values.unsqueeze(1)).abs()
                    loss += const_loss * const_reg
                    
                # Select best across B dimension (dim=1) for each problem in the batch
                if self._maximize:
                    _, best_idx_in_trial = torch.max(loss, dim=1)
                else:
                    _, best_idx_in_trial = torch.min(loss, dim=1)
                
                # Extract the best x values using the selected indices
                # best_idx_in_trial has shape (problem_batch_size,)
                # We need to select from x which has shape (problem_batch_size, B, len(in_cols))
                batch_indices = torch.arange(problem_batch_size, device=self._device)
                best_x = x[batch_indices, best_idx_in_trial]  # Shape: (problem_batch_size, len(in_cols))
                best_y_vals = y[batch_indices, best_idx_in_trial]  # Shape: (problem_batch_size, len(out_cols))
            else:
                # Get the minimum loss across all trials for each problem
                _, best_idx_in_trial = torch.min(loss.sum(dim=0), dim=0)
                best_x = x[:, best_idx_in_trial, :]
                best_y_vals = y[:, best_idx_in_trial, :]

            # Denormalize the best solutions
            results = []
            for i in range(problem_batch_size):
                best_x_dict = {}
                for j, col in enumerate(in_cols):
                    best_x_dict[col] = best_x[i, j].item()
                
                # Add the objective and constraint values to the result
                for j, col in enumerate(out_cols):
                    best_x_dict[col] = best_y_vals[i, j].item()
                
                # Denormalize the input values
                best_x_denorm = self._model.denormalize(best_x_dict)
                results.append(best_x_denorm)
            
            return results
    
    @property
    def model(self):
        return self._model

    @property
    def param_range(self) -> Dict[str, Tuple[float, float]]:
        out = {}
        for field in self._model.input_col:
            out[field] = (self._param_min[field], self._param_max[field])
        return out
