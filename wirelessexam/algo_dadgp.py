# -*- coding: utf-8 -*-
"""
DADGP 核心算法。
"""

import copy
import math

import numpy as np
import torch
import tqdm

from common import IndexedTensorDataset
from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LR_GAMMA,
    MAX_GRAD_NORM,
    META_LEARNING_RATE,
    META_LR_GAMMA,
    WEIGHT_INIT,
)


class DADGP:
    """通过双层优化动态学习任务权重。"""

    def __init__(
        self,
        model,
        device,
        train_tasks,
        pri_tasks,
        weight_init=WEIGHT_INIT,
        sample_weights=None,
    ):
        self.model = model
        self.model_ = copy.deepcopy(model)
        self.device = device
        self.meta_weights = torch.tensor(
            [weight_init] * len(train_tasks), requires_grad=True, device=device
        )
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks
        self.train_task_ids = list(train_tasks.keys())
        self.loss_fn = None
        self.sample_weights = sample_weights

    def get_normalized_weights(self):
        return torch.softmax(self.meta_weights, dim=0)

    @staticmethod
    def _clip_raw_gradients(gradients, max_norm):
        """裁剪 torch.autograd.grad 返回的梯度张量。"""
        valid_gradients = [gradient for gradient in gradients if gradient is not None]
        if not valid_gradients:
            return tuple(gradients)

        total_norm = torch.linalg.vector_norm(
            torch.stack([gradient.detach().norm(2) for gradient in valid_gradients]),
            ord=2,
        )
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef.item() >= 1.0:
            return tuple(gradients)
        return tuple(
            None if gradient is None else gradient * clip_coef
            for gradient in gradients
        )

    def _get_adam_virtual_state(self, param, optimizer_state):
        state = optimizer_state.get(param, {})
        if len(state) == 0:
            return torch.zeros_like(param), torch.zeros_like(param), 0
        return state["exp_avg"].clone(), state["exp_avg_sq"].clone(), state["step"]

    def _simulate_adam_step(
        self,
        param,
        grad,
        virtual_exp_avg,
        virtual_exp_avg_sq,
        step,
        optimizer_params,
        alpha,
    ):
        beta1, beta2, eps, weight_decay = optimizer_params
        grad_with_decay = grad.add(param, alpha=weight_decay)
        virtual_exp_avg.mul_(beta1).add_(grad_with_decay, alpha=1 - beta1)
        virtual_exp_avg_sq.mul_(beta2).addcmul_(
            grad_with_decay, grad_with_decay, value=1 - beta2
        )

        step += 1
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = alpha / bias_correction1
        denom = (virtual_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        return param - step_size * (virtual_exp_avg / denom)

    def virtual_step(self, train_x, train_y, train_indices, alpha, model_optim):
        train_pred = self.model(train_x)
        train_loss_list = self.model_fit(
            self.model, train_pred, train_y, train_indices, is_val=False
        )
        normalized_weights = self.get_normalized_weights()
        loss = self.loss_fn.combine_task_losses(
            self.model,
            train_loss_list,
            normalized_weights,
            is_val=False,
        )

        gradients = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
        gradients = self._clip_raw_gradients(gradients, MAX_GRAD_NORM)

        group = model_optim.param_groups[0]
        optimizer_params = (
            group["betas"][0],
            group["betas"][1],
            group["eps"],
            group["weight_decay"],
        )

        with torch.no_grad():
            for param, virtual_param, grad in zip(
                self.model.parameters(), self.model_.parameters(), gradients
            ):
                if grad is None:
                    virtual_param.copy_(param)
                    continue

                exp_avg, exp_avg_sq, step = self._get_adam_virtual_state(
                    param, model_optim.state
                )
                virtual_param.copy_(
                    self._simulate_adam_step(
                        param,
                        grad,
                        exp_avg,
                        exp_avg_sq,
                        step,
                        optimizer_params,
                        alpha,
                    )
                )

    def _get_primary_task_weights(self):
        weights = torch.tensor(
            [1.0 if task_id in self.pri_tasks else 0.0 for task_id in self.train_tasks],
            device=self.device,
            dtype=self.meta_weights.dtype,
        )
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return torch.ones_like(weights) / max(weights.numel(), 1)
        return weights / weight_sum

    def _finite_difference_step(
        self, d_model, train_x, train_y, train_indices, eps, direction=1
    ):
        with torch.no_grad():
            for param, direction_grad in zip(self.model.parameters(), d_model):
                if direction_grad is not None:
                    param.add_(direction_grad, alpha=direction * eps)

        train_pred = self.model(train_x)
        train_loss_list = self.model_fit(
            self.model, train_pred, train_y, train_indices, is_val=False
        )
        normalized_weights = self.get_normalized_weights()
        loss = self.loss_fn.combine_task_losses(
            self.model,
            train_loss_list,
            normalized_weights,
            is_val=False,
        )
        return torch.autograd.grad(loss, self.meta_weights)[0]

    def unrolled_backward(
        self,
        train_x,
        train_y,
        train_indices,
        val_x,
        val_y,
        val_indices,
        alpha,
        model_optim,
    ):
        self.virtual_step(train_x, train_y, train_indices, alpha, model_optim)

        primary_weights = self._get_primary_task_weights()
        val_pred = self.model_(val_x)
        val_loss_list = self.model_fit(
            self.model_, val_pred, val_y, val_indices, is_val=True
        )
        loss = self.loss_fn.combine_task_losses(
            self.model_,
            val_loss_list,
            primary_weights,
            is_val=True,
        )

        model_weights_ = tuple(self.model_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        d_model = self._clip_raw_gradients(d_model, MAX_GRAD_NORM)

        hessian = self.compute_hessian(d_model, train_x, train_y, train_indices)
        hessian = self._clip_raw_gradients(hessian, MAX_GRAD_NORM)

        with torch.no_grad():
            if hessian and hessian[0] is not None:
                self.meta_weights.grad = -alpha * hessian[0]
            elif self.meta_weights.grad is not None:
                self.meta_weights.grad.zero_()

        return val_loss_list, loss.detach()

    def compute_hessian(self, d_model, train_x, train_y, train_indices):
        d_model_list = [gradient for gradient in d_model if gradient is not None]
        if not d_model_list:
            return [torch.zeros_like(self.meta_weights)]

        norm = torch.cat([gradient.reshape(-1) for gradient in d_model_list]).norm()
        eps = 0.01 / (norm + 1e-8)

        d_weight_positive = self._finite_difference_step(
            d_model, train_x, train_y, train_indices, eps, direction=1
        )
        d_weight_negative = self._finite_difference_step(
            d_model, train_x, train_y, train_indices, eps, direction=-2
        )

        with torch.no_grad():
            for param, direction_grad in zip(self.model.parameters(), d_model):
                if direction_grad is not None:
                    param.add_(direction_grad, alpha=eps)

        return [(d_weight_positive - d_weight_negative) / (2.0 * eps)]

    def model_fit(self, model, pred, targets, indices, is_val=False):
        return self.loss_fn.task_losses(
            model,
            pred,
            targets,
            self.train_task_ids,
            indices,
            is_val=is_val,
        )


def run_training_loop(model, dadgp, datasets, num_epochs):
    """执行 DADGP 训练循环。"""
    train_data, val_data = datasets["train"], datasets["val"]

    train_dataset = IndexedTensorDataset(train_data["x"], train_data["y"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    val_dataset = IndexedTensorDataset(val_data["x"], val_data["y"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False
    )
    val_x_full, val_y_full, val_indices_full = next(iter(val_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    meta_optimizer = torch.optim.Adam([dadgp.meta_weights], lr=META_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
    meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        meta_optimizer, gamma=META_LR_GAMMA
    )

    history = {
        "weights": [],
        "losses": [],
        "val_meta_loss": [],
        "val_task_losses": [],
        "meta_grad": [],
        "meta_grad_sign": [],
        "step_meta_grad": [],
        "step_val_meta_loss": [],
        "step_val_task_losses": [],
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        num_batches = 0
        last_val_meta = None
        last_val_losses = None
        last_meta_grad = None

        epochs_iter = tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        )
        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            optimizer.zero_grad()
            output = model(train_x_batch)
            task_losses = dadgp.model_fit(
                model, output, train_y_batch, train_indices, is_val=False
            )
            normalized_weights = dadgp.get_normalized_weights()
            total_loss = dadgp.loss_fn.combine_task_losses(
                model,
                task_losses,
                normalized_weights,
                is_val=False,
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            meta_optimizer.zero_grad()
            val_loss_list, val_meta_obj = dadgp.unrolled_backward(
                train_x_batch,
                train_y_batch,
                train_indices,
                val_x_full,
                val_y_full,
                val_indices_full,
                optimizer.param_groups[0]["lr"],
                optimizer,
            )

            if dadgp.meta_weights.grad is not None:
                meta_grad_snapshot = (
                    dadgp.meta_weights.grad.detach().cpu().numpy().copy()
                )
            else:
                meta_grad_snapshot = None
            meta_optimizer.step()

            with torch.no_grad():
                epoch_total_loss += total_loss.item()
                last_val_meta = val_meta_obj.item()
                last_val_losses = [loss.item() for loss in val_loss_list]
                last_meta_grad = meta_grad_snapshot

            history["weights"].append(
                dadgp.get_normalized_weights().detach().cpu().numpy().copy()
            )
            history["step_meta_grad"].append(
                None if meta_grad_snapshot is None else meta_grad_snapshot.copy()
            )
            history["step_val_meta_loss"].append(last_val_meta)
            history["step_val_task_losses"].append(
                None if last_val_losses is None else list(last_val_losses)
            )
            epochs_iter.set_postfix(loss=total_loss.item())
            num_batches += 1

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        history["losses"].append(avg_epoch_loss)
        history["val_meta_loss"].append(last_val_meta)
        history["val_task_losses"].append(last_val_losses)

        if last_meta_grad is not None:
            history["meta_grad"].append(last_meta_grad.copy())
            history["meta_grad_sign"].append(np.sign(last_meta_grad).copy())
        else:
            history["meta_grad"].append(None)
            history["meta_grad_sign"].append(None)

        scheduler.step()
        meta_scheduler.step()

        val_losses_str = (
            ", ".join(f"{value:.4f}" for value in last_val_losses)
            if last_val_losses is not None
            else "N/A"
        )
        grad_str = (
            np.array2string(last_meta_grad, precision=4, separator=", ")
            if last_meta_grad is not None
            else "N/A"
        )
        grad_sign_str = (
            np.array2string(np.sign(last_meta_grad), separator=", ")
            if last_meta_grad is not None
            else "N/A"
        )
        print(
            f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}, "
            f"Current LR: {optimizer.param_groups[0]['lr']}, "
            f"Val Meta: {last_val_meta if last_val_meta is not None else float('nan'):.4f}, "
            f"Val Tasks: [{val_losses_str}], Meta Grad: {grad_str}, Sign: {grad_sign_str}"
        )

    return history
