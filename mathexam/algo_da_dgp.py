# -*- coding: utf-8 -*-
"""
DA-DGP算法：通过双层优化实现多任务学习中的任务权重自动调整

本文件包含DA-DGP的核心实现：
- DADGP类：元学习任务权重的算法
- run_training_loop：DA-DGP的训练循环
"""

import torch
import tqdm
import copy
import math
import numpy as np

from config import (
    LEARNING_RATE,
    META_LEARNING_RATE,
    LR_GAMMA,
    META_LR_GAMMA,
    MAX_GRAD_NORM,
    BATCH_SIZE,
    WEIGHT_INIT,
)
from common import IndexedTensorDataset, create_loss_function
from models_dgp import MultitaskDeepGP


# ==========================================================================================
# DA-DGP核心类
# ==========================================================================================

class DADGP:
    """DA-DGP: 通过双层优化实现多任务学习中的任务权重自动调整。

    该算法使用双层优化策略：
    - 内层循环：优化模型参数（使用当前任务权重）
    - 外层循环：优化任务权重（使用验证集性能作为元目标）

    核心思想是通过验证集性能来自动调整各任务的重要性权重，
    避免了手动设置权重的困难和主观性。

    Args:
        model: 多任务DGP模型实例
        device: 计算设备（如torch.device("cuda:0")）
        train_tasks: 训练任务字典，格式为{"task_id": weight, ...}
        pri_tasks: 主要任务字典，用于定义元优化目标
        weight_init: 元权重的初始值（softmax前的值）
        sample_weights: 样本权重字典（用于局部优化）

    属性:
        model: 主模型（用于训练）
        model_: 模型的深拷贝（用于计算虚拟步骤）
        meta_weights: 可学习的任务权重参数
        loss_fn: 损失函数
        sample_weights: 样本权重
    """

    def __init__(
        self,
        model,
        device,
        train_tasks,
        pri_tasks,
        weight_init=WEIGHT_INIT,
        sample_weights=None,
    ):
        self.model = model  # 存储主模型
        self.model_ = copy.deepcopy(model)  # 创建深拷贝用于虚拟步骤
        self.device = device

        # 初始化元权重（每个任务一个权重，设置为可求导）
        self.meta_weights = torch.tensor(
            [weight_init] * len(train_tasks), requires_grad=True, device=device
        )

        self.train_tasks = train_tasks  # 存储训练任务
        self.pri_tasks = pri_tasks  # 存储主要任务（用于元目标）
        self.train_task_ids = list(train_tasks.keys())
        self.loss_fn = None  # 损伤函数（将在后续设置）
        self.sample_weights = sample_weights  # 样本权重

    def get_normalized_weights(self):
        """对元权重应用softmax函数，获得归一化的任务权重。

        Returns:
            归一化的任务权重张量，形状为[num_tasks]
        """
        return torch.softmax(self.meta_weights, dim=0)

    def _get_adam_virtual_state(self, param, optimizer_state):
        """获取虚拟的Adam优化器状态（不修改实际优化器状态）。

        Args:
            param: 模型参数
            optimizer_state: 优化器状态字典

        Returns:
            (exp_avg, exp_avg_sq, step): Adam的动量状态
        """
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
        """模拟单步Adam优化过程。

        Args:
            param: 当前参数值
            grad: 参数梯度
            virtual_exp_avg: 一阶动量
            virtual_exp_avg_sq: 二阶动量
            step: 当前步数
            optimizer_params: Adam超参数 (beta1, beta2, eps, weight_decay)
            alpha: 学习率

        Returns:
            更新后的参数值
        """
        beta1, beta2, eps, weight_decay = optimizer_params

        # 应用权重衰减
        grad_with_decay = grad.add(param, alpha=weight_decay)

        # 更新动量估计
        virtual_exp_avg.mul_(beta1).add_(grad_with_decay, alpha=1 - beta1)
        virtual_exp_avg_sq.mul_(beta2).addcmul_(
            grad_with_decay, grad_with_decay, value=1 - beta2
        )

        # 偏差修正
        step += 1
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        # 计算参数更新量
        step_size = alpha / bias_correction1
        denom = (virtual_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        return param - step_size * (virtual_exp_avg / denom)

    def virtual_step(self, train_x, train_y, train_indices, alpha, model_optim):
        """计算展开网络的theta'（虚拟步骤）。

        模拟一个Adam优化步骤，但不修改实际的模型参数，
        只更新虚拟模型self.model_的参数。

        Args:
            train_x: 训练输入
            train_y: 训练目标
            train_indices: 样本索引
            alpha: 学习率
            model_optim: 模型优化器
        """
        # 前向传播并计算损失
        train_pred = self.model(train_x)
        train_loss_list = self.model_fit(
            self.model, train_pred, train_y, train_indices, is_val=False
        )
        normalized_weights = self.get_normalized_weights()
        loss = sum(
            weight * task_loss
            for weight, task_loss in zip(normalized_weights, train_loss_list)
        )

        # 计算梯度
        gradients = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)

        # 梯度裁剪
        grad_list = [g for g in gradients if g is not None]
        if grad_list:
            torch.nn.utils.clip_grad_norm_(grad_list, max_norm=MAX_GRAD_NORM)

        # 提取优化器参数
        group = model_optim.param_groups[0]
        optimizer_params = (
            group["betas"][0],
            group["betas"][1],
            group["eps"],
            group["weight_decay"],
        )

        # 执行虚拟Adam步骤
        with torch.no_grad():
            for p, p_, grad in zip(self.model.parameters(), self.model_.parameters(), gradients):
                if grad is None:
                    p_.copy_(p)
                    continue

                exp_avg, exp_avg_sq, step = self._get_adam_virtual_state(p, model_optim.state)
                p_.copy_(
                    self._simulate_adam_step(p, grad, exp_avg, exp_avg_sq, step, optimizer_params, alpha)
                )

    def _get_primary_task_weights(self):
        """获取主要任务的二进制权重（1.0表示主要任务，0.0表示非主要任务）。

        Returns:
            权重列表，长度为len(train_tasks)
        """
        return [1.0 if task in self.pri_tasks else 0.0 for task in self.train_tasks]

    def _finite_difference_step(self, d_model, train_x, train_y, train_indices, eps, direction=1):
        """执行有限差分步骤并计算损失。

        Args:
            d_model: 模型参数的梯度方向
            train_x: 训练输入
            train_y: 训练目标
            train_indices: 样本索引
            eps: 步长
            direction: 方向（1或-1）

        Returns:
            损失相对于元权重的梯度
        """
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                if d is not None:
                    p.add_(d, alpha=direction * eps)

        train_pred = self.model(train_x)
        train_loss_list = self.model_fit(
            self.model, train_pred, train_y, train_indices, is_val=False
        )
        normalized_weights = self.get_normalized_weights()
        loss = sum(
            weight * task_loss
            for weight, task_loss in zip(normalized_weights, train_loss_list)
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
        """计算展开后的损失，并反向传播梯度以更新元权重。

        这是DA-DGP的核心步骤：
        1. 执行虚拟步骤，得到虚拟模型参数
        2. 在虚拟模型上计算验证损失（元目标）
        3. 通过有限差分近似计算Hessian向量积
        4. 更新元权重的梯度

        Args:
            train_x, train_y, train_indices: 训练数据（用于虚拟步骤）
            val_x, val_y, val_indices: 验证数据（用于元目标）
            alpha: 当前学习率
            model_optim: 模型优化器

        Returns:
            (val_loss_list, val_meta_obj): 验证损失列表和元目标值
        """
        # 执行虚拟步骤
        self.virtual_step(train_x, train_y, train_indices, alpha, model_optim)

        # 计算验证损失（元目标）
        pri_weights = self._get_primary_task_weights()
        val_pred = self.model_(val_x)
        val_loss_list = self.model_fit(
            self.model_, val_pred, val_y, val_indices, is_val=True
        )
        loss = sum(
            weight * task_loss
            for weight, task_loss in zip(pri_weights, val_loss_list)
        )

        # 计算验证损失相对于虚拟模型参数的梯度
        model_weights_ = tuple(self.model_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)

        # 梯度裁剪
        d_model_list = [g for g in d_model if g is not None]
        if d_model_list:
            torch.nn.utils.clip_grad_norm_(d_model_list, max_norm=MAX_GRAD_NORM)

        # 使用有限差分计算Hessian向量积
        hessian = self.compute_hessian(d_model, train_x, train_y, train_indices)

        # 更新元权重的梯度
        if hessian and hessian[0] is not None:
            torch.nn.utils.clip_grad_norm_(hessian, max_norm=MAX_GRAD_NORM)

        with torch.no_grad():
            if hessian and hessian[0] is not None:
                self.meta_weights.grad = -alpha * hessian[0]
            else:
                if self.meta_weights.grad is not None:
                    self.meta_weights.grad.zero_()

        return val_loss_list, loss.detach()

    def compute_hessian(self, d_model, train_x, train_y, train_indices):
        """使用有限差分近似计算Hessian向量积。

        Args:
            d_model: 模型参数的梯度方向
            train_x, train_y, train_indices: 训练数据

        Returns:
            Hessian向量积（相对于元权重）
        """
        d_model_list = [w for w in d_model if w is not None]
        if not d_model_list:
            return [torch.zeros_like(self.meta_weights)]

        # 计算步长eps
        norm = torch.cat([w.reshape(-1) for w in d_model_list]).norm()
        eps = 0.01 / (norm + 1e-8)

        # 正向差分
        d_weight_p = self._finite_difference_step(d_model, train_x, train_y, train_indices, eps, direction=1)

        # 反向差分
        d_weight_n = self._finite_difference_step(d_model, train_x, train_y, train_indices, eps, direction=-2)

        # 恢复原始参数
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                if d is not None:
                    p.add_(d, alpha=eps)

        # 中心差分公式
        return [(d_weight_p - d_weight_n) / (2.0 * eps)]

    def model_fit(self, model, pred, targets, indices, is_val=False):
        """定义特定于任务的损失计算。

        Args:
            pred: 模型预测分布
            targets: 目标值张量
            indices: 样本索引
            is_val: 是否为验证阶段

        Returns:
            各任务的损失列表
        """
        return self.loss_fn.task_losses(
            model,
            pred,
            targets,
            self.train_task_ids,
            indices,
            is_val=is_val,
        )


# ==========================================================================================
# DA-DGP训练循环
# ==========================================================================================

def run_training_loop(model, dadgp, datasets, num_epochs):
    """执行带有DA-DGP优化的训练循环。

    该函数实现双层优化：
    - 内层：优化模型参数
    - 外层：优化任务权重（通过验证集元目标）

    Args:
        model: 多任务DGP模型
        dadgp: DADGP实例
        datasets: 数据集字典，包含"train"和"val"
        num_epochs: 训练轮数

    Returns:
        history: 训练历史字典，包含：
            - "weights": 各步的任务权重记录
            - "losses": 各epoch的平均损失
            - "val_meta_loss": 各epoch的验证元目标
            - "val_task_losses": 各epoch各任务的验证损失
            - "meta_grad": 各epoch元权重的梯度
            - "meta_grad_sign": 各epoch元权重梯度的符号
    """
    train_data, val_data = datasets["train"], datasets["val"]

    # 设置数据加载器
    train_dataset = IndexedTensorDataset(train_data["x"], train_data["y"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 使用完整训练集进行虚拟步骤
    full_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=False
    )
    train_x_full, train_y_full, train_indices_full = next(iter(full_train_loader))

    # 验证集用于元目标
    val_dataset = IndexedTensorDataset(val_data["x"], val_data["y"])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    val_x_full, val_y_full, val_indices_full = next(iter(val_loader))

    # 设置优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    meta_optimizer = torch.optim.Adam([dadgp.meta_weights], lr=META_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
    meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_optimizer, gamma=META_LR_GAMMA)

    # 训练历史记录
    history = {
        "weights": [],
        "losses": [],
        "val_meta_loss": [],
        "val_task_losses": [],
        "meta_grad": [],
        "meta_grad_sign": [],
    }

    for epoch in range(num_epochs):
        model.train()
        epochs_iter = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        epoch_total_loss, num_batches = 0.0, 0
        last_val_meta = None
        last_val_losses = None
        last_meta_grad = None

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            # --- 模型参数更新步骤（内层循环） ---
            optimizer.zero_grad()
            output = model(train_x_batch)
            task_losses = dadgp.model_fit(
                model, output, train_y_batch, train_indices, is_val=False
            )
            normalized_weights = dadgp.get_normalized_weights()
            total_loss = sum(
                weight * task_loss
                for weight, task_loss in zip(normalized_weights, task_losses)
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            # --- 元更新步骤（外层循环） ---
            meta_optimizer.zero_grad()
            val_loss_list, val_meta_obj = dadgp.unrolled_backward(
                train_x_full, train_y_full, train_indices_full,
                val_x_full, val_y_full, val_indices_full,
                optimizer.param_groups[0]["lr"], optimizer,
            )

            if dadgp.meta_weights.grad is not None:
                meta_grad_snapshot = dadgp.meta_weights.grad.detach().cpu().numpy().copy()
            else:
                meta_grad_snapshot = None
            meta_optimizer.step()

            # 记录日志
            with torch.no_grad():
                epoch_total_loss += total_loss.item()
                last_val_meta = val_meta_obj.item()
                last_val_losses = [vl.item() for vl in val_loss_list]
                last_meta_grad = meta_grad_snapshot

            epochs_iter.set_postfix(loss=total_loss.item())
            num_batches += 1
            history["weights"].append(dadgp.get_normalized_weights().detach().cpu().numpy().copy())

        # Epoch结束后的统计
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
        history["losses"].append(avg_epoch_loss)

        if last_val_meta is not None:
            history["val_meta_loss"].append(last_val_meta)
            history["val_task_losses"].append(last_val_losses)
            if last_meta_grad is not None:
                history["meta_grad"].append(last_meta_grad.copy())
                history["meta_grad_sign"].append(np.sign(last_meta_grad).copy())
            else:
                history["meta_grad"].append(None)
                history["meta_grad_sign"].append(None)
        else:
            history["val_meta_loss"].append(None)
            history["val_task_losses"].append(None)
            history["meta_grad"].append(None)
            history["meta_grad_sign"].append(None)

        scheduler.step()
        meta_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # 打印日志
        val_losses_str = ", ".join(f"{v:.4f}" for v in last_val_losses) if last_val_losses else "N/A"
        grad_str = np.array2string(last_meta_grad, precision=4, separator=", ") if last_meta_grad is not None else "N/A"
        grad_sign_str = np.array2string(np.sign(last_meta_grad), separator=", ") if last_meta_grad is not None else "N/A"

        val_meta_str = f"{last_val_meta:.4f}" if last_val_meta is not None else "nan"
        print(
            f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}, "
            f"Current LR: {current_lr}, Val Meta: {val_meta_str}, "
            f"Val Tasks: [{val_losses_str}], Meta Grad: {grad_str}, Sign: {grad_sign_str}"
        )

    return history
