# -*- coding: utf-8 -*-
"""
公共组件：包含项目中通用的工具类和备份版任务损失定义
"""

import torch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from torch.utils.data import TensorDataset

from models_dgp import WeightedVariationalELBO


# ==========================================================================================
# 数据集工具类
# ==========================================================================================


class IndexedTensorDataset(TensorDataset):
    """一个同时返回样本索引的TensorDataset。"""

    def __getitem__(self, index):
        return super().__getitem__(index) + (index,)


# ==========================================================================================
# 备份版任务损失定义
# ==========================================================================================


class BackupStyleLoss:
    """备份版损失语义：每个任务损失都是一个完整ELBO。"""

    def __init__(self, model=None, sample_weights=None, split_sizes=None):
        self.model = model
        self.sample_weights = sample_weights
        self.split_sizes = split_sizes or {}

    @staticmethod
    def _split_name(is_val: bool) -> str:
        return "val" if is_val else "train"

    def _get_sample_weights(self, task_id, indices, is_val):
        if self.sample_weights is None:
            return None

        current_weights = self.sample_weights[self._split_name(is_val)]
        if task_id not in current_weights:
            return None
        return current_weights[task_id][indices]

    def _resolve_model(self, model):
        resolved = model if model is not None else self.model
        if resolved is None:
            raise ValueError("model is required for backup-style ELBO loss.")
        return resolved

    def _num_data(self, targets, is_val):
        return self.split_sizes.get(self._split_name(is_val), targets.size(0))

    def __call__(self, pred, targets, task_id, indices, is_val=False, model=None):
        resolved_model = self._resolve_model(model)
        num_data = self._num_data(targets, is_val)

        if task_id == "global_fit":
            mll = DeepApproximateMLL(
                VariationalELBO(
                    resolved_model.likelihood,
                    resolved_model,
                    num_data=num_data,
                )
            )
            return -mll(pred, targets)

        weights = self._get_sample_weights(task_id, indices, is_val)
        mll = DeepApproximateMLL(
            WeightedVariationalELBO(
                resolved_model.likelihood,
                resolved_model,
                num_data=num_data,
                weights=weights,
            )
        )
        return -mll(pred, targets)

    def task_losses(self, model, pred, targets, task_ids, indices, is_val=False):
        return [
            self(pred, targets, task_id, indices, is_val=is_val, model=model)
            for task_id in task_ids
        ]

    def shared_kl_loss(self, model, is_val=False):
        """兼容旧调用；备份版语义中KL已经包含在每个任务损失里。"""
        return torch.zeros((), device=next(model.parameters()).device)

    def combine_task_losses(self, model, task_losses, task_weights=None, is_val=False):
        if task_weights is None:
            return torch.mean(torch.stack(task_losses))
        return sum(weight * loss for weight, loss in zip(task_weights, task_losses))


def create_loss_function(
    model=None,
    sample_weights=None,
    split_sizes=None,
    task_to_index=None,
):
    """创建备份版语义的损失对象。

    split_sizes/task_to_index 保留为兼容参数；备份语义不再按输出维度拆分数据项，
    也不再单独添加共享KL。
    """
    return BackupStyleLoss(
        model=model,
        sample_weights=sample_weights,
        split_sizes=split_sizes,
    )
