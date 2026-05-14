# -*- coding: utf-8 -*-
"""
实验编排层的共享工具。
"""

from pathlib import Path

import torch

from config import LMC_NUM_LATENTS, NUM_HIDDEN_DGP_DIMS
from models_dgp import (
    IndependentDeepGP,
    IndependentHeteroscedasticGP,
    LMCDeepGP,
    MultitaskDeepGP,
)


MODEL_TYPE_MULTITASK_DGP = "multitask_dgp"
MODEL_TYPE_INDEPENDENT_DGP = "independent_dgp"
MODEL_TYPE_INDEPENDENT_HETGP = "independent_hetgp"
MODEL_TYPE_LMC_DGP = "lmc_dgp"


def build_model_from_shape(
    train_x_shape,
    num_tasks,
    device,
    num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS,
):
    """根据输入形状与任务数创建多任务 DGP 模型。"""
    return MultitaskDeepGP(
        train_x_shape,
        num_hidden_dgp_dims=num_hidden_dgp_dims,
        num_tasks=num_tasks,
    ).to(device)


def build_model(train_data, num_tasks, device):
    """创建多任务 DGP 模型。"""
    return build_model_from_shape(train_data["x"].shape, num_tasks, device)


def infer_model_type(model):
    """根据模型实例推断 checkpoint 中的模型类型。"""
    if isinstance(model, IndependentHeteroscedasticGP):
        return MODEL_TYPE_INDEPENDENT_HETGP
    if isinstance(model, IndependentDeepGP):
        return MODEL_TYPE_INDEPENDENT_DGP
    if isinstance(model, LMCDeepGP):
        return MODEL_TYPE_LMC_DGP
    return MODEL_TYPE_MULTITASK_DGP


def infer_num_hidden_dgp_dims(model):
    """尽量从模型实例中推断隐藏层输出维度。"""
    if hasattr(model, "hidden_layer"):
        return int(model.hidden_layer.output_dims)
    if isinstance(model, IndependentDeepGP) and len(model.models) > 0:
        return int(model.models[0].hidden_layer.output_dims)
    return NUM_HIDDEN_DGP_DIMS


def infer_lmc_num_latents(model):
    """从 LMC 变分策略中推断 latent 数量。"""
    if not isinstance(model, LMCDeepGP):
        return LMC_NUM_LATENTS
    strategy = model.last_layer.variational_strategy
    return int(getattr(strategy, "num_latents", LMC_NUM_LATENTS))


def build_model_from_checkpoint(checkpoint, device):
    """根据 checkpoint 元数据重建对应模型。"""
    model_type = checkpoint.get("model_type", MODEL_TYPE_MULTITASK_DGP)
    train_x_shape = checkpoint["train_x_shape"]
    num_tasks = checkpoint["num_tasks"]
    num_hidden_dgp_dims = checkpoint.get("num_hidden_dgp_dims", NUM_HIDDEN_DGP_DIMS)

    if model_type == MODEL_TYPE_MULTITASK_DGP:
        return build_model_from_shape(
            train_x_shape,
            num_tasks,
            device,
            num_hidden_dgp_dims=num_hidden_dgp_dims,
        )

    if model_type == MODEL_TYPE_INDEPENDENT_DGP:
        return IndependentDeepGP(
            train_x_shape,
            num_hidden_dgp_dims=num_hidden_dgp_dims,
            num_tasks=num_tasks,
        ).to(device)

    if model_type == MODEL_TYPE_LMC_DGP:
        return LMCDeepGP(
            train_x_shape,
            num_hidden_dgp_dims=num_hidden_dgp_dims,
            num_tasks=num_tasks,
            num_latents=checkpoint.get("lmc_num_latents", LMC_NUM_LATENTS),
        ).to(device)

    if model_type == MODEL_TYPE_INDEPENDENT_HETGP:
        train_x = checkpoint.get("train_x")
        train_y = checkpoint.get("train_y")
        if train_x is None or train_y is None:
            raise ValueError(
                "independent_hetgp checkpoint requires train_x and train_y."
            )
        return IndependentHeteroscedasticGP(
            train_x.to(device),
            train_y.to(device),
        ).to(device)

    raise ValueError(f"Unsupported model_type in checkpoint: {model_type}")


def _torch_load_checkpoint(model_path, device):
    """兼容不同 PyTorch 版本的 checkpoint 加载。"""
    load_kwargs = {"map_location": device}
    try:
        return torch.load(model_path, weights_only=True, **load_kwargs)
    except TypeError:
        return torch.load(model_path, **load_kwargs)


def load_model_checkpoint(model_path, device):
    """加载 checkpoint，并重建对应的多任务 DGP 模型。"""
    model_path = Path(model_path)
    checkpoint = _torch_load_checkpoint(model_path, device)

    model = build_model_from_checkpoint(checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalization_params = checkpoint.get("normalization_params")
    if normalization_params is not None:
        normalization_params = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in normalization_params.items()
        }

    return checkpoint, model, normalization_params


def save_model_checkpoint(
    model,
    model_path,
    train_data,
    normalization_params,
    method_name,
    dataset_id="single_run",
    sample_attention_config=None,
):
    """按 state_dict 方式保存模型与必要元数据。"""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_type = infer_model_type(model)
    checkpoint = {
        "method": method_name,
        "dataset_id": dataset_id,
        "model_type": model_type,
        "train_x_shape": tuple(train_data["x"].shape),
        "num_tasks": int(train_data["y"].shape[-1]),
        "num_hidden_dgp_dims": infer_num_hidden_dgp_dims(model),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "normalization_params": None,
        "sample_attention_config": sample_attention_config,
    }
    if model_type == MODEL_TYPE_LMC_DGP:
        checkpoint["lmc_num_latents"] = infer_lmc_num_latents(model)
    if model_type == MODEL_TYPE_INDEPENDENT_HETGP:
        checkpoint["train_x"] = train_data["x"].detach().cpu()
        checkpoint["train_y"] = train_data["y"].detach().cpu()
    if normalization_params is not None:
        checkpoint["normalization_params"] = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in normalization_params.items()
        }

    torch.save(checkpoint, model_path)
