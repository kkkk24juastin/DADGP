# -*- coding: utf-8 -*-
"""
MGDA 基线。
"""

import torch
import tqdm

from common import IndexedTensorDataset, create_loss_function
from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LR_GAMMA,
    MAX_GRAD_NORM,
    MGDA_GRAD_EPS,
    MGDA_MAX_ITER,
    MGDA_WEIGHT_TOL,
)


def _flatten_task_gradients(task_gradients, params):
    """将每个任务的梯度展平成向量，缺失梯度按参数形状补零。"""
    flat_grads = []
    for task_grads in task_gradients:
        flat_grads.append(
            torch.cat(
                [
                    grad.flatten()
                    if grad is not None
                    else torch.zeros_like(param).flatten()
                    for grad, param in zip(task_grads, params)
                ]
            )
        )
    return flat_grads


def mgda_solver(flat_grads, device, init_weights=None):
    """使用 Frank-Wolfe 算法求解最优梯度组合权重。"""
    num_tasks = len(flat_grads)
    if num_tasks == 0:
        return torch.tensor([], device=device)

    flat_grad_matrix = torch.stack(flat_grads)
    gram_matrix = torch.matmul(flat_grad_matrix, flat_grad_matrix.t())

    if init_weights is None:
        weights = torch.ones(num_tasks, device=device, dtype=gram_matrix.dtype)
        weights = weights / num_tasks
    else:
        weights = init_weights.to(device=device, dtype=gram_matrix.dtype).clone()
        weights = weights.clamp_min(0.0)
        weights = weights / weights.sum().clamp_min(MGDA_GRAD_EPS)

    for _ in range(MGDA_MAX_ITER):
        grad_w = 2.0 * torch.mv(gram_matrix, weights)
        min_idx = torch.argmin(grad_w)

        direction = torch.zeros(num_tasks, device=device, dtype=weights.dtype)
        direction[min_idx] = 1.0
        delta = direction - weights

        a_value = torch.dot(delta, torch.mv(gram_matrix, delta))
        b_value = 2.0 * torch.dot(weights, torch.mv(gram_matrix, delta))
        if a_value > MGDA_GRAD_EPS:
            gamma = torch.clamp(-b_value / (2.0 * a_value), 0.0, 1.0)
        else:
            gamma = weights.new_tensor(0.0)

        new_weights = weights + gamma * delta
        if torch.norm(new_weights - weights) < MGDA_WEIGHT_TOL:
            weights = new_weights
            break
        weights = new_weights

    return weights / weights.sum().clamp_min(MGDA_GRAD_EPS)


def run_mgda_training_loop(model, datasets, num_epochs, task_ids, sample_weights):
    """执行 MGDA 训练。"""
    train_data = datasets["train"]
    val_data = datasets["val"]
    train_dataset = IndexedTensorDataset(train_data["x"], train_data["y"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
    loss_fn = create_loss_function(
        sample_weights=sample_weights,
        split_sizes={
            "train": train_data["y"].size(0),
            "val": val_data["y"].size(0),
        },
    )
    device = next(model.parameters()).device
    params = list(model.parameters())
    history = {"weights": [], "losses": []}
    previous_weights = torch.ones(len(task_ids), device=device) / len(task_ids)

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        num_batches = 0
        epoch_weights_sum = None

        epochs_iter = tqdm.tqdm(
            train_loader, desc=f"MGDA Epoch {epoch + 1}/{num_epochs}", leave=False
        )
        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            optimizer.zero_grad()
            task_gradients = []
            task_losses = []
            output = model(train_x_batch)
            task_loss_values = loss_fn.task_losses(
                model,
                output,
                train_y_batch,
                task_ids,
                train_indices,
                is_val=False,
            )

            for task_loss in task_loss_values:
                task_losses.append(task_loss.item())
                grads = torch.autograd.grad(
                    task_loss,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                task_gradients.append(
                    [grad.clone() if grad is not None else None for grad in grads]
                )

            flat_task_gradients = _flatten_task_gradients(task_gradients, params)
            optimal_weights = mgda_solver(
                flat_task_gradients,
                device,
                init_weights=previous_weights,
            )
            previous_weights = optimal_weights.detach()

            aggregated_grads = []
            for param_idx in range(len(params)):
                grad_sum = None
                for task_idx, task_grads in enumerate(task_gradients):
                    task_grad = task_grads[param_idx]
                    if task_grad is None:
                        continue
                    weighted_grad = optimal_weights[task_idx] * task_grad
                    grad_sum = (
                        weighted_grad.clone()
                        if grad_sum is None
                        else grad_sum + weighted_grad
                    )
                aggregated_grads.append(grad_sum)

            for param, grad in zip(params, aggregated_grads):
                if grad is not None:
                    param.grad = grad.clone()

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            if epoch_weights_sum is None:
                epoch_weights_sum = optimal_weights.detach().cpu()
            else:
                epoch_weights_sum += optimal_weights.detach().cpu()

            avg_task_loss = sum(task_losses) / len(task_losses)
            epoch_total_loss += avg_task_loss
            num_batches += 1
            history["weights"].append(optimal_weights.detach().cpu().numpy().copy())
            weight_list = [round(weight.item(), 4) for weight in optimal_weights]
            epochs_iter.set_postfix(loss=avg_task_loss, w=weight_list)

        scheduler.step()
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        history["losses"].append(avg_epoch_loss)
        if epoch_weights_sum is None:
            weights_print = []
        else:
            avg_weights = epoch_weights_sum / max(num_batches, 1)
            weights_print = [round(weight, 4) for weight in avg_weights.tolist()]
        print(
            f"MGDA Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}, "
            f"Average Weights: {weights_print}"
        )

    return history
