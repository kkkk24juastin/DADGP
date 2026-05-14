# -*- coding: utf-8 -*-
"""
补充实验基线训练循环：
- Indep-DGP：每任务独立两层 DGP
- Indep-HetGP：每任务独立固定异方差 Exact GP
"""

import torch
import tqdm
from gpytorch.mlls import DeepApproximateMLL, ExactMarginalLogLikelihood, VariationalELBO

from common import IndexedTensorDataset
from config import BATCH_SIZE, LEARNING_RATE, LR_GAMMA, MAX_GRAD_NORM


def run_independent_dgp_training_loop(model, datasets, num_epochs):
    """训练每任务独立的两层 DGP。"""
    train_data = datasets["train"]
    train_dataset = IndexedTensorDataset(train_data["x"], train_data["y"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
    mlls = [
        DeepApproximateMLL(
            VariationalELBO(
                task_model.likelihood,
                task_model,
                num_data=train_data["y"].size(0),
            )
        )
        for task_model in model.models
    ]
    history = {"weights": [], "losses": []}
    equal_weights = torch.ones(len(model.models)) / max(len(model.models), 1)

    for epoch in range(num_epochs):
        model.train()
        epochs_iter = tqdm.tqdm(
            train_loader,
            desc=f"Indep-DGP Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        )
        epoch_total_loss = 0.0
        num_batches = 0

        for train_x_batch, train_y_batch, _ in epochs_iter:
            optimizer.zero_grad()

            task_losses = []
            for task_idx, task_model in enumerate(model.models):
                output = task_model(train_x_batch)
                task_loss = -mlls[task_idx](output, train_y_batch[:, task_idx])
                task_losses.append(task_loss)

            total_loss = torch.mean(torch.stack(task_losses))
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            num_batches += 1
            history["weights"].append(equal_weights.numpy().copy())
            epochs_iter.set_postfix(loss=total_loss.item())

        scheduler.step()
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        history["losses"].append(avg_epoch_loss)
        print(
            f"Indep-DGP Epoch {epoch + 1} finished. "
            f"Average Loss: {avg_epoch_loss:.4f}"
        )

    return history


def run_independent_hetgp_training_loop(model, datasets, num_epochs):
    """训练每任务独立的固定异方差 Exact GP。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
    mlls = [
        ExactMarginalLogLikelihood(task_model.likelihood, task_model)
        for task_model in model.models
    ]
    history = {"weights": [], "losses": []}
    equal_weights = torch.ones(len(model.models)) / max(len(model.models), 1)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        task_losses = []
        for task_idx, task_model in enumerate(model.models):
            train_x = task_model.train_inputs[0]
            train_y = task_model.train_targets
            output = task_model(train_x)
            task_losses.append(-mlls[task_idx](output, train_y))

        total_loss = torch.mean(torch.stack(task_losses))
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        history["weights"].append(equal_weights.numpy().copy())
        history["losses"].append(total_loss.item())
        print(
            f"Indep-HetGP Epoch {epoch + 1}/{num_epochs} finished. "
            f"Average Loss: {total_loss.item():.4f}"
        )

    return history
