# -*- coding: utf-8 -*-
"""Utilities for supplemental DA-DGP training scripts.

These helpers are intentionally isolated from the existing experiment pipeline.
They only read from ``Achievements`` and write to ``SupplementalModels``.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.models.deep_gps import DeepGP

from algo_da_dgp import DADGP, run_training_loop
from common import create_loss_function
from config import (
    BASE_DIR,
    BOUNDS,
    DIMENSIONS,
    N_RUNS,
    NUM_EPOCHS,
    TARGET_VALUES,
    TRAIN_TASKS,
    PRI_TASKS,
    WEIGHT_INIT,
)
from data_generation import generate_candidates, setup_sample_weights, three_task_function
from models_dgp import DGPHiddenLayer, MultitaskDeepGP


SUPPLEMENTAL_DIR = BASE_DIR / "SupplementalModels"


class ConfigurableMultitaskDeepGP(DeepGP):
    """Supplemental-only multitask DGP with configurable total depth."""

    def __init__(
        self,
        train_x_shape,
        num_hidden_dgp_dims=5,
        num_tasks=3,
        num_dgp_layers=2,
    ):
        super().__init__()

        if num_dgp_layers < 1:
            raise ValueError("num_dgp_layers must be >= 1.")

        self.num_dgp_layers = int(num_dgp_layers)
        self.num_hidden_dgp_dims = int(num_hidden_dgp_dims)
        self.num_tasks = int(num_tasks)
        input_dims = train_x_shape[-1]

        self.extra_hidden_layers = torch.nn.ModuleList()

        if self.num_dgp_layers == 1:
            self.hidden_layer = None
            last_layer_input_dims = input_dims
        else:
            self.hidden_layer = DGPHiddenLayer(
                input_dims=input_dims,
                output_dims=self.num_hidden_dgp_dims,
                use_constant_mean=True,
            )
            last_layer_input_dims = self.hidden_layer.output_dims

            for _ in range(self.num_dgp_layers - 2):
                self.extra_hidden_layers.append(
                    DGPHiddenLayer(
                        input_dims=self.num_hidden_dgp_dims,
                        output_dims=self.num_hidden_dgp_dims,
                        use_constant_mean=True,
                    )
                )

        self.last_layer = DGPHiddenLayer(
            input_dims=last_layer_input_dims,
            output_dims=self.num_tasks,
            use_constant_mean=False,
        )
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_tasks)

    def forward(self, inputs):
        hidden_rep = inputs
        if self.hidden_layer is not None:
            hidden_rep = self.hidden_layer(hidden_rep)
            for hidden_layer in self.extra_hidden_layers:
                hidden_rep = hidden_layer(hidden_rep)
        return self.last_layer(hidden_rep)

    def predict(self, test_x, batch_size=50):
        self.eval()
        with torch.no_grad():
            means, vars_ = [], []
            for i in range(0, test_x.shape[0], batch_size):
                batch_x = test_x[i : i + batch_size]
                preds = self.likelihood(self(batch_x)).to_data_independent_dist()
                mean, var = preds.mean, preds.variance
                batch_size_in, num_tasks = batch_x.shape[0], self.likelihood.num_tasks
                means.append(
                    MultitaskDeepGP._align_to_bxt(mean, batch_size_in, num_tasks)
                )
                vars_.append(
                    MultitaskDeepGP._align_to_bxt(var, batch_size_in, num_tasks)
                )
            return torch.cat(means, dim=0), torch.cat(vars_, dim=0)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def default_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_arg_parser(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--start-run", type=int, default=1)
    parser.add_argument("--end-run", type=int, default=N_RUNS)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def run_seed(base_seed: int, run_id: int) -> int:
    return int(base_seed) + int(run_id)


def source_run_dir(run_id: int) -> Path:
    return BASE_DIR / "Achievements" / f"{run_id:02d}"


def load_excel_data(data_path: Path, device: torch.device):
    df = pd.read_excel(data_path)
    x_cols = [col for col in df.columns if str(col).startswith("x")]
    y_cols = [col for col in df.columns if str(col).startswith("y")]
    if not x_cols or not y_cols:
        raise ValueError(f"Invalid data file columns: {data_path}")
    return {
        "x": torch.from_numpy(df[x_cols].values).float().to(device),
        "y": torch.from_numpy(df[y_cols].values).float().to(device),
    }


def save_data_excel(data: dict, save_path: Path):
    x_np = data["x"].detach().cpu().numpy()
    y_np = data["y"].detach().cpu().numpy()
    columns = (
        [f"x{i + 1}" for i in range(x_np.shape[1])]
        + [f"y{i + 1}" for i in range(y_np.shape[1])]
    )
    df = pd.DataFrame(np.hstack([x_np, y_np]), columns=columns)
    df.to_excel(save_path, index=False)


def load_existing_train_val(run_id: int, device: torch.device):
    run_dir = source_run_dir(run_id)
    train_path = run_dir / "train_data.xlsx"
    val_path = run_dir / "val_data.xlsx"
    missing = [str(path) for path in [train_path, val_path] if not path.exists()]
    if missing:
        print(f"[skip] run {run_id:02d}: missing data files: {', '.join(missing)}")
        return None, None
    return (
        {"train": load_excel_data(train_path, device), "val": load_excel_data(val_path, device)},
        {"train_data": train_path, "val_data": val_path},
    )


def load_existing_val(run_id: int, device: torch.device):
    val_path = source_run_dir(run_id) / "val_data.xlsx"
    if not val_path.exists():
        print(f"[skip] run {run_id:02d}: missing data file: {val_path}")
        return None, None
    return load_excel_data(val_path, device), {"val_data": val_path}


def generate_lhs_train_data(n_train: int, seed: int, device: torch.device):
    lower_bounds = [BOUNDS[0]] * DIMENSIONS
    upper_bounds = [BOUNDS[1]] * DIMENSIONS
    x_np = generate_candidates(n_train, lower_bounds, upper_bounds, seed=seed)
    y_np = three_task_function(x_np)
    return {
        "x": torch.from_numpy(x_np).float().to(device),
        "y": torch.from_numpy(y_np).float().to(device),
    }


def save_weight_history(history: dict, save_dir: Path):
    if not history or "weights" not in history:
        return

    weights_array = np.array(history["weights"])
    if weights_array.size == 0:
        return

    n_steps = weights_array.shape[0]
    losses_per_epoch = len(history.get("losses", []))
    steps_per_epoch = n_steps // losses_per_epoch if losses_per_epoch > 0 else n_steps
    task_ids = list(TRAIN_TASKS)
    weight_records = []

    for step_idx in range(n_steps):
        epoch = step_idx // steps_per_epoch + 1 if steps_per_epoch > 0 else 1
        batch = step_idx % steps_per_epoch + 1 if steps_per_epoch > 0 else step_idx + 1
        record = {"step": step_idx + 1, "epoch": epoch, "batch": batch}
        for task_idx, task_id in enumerate(task_ids):
            record[f"weight_{task_id}"] = weights_array[step_idx, task_idx]
        weight_records.append(record)

    for epoch_idx, loss in enumerate(history.get("losses", [])):
        record_idx = epoch_idx * steps_per_epoch
        if 0 <= record_idx < len(weight_records):
            weight_records[record_idx]["epoch_loss"] = loss

    pd.DataFrame(weight_records).to_excel(
        save_dir / "weight_history_da_dgp.xlsx", index=False
    )


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_metadata(save_dir: Path, metadata: dict):
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(metadata), f, indent=2, ensure_ascii=False)


def should_skip_output(save_dir: Path, skip_existing: bool):
    model_path = save_dir / "model_da_dgp.pt"
    if skip_existing and model_path.exists():
        print(f"[skip] existing model: {model_path}")
        return True
    return False


def train_da_dgp_condition(
    datasets: dict,
    save_dir: Path,
    device: torch.device,
    epochs: int,
    seed: int,
    sigma_values,
    num_hidden_dgp_dims: int = 5,
    num_dgp_layers: int = 2,
    metadata: dict = None,
):
    set_seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data = datasets["train"], datasets["val"]
    num_tasks = train_data["y"].size(-1)
    sample_weights = setup_sample_weights(
        train_data, val_data, TARGET_VALUES, tuple(sigma_values)
    )

    model = ConfigurableMultitaskDeepGP(
        train_data["x"].shape,
        num_hidden_dgp_dims=num_hidden_dgp_dims,
        num_tasks=num_tasks,
        num_dgp_layers=num_dgp_layers,
    ).to(device)
    dadgp = DADGP(
        model,
        device,
        TRAIN_TASKS,
        PRI_TASKS,
        weight_init=WEIGHT_INIT,
        sample_weights=sample_weights,
    )
    dadgp.loss_fn = create_loss_function(
        sample_weights=sample_weights,
        split_sizes={
            "train": train_data["y"].size(0),
            "val": val_data["y"].size(0),
        },
    )

    history = run_training_loop(model, dadgp, datasets, epochs)
    torch.save(model.state_dict(), save_dir / "model_da_dgp.pt")
    save_weight_history(history, save_dir)

    final_weights = dadgp.get_normalized_weights().detach().cpu().numpy()
    save_metadata(
        save_dir,
        {
            **(metadata or {}),
            "seed": seed,
            "epochs": epochs,
            "sigma_values": tuple(sigma_values),
            "num_hidden_dgp_dims": num_hidden_dgp_dims,
            "num_dgp_layers": num_dgp_layers,
            "num_tasks": num_tasks,
            "train_size": train_data["y"].size(0),
            "val_size": val_data["y"].size(0),
            "final_task_weights": final_weights,
            "output_dir": save_dir,
        },
    )
    return model
