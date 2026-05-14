# -*- coding: utf-8 -*-
"""Run DA-DGP depth and complexity sensitivity training only."""

from config import SIGMA_VALUES
from supplemental_training_utils import (
    SUPPLEMENTAL_DIR,
    build_arg_parser,
    default_device,
    load_existing_train_val,
    run_seed,
    should_skip_output,
    train_da_dgp_condition,
)


DEPTH_CONFIGS = [
    {"label": "2L-D2", "num_dgp_layers": 2, "num_hidden_dgp_dims": 2},
    {"label": "2L-D5", "num_dgp_layers": 2, "num_hidden_dgp_dims": 5},
    {"label": "3L-D5", "num_dgp_layers": 3, "num_hidden_dgp_dims": 5},
    {"label": "4L-D5", "num_dgp_layers": 4, "num_hidden_dgp_dims": 5},
    {"label": "5L-D5", "num_dgp_layers": 5, "num_hidden_dgp_dims": 5},
]


def select_depth_configs(only_labels=None):
    if not only_labels:
        return DEPTH_CONFIGS

    requested_labels = [
        label.strip() for label in only_labels.split(",") if label.strip()
    ]
    configs_by_label = {config["label"]: config for config in DEPTH_CONFIGS}
    unknown_labels = [
        label for label in requested_labels if label not in configs_by_label
    ]
    if unknown_labels:
        valid_labels = ", ".join(configs_by_label)
        raise ValueError(
            f"Unknown depth condition(s): {', '.join(unknown_labels)}. "
            f"Valid labels: {valid_labels}"
        )

    return [configs_by_label[label] for label in requested_labels]


def main():
    parser = build_arg_parser("DA-DGP depth/complexity sensitivity training")
    parser.add_argument(
        "--only-labels",
        type=str,
        default=None,
        help="Comma-separated depth condition labels to run, e.g. 5L-D5.",
    )
    args = parser.parse_args()

    if args.start_run > args.end_run:
        raise ValueError("--start-run must be <= --end-run")

    device = default_device()
    print(f"Using device: {device}")
    print("Output root: SupplementalModels/depth_complexity")

    selected_configs = select_depth_configs(args.only_labels)
    print(
        "Selected conditions: "
        + ", ".join(config["label"] for config in selected_configs)
    )

    for config in selected_configs:
        condition_label = config["label"]
        print(f"\n=== Depth condition: {condition_label} ===")

        for run_id in range(args.start_run, args.end_run + 1):
            save_dir = (
                SUPPLEMENTAL_DIR
                / "depth_complexity"
                / condition_label
                / f"{run_id:02d}"
            )
            if should_skip_output(save_dir, args.skip_existing):
                continue

            datasets, data_sources = load_existing_train_val(run_id, device)
            if datasets is None:
                continue

            seed = run_seed(args.base_seed, run_id)
            print(f"\n[run {run_id:02d}] depth={condition_label}, seed={seed}")
            train_da_dgp_condition(
                datasets=datasets,
                save_dir=save_dir,
                device=device,
                epochs=args.epochs,
                seed=seed,
                sigma_values=SIGMA_VALUES,
                num_hidden_dgp_dims=config["num_hidden_dgp_dims"],
                num_dgp_layers=config["num_dgp_layers"],
                metadata={
                    "experiment": "depth_complexity",
                    "condition": condition_label,
                    "run_id": run_id,
                    "data_sources": data_sources,
                    "depth_config": config,
                },
            )


if __name__ == "__main__":
    main()
