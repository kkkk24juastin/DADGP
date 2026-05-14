# -*- coding: utf-8 -*-
"""Run DA-DGP training-sample-size sensitivity training only."""

from config import NUM_HIDDEN_DGP_DIMS, SIGMA_VALUES
from supplemental_training_utils import (
    SUPPLEMENTAL_DIR,
    build_arg_parser,
    default_device,
    generate_lhs_train_data,
    load_existing_val,
    run_seed,
    save_data_excel,
    should_skip_output,
    train_da_dgp_condition,
)


TRAIN_SIZE_GRID = [20, 50, 100, 200, 400]


def main():
    parser = build_arg_parser("DA-DGP training sample size sensitivity training")
    args = parser.parse_args()

    if args.start_run > args.end_run:
        raise ValueError("--start-run must be <= --end-run")

    device = default_device()
    print(f"Using device: {device}")
    print("Output root: SupplementalModels/sample_size")

    for n_train in TRAIN_SIZE_GRID:
        condition_label = f"n_train_{n_train}"
        print(f"\n=== Sample-size condition: {condition_label} ===")

        for run_id in range(args.start_run, args.end_run + 1):
            save_dir = (
                SUPPLEMENTAL_DIR
                / "sample_size"
                / condition_label
                / f"{run_id:02d}"
            )
            if should_skip_output(save_dir, args.skip_existing):
                continue

            val_data, data_sources = load_existing_val(run_id, device)
            if val_data is None:
                continue

            seed = run_seed(args.base_seed, run_id)
            train_data = generate_lhs_train_data(n_train, seed, device)
            datasets = {"train": train_data, "val": val_data}

            save_dir.mkdir(parents=True, exist_ok=True)
            save_data_excel(train_data, save_dir / "train_data.xlsx")

            print(f"\n[run {run_id:02d}] n_train={n_train}, lhs_seed={seed}")
            train_da_dgp_condition(
                datasets=datasets,
                save_dir=save_dir,
                device=device,
                epochs=args.epochs,
                seed=seed,
                sigma_values=SIGMA_VALUES,
                num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS,
                num_dgp_layers=2,
                metadata={
                    "experiment": "sample_size",
                    "condition": condition_label,
                    "run_id": run_id,
                    "n_train": n_train,
                    "lhs_train_seed": seed,
                    "saved_train_data": save_dir / "train_data.xlsx",
                    "data_sources": data_sources,
                },
            )


if __name__ == "__main__":
    main()
