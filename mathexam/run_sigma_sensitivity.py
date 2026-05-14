# -*- coding: utf-8 -*-
"""Run DA-DGP sigma sensitivity training only."""

from config import NUM_HIDDEN_DGP_DIMS
from supplemental_training_utils import (
    SUPPLEMENTAL_DIR,
    build_arg_parser,
    default_device,
    load_existing_train_val,
    run_seed,
    should_skip_output,
    train_da_dgp_condition,
)


SIGMA_GRID = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]


def main():
    parser = build_arg_parser("DA-DGP sigma sensitivity training")
    args = parser.parse_args()

    if args.start_run > args.end_run:
        raise ValueError("--start-run must be <= --end-run")

    device = default_device()
    print(f"Using device: {device}")
    print("Output root: SupplementalModels/sigma_sensitivity")

    for sigma in SIGMA_GRID:
        condition_label = f"sigma_{sigma}"
        sigma_values = (sigma, sigma, sigma)
        print(f"\n=== Sigma condition: {condition_label} ===")

        for run_id in range(args.start_run, args.end_run + 1):
            save_dir = (
                SUPPLEMENTAL_DIR
                / "sigma_sensitivity"
                / condition_label
                / f"{run_id:02d}"
            )
            if should_skip_output(save_dir, args.skip_existing):
                continue

            datasets, data_sources = load_existing_train_val(run_id, device)
            if datasets is None:
                continue

            seed = run_seed(args.base_seed, run_id)
            print(f"\n[run {run_id:02d}] sigma={sigma}, seed={seed}")
            train_da_dgp_condition(
                datasets=datasets,
                save_dir=save_dir,
                device=device,
                epochs=args.epochs,
                seed=seed,
                sigma_values=sigma_values,
                num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS,
                num_dgp_layers=2,
                metadata={
                    "experiment": "sigma_sensitivity",
                    "condition": condition_label,
                    "run_id": run_id,
                    "data_sources": data_sources,
                },
            )


if __name__ == "__main__":
    main()
