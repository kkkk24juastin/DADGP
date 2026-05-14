# -*- coding: utf-8 -*-
"""Evaluate DA-DGP training-sample-size sensitivity results."""

from run_sample_size_sensitivity import TRAIN_SIZE_GRID
from supplemental_evaluation_utils import (
    build_evaluation_arg_parser,
    run_sensitivity_evaluation,
)


def main():
    parser = build_evaluation_arg_parser(
        "Evaluate DA-DGP training-sample-size sensitivity results"
    )
    args = parser.parse_args()
    condition_specs = [
        {"label": f"n_train_{n_train}", "display": str(n_train)}
        for n_train in TRAIN_SIZE_GRID
    ]
    run_sensitivity_evaluation(
        "sample_size",
        condition_specs,
        "Training size",
        args,
    )


if __name__ == "__main__":
    main()
