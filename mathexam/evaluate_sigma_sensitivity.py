# -*- coding: utf-8 -*-
"""Evaluate DA-DGP sigma sensitivity results."""

from run_sigma_sensitivity import SIGMA_GRID
from supplemental_evaluation_utils import (
    build_evaluation_arg_parser,
    run_sensitivity_evaluation,
)


def main():
    parser = build_evaluation_arg_parser("Evaluate DA-DGP sigma sensitivity results")
    args = parser.parse_args()
    condition_specs = [
        {"label": f"sigma_{sigma}", "display": str(sigma)}
        for sigma in SIGMA_GRID
    ]
    run_sensitivity_evaluation(
        "sigma_sensitivity",
        condition_specs,
        "Sigma",
        args,
    )


if __name__ == "__main__":
    main()
