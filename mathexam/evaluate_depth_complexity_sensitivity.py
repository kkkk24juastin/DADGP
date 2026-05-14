# -*- coding: utf-8 -*-
"""Evaluate DA-DGP depth/complexity sensitivity results."""

from run_depth_complexity_sensitivity import DEPTH_CONFIGS
from supplemental_evaluation_utils import (
    build_evaluation_arg_parser,
    run_sensitivity_evaluation,
)


def main():
    parser = build_evaluation_arg_parser(
        "Evaluate DA-DGP depth/complexity sensitivity results"
    )
    args = parser.parse_args()
    condition_specs = [
        {"label": config["label"], "display": config["label"]}
        for config in DEPTH_CONFIGS
    ]
    run_sensitivity_evaluation(
        "depth_complexity",
        condition_specs,
        "Depth/width",
        args,
    )


if __name__ == "__main__":
    main()
