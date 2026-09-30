"""Run the fixed Phase 1 PaRSEL baseline (paper parameters, ROS, no reduction)."""

from __future__ import annotations

import pandas as pd

from scripts.run_experiments import PAPER_PATHS, run_combination


def main() -> None:
    train = pd.read_csv(PAPER_PATHS["train"])
    test = pd.read_csv(PAPER_PATHS["test"])
    run_combination(
        train,
        test,
        split="paper_faithful",
        dataset_variant="paper_faithful",
        sampler="ROS",
        reducer="none",
        models=["PaRSEL"],
    )


if __name__ == "__main__":
    main()
