from __future__ import annotations

import argparse
import logging

from src.data import load_config, load_split, run_eda, set_seed
from src.evaluate import run_evaluation
from src.tune import run_sensitivity, run_tuning


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="WDBC XGBoost pipeline")
    ap.add_argument("--stage", choices=["data", "tune", "sensitivity", "evaluate", "all"], default="all")
    ap.add_argument("--config", default="configs/config.json")
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    if args.stage in ("data", "all"):
        load_split(config)  # triggers download + schema assertions
        run_eda(config)
    if args.stage in ("tune", "all"):
        run_tuning(config)
    if args.stage == "sensitivity":  # standalone regen from saved best config
        run_sensitivity(config)
    if args.stage in ("evaluate", "all"):  # test set touched exactly once, always last
        run_evaluation(config)


if __name__ == "__main__":
    main()
