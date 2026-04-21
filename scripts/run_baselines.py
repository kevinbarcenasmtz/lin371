"""Run all baseline classifiers and log results to experiments.jsonl.

Baselines: majority class, Kalshi consensus, buy-all-Yes, historical frequency.

Usage:
    python scripts/run_baselines.py
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import RANDOM_SEED, SPLITS_DIR
from src.evaluate import (
    evaluate_classification,
    log_experiment,
    roi_backtest,
    write_results_table,
)
from src.models.baselines import (
    BuyAllYes,
    HistFreqClassifier,
    KalshiConsensus,
    MajorityClassifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


UNCERTAIN_LO, UNCERTAIN_HI = 0.2, 0.8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline classifiers.")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    p.add_argument(
        "--uncertain",
        action="store_true",
        help=f"Restrict dev/test eval to {UNCERTAIN_LO} <= implied_prob <= {UNCERTAIN_HI}.",
    )
    return p.parse_args()


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(SPLITS_DIR / "train.csv")
    dev = pd.read_csv(SPLITS_DIR / "dev.csv")
    test = pd.read_csv(SPLITS_DIR / "test.csv")
    logger.info("Loaded splits: %d train / %d dev / %d test", len(train), len(dev), len(test))
    return train, dev, test


def _restrict_uncertain(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["implied_prob"] >= UNCERTAIN_LO) & (df["implied_prob"] <= UNCERTAIN_HI)
    return df.loc[mask].reset_index(drop=True)


def _run_and_log(
    model_name: str,
    preds_dev: np.ndarray,
    preds_test: np.ndarray,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_suffix: str,
    config_extras: dict | None = None,
) -> None:
    config = {"model": model_name, "phase": "baseline"}
    if config_extras:
        config.update(config_extras)

    dev_metrics = evaluate_classification(dev_df["mentioned"].tolist(), preds_dev.tolist())
    test_metrics = evaluate_classification(test_df["mentioned"].tolist(), preds_test.tolist())
    test_roi = roi_backtest(test_df, preds_test.tolist())
    test_metrics.update(test_roi)

    log_experiment({**config, "split": f"dev{split_suffix}"}, dev_metrics)
    log_experiment({**config, "split": f"test{split_suffix}"}, test_metrics)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    train, dev_full, test_full = _load_splits()
    y_train = train["mentioned"].to_numpy(dtype=int)
    y_dev_full = dev_full["mentioned"].to_numpy(dtype=int)

    if args.uncertain:
        dev = _restrict_uncertain(dev_full)
        test = _restrict_uncertain(test_full)
        logger.info(
            "Restricted to uncertain markets (%.2f <= implied_prob <= %.2f): dev %d -> %d, test %d -> %d",
            UNCERTAIN_LO, UNCERTAIN_HI, len(dev_full), len(dev), len(test_full), len(test),
        )
        split_suffix = "_uncertain"
    else:
        dev, test = dev_full, test_full
        split_suffix = ""

    logger.info("=== Majority ===")
    maj = MajorityClassifier().fit(None, y_train)
    _run_and_log("majority", maj.predict(dev), maj.predict(test), dev, test, split_suffix,
                 config_extras={"majority_class": int(maj.majority_)})

    logger.info("=== Kalshi consensus ===")
    cons = KalshiConsensus()
    _run_and_log("consensus",
                 cons.predict(dev["implied_prob"].to_numpy()),
                 cons.predict(test["implied_prob"].to_numpy()),
                 dev, test, split_suffix)

    logger.info("=== Buy-all-Yes ===")
    byes = BuyAllYes()
    _run_and_log("buy_all_yes", byes.predict(dev), byes.predict(test), dev, test, split_suffix)

    logger.info("=== Historical frequency ===")
    hist = HistFreqClassifier().fit(
        dev_full["hist_rate"].to_numpy(dtype=float),
        y_dev_full,
    )
    _run_and_log(
        "hist_freq",
        hist.predict(dev["hist_rate"].to_numpy(dtype=float)),
        hist.predict(test["hist_rate"].to_numpy(dtype=float)),
        dev, test, split_suffix,
        config_extras={"threshold": float(hist.threshold)},
    )

    write_results_table()


if __name__ == "__main__":
    main()
