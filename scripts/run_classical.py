"""Run TF-IDF + Logistic Regression and Decision Tree classifiers.

Per-row input: TF-IDF of the ticker's transcripts strictly before the market's
close date, concatenated with numeric features (implied_prob, hist_rate,
in_recent_news). Vectorizer is fit on training-split row contexts only.

Follow-up analyses:
  --drop-implied-prob : ablation; exclude implied_prob from numeric features
  --uncertain         : restrict dev/test eval to 0.2 <= implied_prob <= 0.8

Training and hyperparameter tuning always use the full dev split; --uncertain
only restricts the evaluation slice. This mirrors the baseline convention.

Usage:
    python scripts/run_classical.py --model logreg
    python scripts/run_classical.py --model all --drop-implied-prob
    python scripts/run_classical.py --model all --uncertain
    python scripts/run_classical.py --model all --drop-implied-prob --uncertain
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
from src.features import (
    NUMERIC_COLS,
    build_numeric_features,
    build_row_context_texts,
    fit_tfidf,
    stack_tfidf_and_numeric,
)
from src.models.logreg import train_logreg
from src.models.tree import train_tree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UNCERTAIN_LO, UNCERTAIN_HI = 0.2, 0.8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run classical ML classifiers.")
    p.add_argument(
        "--model",
        choices=["logreg", "tree", "all"],
        default="all",
        help="Which model to train (default: all).",
    )
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    p.add_argument(
        "--drop-implied-prob",
        action="store_true",
        help="Ablation: exclude implied_prob from the numeric feature set.",
    )
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


def _build_features(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    numeric_cols: list[str],
):
    logger.info("Building per-row company-context texts...")
    train_texts = build_row_context_texts(train)
    dev_texts = build_row_context_texts(dev)
    test_texts = build_row_context_texts(test)

    logger.info("Fitting TF-IDF on training contexts...")
    vectorizer = fit_tfidf(train_texts)
    X_train_tfidf = vectorizer.transform(train_texts)
    X_dev_tfidf = vectorizer.transform(dev_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    X_train_num = build_numeric_features(train, cols=numeric_cols)
    X_dev_num = build_numeric_features(dev, cols=numeric_cols)
    X_test_num = build_numeric_features(test, cols=numeric_cols)

    X_train = stack_tfidf_and_numeric(X_train_tfidf, X_train_num)
    X_dev = stack_tfidf_and_numeric(X_dev_tfidf, X_dev_num)
    X_test = stack_tfidf_and_numeric(X_test_tfidf, X_test_num)
    logger.info("Feature shapes: train=%s dev=%s test=%s (numeric cols=%s)",
                X_train.shape, X_dev.shape, X_test.shape, numeric_cols)
    return X_train, X_dev, X_test


def _uncertain_mask(df: pd.DataFrame) -> np.ndarray:
    p = df["implied_prob"].to_numpy(dtype=float)
    return (p >= UNCERTAIN_LO) & (p <= UNCERTAIN_HI)


def _eval_and_log(
    model_name: str,
    preds_dev: np.ndarray,
    preds_test: np.ndarray,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_suffix: str,
    config_extras: dict | None = None,
) -> None:
    config = {"model": model_name, "phase": "classical"}
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

    if args.drop_implied_prob:
        numeric_cols = [c for c in NUMERIC_COLS if c != "implied_prob"]
        feature_tag = "no_implied_prob"
    else:
        numeric_cols = list(NUMERIC_COLS)
        feature_tag = "all"

    split_suffix = "_uncertain" if args.uncertain else ""
    if args.drop_implied_prob:
        split_suffix = f"_ablation{split_suffix}"

    train, dev_full, test_full = _load_splits()
    X_train, X_dev, X_test = _build_features(train, dev_full, test_full, numeric_cols)
    y_train = train["mentioned"].to_numpy(dtype=int)
    y_dev = dev_full["mentioned"].to_numpy(dtype=int)

    if args.uncertain:
        dev_mask = _uncertain_mask(dev_full)
        test_mask = _uncertain_mask(test_full)
        dev_eval = dev_full.loc[dev_mask].reset_index(drop=True)
        test_eval = test_full.loc[test_mask].reset_index(drop=True)
        logger.info(
            "Restricted eval: dev %d -> %d, test %d -> %d",
            len(dev_full), len(dev_eval), len(test_full), len(test_eval),
        )
    else:
        dev_mask = np.ones(len(dev_full), dtype=bool)
        test_mask = np.ones(len(test_full), dtype=bool)
        dev_eval, test_eval = dev_full, test_full

    def _slice_preds(preds_full: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return preds_full[mask]

    shared_cfg = {"feature_set": feature_tag}

    if args.model in ("logreg", "all"):
        logger.info("=== Logistic Regression (TF-IDF + numeric=%s) ===", feature_tag)
        lr = train_logreg(X_train, y_train)
        preds_dev_full = lr.predict(X_dev)
        preds_test_full = lr.predict(X_test)
        _eval_and_log(
            "logreg",
            _slice_preds(preds_dev_full, dev_mask),
            _slice_preds(preds_test_full, test_mask),
            dev_eval, test_eval, split_suffix,
            config_extras={**shared_cfg, "C": float(lr.get_params()["C"])},
        )

    if args.model in ("tree", "all"):
        logger.info("=== Decision Tree (TF-IDF + numeric=%s) ===", feature_tag)
        dt = train_tree(X_train, y_train, X_dev, y_dev)
        preds_dev_full = dt.predict(X_dev)
        preds_test_full = dt.predict(X_test)
        _eval_and_log(
            "tree",
            _slice_preds(preds_dev_full, dev_mask),
            _slice_preds(preds_test_full, test_mask),
            dev_eval, test_eval, split_suffix,
            config_extras={**shared_cfg, "max_depth": dt.get_params().get("max_depth")},
        )

    write_results_table()


if __name__ == "__main__":
    main()
