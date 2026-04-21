"""Evaluation: classification metrics, ROI backtest, experiment logging, results table."""
import json
import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.constants import RESULTS_DIR

logger = logging.getLogger(__name__)


def evaluate_classification(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute macro F1, per-class precision/recall, accuracy, and confusion matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    target_names = label_names or ["not_mentioned", "mentioned"]
    n = int(len(y_true))
    if n == 0:
        logger.warning("evaluate_classification called with n=0; returning NaN metrics")
        nan_block = {"precision": float("nan"), "recall": float("nan"), "f1-score": float("nan")}
        report = {target_names[0]: nan_block, target_names[1]: nan_block}
        cm = [[0, 0], [0, 0]]
        acc = float("nan")
        f1m = float("nan")
        f1w = float("nan")
    else:
        report = classification_report(
            y_true, y_pred, labels=[0, 1], target_names=target_names,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0))
        f1w = float(f1_score(y_true, y_pred, labels=[0, 1], average="weighted", zero_division=0))
    metrics: dict[str, Any] = {
        "n": n,
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "precision_pos": float(report[target_names[1]]["precision"]),
        "recall_pos": float(report[target_names[1]]["recall"]),
        "f1_pos": float(report[target_names[1]]["f1-score"]),
        "precision_neg": float(report[target_names[0]]["precision"]),
        "recall_neg": float(report[target_names[0]]["recall"]),
        "f1_neg": float(report[target_names[0]]["f1-score"]),
        "confusion_matrix": cm,
    }
    if n > 0:
        logger.info(
            "Eval: n=%d acc=%.4f macro-F1=%.4f (pos P/R/F1=%.3f/%.3f/%.3f)",
            metrics["n"], metrics["accuracy"], metrics["f1_macro"],
            metrics["precision_pos"], metrics["recall_pos"], metrics["f1_pos"],
        )
    return metrics


def roi_backtest(
    df: pd.DataFrame,
    preds: list[int],
    fee: float = 0.03,
) -> dict[str, float]:
    """Simulate Kalshi trading on the test set.

    Predict 1 -> buy Yes: stake=implied_prob, payout=1 if mentioned=1 else 0
    Predict 0 -> buy No:  stake=1-implied_prob, payout=1 if mentioned=0 else 0
    A flat 3% fee is deducted from each trade's stake.
    """
    preds = np.asarray(preds, dtype=int)
    implied = df["implied_prob"].to_numpy(dtype=float)
    outcomes = df["mentioned"].to_numpy(dtype=int)

    mask = ~np.isnan(implied)
    n_skipped = int((~mask).sum())
    preds, implied, outcomes = preds[mask], implied[mask], outcomes[mask]

    stakes = np.where(preds == 1, implied, 1.0 - implied)
    wins = np.where(preds == outcomes, 1, 0)
    payouts = wins.astype(float)
    pnl = payouts - stakes - fee
    correct = int(wins.sum())

    n_trades = int(len(preds))
    total_roi = float(pnl.sum())
    metrics = {
        "n_trades": n_trades,
        "n_correct": correct,
        "accuracy_trades": float(correct / n_trades) if n_trades else float("nan"),
        "total_pnl": total_roi,
        "roi_per_trade": float(total_roi / n_trades) if n_trades else float("nan"),
        "avg_stake": float(stakes.mean()) if n_trades else float("nan"),
        "n_skipped_nan_prob": n_skipped,
    }
    logger.info(
        "ROI: n=%d correct=%d total_pnl=%.3f per_trade=%.4f (skipped %d NaN-prob)",
        n_trades, correct, total_roi, metrics["roi_per_trade"], n_skipped,
    )
    return metrics


def log_experiment(config: dict, metrics: dict, notes: str = "") -> None:
    """Append an experiment record to outputs/results/experiments.jsonl."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "notes": notes,
    }
    out_path = RESULTS_DIR / "experiments.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Logged experiment: %s", config.get("model", "unknown"))


_TABLE_COLUMNS = [
    "model", "split", "n", "accuracy", "f1_macro", "f1_pos",
    "precision_pos", "recall_pos", "roi_per_trade", "total_pnl",
]


def write_results_table(
    out_path=None,
    jsonl_path=None,
) -> pd.DataFrame:
    """Rebuild outputs/results/results_table.md from experiments.jsonl.

    For each (model, split) pair we keep the most recent record (by timestamp).
    """
    jsonl_path = jsonl_path or (RESULTS_DIR / "experiments.jsonl")
    out_path = out_path or (RESULTS_DIR / "results_table.md")
    if not jsonl_path.exists():
        logger.warning("No experiments.jsonl at %s; skipping table write", jsonl_path)
        return pd.DataFrame()

    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    rows = []
    for r in records:
        cfg = r.get("config", {})
        m = r.get("metrics", {})
        rows.append({
            "timestamp": r.get("timestamp"),
            "model": cfg.get("model", "?"),
            "split": cfg.get("split", "?"),
            "n": m.get("n"),
            "accuracy": m.get("accuracy"),
            "f1_macro": m.get("f1_macro"),
            "f1_pos": m.get("f1_pos"),
            "precision_pos": m.get("precision_pos"),
            "recall_pos": m.get("recall_pos"),
            "roi_per_trade": m.get("roi_per_trade"),
            "total_pnl": m.get("total_pnl"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("experiments.jsonl is empty; skipping table write")
        return df

    df = df.sort_values("timestamp").drop_duplicates(
        subset=["model", "split"], keep="last"
    )
    df = df[_TABLE_COLUMNS].sort_values(["split", "f1_macro"], ascending=[True, False])

    lines = ["# Results Table", "",
             f"_Regenerated from `experiments.jsonl` at {datetime.now().isoformat(timespec='seconds')}._",
             ""]
    for split_name, sub in df.groupby("split"):
        lines.append(f"## Split: `{split_name}`")
        lines.append("")
        lines.append(sub.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote results table: %s (%d rows)", out_path, len(df))
    return df
