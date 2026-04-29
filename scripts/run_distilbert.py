"""Fine-tune DistilBERT for binary mention classification.

Per-row input = (target_word, most-recent-2 prior transcripts), fed as a
sentence-pair task so the tokenizer keeps the word intact and head-truncates
the context to 512 tokens. Trained on the full train split, tuned on dev via
macro-F1, evaluated once on test at the end.

Run on UTCS SSH box inside tmux (activate `pytorch-cuda` + pip install --user
transformers datasets). Checkpoints saved to outputs/models/distilbert/best/.

Usage:
    python scripts/run_distilbert.py
    python scripts/run_distilbert.py --epochs 3 --batch-size 16 --max-seq-len 512
    python scripts/run_distilbert.py --smoke-test    # tiny subset, 1 epoch
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
from src.features import build_recent_context_texts
from src.models.distilbert import (
    DEFAULT_OUTPUT_DIR,
    MODEL_NAME,
    predict_distilbert,
    train_distilbert,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-recent", type=int, default=2,
                   help="How many most-recent prior transcripts to include as context.")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--output-dir", type=str, default=None,
                   help=f"Override checkpoint dir (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny subset, 1 epoch. For verifying training loop before SSH run.")
    return p.parse_args()


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(SPLITS_DIR / "train.csv")
    dev = pd.read_csv(SPLITS_DIR / "dev.csv")
    test = pd.read_csv(SPLITS_DIR / "test.csv")
    logger.info("Loaded splits: %d train / %d dev / %d test", len(train), len(dev), len(test))
    return train, dev, test


def _subset_for_smoke_test(
    train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Smoke test: subsampling to 32/16/16 rows")
    return (
        train.sample(n=min(32, len(train)), random_state=0).reset_index(drop=True),
        dev.sample(n=min(16, len(dev)), random_state=0).reset_index(drop=True),
        test.sample(n=min(16, len(test)), random_state=0).reset_index(drop=True),
    )


def _eval_and_log(
    split_name: str,
    preds: np.ndarray,
    df: pd.DataFrame,
    config: dict,
) -> None:
    metrics = evaluate_classification(df["mentioned"].tolist(), preds.tolist())
    if split_name.startswith("test"):
        metrics.update(roi_backtest(df, preds.tolist()))
    log_experiment({**config, "split": split_name}, metrics)


def main() -> None:
    args = parse_args()

    train, dev, test = _load_splits()
    if args.smoke_test:
        train, dev, test = _subset_for_smoke_test(train, dev, test)
        args.epochs = 1

    logger.info("Building per-row most-recent-%d context windows...", args.max_recent)
    train_ctx = build_recent_context_texts(train, max_recent=args.max_recent)
    dev_ctx = build_recent_context_texts(dev, max_recent=args.max_recent)
    test_ctx = build_recent_context_texts(test, max_recent=args.max_recent)

    train_words = train["word"].astype(str).tolist()
    dev_words = dev["word"].astype(str).tolist()
    test_words = test["word"].astype(str).tolist()

    train_labels = train["mentioned"].astype(int).tolist()
    dev_labels = dev["mentioned"].astype(int).tolist()

    output_dir = args.output_dir or str(DEFAULT_OUTPUT_DIR)

    logger.info("=== DistilBERT fine-tune (%s, epochs=%d, lr=%g, bs=%d) ===",
                MODEL_NAME, args.epochs, args.lr, args.batch_size)
    model, tokenizer, best_dev_metrics = train_distilbert(
        train_words, train_ctx, train_labels,
        dev_words, dev_ctx, dev_labels,
        output_dir=output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )

    logger.info("Predicting on dev + test with best checkpoint...")
    dev_preds, _ = predict_distilbert(
        model, tokenizer, dev_words, dev_ctx,
        batch_size=args.batch_size * 2, max_seq_len=args.max_seq_len,
    )
    test_preds, test_probs = predict_distilbert(
        model, tokenizer, test_words, test_ctx,
        batch_size=args.batch_size * 2, max_seq_len=args.max_seq_len,
    )

    config = {
        "model": "distilbert",
        "phase": "distilbert",
        "feature_set": "text_only",
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "max_recent": args.max_recent,
        "seed": args.seed,
    }
    if args.smoke_test:
        config["smoke_test"] = True

    _eval_and_log("dev", dev_preds, dev, config)
    _eval_and_log("test", test_preds, test, config)

    probs_name = "test_probs_smoke.npy" if args.smoke_test else "test_probs.npy"
    np.save(Path(output_dir) / probs_name, test_probs)
    logger.info("Saved test positive-class probs to %s/%s", output_dir, probs_name)

    write_results_table()


if __name__ == "__main__":
    main()
