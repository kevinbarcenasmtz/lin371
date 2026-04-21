"""Unified training entry point for all classification models.

Usage:
    python src/train.py --model logreg
    python src/train.py --model distilbert --epochs 3 --lr 2e-5
    python src/train.py --model majority
"""
import argparse
import logging

from src.constants import RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train a classification model.")
    p.add_argument(
        "--model",
        choices=["majority", "consensus", "buy_all_yes", "hist_freq", "logreg", "tree", "distilbert"],
        required=True,
        help="Model to train.",
    )
    p.add_argument("--epochs", type=int, default=3, help="Training epochs (DistilBERT only).")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate (DistilBERT only).")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size (DistilBERT only).")
    p.add_argument("--max-depth", type=int, default=None, help="Decision tree max depth.")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    return p.parse_args()


def main() -> None:
    """Training entry point — implement in P3."""
    raise NotImplementedError("train.py — implement in P3.")


if __name__ == "__main__":
    main()
