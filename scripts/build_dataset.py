"""Build labels.csv and cross-company splits from raw transcripts and Kalshi markets.

Reads Kalshi markets from outputs/results/kalshi_markets_flat.csv, cleans any
new raw transcripts, computes hist_rate using training-split companies only
(anti-leakage), and writes labels.csv plus train/dev/test splits under
data/processed/.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --skip-preprocess  # skip transcript cleaning
    python scripts/build_dataset.py --dry-run
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import PROCESSED_DIR, RESULTS_DIR, SPLITS_DIR, TRANSCRIPTS_DIR
from src.data import (
    assign_company_splits,
    build_labels,
    cross_company_split,
    load_transcripts,
    preprocess_all_transcripts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KALSHI_MARKETS_CSV = RESULTS_DIR / "kalshi_markets_flat.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build labels.csv and cross-company splits.")
    p.add_argument("--skip-preprocess", action="store_true", help="Skip transcript cleaning.")
    p.add_argument("--dry-run", action="store_true", help="Log steps without writing output files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_preprocess:
        counts = preprocess_all_transcripts()
        logger.info("Preprocessed %d tickers, %d transcripts total",
                    len(counts), sum(counts.values()))

    markets = pd.read_csv(KALSHI_MARKETS_CSV).to_dict("records")
    logger.info("Loaded %d Kalshi markets from %s", len(markets), KALSHI_MARKETS_CSV)

    market_tickers = sorted({m["company_label"] for m in markets})
    train_tickers, dev_tickers, test_tickers = assign_company_splits(market_tickers)
    logger.info("Split assignment: train=%d dev=%d test=%d tickers",
                len(train_tickers), len(dev_tickers), len(test_tickers))

    all_transcripts: list[dict] = []
    for tkr in market_tickers:
        all_transcripts.extend(load_transcripts(tkr))
    logger.info("Loaded %d transcripts across %d tickers for hist_rate",
                len(all_transcripts), len({t["ticker"] for t in all_transcripts}))

    labels = build_labels(all_transcripts, markets)

    train_df, dev_df, test_df = cross_company_split(labels)

    if args.dry_run:
        logger.info("DRY RUN — skipping writes.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = PROCESSED_DIR / "labels.csv"
    labels.to_csv(labels_path, index=False)
    logger.info("Wrote %s (%d rows)", labels_path, len(labels))

    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        out_path = SPLITS_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
