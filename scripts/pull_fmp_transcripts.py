"""Reference template for pulling FMP earnings call transcripts.

NOTE: FMP transcript collection. Mayank has the working
pull code and API access. This file is a template showing the expected interface
so Kevin's downstream pipeline (build_dataset.py, src/data.py) can document its
assumptions about the cache layout.

Mayank should deliver transcripts as either:
  - FMP JSON:  data/raw/fmp/{TICKER}/{YEAR}Q{Q}.json
  - Plain text: data/raw/fmp/{TICKER}_YEAR_QQ.txt  (e.g. KO_2025_Q3.txt)

Kevin's fmp_client.get_all_cached_for_ticker() reads both formats automatically.
Kevin already has two KO transcripts (2025 Q3, Q4) as local .txt files.

Usage :
    python scripts/pull_fmp_transcripts.py --ticker KO --years 2022 2023 2024
    python scripts/pull_fmp_transcripts.py --all --years 2021 2022 2023 2024
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import TARGET_TICKERS
from src.fmp_client import FMPClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUARTERS = [1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Pull FMP earnings transcripts (Mayank's script — see module docstring)."
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", type=str, help="Single ticker, e.g. KO")
    group.add_argument("--all", action="store_true", help="Pull all TARGET_TICKERS")
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023, 2024],
        help="Years to pull (default: 2021-2024)",
    )
    return p.parse_args()


def pull_for_ticker(client: FMPClient, ticker: str, years: list[int]) -> None:
    """Pull and cache all transcripts for one ticker across specified years and quarters."""
    for year in years:
        for quarter in QUARTERS:
            result = client.get_transcript(ticker, year, quarter)
            if result and result["content"]:
                word_count = len(result["content"].split())
                logger.info("OK  %s %dQ%d -- %d words", ticker, year, quarter, word_count)
            elif result:
                logger.warning("EMPTY  %s %dQ%d -- content missing", ticker, year, quarter)
            else:
                logger.warning("SKIP  %s %dQ%d -- not available from FMP", ticker, year, quarter)


def main() -> None:
    """Entry point — requires FMP_API_KEY in .env (Mayank's setup)."""
    args = parse_args()
    client = FMPClient()
    tickers = TARGET_TICKERS if args.all else [args.ticker]
    for ticker in tickers:
        logger.info("-- %s ------------------------------------------", ticker)
        pull_for_ticker(client, ticker, args.years)
    logger.info("Done.")


if __name__ == "__main__":
    main()
