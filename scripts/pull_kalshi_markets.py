"""Pull Kalshi settled markets for a company's earnings-mention series.

Use --search first to discover the series ticker for each company, then
use --series-ticker with --ticker to pull and cache market data.

Usage:
    # Discover series tickers
    python scripts/pull_kalshi_markets.py --search earnings
    python scripts/pull_kalshi_markets.py --search KO

    # Pull settled markets for a known series
    python scripts/pull_kalshi_markets.py --ticker KO --series-ticker KXEARNINGSMENTIONKO
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kalshi_client import KalshiClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Pull Kalshi settled earnings-mention markets."
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--search",
        type=str,
        metavar="QUERY",
        help="Search for Kalshi series by keyword and print matching series.",
    )
    group.add_argument(
        "--series-ticker",
        type=str,
        help="Kalshi series ticker to pull (e.g. KXEARNINGSCALL-KO).",
    )
    p.add_argument(
        "--ticker",
        type=str,
        default="UNKNOWN",
        help="Company ticker label for logging (e.g. KO).",
    )
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    client = KalshiClient()

    if args.search:
        results = client.search_series(args.search)
        if results:
            print(json.dumps(results, indent=2))
        else:
            logger.info("No series found matching: %s", args.search)
        return

    markets = client.pull_all_markets_for_ticker(args.ticker, args.series_ticker)
    logger.info(
        "Pulled %d total markets for %s (series=%s)",
        len(markets),
        args.ticker,
        args.series_ticker,
    )


if __name__ == "__main__":
    main()
