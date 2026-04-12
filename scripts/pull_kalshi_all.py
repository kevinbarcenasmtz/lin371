"""Pull all Kalshi earnings-mention series and report what has settled market data.

Discovers every KXEARNINGMENTION* series from the API, attempts to pull settled
markets for each, and writes a summary CSV to outputs/results/kalshi_coverage.csv.

Usage:
    python scripts/pull_kalshi_all.py
    python scripts/pull_kalshi_all.py --dry-run   # discovery only, no market pulls
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kalshi_client import KalshiClient
from src.constants import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_INTER_SERIES_SLEEP = 2.0  # seconds between series pulls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull all Kalshi earnings-mention series.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover series only; do not pull markets.",
    )
    p.add_argument(
        "--query",
        type=str,
        default="earnings",
        help="Keyword to filter series (default: 'earnings').",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = KalshiClient()

    logger.info("Discovering earnings-mention series (query=%r)...", args.query)
    all_series = client.search_series(args.query, limit=500)
    # Keep only KXEARNINGMENTION* series (the standardized format)
    series = [s for s in all_series if "MENTION" in s.get("ticker", "").upper()]
    logger.info("Found %d earnings-mention series", len(series))

    if args.dry_run:
        for s in sorted(series, key=lambda x: x.get("ticker", "")):
            print(f"{s['ticker']:45s} {s.get('title','')}")
        return

    rows = []
    for i, s in enumerate(series):
        series_ticker = s.get("ticker", "")
        title = s.get("title", "")
        logger.info("[%d/%d] %s — %s", i + 1, len(series), series_ticker, title)

        try:
            events = client.get_events(series_ticker)
            total_markets = 0
            call_dates = []
            for event in events:
                event_ticker = event.get("event_ticker") or event.get("ticker", "")
                if not event_ticker:
                    continue
                markets = client.get_markets(event_ticker)
                total_markets += len(markets)
                for m in markets:
                    ct = m.get("close_time", "")[:10]
                    if ct:
                        call_dates.append(ct)
                time.sleep(
                    client._INTER_EVENT_SLEEP
                    if hasattr(client, "_INTER_EVENT_SLEEP")
                    else 1.5
                )

            earliest = min(call_dates) if call_dates else ""
            latest = max(call_dates) if call_dates else ""
            rows.append(
                {
                    "series_ticker": series_ticker,
                    "title": title,
                    "n_events": len(events),
                    "n_markets": total_markets,
                    "earliest_call": earliest,
                    "latest_call": latest,
                }
            )
            logger.info(
                "  -> %d markets across %d events (calls: %s – %s)",
                total_markets,
                len(events),
                earliest,
                latest,
            )

        except Exception as e:
            logger.warning("  -> FAILED: %s", e)
            rows.append(
                {
                    "series_ticker": series_ticker,
                    "title": title,
                    "n_events": -1,
                    "n_markets": -1,
                    "earliest_call": "",
                    "latest_call": "",
                }
            )

        if i < len(series) - 1:
            time.sleep(_INTER_SERIES_SLEEP)

    # Write summary CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "kalshi_coverage.csv"
    fieldnames = [
        "series_ticker",
        "title",
        "n_events",
        "n_markets",
        "earliest_call",
        "latest_call",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    with_data = [r for r in rows if r["n_markets"] > 0]
    total = sum(r["n_markets"] for r in with_data)
    logger.info("=== SUMMARY ===")
    logger.info("Series with data: %d / %d", len(with_data), len(rows))
    logger.info("Total settled markets: %d", total)
    logger.info("Coverage report: %s", out_path)
    print("\nSeries with settled markets:")
    for r in sorted(with_data, key=lambda x: -x["n_markets"]):
        print(
            f"  {r['series_ticker']:45s} {r['n_markets']:3d} markets  {r['earliest_call']} – {r['latest_call']}"
        )


if __name__ == "__main__":
    main()
