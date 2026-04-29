"""Flatten cached Kalshi market JSON into the canonical wide CSV.

Walks every per-event file in `data/raw/kalshi/*.json` (skipping the
`events_*.json` indices) and writes one row per market to
`outputs/results/kalshi_markets_flat.csv`. This is the file that
`scripts/build_dataset.py` consumes when constructing `labels.csv`.

Run after `scripts/pull_kalshi_all.py` has populated the raw cache.

Usage:
    python scripts/flatten_kalshi.py
    python scripts/flatten_kalshi.py --output outputs/results/kalshi_markets_flat.csv
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import RAW_KALSHI_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = RESULTS_DIR / "kalshi_markets_flat.csv"

FIELDNAMES = [
    "series_ticker", "event_ticker", "market_ticker", "company_label",
    "word", "result", "label", "implied_prob",
    "close_date", "settlement_date", "status",
    "yes_bid", "no_bid", "volume", "open_interest", "rules_primary",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flatten cached Kalshi JSON to a wide CSV.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_KALSHI_DIR,
        help=f"Directory containing per-event JSON files (default: {RAW_KALSHI_DIR}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    p.add_argument(
        "--settled-only",
        action="store_true",
        help="Skip markets whose status is not 'finalized' or 'settled'.",
    )
    return p.parse_args()


_SERIES_PREFIXES = ("KXEARNINGSMENTION", "KXMENTIONEARN")


def _company_label(series_ticker: str) -> str:
    """Strip the Kalshi earnings-mention series prefix to recover the ticker.

    Two naming conventions exist in the corpus:
      KXEARNINGSMENTION{TICKER}   — 86 series (e.g. KXEARNINGSMENTIONBLK -> BLK)
      KXMENTIONEARN{TICKER}       —  7 series (e.g. KXMENTIONEARNAIR     -> AIR)

    The earlier split-on-MENTION rule yielded "EARNAIR" for the second
    family because "MENTION" appears mid-ticker. Both prefixes are now
    stripped explicitly so the company_label matches Mayank's transcript
    drops at data/raw/transcripts/{TICKER}_transcripts/.
    """
    for prefix in _SERIES_PREFIXES:
        if series_ticker.startswith(prefix):
            return series_ticker[len(prefix):]
    return series_ticker


def _row_from_market(market: dict, series_ticker: str) -> dict:
    raw_prob = market.get("previous_price_dollars")
    result = (market.get("result") or "").lower()
    custom_strike = market.get("custom_strike") or {}
    return {
        "series_ticker": series_ticker,
        "event_ticker": market.get("event_ticker", ""),
        "market_ticker": market.get("ticker", ""),
        "company_label": _company_label(series_ticker),
        "word": custom_strike.get("Word", ""),
        "result": result,
        "label": 1 if result == "yes" else 0,
        "implied_prob": float(raw_prob) if raw_prob not in (None, "") else "",
        "close_date": (market.get("close_time") or "")[:10],
        "settlement_date": (market.get("settlement_ts") or "")[:10],
        "status": market.get("status", ""),
        "yes_bid": float(market.get("yes_bid_dollars") or 0.0),
        "no_bid": float(market.get("no_bid_dollars") or 0.0),
        "volume": float(market.get("volume_fp") or 0.0),
        "open_interest": float(market.get("open_interest_fp") or 0.0),
        "rules_primary": market.get("rules_primary", ""),
    }


def _format_row(row: dict) -> dict:
    out = dict(row)
    if isinstance(out["implied_prob"], float):
        out["implied_prob"] = f"{out['implied_prob']:.4f}"
    out["yes_bid"] = f"{out['yes_bid']:.4f}"
    out["no_bid"] = f"{out['no_bid']:.4f}"
    out["volume"] = f"{out['volume']:.2f}"
    out["open_interest"] = f"{out['open_interest']:.2f}"
    return out


def flatten(input_dir: Path, output: Path, settled_only: bool = False) -> int:
    paths = sorted(p for p in input_dir.glob("*.json") if not p.stem.startswith("events_"))
    logger.info("Flattening %d event files from %s", len(paths), input_dir)

    rows: list[dict] = []
    skipped_status = 0
    for path in paths:
        markets = json.loads(path.read_text(encoding="utf-8"))
        if not markets:
            continue
        # series_ticker is the prefix before the event date suffix:
        #   KXEARNINGSMENTIONBLK-26JUN30 -> KXEARNINGSMENTIONBLK
        series_ticker = path.stem.split("-")[0]
        for m in markets:
            if settled_only and m.get("status") not in ("finalized", "settled"):
                skipped_status += 1
                continue
            rows.append(_row_from_market(m, series_ticker))

    rows.sort(key=lambda r: (r["series_ticker"], r["event_ticker"], r["market_ticker"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow(_format_row(row))

    logger.info("Wrote %d markets to %s (skipped %d non-finalized)",
                len(rows), output, skipped_status)
    return len(rows)


def main() -> None:
    args = parse_args()
    flatten(args.input_dir, args.output, settled_only=args.settled_only)


if __name__ == "__main__":
    main()
