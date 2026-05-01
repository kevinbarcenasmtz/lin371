"""FMP API client with disk caching for earnings transcripts and stock news.

Handles two input formats transparently:
  - FMP JSON responses (from API): content lives in response[0]["content"]
  - Local .txt files (e.g. KO transcripts Kevin drops in manually)

Both produce the same normalized dict so downstream code doesn't care.
"""

import json
import logging
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

from src.constants import FMP_BASE_URL, RAW_FMP_DIR, RAW_NEWS_DIR

load_dotenv()
logger = logging.getLogger(__name__)

_RATE_LIMIT_SLEEP = 0.5  # seconds between API calls; FMP free tier ~10 req/min


class FMPClient:
    """Wrapper for Financial Modeling Prep API with local disk caching.

    Cache layout:
      data/raw/fmp/{TICKER}/{YEAR}Q{Q}.json   ← API responses
      data/raw/fmp/{TICKER}/*.txt              ← local transcripts (KO)
      data/raw/news/{TICKER}/{YEAR}Q{Q}.json  ← stock news
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_root: Path = RAW_FMP_DIR,
        news_cache_root: Path = RAW_NEWS_DIR,
    ) -> None:
        """Load API key from env if not provided; set cache directories."""
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FMP_API_KEY not set. Add it to .env or pass api_key= directly."
            )
        self.cache_root = Path(cache_root)
        self.news_cache_root = Path(news_cache_root)

    # ── Transcripts ──────────────────────────────────────────────────────────

    def get_transcript(self, ticker: str, year: int, quarter: int) -> Optional[dict]:
        """Fetch earnings call transcript for (ticker, year, quarter).

        Returns a normalized dict:
          {"ticker", "year", "quarter", "content", "date", "source"}
        Returns None if FMP returns an empty response (quarter not available).
        Caches raw API response to data/raw/fmp/{TICKER}/{YEAR}Q{Q}.json.
        """
        cache_path = self.cache_root / ticker / f"{year}Q{quarter}.json"
        if cache_path.exists():
            logger.info("Cache hit: %s", cache_path)
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return _normalize_fmp_json(raw, ticker, year, quarter)

        url = f"{FMP_BASE_URL}/earning_call_transcript/{ticker}"
        params = {"year": year, "quarter": quarter, "apikey": self.api_key}
        logger.info("Fetching FMP transcript: %s %dQ%d", ticker, year, quarter)
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        time.sleep(_RATE_LIMIT_SLEEP)

        if not raw:
            logger.warning("Empty FMP response: %s %dQ%d", ticker, year, quarter)
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        logger.info("Cached: %s", cache_path)
        return _normalize_fmp_json(raw, ticker, year, quarter)

    def get_all_cached_for_ticker(self, ticker: str) -> list[dict]:
        """Return all cached transcripts for a ticker (JSON and .txt formats).

        Scans two locations:
          1. data/raw/fmp/{TICKER}/  — standard subdir (JSON from API, or renamed .txt)
          2. data/raw/fmp/            — root dir, for files named {TICKER}_YEAR_QQ.txt

        This handles both the expected layout and the case where local .txt files
        are dropped directly into data/raw/fmp/ with names like KO_2025_Q3.txt.
        """
        results: list[dict] = []
        seen: set[Path] = set()

        # ── Subdir: data/raw/fmp/{TICKER}/ ──────────────────────────────────
        ticker_dir = self.cache_root / ticker
        if ticker_dir.exists():
            for p in sorted(ticker_dir.iterdir()):
                seen.add(p)
                if p.suffix == ".json":
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    year, quarter = _parse_filename(p.stem)
                    results.append(_normalize_fmp_json(raw, ticker, year, quarter))
                elif p.suffix == ".txt":
                    results.append(_load_txt_transcript(p, ticker))

        # ── Root: data/raw/fmp/{TICKER}_*.txt ───────────────────────────────
        for p in sorted(self.cache_root.glob(f"{ticker}_*.txt")):
            if p not in seen:
                results.append(_load_txt_transcript(p, ticker))

        if not results:
            logger.warning("No cached transcripts found for ticker: %s", ticker)
        return results

    # ── News ─────────────────────────────────────────────────────────────────

    def get_news(
        self, ticker: str, year: int, quarter: int, limit: int = 100
    ) -> list[dict]:
        """Fetch stock news for the 30-day window before the earnings call.

        Caches to data/raw/news/{TICKER}/{YEAR}Q{Q}.json.
        The call date is approximated as mid-first-month of the following quarter.
        """
        cache_path = self.news_cache_root / ticker / f"{year}Q{quarter}.json"
        if cache_path.exists():
            logger.info("Cache hit (news): %s", cache_path)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        from_date, to_date = _quarter_news_window(year, quarter)
        url = f"{FMP_BASE_URL}/stock_news"
        params = {
            "tickers": ticker,
            "limit": limit,
            "from": from_date,
            "to": to_date,
            "apikey": self.api_key,
        }
        logger.info(
            "Fetching FMP news: %s %dQ%d (%s → %s)",
            ticker,
            year,
            quarter,
            from_date,
            to_date,
        )
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        time.sleep(_RATE_LIMIT_SLEEP)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Cached (news): %s", cache_path)
        return data


# ── Module-level helpers ──────────────────────────────────────────────────────


def _normalize_fmp_json(raw: list | dict, ticker: str, year: int, quarter: int) -> dict:
    """Extract transcript text from FMP's array response into a normalized dict."""
    if isinstance(raw, list) and raw:
        entry = raw[0]
    elif isinstance(raw, dict):
        entry = raw
    else:
        entry = {}
    return {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "content": entry.get("content") or "",  # guard against None from FMP
        "date": entry.get("date"),
        "source": "fmp_api",
    }


def _load_txt_transcript(path: Path, ticker: str) -> dict:
    """Load a local .txt transcript. Infers year/quarter from the filename.

    Strips the header block present in Mayank's transcript files:
      Symbol: KO
      Period: Q3 2006
      Date: 2006-10-18
      ================

    The `Date:` header value (the actual call date) is preserved on the
    returned dict as `date` so downstream code can filter priors by
    real calendar date rather than fiscal (year, quarter) — fiscal-vs-
    calendar mismatch leaks the target call's own transcript into its
    own "prior" set on shifted-fiscal tickers.
    """
    year, quarter = _parse_filename(path.stem)
    content = path.read_text(encoding="utf-8")
    date_match = re.search(r"^Date:\s*(\S+)", content, re.MULTILINE)
    date_str = date_match.group(1) if date_match else None
    content = re.sub(r"^(Symbol|Period|Date):.*\n", "", content, flags=re.MULTILINE)
    content = re.sub(r"=+\n", "", content)
    return {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "content": content.strip(),
        "date": date_str,
        "source": "local_txt",
        "filename": path.name,
    }


def _parse_filename(stem: str) -> tuple[int, int]:
    """Parse year and quarter from a filename stem. Handles two formats:

      '2024Q2'       → (2024, 2)   (FMP API cache format)
      'KO_2024_Q2'   → (2024, 2)   (local transcript naming convention)

    Returns (0, 0) if the format is unrecognized.
    """
    # Format 1: YEARQQ  e.g. "2024Q2"
    m = re.match(r"^(\d{4})Q(\d)$", stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Format 2: TICKER_YEAR_QQ  e.g. "KO_2025_Q3"
    m = re.match(r"^[A-Za-z]+_(\d{4})_Q(\d)$", stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    logger.warning("Could not parse year/quarter from filename stem: %s", stem)
    return 0, 0


def _quarter_news_window(year: int, quarter: int) -> tuple[str, str]:
    """Return (from_date, to_date) ISO strings for the 30-day window before the earnings call.

    Approximates the call date as the 15th of the first month of the following quarter.
    """
    quarter_end_month = {1: 3, 2: 6, 3: 9, 4: 12}[quarter]
    call_month = (quarter_end_month % 12) + 1
    call_year = year + (1 if call_month == 1 else 0)
    call_date = date(call_year, call_month, 15)
    from_date = call_date - timedelta(days=30)
    return from_date.isoformat(), call_date.isoformat()
