"""Kalshi Trade API v2 client. Has public endpoints, no auth required.

Endpoints used:
  GET /series              — list/search series tickers
  GET /events              — list events for a series (with cursor pagination)
  GET /markets             — list markets for an event (with cursor pagination)

Cache layout:
  data/raw/kalshi/events_{series_ticker}.json
  data/raw/kalshi/{event_ticker}.json
"""

import json
import logging
import time
from collections.abc import Generator
from pathlib import Path

import requests

from src.constants import KALSHI_BASE_URL, RAW_KALSHI_DIR

logger = logging.getLogger(__name__)

_RATE_LIMIT_SLEEP = 1.0  # seconds between paginated requests within an event
_INTER_EVENT_SLEEP = 1.5  # seconds between events in pull_all_markets_for_ticker
_PAGE_LIMIT = 100  # max results per page
_MAX_RETRIES = 3  # retries on 429 before giving up


class KalshiClient:
    """Public REST client for Kalshi Trade API v2.

    No authentication is required for reading settled market data.
    All results are cached to disk on first fetch.
    """

    def __init__(self, cache_root: Path = RAW_KALSHI_DIR) -> None:
        self.cache_root = Path(cache_root)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # series discovery

    def search_series(self, query: str, limit: int = 200) -> list[dict]:
        """List series whose title or ticker contains query (case-insensitive).

        Use this interactively to find the earnings-mention series ticker for
        each company before running the full pull.

        Args:
            query: keyword to filter on (e.g. "earnings", "KO", "AAPL").
            limit: max series to fetch from the API before filtering.
        """
        url = f"{KALSHI_BASE_URL}/series"
        params: dict = {"limit": limit}
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        series_list: list[dict] = resp.json().get("series", [])
        q = query.lower()
        return [
            s
            for s in series_list
            if q in s.get("title", "").lower() or q in s.get("ticker", "").lower()
        ]

    # events

    def get_events(self, series_ticker: str, status: str = "settled") -> list[dict]:
        """Fetch all events for a series ticker. Handles cursor-based pagination.

        Caches to data/raw/kalshi/events_{series_ticker}.json.
        """
        cache_path = self.cache_root / f"events_{series_ticker}.json"
        if cache_path.exists():
            logger.info("Cache hit (events): %s", cache_path)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        events = list(self._paginate_events(series_ticker, status))
        self.cache_root.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
        logger.info(
            "Cached %d events for series %s → %s",
            len(events),
            series_ticker,
            cache_path,
        )
        return events

    def _paginate_events(
        self, series_ticker: str, status: str
    ) -> Generator[dict, None, None]:
        """Yield all events for a series across pages."""
        url = f"{KALSHI_BASE_URL}/events"
        cursor: str | None = None
        while True:
            params: dict = {
                "series_ticker": series_ticker,
                "status": status,
                "limit": _PAGE_LIMIT,
            }
            if cursor:
                params["cursor"] = cursor
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            events: list[dict] = body.get("events", [])
            yield from events
            cursor = body.get("cursor")
            if not cursor or not events:
                break
            time.sleep(_RATE_LIMIT_SLEEP)

    # ── Markets ──────────────────────────────────────────────────────────────

    def get_markets(self, event_ticker: str, status: str = "settled") -> list[dict]:
        """Fetch all markets for an event ticker. Handles cursor-based pagination.

        Caches to data/raw/kalshi/{event_ticker}.json.
        Each market dict includes 'result' ("yes"/"no") and 'previous_price'
        (pre-call implied probability in dollars, i.e. 0.0–1.0).
        """
        cache_path = self.cache_root / f"{event_ticker}.json"
        if cache_path.exists():
            logger.info("Cache hit (markets): %s", cache_path)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        markets = list(self._paginate_markets(event_ticker, status))
        self.cache_root.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(markets, indent=2), encoding="utf-8")
        logger.info(
            "Cached %d markets for event %s → %s",
            len(markets),
            event_ticker,
            cache_path,
        )
        return markets

    def _paginate_markets(
        self, event_ticker: str, status: str
    ) -> Generator[dict, None, None]:
        """Yield all markets for an event across pages. Retries on 429."""
        url = f"{KALSHI_BASE_URL}/markets"
        cursor: str | None = None
        while True:
            params: dict = {
                "event_ticker": event_ticker,
                "status": status,
                "limit": _PAGE_LIMIT,
            }
            if cursor:
                params["cursor"] = cursor
            for attempt in range(_MAX_RETRIES):
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = 2**attempt * 2
                    logger.warning(
                        "429 rate limited; retrying in %ds (attempt %d/%d)",
                        wait,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            body = resp.json()
            markets: list[dict] = body.get("markets", [])
            yield from markets
            cursor = body.get("cursor")
            if not cursor or not markets:
                break
            time.sleep(_RATE_LIMIT_SLEEP)

    # High-level pull

    def pull_all_markets_for_ticker(
        self, ticker: str, series_ticker: str
    ) -> list[dict]:
        """Fetch all settled markets for a company's earnings-mention series.

        Args:
            ticker: company ticker (e.g. "KO") — used for logging only.
            series_ticker: Kalshi series ticker (e.g. "KXEARNINGSMENTIONKO" for KO).
                           Discover others with search_series("earnings").

        Returns:
            Flat list of market dicts. Relevant fields:
              market_ticker, title, result ("yes"/"no"), previous_price (0–1).
        """
        logger.info("Pulling Kalshi markets for %s (series=%s)", ticker, series_ticker)
        events = self.get_events(series_ticker)
        all_markets: list[dict] = []
        for i, event in enumerate(events):
            event_ticker = event.get("event_ticker") or event.get("ticker", "")
            if not event_ticker:
                logger.warning("Skipping event with no ticker: %s", event)
                continue
            markets = self.get_markets(event_ticker)
            all_markets.extend(markets)
            if i < len(events) - 1:
                time.sleep(_INTER_EVENT_SLEEP)
        logger.info("Total markets fetched for %s: %d", ticker, len(all_markets))
        return all_markets
