"""Transcript cleaning, label construction, and cross-company splitting."""
import json
import logging
import random
import re
import unicodedata

import pandas as pd

from src.constants import (
    EXPANDED_TRANSCRIPTS_DIR,
    MIN_TRANSCRIPT_WORDS,
    RANDOM_SEED,
    RAW_FMP_DIR,
    TRANSCRIPTS_DIR,
)
from src.fmp_client import (
    _load_txt_transcript,
    _normalize_fmp_json,
    _parse_filename,
)

logger = logging.getLogger(__name__)


# Conservative safe-harbor sentence patterns. Only strip sentences that are
# clearly SEC boilerplate; leave organic speaker use of the same vocabulary
# intact (e.g. a CEO saying "forward-looking agreement" is not boilerplate).
_SAFE_HARBOR_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"[^.?!]*\bforward[- ]looking statements?\b[^.?!]*[.?!]",
        r"[^.?!]*\bactual results\b[^.?!]*\b(?:differ|vary)\s+materially\b[^.?!]*[.?!]",
        r"[^.?!]*\b(?:risk factors|safe harbor)\b[^.?!]*[.?!]",
        r"[^.?!]*\bnon[- ]?gaap\b[^.?!]*\b(?:gaap|reconcil\w*)\b[^.?!]*[.?!]",
        r"[^.?!]*\bprivate securities litigation reform act\b[^.?!]*[.?!]",
    )
]

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_RELATED_FOOTER_RE = re.compile(r"\n\s*Related\s*:.*\Z", re.DOTALL | re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_transcript(text: str) -> str:
    """NFKC normalize, strip footer/URLs/safe-harbor sentences, lowercase, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _RELATED_FOOTER_RE.sub("", text)
    text = _URL_RE.sub(" ", text)
    for pattern in _SAFE_HARBOR_PATTERNS:
        text = pattern.sub(" ", text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


_PROCESSED_NAME_RE = re.compile(r"^([A-Za-z]+)_(\d{4})Q(\d)$")
_DATES_SIDECAR = "_dates.csv"
_DATES_CACHE: dict[tuple[str, int, int], str | None] | None = None


def _load_dates_cache() -> dict[tuple[str, int, int], str | None]:
    """Lazily load the (ticker, year, quarter) → call_date sidecar written by preprocess."""
    global _DATES_CACHE
    if _DATES_CACHE is None:
        path = TRANSCRIPTS_DIR / _DATES_SIDECAR
        if path.exists():
            df = pd.read_csv(path)
            _DATES_CACHE = {
                (str(r.ticker), int(r.year), int(r.quarter)):
                    (None if pd.isna(r.call_date) else str(r.call_date))
                for r in df.itertuples(index=False)
            }
        else:
            _DATES_CACHE = {}
    return _DATES_CACHE


def load_transcripts(ticker: str) -> list[dict]:
    """Return cleaned transcripts for a ticker, sorted by (year, quarter) ascending.

    Each dict carries a `date` field (ISO string or None) sourced from
    `TRANSCRIPTS_DIR/_dates.csv`. Downstream code MUST filter priors by
    this `date` rather than by `(year, quarter)` — see FIX_LEAKAGE.md.
    """
    dates = _load_dates_cache()
    results: list[dict] = []
    for path in sorted(TRANSCRIPTS_DIR.glob(f"{ticker}_*Q*.txt")):
        m = _PROCESSED_NAME_RE.match(path.stem)
        if not m:
            logger.warning("Unexpected filename in processed/transcripts: %s", path.name)
            continue
        year = int(m.group(2))
        quarter = int(m.group(3))
        results.append(
            {
                "ticker": m.group(1),
                "year": year,
                "quarter": quarter,
                "content": path.read_text(encoding="utf-8"),
                "date": dates.get((ticker, year, quarter)),
            }
        )
    results.sort(key=lambda d: (d["year"], d["quarter"]))
    if not results:
        logger.warning("No cleaned transcripts found for ticker: %s", ticker)
    return results


def _iter_raw_transcripts(ticker: str) -> list[dict]:
    """Read raw transcripts from both data/raw/transcripts/ and data/raw/fmp/.

    Expanded corpus wins on (year, quarter) collisions.
    """
    out: list[dict] = []
    seen: set[tuple[int, int]] = set()

    expanded_dir = EXPANDED_TRANSCRIPTS_DIR / f"{ticker}_transcripts"
    if expanded_dir.exists():
        for p in sorted(expanded_dir.iterdir()):
            if p.suffix != ".txt":
                continue
            raw = _load_txt_transcript(p, ticker)
            key = (raw["year"], raw["quarter"])
            if key in seen:
                continue
            seen.add(key)
            out.append(raw)

    ticker_dir = RAW_FMP_DIR / ticker
    if ticker_dir.exists():
        for p in sorted(ticker_dir.iterdir()):
            if p.suffix == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                year, quarter = _parse_filename(p.stem)
                if (year, quarter) in seen:
                    continue
                seen.add((year, quarter))
                out.append(_normalize_fmp_json(data, ticker, year, quarter))
            elif p.suffix == ".txt":
                raw = _load_txt_transcript(p, ticker)
                key = (raw["year"], raw["quarter"])
                if key in seen:
                    continue
                seen.add(key)
                out.append(raw)
    return out


def preprocess_all_transcripts(tickers: list[str] | None = None) -> dict[str, int]:
    """Clean raw transcripts and write to data/processed/transcripts/. Returns per-ticker counts.

    Transcripts under MIN_TRANSCRIPT_WORDS after cleaning are logged and excluded.

    Also writes a sidecar `_dates.csv` with the actual call date extracted
    from each raw source (local txt header `Date:` line, or FMP's `date`
    response field). This sidecar is required by `load_transcripts` and
    by downstream `build_labels` / context builders to filter priors by
    real calendar date — see FIX_LEAKAGE.md.
    """
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    if tickers is None:
        discovered: set[str] = set()
        if RAW_FMP_DIR.exists():
            discovered.update(d.name for d in RAW_FMP_DIR.iterdir() if d.is_dir())
        if EXPANDED_TRANSCRIPTS_DIR.exists():
            for d in EXPANDED_TRANSCRIPTS_DIR.iterdir():
                if d.is_dir() and d.name.endswith("_transcripts"):
                    discovered.add(d.name.removesuffix("_transcripts"))
        tickers = sorted(discovered)

    counts: dict[str, int] = {}
    date_rows: list[dict] = []
    for ticker in tickers:
        written = 0
        for raw in _iter_raw_transcripts(ticker):
            year, quarter = raw["year"], raw["quarter"]
            if year == 0 or quarter == 0:
                logger.warning(
                    "Skipping %s: unparseable year/quarter (source=%s)",
                    ticker,
                    raw.get("filename", "?"),
                )
                continue
            cleaned = clean_transcript(raw["content"])
            word_count = len(cleaned.split())
            if word_count < MIN_TRANSCRIPT_WORDS:
                logger.warning(
                    "Skipping %s %dQ%d: only %d words (<%d)",
                    ticker,
                    year,
                    quarter,
                    word_count,
                    MIN_TRANSCRIPT_WORDS,
                )
                continue
            out_path = TRANSCRIPTS_DIR / f"{ticker}_{year}Q{quarter}.txt"
            out_path.write_text(cleaned, encoding="utf-8")
            written += 1
            date_rows.append({
                "ticker": ticker,
                "year": year,
                "quarter": quarter,
                "call_date": raw.get("date"),
                "source": raw.get("source"),
            })
        counts[ticker] = written
        logger.info("Preprocessed %s: %d transcripts written", ticker, written)

    if date_rows:
        dates_path = TRANSCRIPTS_DIR / _DATES_SIDECAR
        pd.DataFrame(date_rows).to_csv(dates_path, index=False)
        missing = sum(1 for r in date_rows if r["call_date"] in (None, "")
                      or (isinstance(r["call_date"], float) and pd.isna(r["call_date"])))
        logger.info("Wrote %s (%d rows, %d missing call_date)",
                    dates_path, len(date_rows), missing)
        # Invalidate the in-process cache so load_transcripts sees the fresh sidecar.
        global _DATES_CACHE
        _DATES_CACHE = None

    return counts


def _mention_rate(word: str, docs: list[str]) -> float:
    """Fraction of docs in which `word` appears as a whole-word match, case-insensitive."""
    if not docs:
        return float("nan")
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    hits = sum(1 for d in docs if pattern.search(d))
    return hits / len(docs)


def _calendar_quarter(month: int) -> int:
    return (month - 1) // 3 + 1


def build_labels(
    transcripts: list[dict],
    markets: list[dict],
) -> pd.DataFrame:
    """Build labels.csv from Kalshi markets + the full transcript corpus.

    `mentioned` is Kalshi's settled `result` field. `hist_rate` is the
    fraction of the row's ticker's transcripts **strictly before the
    market's close date** that contain `word` as a case-insensitive
    whole-word match. Rows whose ticker has no prior transcripts (or
    whose transcripts lack usable dates) get `hist_rate = NaN`.

    NOTE on the strictly-prior filter: we filter by actual call date
    (transcript `Date:` header / FMP `date` field) compared against the
    Kalshi market's `close_date` (fall back to `settlement_date`). The
    prior (year, quarter)-tuple comparison leaked the target call's
    own transcript into the "prior" set for calendar-year-fiscal
    tickers — see FIX_LEAKAGE.md for the diagnosis.
    """
    docs_by_ticker: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    missing_dates = 0
    for t in transcripts:
        tdate = pd.to_datetime(t.get("date"), errors="coerce")
        if pd.isna(tdate):
            missing_dates += 1
            continue
        docs_by_ticker.setdefault(t["ticker"], []).append((tdate, t["content"]))
    if missing_dates:
        logger.warning(
            "build_labels: %d transcripts lack a parseable date and will be excluded "
            "from hist_rate priors (fix by re-running preprocess_all_transcripts).",
            missing_dates,
        )

    hist_cache: dict[tuple[str, pd.Timestamp, str], float] = {}

    rows = []
    for m in markets:
        ticker = m["company_label"]
        word = m["word"]

        # Prediction time = close_date (trading halted) preferred over
        # settlement_date (payout). For earnings-mention markets these
        # are almost always the same day, but close_date is semantically
        # the earlier of the two and therefore safer as a cutoff.
        pred_time_str = m.get("close_date") or m.get("settlement_date")
        pred_time = pd.to_datetime(pred_time_str, errors="coerce")
        if pd.isna(pred_time):
            logger.warning(
                "Unparseable close/settlement for %s: %r",
                m.get("market_ticker"), pred_time_str,
            )
            year, quarter, call_date_iso = 0, 0, None
        else:
            year = int(pred_time.year)
            quarter = _calendar_quarter(int(pred_time.month))
            call_date_iso = pred_time.date().isoformat()

        if "label" in m and pd.notna(m["label"]):
            mentioned = int(m["label"])
        else:
            mentioned = 1 if str(m.get("result", "")).lower() == "yes" else 0

        implied_prob = float(m["implied_prob"]) if pd.notna(m.get("implied_prob")) else float("nan")
        market_ticker = m.get("market_ticker", "")

        cache_key = (ticker, pred_time, word.lower())
        if cache_key not in hist_cache:
            if pd.isna(pred_time):
                hist_cache[cache_key] = float("nan")
            else:
                prior_docs = [
                    content for (tdate, content) in docs_by_ticker.get(ticker, [])
                    if tdate < pred_time
                ]
                hist_cache[cache_key] = _mention_rate(word, prior_docs)
        hist_rate = hist_cache[cache_key]

        in_recent_news = 0

        rows.append(
            {
                "ticker": ticker,
                "year": year,
                "quarter": quarter,
                "call_date": call_date_iso,
                "word": word,
                "mentioned": mentioned,
                "implied_prob": implied_prob,
                "hist_rate": hist_rate,
                "in_recent_news": in_recent_news,
                "market_ticker": market_ticker,
            }
        )

    df = pd.DataFrame(rows, columns=[
        "ticker", "year", "quarter", "call_date", "word", "mentioned",
        "implied_prob", "hist_rate", "in_recent_news", "market_ticker",
    ])
    logger.info(
        "build_labels: %d rows, %d unique tickers, hist_rate non-null: %d",
        len(df), df["ticker"].nunique(), df["hist_rate"].notna().sum(),
    )
    return df


def assign_company_splits(
    tickers: list[str],
    train_frac: float = 0.70,
    dev_frac: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Partition tickers into (train, dev, test) by random shuffle at `seed`.

    Test fraction is `1 - train_frac - dev_frac`. Uses floor rounding for
    train and dev so any leftover tickers fall into test.
    """
    shuffled = sorted(tickers)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    train = sorted(shuffled[:n_train])
    dev = sorted(shuffled[n_train : n_train + n_dev])
    test = sorted(shuffled[n_train + n_dev :])
    return train, dev, test


def cross_company_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    dev_frac: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train/dev/test by company. All rows for a ticker go to one split."""
    tickers = sorted(df["ticker"].unique())
    train_tickers, dev_tickers, test_tickers = assign_company_splits(
        tickers, train_frac=train_frac, dev_frac=dev_frac, seed=seed
    )
    train = df[df["ticker"].isin(train_tickers)].reset_index(drop=True)
    dev = df[df["ticker"].isin(dev_tickers)].reset_index(drop=True)
    test = df[df["ticker"].isin(test_tickers)].reset_index(drop=True)
    logger.info(
        "cross_company_split: %d train / %d dev / %d test rows "
        "(%d / %d / %d tickers)",
        len(train), len(dev), len(test),
        len(train_tickers), len(dev_tickers), len(test_tickers),
    )
    return train, dev, test
