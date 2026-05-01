"""Feature extraction: TF-IDF over transcript text and numeric feature assembly.

Vectorizers are ALWAYS fit on training-company transcripts only, then transform
dev/test. This is enforced by passing train_texts separately.
"""
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import load_transcripts

logger = logging.getLogger(__name__)


NUMERIC_COLS = ["implied_prob", "hist_rate", "in_recent_news"]


def fit_tfidf(
    train_texts: list[str],
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 3,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on training texts only and return it."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )
    vectorizer.fit(train_texts)
    logger.info("TF-IDF fit: %d train docs, %d features",
                len(train_texts), len(vectorizer.vocabulary_))
    return vectorizer


def build_tfidf_features(
    train_texts: list[str],
    eval_texts: list[str],
    **kwargs,
) -> tuple[sp.csr_matrix, sp.csr_matrix, TfidfVectorizer]:
    """Fit TF-IDF on train_texts, transform both splits. Kept for single-eval callers."""
    vectorizer = fit_tfidf(train_texts, **kwargs)
    X_train = vectorizer.transform(train_texts)
    X_eval = vectorizer.transform(eval_texts)
    return X_train, X_eval, vectorizer


def build_numeric_features(df: pd.DataFrame, cols: list[str] | None = None) -> np.ndarray:
    """Numeric feature matrix from labels columns.

    `cols` defaults to NUMERIC_COLS (implied_prob, hist_rate, in_recent_news).
    Pass a subset to drop features — e.g. cols=["hist_rate", "in_recent_news"]
    for the implied_prob ablation. NaN values are imputed with 0.0.
    """
    cols = cols if cols is not None else NUMERIC_COLS
    X = df[cols].to_numpy(dtype=float, copy=True)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        n_nan = int(nan_mask.any(axis=1).sum())
        logger.info("build_numeric_features(%s): imputing 0.0 for %d rows with NaN", cols, n_nan)
        X[nan_mask] = 0.0
    return X


def build_row_context_texts(df: pd.DataFrame) -> list[str]:
    """For each row, concatenate the ticker's cleaned transcripts strictly before the call date.

    Returns a list of strings aligned with df's rows. Empty string for rows
    whose ticker has no prior transcripts, whose row lacks a usable
    `call_date`, or whose transcripts all lack parseable dates. The
    strictly-prior filter compares transcript `date` (from the
    `_dates.csv` sidecar) against the row's `call_date`; (year, quarter)
    tuples are unsafe here because fiscal years can be shifted relative
    to the calendar year.
    """
    transcripts_by_ticker: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for ticker in df["ticker"].unique():
        for t in load_transcripts(ticker):
            tdate = pd.to_datetime(t.get("date"), errors="coerce")
            if pd.isna(tdate):
                continue
            transcripts_by_ticker.setdefault(ticker, []).append((tdate, t["content"]))

    texts: list[str] = []
    empty_count = 0
    for _, row in df.iterrows():
        ticker = row["ticker"]
        pred_time = pd.to_datetime(row.get("call_date"), errors="coerce")
        if pd.isna(pred_time):
            empty_count += 1
            texts.append("")
            continue
        prior = [
            content for (tdate, content) in transcripts_by_ticker.get(ticker, [])
            if tdate < pred_time
        ]
        if not prior:
            empty_count += 1
            texts.append("")
        else:
            texts.append(" ".join(prior))
    if empty_count:
        logger.info("build_row_context_texts: %d/%d rows have no prior transcripts (empty context)",
                    empty_count, len(df))
    return texts


def stack_tfidf_and_numeric(X_tfidf: sp.spmatrix, X_numeric: np.ndarray) -> sp.csr_matrix:
    """Horizontally stack sparse TF-IDF with dense numeric features as sparse."""
    return sp.hstack([X_tfidf, sp.csr_matrix(X_numeric)], format="csr")


def build_recent_context_texts(df: pd.DataFrame, max_recent: int = 2) -> list[str]:
    """Per-row context built from the N most-recent strictly-prior transcripts, newest first.

    Differs from `build_row_context_texts` in two ways:
      1. Keeps only the `max_recent` newest prior transcripts (not all prior).
      2. Concatenates newest-first, so head-truncation at 512 tokens preserves
         the freshest content — matches the DistilBERT input convention.

    Strictly-prior is date-based: transcript `date` (from `_dates.csv`)
    compared against row `call_date`.
    """
    transcripts_by_ticker: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for ticker in df["ticker"].unique():
        for t in load_transcripts(ticker):
            tdate = pd.to_datetime(t.get("date"), errors="coerce")
            if pd.isna(tdate):
                continue
            transcripts_by_ticker.setdefault(ticker, []).append((tdate, t["content"]))

    texts: list[str] = []
    empty_count = 0
    for _, row in df.iterrows():
        ticker = row["ticker"]
        pred_time = pd.to_datetime(row.get("call_date"), errors="coerce")
        if pd.isna(pred_time):
            empty_count += 1
            texts.append("")
            continue
        prior = [
            (tdate, content) for (tdate, content) in transcripts_by_ticker.get(ticker, [])
            if tdate < pred_time
        ]
        if not prior:
            empty_count += 1
            texts.append("")
            continue
        prior.sort(key=lambda r: r[0], reverse=True)
        chosen = [content for (_, content) in prior[:max_recent]]
        texts.append(" ".join(chosen))
    if empty_count:
        logger.info(
            "build_recent_context_texts: %d/%d rows have no prior transcripts (empty context)",
            empty_count, len(df),
        )
    return texts
