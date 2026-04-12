"""Feature extraction: TF-IDF over transcript text and numeric feature assembly.

Vectorizers are ALWAYS fit on training data only, then transform dev/test.
This is enforced by the function signatures (train_texts passed separately).
"""
import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.constants import RANDOM_SEED

logger = logging.getLogger(__name__)


def build_tfidf_features(
    train_texts: list[str],
    eval_texts: list[str],
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 3,
    max_df: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Fit TF-IDF on train_texts, transform both splits.

    Vectorizer is fit on training data only — transform only on eval/test.

    Args:
        train_texts: list of cleaned transcript strings for training.
        eval_texts: list of cleaned transcript strings for dev or test.

    Returns:
        (X_train, X_eval, fitted_vectorizer)
    """
    raise NotImplementedError("build_tfidf_features — implement in P3.")


def build_numeric_features(df: pd.DataFrame) -> np.ndarray:
    """Assemble numeric feature matrix from labels.csv columns.

    Features: implied_prob, hist_rate, in_recent_news.

    Args:
        df: DataFrame with at least those three columns.

    Returns:
        np.ndarray of shape (n_examples, 3).
    """
    raise NotImplementedError("build_numeric_features — implement in P3.")
