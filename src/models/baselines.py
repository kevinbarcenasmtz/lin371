"""Baseline classifiers.

  MajorityClassifier  — predict the dominant training class
  KalshiConsensus     — predict 1 if implied_prob > 0.5
  BuyAllYes           — always predict 1
  HistFreqClassifier  — predict 1 if hist_rate > threshold (threshold tuned on dev)
"""
import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Predicts the majority class observed in training data."""

    def fit(self, X: Any, y: Any) -> "MajorityClassifier":
        values, counts = np.unique(np.asarray(y), return_counts=True)
        self.majority_ = int(values[int(np.argmax(counts))])
        return self

    def predict(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.full(n, self.majority_, dtype=int)


class KalshiConsensus(BaseEstimator, ClassifierMixin):
    """Predict 1 if implied_prob > 0.5, else 0. Takes implied_prob array as X."""

    def fit(self, X: Any, y: Any) -> "KalshiConsensus":
        return self

    def predict(self, implied_probs: Any) -> np.ndarray:
        return (np.asarray(implied_probs, dtype=float) > 0.5).astype(int)


class BuyAllYes(BaseEstimator, ClassifierMixin):
    """Always predict 1 (mentioned)."""

    def fit(self, X: Any, y: Any) -> "BuyAllYes":
        return self

    def predict(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.ones(n, dtype=int)


class HistFreqClassifier(BaseEstimator, ClassifierMixin):
    """Predict 1 if hist_rate > threshold; threshold tuned on dev set.

    Takes hist_rate array as X (not transcript features).
    NaN hist_rates are treated as 0.0 for thresholding (no history -> predict 0).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def _tune(self, hist_rates: np.ndarray, y: np.ndarray) -> float:
        rates = np.where(np.isnan(hist_rates), 0.0, hist_rates)
        grid = np.round(np.arange(0.0, 1.0, 0.05), 2)
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            preds = (rates > t).astype(int)
            f1 = f1_score(y, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        logger.info("HistFreq threshold tuned: %.2f (dev macro-F1=%.4f)", best_t, best_f1)
        return best_t

    def fit(
        self,
        X_dev: Any,
        y_dev: Any,
    ) -> "HistFreqClassifier":
        """Tune threshold on dev hist_rates by grid search over macro-F1."""
        self.threshold = self._tune(np.asarray(X_dev, dtype=float), np.asarray(y_dev))
        return self

    def predict(self, hist_rates: Any) -> np.ndarray:
        rates = np.asarray(hist_rates, dtype=float)
        rates = np.where(np.isnan(rates), 0.0, rates)
        return (rates > self.threshold).astype(int)
