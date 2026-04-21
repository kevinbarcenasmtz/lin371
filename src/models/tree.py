"""Decision tree classifier (section 6 of IMPLEMENTATION.md).

max_depth is tuned on the dev set; included for interpretability.
"""
import logging

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from src.constants import RANDOM_SEED

logger = logging.getLogger(__name__)


def train_tree(
    X_train: sp.csr_matrix | np.ndarray,
    y_train: np.ndarray,
    X_dev: sp.csr_matrix | np.ndarray,
    y_dev: np.ndarray,
    max_depth_range: list[int | None] | None = None,
) -> DecisionTreeClassifier:
    """Fit a decision tree with max_depth tuned on dev macro-F1, then refit on full train."""
    if max_depth_range is None:
        max_depth_range = [3, 5, 10, 15, None]

    best_depth, best_f1 = None, -1.0
    for d in max_depth_range:
        clf = DecisionTreeClassifier(
            max_depth=d,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_dev)
        f1 = f1_score(y_dev, preds, average="macro", zero_division=0)
        logger.info("Tree depth=%s dev macro-F1=%.4f", d, f1)
        if f1 > best_f1:
            best_depth, best_f1 = d, f1

    logger.info("Tree best depth=%s (dev macro-F1=%.4f)", best_depth, best_f1)
    best = DecisionTreeClassifier(
        max_depth=best_depth,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    best.fit(X_train, y_train)
    return best
