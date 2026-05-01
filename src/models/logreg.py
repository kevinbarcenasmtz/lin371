"""TF-IDF + Logistic Regression classifier.

Regularization C is tuned via 5-fold CV on training data only.
"""
import logging

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.constants import RANDOM_SEED

logger = logging.getLogger(__name__)


def train_logreg(
    X_train: sp.csr_matrix | np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    c_values: list[float] | None = None,
) -> LogisticRegression:
    """Fit logistic regression with C tuned via cross-validation on training data."""
    if c_values is None:
        c_values = [0.01, 0.1, 1.0, 10.0]
    base = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    grid = GridSearchCV(
        base,
        param_grid={"C": c_values},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    logger.info("LogReg CV best C=%s (macro-F1=%.4f)",
                grid.best_params_["C"], grid.best_score_)
    return grid.best_estimator_
