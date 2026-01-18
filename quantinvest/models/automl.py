"""AutoML utilities for time-series regression.

This module implements a lightweight AutoML helper tailored for
financial time-series problems. It wraps several advanced models
(e.g., gradient boosting, random forests, neural networks) and
performs walk-forward cross-validation to pick the best performer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ScoreType = Union[str, callable]


def _check_array(X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()
    return np.asarray(X)


def _default_estimators(random_state: Optional[int]) -> Dict[str, BaseEstimator]:
    """Return a dictionary of reasonably strong baseline estimators."""

    return {
        "hgb": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            random_state=random_state,
        ),
        "rf": RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        ),
        "lasso": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.0005, max_iter=5000, random_state=random_state)),
            ]
        ),
        "mlp": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        learning_rate_init=1e-3,
                        batch_size=128,
                        max_iter=500,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


class TimeSeriesAutoMLRegressor:
    """Simple AutoML helper for chronological datasets.

    Parameters
    ----------
    estimators : dict, optional
        Mapping of estimator name -> estimator instance. If omitted, a curated
        list including histogram gradient boosting, random forest, lasso, and
        a shallow MLP is used.
    scoring : {"neg_mean_squared_error", "neg_mean_absolute_error", "r2"}
        Metric optimized during cross-validation. Higher is better.
    n_splits : int
        Number of splits for :class:`sklearn.model_selection.TimeSeriesSplit`.
    max_train_size : int, optional
        Forward expanding window size. Uses all history by default.
    random_state : int, optional
        Used for estimator reproducibility where applicable.
    verbose : bool
        Whether to log per-fold progress.
    """

    def __init__(
        self,
        estimators: Optional[Dict[str, BaseEstimator]] = None,
        *,
        scoring: str = "neg_mean_squared_error",
        n_splits: int = 5,
        max_train_size: Optional[int] = None,
        random_state: Optional[int] = 0,
        verbose: bool = False,
    ) -> None:
        self.estimators = estimators or _default_estimators(random_state)
        self.scoring = scoring
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.random_state = random_state
        self.verbose = verbose

        self.best_estimator_: Optional[BaseEstimator] = None
        self.best_estimator_name_: Optional[str] = None
        self.cv_results_: List[Dict[str, Union[str, float]]] = []

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "TimeSeriesAutoMLRegressor":
        X_arr = _check_array(X)
        y_arr = _check_array(y).ravel()

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)

        best_score = -np.inf
        best_estimator = None
        best_name = None
        self.cv_results_.clear()

        for name, estimator in self.estimators.items():
            fold_scores: List[float] = []
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
                est = clone(estimator)
                est.fit(X_arr[train_idx], y_arr[train_idx])
                preds = est.predict(X_arr[val_idx])
                score = self._score(y_arr[val_idx], preds)
                fold_scores.append(score)
                if self.verbose:
                    print(f"[{name}] fold {fold_idx+1}/{self.n_splits} score={score:.6f}")
            mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
            self.cv_results_.append({
                "estimator": name,
                "mean_score": mean_score,
                "std_score": float(np.std(fold_scores)) if fold_scores else np.nan,
            })
            if mean_score > best_score:
                best_score = mean_score
                best_estimator = clone(estimator)
                best_name = name

        if best_estimator is None:
            raise RuntimeError("No estimator could be trained; check data sufficiency")

        best_estimator.fit(X_arr, y_arr)
        self.best_estimator_ = best_estimator
        self.best_estimator_name_ = best_name
        if self.verbose:
            print(f"Selected estimator: {best_name} (score={best_score:.6f})")
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.best_estimator_ is None:
            raise NotFittedError("TimeSeriesAutoMLRegressor has not been fit")
        X_arr = _check_array(X)
        estimator = cast(Any, self.best_estimator_)
        return estimator.predict(X_arr)

    def get_cv_results(self) -> pd.DataFrame:
        """Return cross-validation summary as a DataFrame."""
        return pd.DataFrame(self.cv_results_)

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.scoring == "neg_mean_squared_error":
            return -mean_squared_error(y_true, y_pred)
        if self.scoring == "neg_mean_absolute_error":
            return -mean_absolute_error(y_true, y_pred)
        if self.scoring == "r2":
            return r2_score(y_true, y_pred)
        raise ValueError(f"Unsupported scoring metric: {self.scoring}")