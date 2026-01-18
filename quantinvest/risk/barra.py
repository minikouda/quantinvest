"""
Barra risk model implementation.

This module implements the Barra risk model with factor structure
and specific risk estimation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union, Iterable, Any, List


class BarraRiskModel:
    
    def __init__(self, 
                 lambda_value: float = 0.94,
                 half_life: int = 42,
                 window: int = 252):
        """
        Initialize Barra risk model.
        
        Parameters:
        -----------
        lambda_value : float
            EWMA decay parameter
        half_life : int
            Half-life for exponential weighting
        window : int
            Estimation window length
        """
        self.lambda_value = lambda_value
        self.half_life = half_life
        self.window = window
        self.factor_returns_ = None
        self.specific_risk_ = None
        self.factor_covariance_ = None
        self.specific_variance_ = None
        self.specific_returns_ = None
        self.factors_ = None
        self.assets_ = None
        
    def fit(self, 
            factor_exposures: pd.DataFrame,
            returns: pd.DataFrame) -> 'BarraRiskModel':
        """
        Fit the Barra risk model.
        Returns:
        --------
        self : BarraRiskModel
        """
        # Estimate factor returns and specific returns
        self.factor_returns_, self.specific_returns_ = self._estimate_factor_returns(
            factor_exposures, returns
        )
        
        # Estimate factor covariance matrix
        self.factor_covariance_ = self._estimate_factor_covariance(self.factor_returns_)
        
        # Estimate specific risk
        self.specific_risk_ = self._estimate_specific_risk(self.specific_returns_)
        
        return self

    def predict_asset_covariance(
        self,
        factor_exposures: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        *,
        date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Compute the implied asset covariance matrix for a given date.

        Uses the standard Barra-style decomposition:

        Sigma_assets = X * Sigma_factor * X^T + diag(specific_variance)

        Parameters
        ----------
        factor_exposures:
            Factor exposures for the requested date. Recommended format is a DataFrame
            with index=dates and MultiIndex columns (factor, asset) as produced by
            `quantinvest.factors.style_factors.StyleFactors.compute`.
            A dict of {factor_name: DataFrame(dates x assets)} is also accepted.
        date:
            Date to use. Defaults to the latest date available in `factor_exposures`.
        """
        if self.factor_covariance_ is None or self.specific_variance_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        exposures_panel = self._coerce_exposures_to_panel(factor_exposures)
        if date is None:
            date = pd.to_datetime(exposures_panel.index.max())
        else:
            date = pd.to_datetime(date)

        X = self._slice_exposures_as_matrix(exposures_panel, date=date)  # assets x factors
        factors = list(X.columns)
        assets = list(X.index)

        factor_cov = self.factor_covariance_.reindex(index=factors, columns=factors)
        spec_var = self.specific_variance_.reindex(index=assets)

        Xv = X.to_numpy(dtype=float)
        F = factor_cov.to_numpy(dtype=float)
        D = np.diag(spec_var.fillna(0.0).to_numpy(dtype=float))

        cov_assets = Xv @ F @ Xv.T + D
        return pd.DataFrame(cov_assets, index=assets, columns=assets)
    
    def _estimate_factor_returns(
        self,
        factor_exposures: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        returns: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Estimate factor returns via cross-sectional regression.

        For each date t, solve (OLS):

            r_t = alpha_t + X_t f_t + eps_t

        where r_t is the cross-section of asset returns, X_t is the exposure matrix
        (assets x factors), f_t are factor returns, and eps_t are specific returns.

        Input format
        ------------
        Recommended `factor_exposures` format is:

        - index: dates
        - columns: MultiIndex with two levels: (factor, asset) or (asset, factor)

        This matches the output of `StyleFactors.compute` (factor first).

        Returns
        -------
        factor_returns: DataFrame (dates x factors)
        specific_returns: DataFrame (dates x assets)
        """
        exposures_panel = self._coerce_exposures_to_panel(factor_exposures)

        common_dates = returns.index.intersection(exposures_panel.index)
        returns_aligned = returns.loc[common_dates].copy()
        exposures_panel = exposures_panel.loc[common_dates]

        factor_names = self._infer_factor_names(exposures_panel)
        asset_names = list(returns_aligned.columns)

        factor_returns = pd.DataFrame(index=common_dates, columns=factor_names, dtype=float)
        specific_returns = pd.DataFrame(index=common_dates, columns=asset_names, dtype=float)

        min_obs = max(10, len(factor_names) + 2)
        window_dates = common_dates[-self.window :] if self.window and len(common_dates) > self.window else common_dates

        for dt in window_dates:
            y = returns_aligned.loc[dt]
            X = self._slice_exposures_as_matrix(exposures_panel, date=dt)  # assets x factors

            # Align cross-section
            X = X.reindex(index=y.index)
            valid_mask = y.notna() & X.notna().all(axis=1)
            if int(valid_mask.sum()) < min_obs:
                continue

            yv = y.loc[valid_mask].to_numpy(dtype=float)
            Xv = X.loc[valid_mask]

            # Add intercept
            design = np.column_stack([np.ones((Xv.shape[0], 1)), Xv.to_numpy(dtype=float)])

            # Solve least squares robustly
            beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
            f = beta[1:]

            factor_returns.loc[dt, Xv.columns] = f

            resid = yv - design @ beta
            specific_returns.loc[dt, Xv.index] = resid

        self.factors_ = list(factor_returns.columns)
        self.assets_ = list(specific_returns.columns)
        return factor_returns, specific_returns
    
    def _estimate_factor_covariance(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate factor covariance matrix using EWMA.

        Returns a single covariance matrix for the latest fitted window.
        """
        alpha = self._ewma_alpha()
        fr = factor_returns.dropna(how="any")
        if self.window and len(fr) > self.window:
            fr = fr.iloc[-self.window :]
        if fr.empty:
            return pd.DataFrame(index=factor_returns.columns, columns=factor_returns.columns, dtype=float)

        cov = self._ewma_covariance(fr.to_numpy(dtype=float), alpha=alpha)
        return pd.DataFrame(cov, index=fr.columns, columns=fr.columns)
    
    def _estimate_specific_risk(self, specific_returns: pd.DataFrame) -> pd.Series:
        """Estimate specific (idiosyncratic) risk using EWMA variance."""
        alpha = self._ewma_alpha()
        sr = specific_returns.copy()
        if self.window and len(sr) > self.window:
            sr = sr.iloc[-self.window :]

        variances: Dict[str, float] = {}
        for col in sr.columns:
            x = pd.to_numeric(sr[col], errors="coerce").dropna().to_numpy(dtype=float)
            if x.size < 2:
                variances[col] = np.nan
                continue
            var = self._ewma_variance(x, alpha=alpha)
            variances[col] = float(var)

        self.specific_variance_ = pd.Series(variances, index=sr.columns, dtype=float)
        # Return as Series (sqrt of variance); keep NaNs where variance is missing
        return self.specific_variance_.pow(0.5)

    # ---------------------------- helpers ----------------------------
    def _ewma_alpha(self) -> float:
        """Convert (half-life, lambda) inputs into an EWMA alpha in (0, 1]."""
        if self.half_life is not None and self.half_life > 0:
            decay = float(np.exp(np.log(0.5) / float(self.half_life)))
            alpha = 1.0 - decay
        else:
            alpha = 1.0 - float(self.lambda_value)

        return float(np.clip(alpha, 1e-6, 1.0))

    @staticmethod
    def _ewma_variance(x: np.ndarray, *, alpha: float) -> float:
        """EWMA variance with online mean update (adjust=False style)."""
        mean = x[0]
        var = 0.0
        for xt in x[1:]:
            prev_mean = mean
            mean = (1.0 - alpha) * mean + alpha * xt
            var = (1.0 - alpha) * var + alpha * (xt - prev_mean) * (xt - mean)
        return float(max(var, 0.0))

    @classmethod
    def _ewma_covariance(cls, X: np.ndarray, *, alpha: float) -> np.ndarray:
        """EWMA covariance with online mean update.

        Parameters
        ----------
        X:
            2D array (n_samples x n_features), assumed finite.
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        mean = X[0].astype(float)
        cov = np.zeros((X.shape[1], X.shape[1]), dtype=float)
        for xt in X[1:]:
            xt = xt.astype(float)
            prev_mean = mean
            mean = (1.0 - alpha) * mean + alpha * xt
            delta = xt - prev_mean
            cov = (1.0 - alpha) * cov + alpha * np.outer(delta, xt - mean)
        return 0.5 * (cov + cov.T)

    @staticmethod
    def _coerce_exposures_to_panel(
        factor_exposures: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ) -> pd.DataFrame:
        """Normalize exposures into a DataFrame with MultiIndex columns (factor, asset)."""
        if isinstance(factor_exposures, dict):
            # dict[factor] -> DataFrame(dates x assets)
            return pd.concat(factor_exposures, axis=1)

        if not isinstance(factor_exposures, pd.DataFrame):
            raise TypeError("factor_exposures must be a DataFrame or dict[str, DataFrame]")

        # If already in (dates x MultiIndex columns) format, return as-is
        if isinstance(factor_exposures.columns, pd.MultiIndex):
            return factor_exposures

        # If provided as (date, asset) MultiIndex rows with factor columns, pivot to panel
        if isinstance(factor_exposures.index, pd.MultiIndex) and factor_exposures.index.nlevels == 2:
            # assume index levels: (date, asset)
            df = factor_exposures.copy()
            df = df.sort_index()
            # stack factors into MultiIndex columns (factor, asset)
            stacked = df.stack()
            stacked.name = "exposure"
            # stacked index: (date, asset, factor)
            stacked.index = stacked.index.set_names(["date", "asset", "factor"])
            panel_any = stacked.unstack(["factor", "asset"]).sort_index(axis=1)
            if isinstance(panel_any, pd.Series):
                panel = panel_any.to_frame()
            else:
                panel = panel_any
            if isinstance(panel.columns, pd.MultiIndex):
                panel.columns = panel.columns.set_names(["factor", "asset"])
            return panel

        raise ValueError(
            "Unsupported factor_exposures format. Expected MultiIndex columns (factor, asset) "
            "or a dict[factor]->DataFrame(dates x assets), or MultiIndex index (date, asset) with factor columns."
        )

    @staticmethod
    def _infer_factor_names(exposures_panel: pd.DataFrame) -> List[str]:
        if not isinstance(exposures_panel.columns, pd.MultiIndex) or exposures_panel.columns.nlevels != 2:
            raise ValueError("exposures_panel must have 2-level MultiIndex columns")
        level0 = list(exposures_panel.columns.get_level_values(0).unique())
        level1 = list(exposures_panel.columns.get_level_values(1).unique())
        if len(level0) <= len(level1):
            return level0
        return level1

    @staticmethod
    def _slice_exposures_as_matrix(exposures_panel: pd.DataFrame, *, date: Any) -> pd.DataFrame:
        """Return exposures at `date` as DataFrame (assets x factors)."""
        if not isinstance(exposures_panel.columns, pd.MultiIndex) or exposures_panel.columns.nlevels != 2:
            raise ValueError("exposures_panel must have 2-level MultiIndex columns")

        dt = pd.to_datetime(date)
        if dt not in exposures_panel.index:
            raise KeyError(f"date {dt} not found in exposures")

        row = exposures_panel.loc[dt]
        if isinstance(row, pd.DataFrame):
            row = row.squeeze(axis=0)
        if not isinstance(row, pd.Series) or not isinstance(row.index, pd.MultiIndex):
            raise ValueError("Expected a Series with MultiIndex for exposures row")

        m0 = row.unstack(0)
        m1 = row.unstack(1)
        mat = m0 if m0.shape[1] <= m0.shape[0] else m1

        return mat.apply(pd.to_numeric, errors="coerce")