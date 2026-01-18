"""Style factor implementations (clean, production-ready).

This module provides implementations for:
- Beta (exponentially-weighted OLS vs market)
- Momentum (RSTR with lookback and lag)
- Liquidity (STOM/STOQ/STOA composite)
- Residual volatility (DASTD/CMRA/HSIGMA composite)

It mirrors the logic from legacy scripts in `src/` but with a stable API.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .base import FactorBase


DataFrameMap = Dict[str, pd.DataFrame]


class StyleFactors(FactorBase):
    def __init__(
        self,
        name: str = "StyleFactors",
        risk_free_rate: float = 0.0001,
        winsorize_limits: Tuple[float, float] = (0.01, 0.99),
    ) -> None:
        super().__init__(name=name, winsorize_limits=winsorize_limits)
        self.default_risk_free_rate = risk_free_rate
        self.beta_residuals_: Optional[pd.DataFrame] = None
        self.raw_components_: Dict[str, pd.DataFrame] = {}

    def compute(
        self,
        data: DataFrameMap,
        *,
        market_returns_key: str = "market_returns",
        returns_key: str = "returns",
        prices_key: str = "close",
        volume_key: str = "volume",
        float_shares_key: str = "float_shares",
        market_cap_key: str = "market_cap",
        turnover_key: str = "turnover",
        risk_free_key: str = "risk_free",
        winsorize: bool = True,
        standardize: bool = True,
        orthogonalize: bool = True,
        beta_window: int = 252,
        beta_half_life: int = 63,
        momentum_lookback: int = 504,
        momentum_lag: int = 21,
        momentum_half_life: int = 126,
        volatility_window: int = 252,
        volatility_half_life: int = 63,
    ) -> pd.DataFrame:
        """Compute style factors and return a concatenated DataFrame."""

        returns = data.get(returns_key)
        prices = data.get(prices_key)
        if returns is None:
            if prices is None:
                raise ValueError("Either returns or prices must be supplied.")
            returns = prices.pct_change()

        market_caps = data.get(market_cap_key)
        float_shares = data.get(float_shares_key)
        if market_caps is None:
            if prices is None or float_shares is None:
                raise ValueError("market_cap missing and cannot be inferred.")
            market_caps = prices * float_shares

        turnover = data.get(turnover_key)
        volumes = data.get(volume_key)
        if turnover is None:
            if volumes is None or float_shares is None:
                raise ValueError("turnover missing and cannot be inferred.")
            turnover = volumes / float_shares.replace(0, np.nan)

        market_returns = self._broadcast_to_frame(
            data.get(market_returns_key), returns.index, returns.columns
        )
        if market_returns is None:
            raise ValueError("market_returns are required for beta estimation.")

        risk_free = self._broadcast_to_frame(
            data.get(risk_free_key, self.default_risk_free_rate),
            returns.index,
            returns.columns,
        )
        if risk_free is None:
            risk_free = pd.DataFrame(
                self.default_risk_free_rate, index=returns.index, columns=returns.columns
            )

        def _align(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.reindex(index=returns.index, columns=returns.columns)

        market_caps = _align(market_caps)
        turnover = _align(turnover)
        market_returns = _align(market_returns)
        risk_free = _align(risk_free)

        factors: Dict[str, pd.DataFrame] = {}

        # Use numpy ufunc on values and wrap back to DataFrame to preserve type for static checkers
        _mc = market_caps.replace(0, np.nan)
        size_raw = pd.DataFrame(np.log(_mc.values), index=_mc.index, columns=_mc.columns)
        factors["size"] = self.preprocess(size_raw, winsorize=winsorize, standardize=standardize)

        # Nonlinear Size (TriCap): z^3 orthogonalized to z
        nonlinear_size_raw = self._compute_nonlinear_size(size_raw)
        factors["nonlinear_size"] = self.preprocess(
            nonlinear_size_raw, winsorize=winsorize, standardize=standardize
        )

        beta_raw, beta_resid = self._compute_beta(
            stock_returns=returns,
            market_returns=market_returns,
            risk_free=risk_free,
            window=beta_window,
            half_life=beta_half_life,
        )
        self.beta_residuals_ = beta_resid
        factors["beta"] = self.preprocess(beta_raw, winsorize=winsorize, standardize=standardize)

        momentum_raw = self._compute_momentum(
            returns=returns,
            risk_free=risk_free,
            lookback=momentum_lookback,
            lag=momentum_lag,
            half_life=momentum_half_life,
        )
        factors["momentum"] = self.preprocess(momentum_raw, winsorize=winsorize, standardize=standardize)

        liquidity_raw = self._compute_liquidity(turnover)
        if orthogonalize:
            liquidity_raw = self._cross_sectional_residual(liquidity_raw, {"size": factors["size"]})
        factors["liquidity"] = self.preprocess(liquidity_raw, winsorize=winsorize, standardize=standardize)

        rv = self._compute_residual_volatility(
            returns=returns,
            risk_free=risk_free,
            beta_residuals=beta_resid,
            window=volatility_window,
            half_life=volatility_half_life,
        )
        if orthogonalize:
            proj = {"beta": factors["beta"], "size": factors["size"]}
            rv = {k: self._cross_sectional_residual(v, proj) for k, v in rv.items()}
        for k, v in rv.items():
            factors[k] = self.preprocess(v, winsorize=winsorize, standardize=standardize)

        self.raw_components_ = {k: v.copy() for k, v in factors.items()}
        return pd.concat(factors, axis=1)

    # ---------------------- helpers ----------------------
    def _compute_beta(
        self,
        *,
        stock_returns: pd.DataFrame,
        market_returns: pd.DataFrame,
        risk_free: pd.DataFrame,
        window: int,
        half_life: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        stock_excess = stock_returns - risk_free
        market_excess = market_returns - risk_free

        lam = np.log(2.0) / half_life
        weights = np.exp(-lam * np.arange(window)[::-1])
        weights /= weights.sum()

        beta = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype=float)
        resid = beta.copy()

        lr = LinearRegression()
        for j, col in enumerate(stock_returns.columns):
            y_all = stock_excess[col].to_numpy()
            x_all = market_excess[col].to_numpy()
            for t in range(window - 1, len(y_all)):
                sl = slice(t - window + 1, t + 1)
                x = x_all[sl]
                y = y_all[sl]
                if np.isnan(x).all() or np.isnan(y).all():
                    continue
                x = self._mean_impute_vector(x)
                y = self._mean_impute_vector(y)
                lr.fit(x.reshape(-1, 1), y, sample_weight=weights)
                beta.iat[t, j] = lr.coef_[0]
                resid.iat[t, j] = y[-1] - lr.predict(x.reshape(-1, 1))[-1]
        return beta, resid

    def _compute_momentum(
        self,
        *,
        returns: pd.DataFrame,
        risk_free: pd.DataFrame,
        lookback: int,
        lag: int,
        half_life: int,
    ) -> pd.DataFrame:
        window = lookback + lag
        lam = np.log(2.0) / half_life
        w = np.exp(-lam * np.arange(window))[::-1]
        w /= w.sum()

        # Ensure log-excess returns remain a DataFrame
        _re = (returns - risk_free)
        log_excess = pd.DataFrame(np.log1p(_re.values), index=_re.index, columns=_re.columns)
        out = np.full((log_excess.shape[0], log_excess.shape[1]), np.nan)
        for t in range(window - 1, log_excess.shape[0]):
            sl = slice(t - window + 1, t + 1)
            block = log_excess.iloc[sl].values
            if np.isnan(block).all():
                continue
            block = self._mean_impute_matrix(block)
            out[t, :] = w @ block
        return pd.DataFrame(out, index=log_excess.index, columns=log_excess.columns).shift(lag)

    def _compute_nonlinear_size(self, log_cap: pd.DataFrame) -> pd.DataFrame:
        """Compute Nonlinear Size (TriCap):
        - cross-sectionally standardize log_cap to z
        - compute z^3
        - regress z^3 on z with intercept per date and return residuals
        """
        # cross-sectional standardization (z-score) row-wise
        def _z_row(row: pd.Series) -> pd.Series:
            m = row.mean(skipna=True)
            s = row.std(skipna=True)
            if s == 0 or np.isnan(s):
                return row * np.nan
            return (row - m) / s

        z = log_cap.apply(_z_row, axis=1)
        z3 = z ** 3
        # orthogonalize z3 to z cross-sectionally
        return self._cross_sectional_residual(z3, {"size_z": z})

    def _compute_liquidity(self, turnover: pd.DataFrame) -> pd.DataFrame:
        # Apply elementwise log using numpy ufuncs and wrap back to DataFrame
        _stom = turnover.rolling(21, min_periods=21).sum().clip(lower=1e-12)
        stom = pd.DataFrame(np.log(_stom.values), index=_stom.index, columns=_stom.columns)
        _stoq = (turnover.rolling(63, min_periods=63).sum() / 3.0).clip(lower=1e-12)
        stoq = pd.DataFrame(np.log(_stoq.values), index=_stoq.index, columns=_stoq.columns)
        _stoa = (turnover.rolling(252, min_periods=252).sum() / 12.0).clip(lower=1e-12)
        stoa = pd.DataFrame(np.log(_stoa.values), index=_stoa.index, columns=_stoa.columns)
        combo = 0.35 * stom + 0.35 * stoq + 0.30 * stoa
        return combo.replace([np.inf, -np.inf], np.nan)

    def _compute_residual_volatility(
        self,
        *,
        returns: pd.DataFrame,
        risk_free: pd.DataFrame,
        beta_residuals: pd.DataFrame,
        window: int,
        half_life: int,
    ) -> Dict[str, pd.DataFrame]:
        lam = np.log(2.0) / half_life
        w = np.exp(-lam * np.arange(window)[::-1])
        w /= w.sum()

        stock_ex = returns - risk_free
        log_ex = pd.DataFrame(np.log1p(stock_ex.values), index=stock_ex.index, columns=stock_ex.columns)

        dastd = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        cmra = dastd.copy()
        hsigma = dastd.copy()

        y = stock_ex.values
        ylog = log_ex.values
        r = beta_residuals.values

        for j in range(y.shape[1]):
            for t in range(window - 1, y.shape[0]):
                sl = slice(t - window + 1, t + 1)
                v = y[sl, j]
                lv = ylog[sl, j]
                rv = r[sl, j]
                if np.isnan(v).all():
                    continue
                v = self._mean_impute_vector(v)
                lv = self._mean_impute_vector(lv)
                rv = self._mean_impute_vector(rv)

                v_dm = v - v.mean()
                dastd.iat[t, j] = np.sqrt(np.sum(w * v_dm**2))

                subs = []
                for k in range(0, window, 21):
                    chunk = lv[k : k + 21]
                    if len(chunk) == 0 or np.isnan(chunk).all():
                        continue
                    subs.append(np.nansum(chunk))
                if subs:
                    c = np.cumsum(subs)
                    cmra.iat[t, j] = np.nanmax(c) - np.nanmin(c)

                rv_dm = rv - rv.mean()
                hsigma.iat[t, j] = np.sqrt(np.sum(w * rv_dm**2))

        combo = 0.74 * dastd + 0.16 * cmra + 0.10 * hsigma
        return {"dastd": dastd, "cmra": cmra, "hsigma": hsigma, "residual_vol": combo}

    # ---------------------- utilities ----------------------
    @staticmethod
    def _mean_impute_vector(v: np.ndarray) -> np.ndarray:
        if np.isnan(v).all():
            return v
        m = np.nanmean(v)
        return np.where(np.isnan(v), m, v)

    @staticmethod
    def _mean_impute_matrix(a: np.ndarray) -> np.ndarray:
        b = a.copy()
        for j in range(b.shape[1]):
            b[:, j] = StyleFactors._mean_impute_vector(b[:, j])
        return b

    @staticmethod
    def _broadcast_to_frame(
        source: Union[pd.DataFrame, pd.Series, float, None],
        index: pd.Index,
        columns: pd.Index,
    ) -> Optional[pd.DataFrame]:
        if source is None:
            return None
        if isinstance(source, pd.DataFrame):
            return source.reindex(index=index, columns=columns)
        if isinstance(source, pd.Series):
            s = source.reindex(index)
            mat = np.repeat(s.to_numpy()[:, None], len(columns), axis=1)
            return pd.DataFrame(mat, index=index, columns=columns)
        return pd.DataFrame(float(source), index=index, columns=columns)

    def _cross_sectional_residual(
        self,
        target: pd.DataFrame,
        projections: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Regress out projection factors cross-sectionally date-by-date.

        For each date, fit y = alpha + X beta using available assets (no NaNs)
        and return the residuals, preserving original NaNs where data is missing.
        """
        if not projections:
            return target.copy()

        # Ensure alignment of indices/columns across all inputs
        idx = target.index
        cols = target.columns
        X_frames = {k: v.reindex(index=idx, columns=cols) for k, v in projections.items()}

        resid = pd.DataFrame(np.nan, index=idx, columns=cols)
        lr = LinearRegression(fit_intercept=True)

        for t in idx:
            y = target.loc[t]
            X_df = pd.DataFrame({k: X_frames[k].loc[t] for k in X_frames}, index=cols)

            # Keep rows where y and all X are non-null
            mask = (~y.isna()) & (~X_df.isna().any(axis=1))
            if mask.sum() < max(3, len(X_frames) + 1):
                # Not enough observations to regress; leave NaNs (or keep centered y?)
                continue

            yv = y[mask].to_numpy()
            Xv = X_df.loc[mask].to_numpy()
            try:
                lr.fit(Xv, yv)
                yhat = lr.predict(Xv)
                res = yv - yhat
                resid.loc[t, mask] = res
            except Exception:
                # In case of any numerical issues, skip this date
                continue

        return resid