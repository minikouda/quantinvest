"""Regularized mean-variance optimization utilities."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np


class RegularizedMeanVarianceOptimizer:
    r"""Mean-variance optimizer with L2 and turnover penalties.

    This optimizer solves the following problem:

    .. math::
        \min_w 0.5 * \lambda w^T \Sigma w - \mu^T w + \gamma \|w\|_2^2
                 + \tau \|w - w_{prev}\|_1 \\
        	ext{ subject to } \sum_i w_i = \mathrm{weight\_sum},\; l_i \le w_i \le u_i

    Parameters
    ----------
    risk_aversion : float
        Controls emphasis on variance (higher -> lower risk portfolios).
    l2_reg : float
        Ridge penalty encouraging diversified, small positions.
    turnover_penalty : float
        Penalty on absolute deviation from `prev_weights` (if provided).
    solver : str, optional
        CVXPY solver name, defaults to "ECOS".
    """

    def __init__(
        self,
        *,
        risk_aversion: float = 1.0,
        l2_reg: float = 1e-4,
        turnover_penalty: float = 0.0,
        solver: Optional[str] = "ECOS",
    ) -> None:
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        self.risk_aversion = risk_aversion
        self.l2_reg = l2_reg
        self.turnover_penalty = turnover_penalty
        self.solver = solver

        self.optimal_weights_: Optional[np.ndarray] = None
        self.expected_return_: Optional[float] = None
        self.realized_risk_: Optional[float] = None

    # ------------------------------------------------------------------
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        *,
        bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        target_return: Optional[float] = None,
        prev_weights: Optional[np.ndarray] = None,
        weight_sum: float = 1.0,
    ) -> np.ndarray:
        expected_returns = np.asarray(expected_returns, dtype=float)
        covariance = np.asarray(covariance, dtype=float)

        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be square")
        if expected_returns.shape[0] != covariance.shape[0]:
            raise ValueError("expected_returns and covariance dimensions mismatch")

        n_assets = expected_returns.shape[0]
        w = cp.Variable(n_assets)

        sigma_psd = self._make_psd(covariance)
        quad = 0.5 * self.risk_aversion * cp.quad_form(w, sigma_psd)
        ret_term = -expected_returns @ w
        reg_term = self.l2_reg * cp.sum_squares(w)

        obj = quad + ret_term + reg_term

        constraints = [cp.sum(w) == weight_sum]

        if bounds is not None:
            lower, upper = bounds
            if len(lower) != n_assets or len(upper) != n_assets:
                raise ValueError("bounds must match number of assets")
            constraints.append(w >= np.asarray(lower))
            constraints.append(w <= np.asarray(upper))

        if target_return is not None:
            constraints.append(expected_returns @ w >= target_return)

        if prev_weights is not None and self.turnover_penalty > 0:
            prev = np.asarray(prev_weights, dtype=float)
            if prev.shape[0] != n_assets:
                raise ValueError("prev_weights dimension mismatch")
            obj += self.turnover_penalty * cp.norm1(w - prev)

        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(solver=self.solver)

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise ValueError(f"Optimization failed: {problem.status}")

        weights = w.value
        self.optimal_weights_ = weights
        self.expected_return_ = float(expected_returns @ weights)
        self.realized_risk_ = float(weights.T @ covariance @ weights)
        return weights

    # ------------------------------------------------------------------
    @staticmethod
    def _make_psd(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Ensure matrix is positive semi-definite by clipping eigenvalues."""
        # Symmetrize
        sym = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals_clipped = np.clip(eigvals, epsilon, None)
        return (eigvecs * eigvals_clipped) @ eigvecs.T
