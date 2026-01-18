"""
Quantitative Investment Framework

A comprehensive framework for robust portfolio optimization with factor-based models,
featuring advanced covariance estimation, factor selection, and stress testing capabilities.

Modules:
    factors: Factor computation and selection (sparse PCA, Lasso, standardization)
    risk: Risk modeling (Barra, EWMA, Ledoit-Wolf, Graphical Lasso)
    optimization: Portfolio optimization (robust mean-variance, CVaR, constraints)
    backtesting: Performance evaluation and testing frameworks
    utils: Data processing, I/O, and configuration utilities
"""

__version__ = "1.0"
__author__ = "Zhang Shizhe"

# Import main classes for convenience
from .factors.base import FactorBase
from .risk.covariance_estimators import CovarianceEstimator
from .optimization.robust_optimizer import RobustPortfolioOptimizer
from .backtesting.engine import BacktestEngine

__all__ = [
    "FactorBase",
    "CovarianceEstimator", 
    "RobustPortfolioOptimizer",
    "BacktestEngine"
]