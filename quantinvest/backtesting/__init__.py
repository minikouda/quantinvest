"""
Backtesting and performance evaluation module.

This module provides:
- Portfolio backtesting engine
- Performance metrics calculation
- Monte Carlo stress testing
- Bootstrap resampling
- Risk-return analysis
"""

from .engine import BacktestEngine
from .metrics import PerformanceMetrics
from .stress_testing import StressTester

__all__ = [
    "BacktestEngine",
    "PerformanceMetrics",
    "StressTester"
]