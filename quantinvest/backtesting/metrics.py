"""
Performance metrics module.

This module provides comprehensive performance evaluation
metrics for portfolio backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict


class PerformanceMetrics:
    """
    Performance metrics calculator.
    
    Provides comprehensive performance metrics calculation
    for portfolio evaluation and comparison.
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        """
        Initialize performance metrics calculator.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free rate for risk-adjusted metrics
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_all_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio return series
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        metrics = {}
        # 
        # Return metrics
        metrics.update(self._calculate_return_metrics(returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        return metrics
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate return-based metrics."""
        total_return = float((1 + returns).prod() - 1)
        annualized_return = float((1 + returns.mean()) ** 252 - 1)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'mean_return': float(returns.mean()),
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        volatility = float(returns.std() * np.sqrt(252))
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdowns.min())
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted metrics."""
        ann_return = float((1 + returns.mean()) ** 252 - 1)
        volatility = float(returns.std() * np.sqrt(252))
        
        # Sharpe ratio
        sharpe_ratio = (ann_return - self.risk_free_rate) / volatility if volatility > 0 else 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
        }