"""
Backtesting engine module.

This module provides comprehensive backtesting capabilities
for portfolio strategies with performance evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class BacktestEngine:
    """
    Portfolio backtesting engine.
    
    Provides comprehensive backtesting with performance metrics,
    transaction costs, and risk analysis.
    """
    
    def __init__(self, 
                 commission: float = 0.001,
                 risk_free_rate: float = 0.0001):
        """
        Initialize backtest engine.
        
        Parameters:
        -----------
        commission : float
            Transaction cost rate
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation
        """
        self.commission = commission
        self.risk_free_rate = risk_free_rate
        self.results_ = None
        
    def run_backtest(self, 
                    weights: pd.DataFrame,
                    returns: pd.DataFrame,
                    initial_capital: float = 1000000) -> Dict:
        """
        Run portfolio backtest.
        
        Parameters:
        -----------
        weights : pd.DataFrame
            Portfolio weights over time
        returns : pd.DataFrame
            Asset returns
        initial_capital : float
            Initial portfolio value
            
        Returns:
        --------
        dict
            Backtest results
        """
        # Align weights and returns
        common_dates = weights.index.intersection(returns.index)
        weights_aligned = weights.loc[common_dates]
        returns_aligned = returns.loc[common_dates]
        
        # Calculate portfolio returns
        portfolio_returns = (weights_aligned.shift(1) * returns_aligned).sum(axis=1)
        
        # Calculate cumulative performance
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(portfolio_returns)
        
        self.results_ = {
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'portfolio_value': portfolio_value,
            'metrics': metrics
        }
        
        return self.results_
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        # Ensure numeric series and drop NaNs/infs for robust metrics
        r = pd.to_numeric(returns, errors='coerce')
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        # enforce float dtype to avoid object/Scalar typing issues
        r = r.astype(float)

        if r.empty:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
            }

        # Basic metrics
        total_prod = float(np.prod((1.0 + r.to_numpy())))
        metrics['total_return'] = total_prod - 1.0
        metrics['annualized_return'] = float((1.0 + float(r.mean())) ** 252) - 1.0
        metrics['volatility'] = float(r.to_numpy().std(ddof=1)) * float(np.sqrt(252.0))
        
        # Risk-adjusted metrics
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = float(metrics['annualized_return'] - float(self.risk_free_rate)) / float(metrics['volatility'])
        else:
            metrics['sharpe_ratio'] = 0.0
            
        # Drawdown metrics
        cumulative = (1.0 + r).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        metrics['max_drawdown'] = float(drawdowns.min(skipna=True))
        
        return metrics