"""
Plotting and visualization utilities.

This module provides plotting functions for portfolio analysis,
performance visualization, and risk analysis.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple


class Plotter:
    """
    Plotting utilities for quantitative analysis.
    
    Provides standardized plotting functions for portfolio
    performance, risk analysis, and factor analysis.
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize plotter.
        
        Parameters:
        -----------
        style : str
            Plotting style
        figsize : tuple
            Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        
    def plot_cumulative_returns(self, 
                               returns: pd.Series,
                               benchmark: Optional[pd.Series] = None,
                               title: str = "Cumulative Returns") -> Figure:
        """
        Plot cumulative returns.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        benchmark : pd.Series, optional
            Benchmark returns for comparison
        title : str
            Plot title
            
        Returns:
        --------
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        ax.plot(cum_returns.index, cum_returns.values, label='Portfolio', linewidth=2)
        
        if benchmark is not None:
            cum_benchmark = (1 + benchmark).cumprod()
            ax.plot(cum_benchmark.index, cum_benchmark.values, label='Benchmark', linewidth=2)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(self, returns: pd.Series, title: str = "Drawdown Analysis") -> Figure:
        """
        Plot drawdown analysis.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        title : str
            Plot title
            
        Returns:
        --------
        Figure
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Cumulative returns
        cum_returns = (1 + returns).cumprod()
        ax1.plot(cum_returns.index, cum_returns.values, linewidth=2)
        ax1.set_title('Cumulative Returns', fontsize=14)
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        
        ax2.fill_between(drawdowns.index, drawdowns.values, 0, alpha=0.3, color='red')
        ax2.plot(drawdowns.index, drawdowns.values, color='red', linewidth=1)
        ax2.set_title('Drawdown', fontsize=14)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_risk_return_scatter(self, 
                               returns_data: Dict[str, pd.Series],
                               title: str = "Risk-Return Analysis") -> Figure:
        """
        Plot risk-return scatter.
        
        Parameters:
        -----------
        returns_data : dict
            Dictionary of strategy returns
        title : str
            Plot title
            
        Returns:
        --------
        Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for name, returns in returns_data.items():
            ann_return = (1 + returns.mean()) ** 252 - 1
            volatility = returns.std() * np.sqrt(252)
            
            ax.scatter(volatility, ann_return, s=100, label=name, alpha=0.7)
        
        ax.set_xlabel('Volatility (Annualized)', fontsize=12)
        ax.set_ylabel('Return (Annualized)', fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig