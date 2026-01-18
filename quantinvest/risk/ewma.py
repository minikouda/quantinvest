"""
EWMA covariance estimation module.

This module implements exponentially weighted moving average
covariance estimation for risk modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class EWMACovarianceEstimator:
    """
    Exponentially Weighted Moving Average covariance estimator.
    
    Provides time-varying covariance estimation with exponential weighting
    for financial time series analysis.
    """
    
    def __init__(self, 
                 lambda_value: float = 0.94,
                 min_periods: int = 30):
        """
        Initialize EWMA estimator.
        
        Parameters:
        -----------
        lambda_value : float
            Decay factor for exponential weighting
        min_periods : int
            Minimum periods for estimation
        """
        self.lambda_value = lambda_value
        self.min_periods = min_periods
        self.covariance_series_ = None
        
    def fit(self, returns: pd.DataFrame) -> 'EWMACovarianceEstimator':
        """
        Fit EWMA covariance model.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Return series (time x assets)
            
        Returns:
        --------
        self : EWMACovarianceEstimator
        """
        # Calculate EWMA covariance
        self.covariance_series_ = returns.ewm(
            span=1/self.lambda_value,
            min_periods=self.min_periods
        ).cov()
        
        return self
    
    def predict(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get covariance matrix for specific date.
        
        Parameters:
        -----------
        date : str, optional
            Date for covariance matrix (latest if None)
            
        Returns:
        --------
        pd.DataFrame
            Covariance matrix
        """
        if self.covariance_series_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if date is None:
            # Return latest covariance matrix
            return self.covariance_series_.iloc[-1]
        else:
            return self.covariance_series_.loc[date]