"""
Data processing utilities.

This module provides data preprocessing, cleaning, and
transformation utilities for quantitative analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataProcessor:
    """
    Data processing utilities for quantitative analysis.
    
    Provides winsorization, standardization, and other
    data preprocessing functions.
    """
    
    def __init__(self, winsorize_limits: Tuple[float, float] = (0.01, 0.99)):
        """
        Initialize data processor.
        
        Parameters:
        -----------
        winsorize_limits : tuple
            Lower and upper quantiles for winsorization
        """
        self.winsorize_limits = winsorize_limits
        
    def winsorize_series(self, series: pd.Series) -> pd.Series:
        """
        Winsorize a pandas Series.
        
        Parameters:
        -----------
        series : pd.Series
            Input series
            
        Returns:
        --------
        pd.Series
            Winsorized series
        """
        lower_limit = series.quantile(self.winsorize_limits[0])
        upper_limit = series.quantile(self.winsorize_limits[1])
        return series.clip(lower=lower_limit, upper=upper_limit)
    
    def standardize_series(self, series: pd.Series) -> pd.Series:
        """
        Standardize a pandas Series.
        
        Parameters:
        -----------
        series : pd.Series
            Input series
            
        Returns:
        --------
        pd.Series
            Standardized series
        """
        return (series - series.mean()) / series.std()
    
    def winsorize_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply winsorization and standardization to DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            Processed DataFrame
        """
        # Apply to each row (cross-sectional)
        winsorized = df.apply(lambda row: self.winsorize_series(row), axis=1)
        standardized = winsorized.apply(lambda row: self.standardize_series(row), axis=1)
        return standardized