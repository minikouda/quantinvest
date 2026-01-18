"""
Base class for factor computation and processing.

This module provides the foundation for factor standardization,
winsorization, and basic factor operations.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union
from scipy.stats import zscore


class FactorBase(ABC):
    """
    Abstract base class for factor computation and processing.
    
    Provides standardized interface for factor calculations,
    data preprocessing, and factor validation.
    """
    
    def __init__(self, name: str, winsorize_limits: Tuple[float, float] = (0.01, 0.99)):
        """
        Initialize factor base class.
        
        Parameters:
        -----------
        name : str
            Name of the factor
        winsorize_limits : tuple
            Lower and upper quantiles for winsorization
        """
        self.name = name
        self.winsorize_limits = winsorize_limits
        
    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute factor values.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data for factor computation
        **kwargs : dict
            Additional parameters for factor computation
            
        Returns:
        --------
        pd.DataFrame
            Computed factor values
        """
        pass
    
    def winsorize_series(self, series: pd.Series) -> pd.Series:
        """
        Winsorize a pandas Series to handle outliers.
        
        Parameters:
        -----------
        series : pd.Series
            Input series to winsorize
            
        Returns:
        --------
        pd.Series
            Winsorized series
        """
        lower_limit = series.quantile(self.winsorize_limits[0])
        upper_limit = series.quantile(self.winsorize_limits[1])
        return series.clip(lower=lower_limit, upper=upper_limit)
    
    def winsorize_dataframe(self, df: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
        """
        Apply winsorization to DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        axis : int
            Axis along which to apply winsorization (0=columns, 1=rows)
            
        Returns:
        --------
        pd.DataFrame
            Winsorized DataFrame
        """
        if axis == 1:  # Apply to each row (cross-sectional)
            return df.apply(lambda row: self.winsorize_series(row), axis=1)
        else:  # Apply to each column (time series)
            return df.apply(lambda col: self.winsorize_series(col), axis=0)
    
    def standardize_series(self, series: pd.Series) -> pd.Series:
        """
        Standardize a pandas Series (z-score normalization).
        
        Parameters:
        -----------
        series : pd.Series
            Input series to standardize
            
        Returns:
        --------
        pd.Series
            Standardized series
        """
        return (series - series.mean()) / series.std()
    
    def standardize_dataframe(self, df: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
        """
        Apply standardization to DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        axis : int
            Axis along which to apply standardization (0=columns, 1=rows)
            
        Returns:
        --------
        pd.DataFrame
            Standardized DataFrame
        """
        if axis == 1:  # Apply to each row (cross-sectional)
            return df.apply(lambda row: self.standardize_series(row), axis=1)
        else:  # Apply to each column (time series)
            return df.apply(lambda col: self.standardize_series(col), axis=0)
    
    def preprocess(self, df: pd.DataFrame, 
                   winsorize: bool = True, 
                   standardize: bool = True,
                   axis: int = 1) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline: winsorization + standardization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        winsorize : bool
            Whether to apply winsorization
        standardize : bool
            Whether to apply standardization
        axis : int
            Axis along which to apply preprocessing
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed DataFrame
        """
        result = df.copy()
        
        if winsorize:
            result = self.winsorize_dataframe(result, axis=axis)
            
        if standardize:
            result = self.standardize_dataframe(result, axis=axis)
            
        return result
    
    def validate_factor(self, factor_data: pd.DataFrame) -> Dict[str, Union[bool, float]]:
        """
        Validate factor data quality.
        
        Parameters:
        -----------
        factor_data : pd.DataFrame
            Factor data to validate
            
        Returns:
        --------
        dict
            Validation results including coverage, uniqueness, etc.
        """
        validation_results = {
            'total_observations': factor_data.size,
            'missing_ratio': factor_data.isna().sum().sum() / factor_data.size,
            'infinite_values': np.isinf(factor_data.select_dtypes(include=[np.number])).sum().sum(),
            'coverage_ratio': 1 - factor_data.isna().all(axis=1).mean(),
            'temporal_coverage': len(factor_data.dropna(how='all')) / len(factor_data),
            'cross_sectional_coverage': (factor_data.notna().sum(axis=1) > 0).mean()
        }
        
        return validation_results