"""
Industry factors module.

This module handles industry factor computation and dummy variable
creation for sector-neutral portfolio optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .base import FactorBase


class IndustryFactors(FactorBase):
    """
    Industry factors computation class.
    
    Handles industry classification and dummy variable creation
    for sector-neutral portfolio construction.
    """
    
    def __init__(self, name: str = "IndustryFactors"):
        super().__init__(name)
        
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute industry dummy variables.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data containing industry classifications
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        pd.DataFrame
            Industry dummy variables
        """
        if 'industry' not in data.columns:
            raise ValueError("Data must contain 'industry' column")
            
        # Create dummy variables for each industry
        industry_dummies = pd.get_dummies(data['industry'], prefix='industry')
        industry_dummies.index = data.index
        
        return industry_dummies
    
    def create_industry_dummies(self, industry_series: pd.Series) -> pd.DataFrame:
        """Create industry dummy variables from industry series."""
        return pd.get_dummies(industry_series, prefix='industry')
    
    def get_industry_exposure_matrix(self, 
                                   stock_industries: pd.Series,
                                   industry_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create industry exposure matrix for portfolio optimization.
        
        Parameters:
        -----------
        stock_industries : pd.Series
            Series mapping stocks to industries
        industry_list : list, optional
            List of industries to include
            
        Returns:
        --------
        pd.DataFrame
            Industry exposure matrix (stocks x industries)
        """
        # Create dummy matrix from the series
        dummies = pd.get_dummies(stock_industries)

        # If a specific industry_list is provided, align columns to it; else keep all
        if industry_list is not None:
            # ensure unique list of strings
            cols = list(dict.fromkeys(industry_list))
            dummies = dummies.reindex(columns=cols, fill_value=0)

        # Ensure index stays aligned with input stock order
        dummies = dummies.reindex(index=stock_industries.index)
        return dummies