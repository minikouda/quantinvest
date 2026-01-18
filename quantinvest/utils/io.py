"""
Data I/O utilities.

This module provides data loading and saving utilities
for various data formats used in quantitative analysis.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Optional, Union 


class DataLoader:
    """
    Data loading utilities for quantitative analysis.
    
    Provides unified interface for loading various data formats
    including CSV, pickle, and custom formats.
    """
    
    def __init__(self, base_path: str = "./data"):
        """
        Initialize data loader.
        
        Parameters:
        -----------
        base_path : str
            Base path for data files
        """
        self.base_path = base_path
        
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file.
        
        Parameters:
        -----------
        filename : str
            Filename or path
        **kwargs : dict
            Additional arguments for pd.read_csv
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        filepath = os.path.join(self.base_path, filename)
        return pd.read_csv(filepath, **kwargs)
    
    def save_csv(self, data: pd.DataFrame, filename: str, **kwargs):
        """
        Save DataFrame to CSV.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to save
        filename : str
            Output filename
        **kwargs : dict
            Additional arguments for to_csv
        """
        filepath = os.path.join(self.base_path, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, **kwargs)
    
    def load_pickle(self, filename: str) -> any:
        """
        Load pickled object.
        
        Parameters:
        -----------
        filename : str
            Pickle filename
            
        Returns:
        --------
        any
            Loaded object
        """
        filepath = os.path.join(self.base_path, filename)
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_pickle(self, obj: any, filename: str):
        """
        Save object to pickle file.
        
        Parameters:
        -----------
        obj : any
            Object to save
        filename : str
            Output filename
        """
        filepath = os.path.join(self.base_path, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    
    def load_factor_data(self, factor_name: str) -> pd.DataFrame:
        """
        Load factor data with standard format.
        
        Parameters:
        -----------
        factor_name : str
            Factor name
            
        Returns:
        --------
        pd.DataFrame
            Factor data
        """
        filename = f"{factor_name}.csv"
        return self.load_csv(filename, index_col='TradingDay', parse_dates=True)