"""
Factor computation and selection module.

This module provides:
- Factor standardization and winsorization
- Sparse PCA for factor dimensionality reduction  
- Lasso-based factor selection
- Industry factor handling
"""

from .base import FactorBase
from .feature_selection import FactorSelector
from .style_factors import StyleFactors
from .industry_factors import IndustryFactors

__all__ = [
    "FactorBase",
    "FactorSelector", 
    "StyleFactors",
    "IndustryFactors"
]