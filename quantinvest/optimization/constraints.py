"""
Constraints builder module.

This module provides utilities for building various portfolio
optimization constraints.
"""

import cvxpy as cp
import numpy as np
from typing import List, Optional


class ConstraintBuilder:
    """
    Builder for portfolio optimization constraints.
    """
    
    def __init__(self):
        self.constraints = []
        
    def add_weight_sum_constraint(self, weights: cp.Variable, target_sum: float = 1.0):
        """Add constraint that weights sum to target value."""
        self.constraints.append(cp.sum(weights) == target_sum)
        return self
    
    def add_long_only_constraint(self, weights: cp.Variable):
        """Add constraint that all weights are non-negative."""
        self.constraints.append(weights >= 0)
        return self
    
    def add_position_limits(self, weights: cp.Variable, 
                          lower_bounds: Optional[np.ndarray] = None,
                          upper_bounds: Optional[np.ndarray] = None):
        """Add individual position limit constraints."""
        if lower_bounds is not None:
            self.constraints.append(weights >= lower_bounds)
        if upper_bounds is not None:
            self.constraints.append(weights <= upper_bounds)
        return self
    
    def get_constraints(self) -> List:
        """Return list of all constraints."""
        return self.constraints