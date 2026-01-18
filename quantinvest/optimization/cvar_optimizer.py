"""
CVaR optimization module.

This module implements Conditional Value at Risk (CVaR) optimization
for downside risk management in portfolio construction.
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from typing import List, Optional, Tuple


class CVaROptimizer:
    """
    Conditional Value at Risk optimizer.
    
    Implements CVaR minimization for portfolio optimization with
    downside risk focus and scenario-based optimization.
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95):
        """
        Initialize CVaR optimizer.
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level for CVaR calculation
        """
        self.confidence_level = confidence_level
        self.optimal_weights_ = None
        self.optimal_cvar_ = None
        
    def optimize(self, 
                return_scenarios: np.ndarray,
                target_return: Optional[float] = None,
                constraints: Optional[List] = None) -> np.ndarray:
        """
        Optimize portfolio to minimize CVaR.
        
        Parameters:
        -----------
        return_scenarios : np.ndarray
            Matrix of return scenarios (scenarios x assets)
        target_return : float, optional
            Target expected return
        constraints : list, optional
            Additional constraints
            
        Returns:
        --------
        np.ndarray
            Optimal portfolio weights
        """
        n_scenarios, n_assets = return_scenarios.shape
        alpha = 1 - self.confidence_level
        
        # Decision variables
        weights = cp.Variable(n_assets)
        var = cp.Variable()
        losses = cp.Variable(n_scenarios)
        
        # Portfolio returns
        portfolio_returns = return_scenarios @ weights
        
        # CVaR constraints
        loss_constraints = [
            losses >= 0,
            losses >= -portfolio_returns - var
        ]
        
        # CVaR objective
        cvar = var + (1/alpha) * cp.sum(losses) / n_scenarios
        objective = cp.Minimize(cvar)
        
        # Constraints
        problem_constraints = [
            cp.sum(weights) == 1,
            *loss_constraints
        ]
        
        if target_return is not None:
            expected_return = cp.sum(portfolio_returns) / n_scenarios
            problem_constraints.append(expected_return >= target_return)
            
        if constraints:
            problem_constraints.extend(constraints)
        
        # Solve
        problem = cp.Problem(objective, problem_constraints)
        problem.solve()
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError(f"Optimization failed: {problem.status}")
        
        self.optimal_weights_ = weights.value
        self.optimal_cvar_ = cvar.value
        
        return self.optimal_weights_