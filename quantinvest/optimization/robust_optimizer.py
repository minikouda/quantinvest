"""
Robust portfolio optimization module.

This module implements advanced portfolio optimization frameworks:
- Robust mean-variance optimization with uncertainty sets
- CVaR (Conditional Value at Risk) minimization
- Sector-neutral and turnover constraints
- Transaction cost modeling
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from typing import Optional, Dict, List, Tuple, Union, Callable
from abc import ABC, abstractmethod
from scipy import sparse
import warnings

warnings.filterwarnings('ignore')


class ConstraintBuilder:
    """
    Builder class for portfolio optimization constraints.
    
    Provides flexible constraint construction for various portfolio
    optimization problems including sector-neutrality, turnover limits,
    and transaction costs.
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
    
    def add_sector_neutral_constraint(self, weights: cp.Variable,
                                   sector_exposure_matrix: np.ndarray,
                                   tolerance: float = 0.0):
        """
        Add sector-neutral constraints.
        
        Parameters:
        -----------
        weights : cp.Variable
            Portfolio weights variable
        sector_exposure_matrix : np.ndarray
            Matrix mapping assets to sectors (n_assets x n_sectors)
        tolerance : float
            Tolerance for sector neutrality
        """
        sector_exposures = sector_exposure_matrix.T @ weights
        self.constraints.extend([
            sector_exposures >= -tolerance,
            sector_exposures <= tolerance
        ])
        return self
    
    def add_turnover_constraint(self, weights: cp.Variable,
                              previous_weights: np.ndarray,
                              max_turnover: float):
        """
        Add turnover constraint.
        
        Parameters:
        -----------
        weights : cp.Variable
            New portfolio weights
        previous_weights : np.ndarray
            Previous portfolio weights
        max_turnover : float
            Maximum allowed turnover (sum of absolute changes)
        """
        turnover = cp.norm(weights - previous_weights, 1)
        self.constraints.append(turnover <= max_turnover)
        return self
    
    def add_tracking_error_constraint(self, weights: cp.Variable,
                                    benchmark_weights: np.ndarray,
                                    covariance_matrix: np.ndarray,
                                    max_tracking_error: float):
        """Add tracking error constraint relative to benchmark."""
        active_weights = weights - benchmark_weights
        tracking_variance = cp.quad_form(active_weights, covariance_matrix)
        self.constraints.append(tracking_variance <= max_tracking_error**2)
        return self
    
    def get_constraints(self) -> List:
        """Return list of all constraints."""
        return self.constraints


class RobustPortfolioOptimizer:
    """
    Robust portfolio optimization with uncertainty sets.
    
    Implements robust mean-variance optimization that accounts for
    parameter uncertainty in expected returns and covariance estimates.
    """
    
    def __init__(self, 
                 uncertainty_set: str = 'ellipsoidal',
                 robustness_parameter: float = 1.0,
                 transaction_cost_model: Optional[str] = None):
        """
        Initialize robust optimizer.
        
        Parameters:
        -----------
        uncertainty_set : str
            Type of uncertainty set ('ellipsoidal', 'box', 'budget')
        robustness_parameter : float
            Controls size of uncertainty set
        transaction_cost_model : str, optional
            Transaction cost model ('linear', 'quadratic', None)
        """
        self.uncertainty_set = uncertainty_set
        self.robustness_parameter = robustness_parameter
        self.transaction_cost_model = transaction_cost_model
        self.optimal_weights_ = None
        self.objective_value_ = None
        
    def optimize_robust_mean_variance(self, 
                                    expected_returns: np.ndarray,
                                    covariance_matrix: np.ndarray,
                                    risk_aversion: float = 1.0,
                                    constraints: Optional[List] = None,
                                    previous_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve robust mean-variance optimization.
        
        Parameters:
        -----------
        expected_returns : np.ndarray
            Expected return estimates
        covariance_matrix : np.ndarray
            Covariance matrix estimate
        risk_aversion : float
            Risk aversion parameter
        constraints : list, optional
            Additional constraints
        previous_weights : np.ndarray, optional
            Previous portfolio weights for transaction costs
            
        Returns:
        --------
        np.ndarray
            Optimal portfolio weights
        """
        n_assets = len(expected_returns)
        weights = cp.Variable(n_assets)
        
        # Robust return term
        if self.uncertainty_set == 'ellipsoidal':
            robust_return = self._ellipsoidal_uncertainty_return(weights, expected_returns, covariance_matrix)
        elif self.uncertainty_set == 'box':
            robust_return = self._box_uncertainty_return(weights, expected_returns)
        else:
            robust_return = weights.T @ expected_returns  # No robustness
        
        # Risk term
        portfolio_risk = cp.quad_form(weights, covariance_matrix)
        
        # Transaction costs
        transaction_costs = 0
        if previous_weights is not None and self.transaction_cost_model:
            transaction_costs = self._compute_transaction_costs(weights, previous_weights)
        
        # Objective: maximize robust return - risk penalty - transaction costs
        objective = cp.Maximize(robust_return - risk_aversion * portfolio_risk - transaction_costs)
        
        # Default constraints
        problem_constraints = [cp.sum(weights) == 1]  # Full investment
        if constraints:
            problem_constraints.extend(constraints)
        
        # Solve problem
        problem = cp.Problem(objective, problem_constraints)
        problem.solve(solver=cp.ECOS)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError(f"Optimization failed with status: {problem.status}")
        
        self.optimal_weights_ = weights.value
        self.objective_value_ = problem.value
        
        return self.optimal_weights_
    
    def _ellipsoidal_uncertainty_return(self, weights: cp.Variable, 
                                      expected_returns: np.ndarray,
                                      covariance_matrix: np.ndarray) -> cp.Expression:
        """Implement ellipsoidal uncertainty set for returns."""
        # Worst-case return under ellipsoidal uncertainty
        # min_{μ} w^T μ s.t. ||μ - μ_0||_Σ <= κ
        # Solution: w^T μ_0 - κ ||Σ^{1/2} w||_2
        
        nominal_return = weights.T @ expected_returns
        
        # Uncertainty penalty
        sqrt_cov = np.linalg.cholesky(covariance_matrix + 1e-8 * np.eye(len(expected_returns)))
        uncertainty_penalty = self.robustness_parameter * cp.norm(sqrt_cov.T @ weights, 2)
        
        return nominal_return - uncertainty_penalty
    
    def _box_uncertainty_return(self, weights: cp.Variable, 
                              expected_returns: np.ndarray) -> cp.Expression:
        """Implement box uncertainty set for returns."""
        # Worst-case return under box uncertainty
        # min_{μ} w^T μ s.t. |μ_i - μ_{0,i}| <= δ_i
        # Solution: w^T μ_0 - Σ_i |w_i| δ_i
        
        nominal_return = weights.T @ expected_returns
        
        # Assume symmetric uncertainty with size proportional to return volatility
        return_volatility = np.sqrt(np.diag(self._get_return_covariance(expected_returns)))
        uncertainty_penalty = self.robustness_parameter * cp.norm(
            cp.multiply(weights, return_volatility), 1
        )
        
        return nominal_return - uncertainty_penalty
    
    def _compute_transaction_costs(self, weights: cp.Variable, 
                                 previous_weights: np.ndarray) -> cp.Expression:
        """Compute transaction costs."""
        weight_changes = weights - previous_weights
        
        if self.transaction_cost_model == 'linear':
            # Linear transaction costs: c * |Δw|
            cost_rate = 0.001  # 10 bps per side
            return cost_rate * cp.norm(weight_changes, 1)
        
        elif self.transaction_cost_model == 'quadratic':
            # Quadratic transaction costs: c * ||Δw||^2
            cost_rate = 0.01
            return cost_rate * cp.sum_squares(weight_changes)
        
        return 0
    
    def _get_return_covariance(self, expected_returns: np.ndarray) -> np.ndarray:
        """Estimate covariance matrix for returns (placeholder)."""
        # In practice, this would be estimated from historical data
        return np.eye(len(expected_returns)) * 0.01  # Simplified assumption


class CVaROptimizer:
    """
    Conditional Value at Risk (CVaR) portfolio optimizer.
    
    Implements CVaR minimization framework for downside risk management
    with support for various constraint types.
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 scenario_method: str = 'historical'):
        """
        Initialize CVaR optimizer.
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level for CVaR calculation (e.g., 0.95 for 95% CVaR)
        scenario_method : str
            Method for generating return scenarios ('historical', 'monte_carlo')
        """
        self.confidence_level = confidence_level
        self.scenario_method = scenario_method
        self.optimal_weights_ = None
        self.optimal_cvar_ = None
        
    def optimize_cvar(self, 
                     return_scenarios: np.ndarray,
                     target_return: Optional[float] = None,
                     constraints: Optional[List] = None) -> np.ndarray:
        """
        Minimize portfolio CVaR.
        
        Parameters:
        -----------
        return_scenarios : np.ndarray
            Return scenarios matrix (n_scenarios x n_assets)
        target_return : float, optional
            Target expected return constraint
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
        var = cp.Variable()  # Value at Risk
        losses = cp.Variable(n_scenarios)  # Auxiliary variables for CVaR
        
        # Portfolio returns for each scenario
        portfolio_returns = return_scenarios @ weights
        
        # CVaR formulation
        # losses[i] = max(0, -portfolio_returns[i] - var)
        loss_constraints = []
        for i in range(n_scenarios):
            loss_constraints.append(losses[i] >= 0)
            loss_constraints.append(losses[i] >= -portfolio_returns[i] - var)
        
        # CVaR = VaR + (1/α) * E[losses]
        cvar = var + (1/alpha) * cp.sum(losses) / n_scenarios
        
        # Objective: minimize CVaR
        objective = cp.Minimize(cvar)
        
        # Constraints
        problem_constraints = [
            cp.sum(weights) == 1,  # Full investment
            *loss_constraints
        ]
        
        # Target return constraint
        if target_return is not None:
            expected_return = cp.sum(return_scenarios @ weights) / n_scenarios
            problem_constraints.append(expected_return >= target_return)
        
        # Additional constraints
        if constraints:
            problem_constraints.extend(constraints)
        
        # Solve
        problem = cp.Problem(objective, problem_constraints)
        problem.solve(solver=cp.ECOS)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError(f"CVaR optimization failed with status: {problem.status}")
        
        self.optimal_weights_ = weights.value
        self.optimal_cvar_ = cvar.value
        
        return self.optimal_weights_
    
    def calculate_portfolio_cvar(self, 
                               weights: np.ndarray,
                               return_scenarios: np.ndarray) -> Tuple[float, float]:
        """
        Calculate portfolio CVaR and VaR for given weights.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        return_scenarios : np.ndarray
            Return scenarios
            
        Returns:
        --------
        tuple
            (CVaR, VaR) values
        """
        portfolio_returns = return_scenarios @ weights
        
        # Calculate VaR
        var = np.percentile(-portfolio_returns, self.confidence_level * 100)
        
        # Calculate CVaR
        tail_losses = -portfolio_returns[portfolio_returns <= -var]
        cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        return cvar, var
    
    def generate_return_scenarios(self, 
                                historical_returns: pd.DataFrame,
                                n_scenarios: int = 1000,
                                method: str = 'bootstrap') -> np.ndarray:
        """
        Generate return scenarios for CVaR optimization.
        
        Parameters:
        -----------
        historical_returns : pd.DataFrame
            Historical return data
        n_scenarios : int
            Number of scenarios to generate
        method : str
            Scenario generation method ('bootstrap', 'parametric')
            
        Returns:
        --------
        np.ndarray
            Generated return scenarios
        """
        if method == 'bootstrap':
            # Bootstrap resampling
            n_periods = len(historical_returns)
            scenario_indices = np.random.choice(n_periods, size=n_scenarios, replace=True)
            return historical_returns.values[scenario_indices]
        
        elif method == 'parametric':
            # Parametric simulation using multivariate normal
            mean_returns = historical_returns.mean().values
            cov_matrix = historical_returns.cov().values
            
            return np.random.multivariate_normal(mean_returns, cov_matrix, size=n_scenarios)
        
        else:
            raise ValueError(f"Unknown scenario generation method: {method}")


class MultiObjectiveOptimizer:
    """
    Multi-objective portfolio optimizer.
    
    Implements various multi-objective optimization techniques for
    portfolio construction including Pareto frontier exploration.
    """
    
    def __init__(self):
        self.pareto_frontier_ = None
        self.efficient_portfolios_ = None
        
    def compute_efficient_frontier(self, 
                                 expected_returns: np.ndarray,
                                 covariance_matrix: np.ndarray,
                                 n_points: int = 50,
                                 constraints: Optional[List] = None) -> Dict[str, np.ndarray]:
        """
        Compute mean-variance efficient frontier.
        
        Parameters:
        -----------
        expected_returns : np.ndarray
            Expected returns
        covariance_matrix : np.ndarray
            Covariance matrix
        n_points : int
            Number of points on frontier
        constraints : list, optional
            Additional constraints
            
        Returns:
        --------
        dict
            Dictionary with 'returns', 'risks', and 'weights' arrays
        """
        n_assets = len(expected_returns)
        
        # Range of target returns
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_weights = []
        frontier_risks = []
        frontier_returns = []
        
        for target_return in target_returns:
            try:
                # Minimize risk for target return
                weights = cp.Variable(n_assets)
                risk = cp.quad_form(weights, covariance_matrix)
                
                problem_constraints = [
                    cp.sum(weights) == 1,
                    weights.T @ expected_returns >= target_return
                ]
                
                if constraints:
                    problem_constraints.extend(constraints)
                
                problem = cp.Problem(cp.Minimize(risk), problem_constraints)
                problem.solve(solver=cp.ECOS)
                
                if problem.status in ['optimal', 'optimal_inaccurate']:
                    frontier_weights.append(weights.value)
                    frontier_risks.append(np.sqrt(risk.value))
                    frontier_returns.append(target_return)
                    
            except Exception:
                continue
        
        self.efficient_portfolios_ = {
            'returns': np.array(frontier_returns),
            'risks': np.array(frontier_risks),
            'weights': np.array(frontier_weights)
        }
        
        return self.efficient_portfolios_