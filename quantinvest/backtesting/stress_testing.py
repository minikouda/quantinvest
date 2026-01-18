"""
Stress testing and robustness analysis module.

This module implements comprehensive stress testing frameworks:
- Monte Carlo stress scenarios 
- Bootstrap block resampling for time series
- Performance robustness analysis
- Risk factor stress testing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import warnings
from scipy import stats
from sklearn.utils import resample
import multiprocessing as mp

warnings.filterwarnings('ignore')


class StressTester:
    """
    Comprehensive stress testing framework for portfolio robustness analysis.
    
    Implements various stress testing methodologies including Monte Carlo
    simulation, bootstrap resampling, and scenario-based stress tests.
    """
    
    def __init__(self, 
                 n_scenarios: int = 10000,
                 confidence_levels: List[float] = [0.95, 0.99],
                 n_jobs: int = -1,
                 random_state: int = 42):
        """
        Initialize stress tester.
        
        Parameters:
        -----------
        n_scenarios : int
            Number of stress scenarios to generate
        confidence_levels : list
            Confidence levels for risk metrics
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        random_state : int
            Random state for reproducibility
        """
        self.n_scenarios = n_scenarios
        self.confidence_levels = confidence_levels
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.random_state = random_state
        self.stress_results_ = None
        
        # Set random seed
        np.random.seed(random_state)
    
    def monte_carlo_stress_test(self, 
                              portfolio_weights: np.ndarray,
                              expected_returns: np.ndarray,
                              covariance_matrix: np.ndarray,
                              time_horizon: int = 252,
                              method: str = 'parametric') -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Conduct Monte Carlo stress testing.
        
        Parameters:
        -----------
        portfolio_weights : np.ndarray
            Portfolio weights to stress test
        expected_returns : np.ndarray
            Expected asset returns
        covariance_matrix : np.ndarray
            Asset return covariance matrix
        time_horizon : int
            Time horizon for stress testing (in days)
        method : str
            Simulation method ('parametric', 'historical_bootstrap')
            
        Returns:
        --------
        dict
            Comprehensive stress test results
        """
        print(f"Running Monte Carlo stress test with {self.n_scenarios:,} scenarios...")
        
        # Generate return scenarios
        if method == 'parametric':
            return_scenarios = self._generate_parametric_scenarios(
                expected_returns, covariance_matrix, time_horizon
            )
        else:
            raise ValueError(f"Method {method} not implemented yet")
        
        # Calculate portfolio returns for all scenarios
        portfolio_returns = self._calculate_portfolio_returns(
            portfolio_weights, return_scenarios, time_horizon
        )
        
        # Compute stress test metrics
        stress_metrics = self._compute_stress_metrics(portfolio_returns)
        
        # Risk factor decomposition
        factor_contributions = self._analyze_factor_contributions(
            portfolio_weights, return_scenarios, covariance_matrix
        )
        
        self.stress_results_ = {
            'portfolio_returns': portfolio_returns,
            'stress_metrics': stress_metrics,
            'factor_contributions': factor_contributions,
            'return_scenarios': return_scenarios,
            'method': method,
            'time_horizon': time_horizon
        }
        
        return self.stress_results_
    
    def _generate_parametric_scenarios(self, 
                                     expected_returns: np.ndarray,
                                     covariance_matrix: np.ndarray,
                                     time_horizon: int) -> np.ndarray:
        """Generate parametric Monte Carlo scenarios."""
        n_assets = len(expected_returns)
        
        # Daily returns simulation
        daily_scenarios = np.random.multivariate_normal(
            expected_returns / 252,  # Daily returns
            covariance_matrix / 252,  # Daily covariance
            size=(self.n_scenarios, time_horizon)
        )
        
        return daily_scenarios
    
    def _calculate_portfolio_returns(self, 
                                   weights: np.ndarray,
                                   return_scenarios: np.ndarray,
                                   time_horizon: int) -> np.ndarray:
        """Calculate portfolio returns for all scenarios."""
        # return_scenarios shape: (n_scenarios, time_horizon, n_assets)
        # weights shape: (n_assets,)
        
        # Portfolio daily returns
        portfolio_daily_returns = np.tensordot(return_scenarios, weights, axes=([2], [0]))
        
        # Cumulative returns over time horizon
        portfolio_cumulative_returns = np.cumprod(1 + portfolio_daily_returns, axis=1)[:, -1] - 1
        
        return portfolio_cumulative_returns
    
    def _compute_stress_metrics(self, portfolio_returns: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive stress test metrics."""
        metrics = {}
        
        # Basic statistics
        metrics['mean_return'] = np.mean(portfolio_returns)
        metrics['std_return'] = np.std(portfolio_returns)
        metrics['skewness'] = stats.skew(portfolio_returns)
        metrics['kurtosis'] = stats.kurtosis(portfolio_returns)
        
        # Risk metrics
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            var = np.percentile(portfolio_returns, alpha * 100)
            cvar = np.mean(portfolio_returns[portfolio_returns <= var])
            
            metrics[f'VaR_{int(conf_level*100)}'] = var
            metrics[f'CVaR_{int(conf_level*100)}'] = cvar
        
        # Tail risk metrics
        metrics['max_drawdown'] = np.min(portfolio_returns)
        metrics['positive_scenarios_pct'] = np.mean(portfolio_returns > 0) * 100
        metrics['extreme_loss_scenarios'] = np.mean(portfolio_returns < -0.1) * 100  # > 10% loss
        
        # Sharpe ratio (annualized)
        if metrics['std_return'] > 0:
            metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['std_return'] * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        return metrics
    
    def _analyze_factor_contributions(self, 
                                    weights: np.ndarray,
                                    return_scenarios: np.ndarray,
                                    covariance_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze factor contributions to portfolio risk."""
        # Portfolio variance decomposition
        portfolio_variance = weights.T @ covariance_matrix @ weights
        
        # Marginal contributions to risk
        marginal_contributions = 2 * covariance_matrix @ weights
        component_contributions = weights * marginal_contributions
        
        # Percentage contributions
        contribution_percentages = component_contributions / portfolio_variance * 100
        
        return {
            'marginal_contributions': marginal_contributions,
            'component_contributions': component_contributions,
            'contribution_percentages': contribution_percentages,
            'portfolio_variance': portfolio_variance
        }
    
    def bootstrap_stress_test(self, 
                            portfolio_weights: np.ndarray,
                            historical_returns: pd.DataFrame,
                            block_size: int = 22,
                            n_bootstrap_samples: Optional[int] = None) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Conduct bootstrap stress testing with block resampling.
        
        Parameters:
        -----------
        portfolio_weights : np.ndarray
            Portfolio weights
        historical_returns : pd.DataFrame
            Historical return data
        block_size : int
            Block size for block bootstrap (22 ≈ 1 month)
        n_bootstrap_samples : int, optional
            Number of bootstrap samples (defaults to self.n_scenarios)
            
        Returns:
        --------
        dict
            Bootstrap stress test results
        """
        if n_bootstrap_samples is None:
            n_bootstrap_samples = self.n_scenarios
            
        print(f"Running bootstrap stress test with {n_bootstrap_samples:,} samples...")
        
        # Block bootstrap resampling
        bootstrap_returns = self._block_bootstrap_resample(
            historical_returns, block_size, n_bootstrap_samples
        )
        
        # Calculate portfolio returns for bootstrap samples
        portfolio_bootstrap_returns = []
        for i in range(n_bootstrap_samples):
            sample_returns = bootstrap_returns[i]
            portfolio_return = (sample_returns @ portfolio_weights).sum()
            portfolio_bootstrap_returns.append(portfolio_return)
        
        portfolio_bootstrap_returns = np.array(portfolio_bootstrap_returns)
        
        # Compute stress metrics
        stress_metrics = self._compute_stress_metrics(portfolio_bootstrap_returns)
        
        return {
            'portfolio_returns': portfolio_bootstrap_returns,
            'stress_metrics': stress_metrics,
            'bootstrap_samples': bootstrap_returns,
            'method': 'block_bootstrap',
            'block_size': block_size
        }
    
    def _block_bootstrap_resample(self, 
                                data: pd.DataFrame,
                                block_size: int,
                                n_samples: int) -> List[np.ndarray]:
        """Perform block bootstrap resampling."""
        n_periods = len(data)
        n_blocks_needed = n_periods // block_size + 1
        
        bootstrap_samples = []
        
        for _ in range(n_samples):
            # Randomly select starting points for blocks
            start_indices = np.random.choice(
                n_periods - block_size + 1, 
                size=n_blocks_needed, 
                replace=True
            )
            
            # Create bootstrap sample by concatenating blocks
            bootstrap_sample = []
            for start_idx in start_indices:
                block = data.iloc[start_idx:start_idx + block_size].values
                bootstrap_sample.append(block)
                
            # Concatenate and trim to original length
            bootstrap_sample = np.vstack(bootstrap_sample)[:n_periods]
            bootstrap_samples.append(bootstrap_sample)
        
        return bootstrap_samples
    
    def scenario_stress_test(self, 
                           portfolio_weights: np.ndarray,
                           stress_scenarios: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Test portfolio against predefined stress scenarios.
        
        Parameters:
        -----------
        portfolio_weights : np.ndarray
            Portfolio weights
        stress_scenarios : dict
            Dictionary of stress scenarios {scenario_name: return_shocks}
            
        Returns:
        --------
        dict
            Results for each stress scenario
        """
        scenario_results = {}
        
        for scenario_name, return_shocks in stress_scenarios.items():
            portfolio_return = portfolio_weights @ return_shocks
            
            scenario_results[scenario_name] = {
                'portfolio_return': portfolio_return,
                'worst_asset_return': np.min(return_shocks),
                'best_asset_return': np.max(return_shocks),
                'scenario_severity': np.std(return_shocks)
            }
        
        return scenario_results
    
    def parallel_stress_test(self, 
                           portfolio_weights: np.ndarray,
                           expected_returns: np.ndarray,
                           covariance_matrix: np.ndarray,
                           test_functions: List[Callable],
                           **kwargs) -> Dict[str, Dict]:
        """
        Run multiple stress tests in parallel.
        
        Parameters:
        -----------
        portfolio_weights : np.ndarray
            Portfolio weights
        expected_returns : np.ndarray
            Expected returns
        covariance_matrix : np.ndarray
            Covariance matrix
        test_functions : list
            List of stress test functions to run
        **kwargs : dict
            Additional arguments for stress test functions
            
        Returns:
        --------
        dict
            Results from all stress tests
        """
        print(f"Running {len(test_functions)} stress tests in parallel...")
        
        # Prepare function calls
        test_calls = []
        for i, test_func in enumerate(test_functions):
            test_call = partial(
                test_func,
                portfolio_weights,
                expected_returns,
                covariance_matrix,
                **kwargs
            )
            test_calls.append((f"test_{i}", test_call))
        
        # Execute in parallel
        results = {}
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_test = {
                executor.submit(test_call): test_name 
                for test_name, test_call in test_calls
            }
            
            for future in as_completed(future_to_test):
                test_name = future_to_test[future]
                try:
                    result = future.result()
                    results[test_name] = result
                except Exception as exc:
                    print(f"Test {test_name} generated an exception: {exc}")
                    results[test_name] = {'error': str(exc)}
        
        return results
    
    def generate_stress_report(self, 
                             stress_results: Dict,
                             benchmark_results: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate comprehensive stress test report.
        
        Parameters:
        -----------
        stress_results : dict
            Results from stress testing
        benchmark_results : dict, optional
            Benchmark stress test results for comparison
            
        Returns:
        --------
        pd.DataFrame
            Formatted stress test report
        """
        if 'stress_metrics' not in stress_results:
            raise ValueError("stress_results must contain 'stress_metrics' key")
        
        metrics = stress_results['stress_metrics']
        
        report_data = []
        for metric_name, value in metrics.items():
            row = {
                'Metric': metric_name,
                'Portfolio': value,
                'Unit': self._get_metric_unit(metric_name)
            }
            
            if benchmark_results and 'stress_metrics' in benchmark_results:
                benchmark_value = benchmark_results['stress_metrics'].get(metric_name, np.nan)
                row['Benchmark'] = benchmark_value
                if not np.isnan(benchmark_value) and benchmark_value != 0:
                    row['Difference'] = value - benchmark_value
                    row['Relative_Difference_%'] = (value / benchmark_value - 1) * 100
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get appropriate unit for metric."""
        if 'return' in metric_name.lower() or 'var' in metric_name.lower() or 'cvar' in metric_name.lower():
            return '%'
        elif 'scenarios' in metric_name.lower():
            return '%'
        elif 'ratio' in metric_name.lower():
            return 'ratio'
        else:
            return ''
    
    def create_stress_scenarios(self, 
                              asset_correlations: np.ndarray,
                              volatility_multipliers: List[float] = [1.5, 2.0, 3.0],
                              correlation_shifts: List[float] = [0.2, 0.5]) -> Dict[str, np.ndarray]:
        """
        Create systematic stress scenarios.
        
        Parameters:
        -----------
        asset_correlations : np.ndarray
            Base correlation matrix
        volatility_multipliers : list
            Volatility stress multipliers
        correlation_shifts : list
            Correlation stress shifts
            
        Returns:
        --------
        dict
            Dictionary of stress scenarios
        """
        n_assets = asset_correlations.shape[0]
        scenarios = {}
        
        # Volatility stress scenarios
        for mult in volatility_multipliers:
            scenario_name = f"vol_stress_{mult}x"
            # Increase volatility while keeping correlations
            stressed_returns = np.random.multivariate_normal(
                np.zeros(n_assets),
                asset_correlations * mult,
                size=1
            )[0]
            scenarios[scenario_name] = stressed_returns
        
        # Correlation stress scenarios
        for shift in correlation_shifts:
            scenario_name = f"corr_stress_{shift}"
            # Increase all correlations
            stressed_corr = asset_correlations.copy()
            mask = ~np.eye(n_assets, dtype=bool)
            stressed_corr[mask] = np.clip(stressed_corr[mask] + shift, -1, 1)
            
            stressed_returns = np.random.multivariate_normal(
                np.zeros(n_assets),
                stressed_corr,
                size=1
            )[0]
            scenarios[scenario_name] = stressed_returns
        
        # Market crash scenario
        scenarios['market_crash'] = np.random.normal(-0.1, 0.05, n_assets)  # -10% +/- 5%
        
        # Sector rotation scenario
        scenarios['sector_rotation'] = np.random.choice([-0.05, 0.05], n_assets)
        
        return scenarios