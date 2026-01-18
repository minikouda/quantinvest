"""
Advanced covariance estimation module.

This module implements ML-enhanced covariance estimation techniques:
- Ledoit-Wolf shrinkage for high-dimensional stability
- Graphical Lasso for sparse covariance estimation
- Cross-validated hyperparameter selection
- Out-of-sample evaluation framework
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Union, List, Literal, Any
from sklearn.covariance import (
    LedoitWolf, OAS, ShrunkCovariance, GraphicalLasso, 
    GraphicalLassoCV, EmpiricalCovariance
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import warnings
from scipy import linalg
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')


class CovarianceEstimator(ABC):
    """
    Abstract base class for covariance estimators.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'CovarianceEstimator':
        """Fit the covariance estimator."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict covariance matrix."""
        pass
    
    def score(self, X: np.ndarray) -> float:
        """Score the covariance estimate using log-likelihood."""
        cov_matrix = self.predict(X)
        return gaussian_log_likelihood(cov_matrix, X)


def gaussian_log_likelihood(cov_matrix: np.ndarray, X: np.ndarray) -> float:
    """Compute Gaussian log-likelihood of data X under covariance cov_matrix.

    Uses a numerically stable formulation with small diagonal regularization
    and pseudo-inverse for precision.
    """
    try:
        n_samples, n_features = X.shape

        # Center X
        Xc = X - np.mean(X, axis=0)

        # Regularize covariance for stability
        eps = 1e-6
        cov_reg = cov_matrix + eps * np.eye(cov_matrix.shape[0])
        sign, logdet = np.linalg.slogdet(cov_reg)
        if sign <= 0:
            return -np.inf

        precision = linalg.pinvh(cov_reg)
        quad = np.trace(Xc @ precision @ Xc.T) / n_samples
        ll = -0.5 * (logdet + quad + n_features * np.log(2 * np.pi))
        return float(ll)
    except Exception:
        return -np.inf


class LedoitWolfEstimator(CovarianceEstimator):
    """
    Ledoit-Wolf shrinkage covariance estimator.
    
    Provides optimal shrinkage towards structured targets for
    high-dimensional covariance estimation with limited samples.
    """
    
    def __init__(self, 
                 store_precision: bool = True,
                 assume_centered: bool = False,
                 block_size: int = 1000):
        """
        Initialize Ledoit-Wolf estimator.
        
        Parameters:
        -----------
        store_precision : bool
            Whether to store precision matrix
        assume_centered : bool
            Whether data is already centered
        block_size : int
            Block size for memory efficiency
        """
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.block_size = block_size
        self.estimator_ = None
        self.shrinkage_ = None
        
    def fit(self, X: np.ndarray) -> 'LedoitWolfEstimator':
        """
        Fit Ledoit-Wolf estimator.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix (n_samples x n_features)
            
        Returns:
        --------
        self : LedoitWolfEstimator
        """
        self.estimator_ = LedoitWolf(
            store_precision=self.store_precision,
            assume_centered=self.assume_centered,
            block_size=self.block_size
        )
        
        self.estimator_.fit(X)
        self.shrinkage_ = self.estimator_.shrinkage_
        
        return self
    
    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get covariance matrix estimate.
        
        Parameters:
        -----------
        X : np.ndarray, optional
            Not used, for interface compatibility
            
        Returns:
        --------
        np.ndarray
            Estimated covariance matrix
        """
        if self.estimator_ is None:
            raise ValueError("Estimator not fitted.")
            
        return self.estimator_.covariance_
    
    def get_precision(self) -> np.ndarray:
        """Get precision matrix (inverse covariance)."""
        if self.estimator_ is None:
            raise ValueError("Estimator not fitted.")
            
        return self.estimator_.precision_


class GraphicalLassoEstimator(CovarianceEstimator):
    """
    Graphical Lasso sparse covariance estimator.
    
    Estimates sparse covariance and precision matrices using
    L1 regularization on the precision matrix.
    """
    
    def __init__(self, 
                 alpha: float = 0.01,
                 mode: Literal['cd', 'lars'] = 'cd',
                 tol: float = 1e-4,
                 max_iter: int = 100,
                 assume_centered: bool = False):
        """
        Initialize Graphical Lasso estimator.
        
        Parameters:
        -----------
        alpha : float
            Regularization parameter
        mode : str
            Algorithm mode ('cd' for coordinate descent, 'lars' for LARS)
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
        assume_centered : bool
            Whether data is centered
        """
        self.alpha = alpha
        self.mode = mode
        self.tol = tol
        self.max_iter = max_iter
        self.assume_centered = assume_centered
        self.estimator_ = None
        
    def fit(self, X: np.ndarray) -> 'GraphicalLassoEstimator':
        """
        Fit Graphical Lasso estimator.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix (n_samples x n_features)
            
        Returns:
        --------
        self : GraphicalLassoEstimator
        """
        mode_value: Literal['cd', 'lars'] = 'lars' if self.mode == 'lars' else 'cd'
        self.estimator_ = GraphicalLasso(
            alpha=self.alpha,
            mode=mode_value,
            tol=self.tol,
            max_iter=self.max_iter,
            assume_centered=self.assume_centered
        )
        
        self.estimator_.fit(X)
        
        return self
    
    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get covariance matrix estimate.
        
        Parameters:
        -----------
        X : np.ndarray, optional
            Not used, for interface compatibility
            
        Returns:
        --------
        np.ndarray
            Estimated covariance matrix
        """
        if self.estimator_ is None:
            raise ValueError("Estimator not fitted.")
            
        return self.estimator_.covariance_
    
    def get_precision(self) -> np.ndarray:
        """Get sparse precision matrix."""
        if self.estimator_ is None:
            raise ValueError("Estimator not fitted.")
            
        return self.estimator_.precision_
    
    def get_sparsity_pattern(self) -> np.ndarray:
        """Get sparsity pattern of precision matrix."""
        precision = self.get_precision()
        return np.abs(precision) > 1e-6


class AdaptiveCovarianceEstimator:
    """
    Adaptive covariance estimator that selects best method via cross-validation.
    
    Compares multiple covariance estimation methods and selects the best
    performing one based on out-of-sample likelihood.
    """
    
    def __init__(self, 
                 estimators: Optional[Dict[str, Any]] = None,
                 cv_folds: int = 5,
                 scoring: str = 'log_likelihood'):
        """
        Initialize adaptive estimator.
        
        Parameters:
        -----------
        estimators : dict, optional
            Dictionary of covariance estimators to compare
        cv_folds : int
            Number of cross-validation folds
        scoring : str
            Scoring method for model selection
        """
        if estimators is None:
            estimators = {
                'empirical': EmpiricalCovariance(),
                'ledoit_wolf': LedoitWolf(),
                'oas': OAS(),
                'graphical_lasso': GraphicalLassoCV()
            }
        
        self.estimators = estimators
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.best_estimator_ = None
        self.best_score_ = None
        self.cv_scores_ = {}
        
    def fit(self, X: np.ndarray) -> 'AdaptiveCovarianceEstimator':
        """
        Fit adaptive estimator by selecting best method.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix
            
        Returns:
        --------
        self : AdaptiveCovarianceEstimator
        """
        best_score = -np.inf
        best_name = None
        
        for name, estimator in self.estimators.items():
            try:
                # Cross-validation score
                scores = self._cross_validate(estimator, X)
                mean_score = np.mean(scores)
                self.cv_scores_[name] = {
                    'mean': mean_score,
                    'std': np.std(scores),
                    'scores': scores
                }
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_name = name
                    
            except Exception as e:
                print(f"Warning: Failed to fit {name}: {str(e)}")
                self.cv_scores_[name] = {
                    'mean': -np.inf,
                    'std': np.inf,
                    'scores': [-np.inf] * self.cv_folds
                }
        
        # Fit best estimator on full data
        if best_name is not None:
            self.best_estimator_ = self.estimators[best_name]
            self.best_estimator_.fit(X)
            self.best_score_ = best_score
            self.best_method_ = best_name
        else:
            raise ValueError("No estimator could be fitted successfully.")
            
        return self
    
    def _cross_validate(self, estimator, X: np.ndarray) -> List[float]:
        """Perform cross-validation for an estimator."""
        n_samples = X.shape[0]
        fold_size = n_samples // self.cv_folds
        scores = []
        
        for i in range(self.cv_folds):
            # Split data
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.cv_folds - 1 else n_samples
            
            # Train and test indices
            test_indices = range(start_idx, end_idx)
            train_indices = list(range(0, start_idx)) + list(range(end_idx, n_samples))
            
            if len(train_indices) < X.shape[1]:  # Need more samples than features
                scores.append(-np.inf)
                continue
                
            X_train = X[train_indices]
            X_test = X[test_indices]
            
            try:
                # Fit on training data
                estimator_copy = type(estimator)(**estimator.get_params())
                estimator_copy.fit(X_train)
                
                # Score on test data
                cov_est = estimator_copy.covariance_
                score = self._compute_score(cov_est, X_test)
                scores.append(score)
                
            except Exception:
                scores.append(-np.inf)
                
        return scores
    
    def _compute_score(self, cov_matrix: np.ndarray, X_test: np.ndarray) -> float:
        """Compute log-likelihood score."""
        try:
            n_samples = X_test.shape[0]
            
            # Center the test data
            X_centered = X_test - np.mean(X_test, axis=0)
            
            # Compute log-likelihood
            sign, logdet = np.linalg.slogdet(cov_matrix)
            if sign <= 0:
                return -np.inf
                
            # Regularize for numerical stability
            cov_reg = cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0])
            precision = linalg.pinvh(cov_reg)
            
            log_likelihood_val = -0.5 * (
                logdet + 
                np.trace(X_centered @ precision @ X_centered.T) / n_samples +
                X_test.shape[1] * np.log(2 * np.pi)
            )
            
            return log_likelihood_val
            
        except Exception:
            return -np.inf
    
    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Get best covariance estimate."""
        if self.best_estimator_ is None:
            raise ValueError("Estimator not fitted.")
            
        return self.best_estimator_.covariance_
    
    def get_selection_results(self) -> pd.DataFrame:
        """Get cross-validation results for all methods."""
        results = []
        for name, scores in self.cv_scores_.items():
            results.append({
                'method': name,
                'mean_score': scores['mean'],
                'std_score': scores['std'],
                'selected': name == self.best_method_
            })
            
        return pd.DataFrame(results).sort_values('mean_score', ascending=False)


class RobustCovarianceEstimator:
    """
    Robust covariance estimation with multiple fallback strategies.
    
    Implements a hierarchy of covariance estimators with robust
    fallback mechanisms for high-dimensional scenarios.
    """
    
    def __init__(self, 
                 min_samples_ratio: float = 2.0,
                 regularization_strength: float = 1e-6):
        """
        Initialize robust estimator.
        
        Parameters:
        -----------
        min_samples_ratio : float
            Minimum ratio of samples to features
        regularization_strength : float
            Regularization for numerical stability
        """
        self.min_samples_ratio = min_samples_ratio
        self.regularization_strength = regularization_strength
        self.estimator_used_ = None
        self.covariance_ = None
        
    def fit(self, X: np.ndarray) -> 'RobustCovarianceEstimator':
        """
        Fit robust covariance estimator.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix
            
        Returns:
        --------
        self : RobustCovarianceEstimator
        """
        n_samples, n_features = X.shape
        sample_ratio = n_samples / n_features
        
        try:
            if sample_ratio >= self.min_samples_ratio:
                # Sufficient samples: try Ledoit-Wolf
                estimator = LedoitWolf()
                estimator.fit(X)
                self.covariance_ = estimator.covariance_
                self.estimator_used_ = 'ledoit_wolf'
                
            elif sample_ratio >= 1.5:
                # Moderate samples: try regularized empirical
                emp_cov = EmpiricalCovariance()
                emp_cov.fit(X)
                self.covariance_ = emp_cov.covariance_ + \
                    self.regularization_strength * np.eye(n_features)
                self.estimator_used_ = 'regularized_empirical'
                
            else:
                # Limited samples: use diagonal estimation
                variances = np.var(X, axis=0)
                self.covariance_ = np.diag(variances)
                self.estimator_used_ = 'diagonal'
                
        except Exception:
            # Fallback to diagonal
            variances = np.var(X, axis=0)
            variances = np.maximum(variances, self.regularization_strength)
            self.covariance_ = np.diag(variances)
            self.estimator_used_ = 'diagonal_fallback'
            
        return self
    
    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Get robust covariance estimate."""
        if self.covariance_ is None:
            raise ValueError("Estimator not fitted.")
            
        return self.covariance_