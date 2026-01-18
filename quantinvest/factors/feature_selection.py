"""
Factor selection module implementing sparse PCA and Lasso-based selection.

This module provides advanced dimensionality reduction and factor selection
techniques to derive interpretable and stable factor structures.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from sklearn.decomposition import SparsePCA, PCA
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')


class FactorSelector:
    """
    Advanced factor selection using sparse PCA and Lasso regression.
    
    This class implements the factor selection methodologies mentioned in the resume:
    - Sparse PCA for interpretable factor extraction
    - Lasso-based factor selection for stability
    - Cross-validation for hyperparameter tuning
    """
    
    def __init__(self, 
                 n_components: Optional[int] = None,
                 alpha_lasso: Optional[float] = None,
                 max_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize FactorSelector.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of components for sparse PCA
        alpha_lasso : float, optional
            Regularization strength for Lasso (if None, will use CV)
        max_iter : int
            Maximum iterations for optimization
        random_state : int
            Random state for reproducibility
        """
        self.n_components = n_components
        self.alpha_lasso = alpha_lasso
        self.max_iter = max_iter
        self.random_state = random_state
        self.sparse_pca_ = None
        self.lasso_selector_ = None
        self.selected_features_ = None
        self.feature_importance_ = None
        
    def fit_sparse_pca(self, 
                       X: pd.DataFrame, 
                       n_components: Optional[int] = None,
                       alpha: float = 1.0) -> 'FactorSelector':
        """
        Fit Sparse PCA to extract interpretable factors.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input factor matrix (time x factors)
        n_components : int, optional
            Number of components to extract
        alpha : float
            Sparsity controlling parameter
            
        Returns:
        --------
        self : FactorSelector
            Fitted selector
        """
        if n_components is None:
            n_components = min(X.shape[1], self.n_components or 10)
            
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Sparse PCA
        self.sparse_pca_ = SparsePCA(
            n_components=n_components,
            alpha=alpha,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        self.sparse_pca_.fit(X_scaled)
        self.scaler_ = scaler
        
        return self
    
    def transform_sparse_pca(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted Sparse PCA.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data to transform
            
        Returns:
        --------
        pd.DataFrame
            Transformed data with sparse principal components
        """
        if self.sparse_pca_ is None:
            raise ValueError("Sparse PCA not fitted. Call fit_sparse_pca first.")
            
        X_scaled = self.scaler_.transform(X)
        components = self.sparse_pca_.transform(X_scaled)
        
        component_names = [f'SparsePCA_{i+1}' for i in range(components.shape[1])]
        
        return pd.DataFrame(
            components, 
            index=X.index, 
            columns=component_names
        )
    
    def get_sparse_pca_loadings(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get factor loadings from Sparse PCA.
        
        Parameters:
        -----------
        feature_names : list
            Names of original features
            
        Returns:
        --------
        pd.DataFrame
            Factor loadings matrix
        """
        if self.sparse_pca_ is None:
            raise ValueError("Sparse PCA not fitted.")
            
        loadings = self.sparse_pca_.components_.T
        component_names = [f'SparsePCA_{i+1}' for i in range(loadings.shape[1])]
        
        return pd.DataFrame(
            loadings,
            index=feature_names,
            columns=component_names
        )
    
    def fit_lasso_selection(self, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           alpha: Optional[float] = None,
                           cv: int = 5) -> 'FactorSelector':
        """
        Fit Lasso regression for factor selection.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Factor matrix (time x factors)
        y : pd.Series
            Target variable (e.g., returns)
        alpha : float, optional
            Regularization parameter (if None, use CV)
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        self : FactorSelector
            Fitted selector
        """
        if alpha is None:
            # Use cross-validation to find optimal alpha
            self.lasso_selector_ = LassoCV(
                cv=cv,
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            self.lasso_selector_ = Lasso(
                alpha=alpha,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.lasso_selector_.fit(X_scaled, y)
        self.lasso_scaler_ = scaler
        
        # Extract selected features (non-zero coefficients)
        non_zero_coefs = np.abs(self.lasso_selector_.coef_) > 1e-6
        self.selected_features_ = X.columns[non_zero_coefs].tolist()
        self.feature_importance_ = pd.Series(
            self.lasso_selector_.coef_[non_zero_coefs],
            index=self.selected_features_
        ).abs().sort_values(ascending=False)
        
        return self
    
    def cross_sectional_lasso_selection(self, 
                                      factor_data: pd.DataFrame,
                                      return_data: pd.DataFrame,
                                      min_periods: int = 252,
                                      alpha: Optional[float] = None) -> pd.DataFrame:
        """
        Perform cross-sectional Lasso selection over time.
        
        Parameters:
        -----------
        factor_data : pd.DataFrame
            Factor exposures (time x stocks x factors in wide format)
        return_data : pd.DataFrame  
            Forward returns (time x stocks)
        min_periods : int
            Minimum periods for estimation
        alpha : float, optional
            Lasso regularization parameter
            
        Returns:
        --------
        pd.DataFrame
            Time series of selected factor coefficients
        """
        results = []
        dates = factor_data.index[min_periods:]
        
        for date in dates:
            # Get factor exposures and returns for cross-section
            # Note: historical window indices not used in this implementation
            # Keeping only the cross-sectional snapshot at the given date
            
            # Cross-sectional data at current date
            X_cross = factor_data.loc[date]  # stocks x factors
            y_cross = return_data.loc[date]  # stock returns
            
            # Remove missing data
            valid_mask = X_cross.notna().all(axis=1) & y_cross.notna()
            if valid_mask.sum() < 10:  # Need minimum observations
                continue
                
            X_valid = X_cross.loc[valid_mask]
            y_valid = y_cross.loc[valid_mask]
            
            # Fit Lasso
            if alpha is None:
                lasso = LassoCV(cv=5, max_iter=self.max_iter, random_state=self.random_state)
            else:
                lasso = Lasso(alpha=alpha, max_iter=self.max_iter, random_state=self.random_state)
                
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_valid)
            
            lasso.fit(X_scaled, y_valid)
            
            # Store results
            result = pd.Series(lasso.coef_, index=X_cross.columns, name=date)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_factor_stability_score(self, 
                                 selection_results: pd.DataFrame,
                                 window: int = 60) -> pd.Series:
        """
        Calculate factor stability scores over time.
        
        Parameters:
        -----------
        selection_results : pd.DataFrame
            Time series of factor selection results
        window : int
            Rolling window for stability calculation
            
        Returns:
        --------
        pd.Series
            Stability scores for each factor
        """
        # Calculate rolling standard deviation of coefficients
        rolling_std = selection_results.rolling(window=window).std()
        
        # Calculate rolling mean of absolute coefficients
        rolling_mean_abs = selection_results.abs().rolling(window=window).mean()
        
        # Stability score = mean(|coef|) / std(coef)
        stability_scores = (rolling_mean_abs / rolling_std).mean()
        
        return stability_scores.sort_values(ascending=False)
    
    def select_stable_factors(self, 
                            factor_data: pd.DataFrame,
                            return_data: pd.DataFrame,
                            stability_threshold: float = 1.0,
                            max_factors: int = 20) -> List[str]:
        """
        Select stable factors based on Lasso regularization path.
        
        Parameters:
        -----------
        factor_data : pd.DataFrame
            Factor exposures
        return_data : pd.DataFrame
            Return data
        stability_threshold : float
            Minimum stability score threshold
        max_factors : int
            Maximum number of factors to select
            
        Returns:
        --------
        list
            Names of selected stable factors
        """
        # Perform cross-sectional Lasso selection
        selection_results = self.cross_sectional_lasso_selection(factor_data, return_data)
        
        # Calculate stability scores
        stability_scores = self.get_factor_stability_score(selection_results)
        
        # Select factors meeting criteria
        stable_factors = stability_scores[
            stability_scores >= stability_threshold
        ].head(max_factors).index.tolist()
        
        return stable_factors
    
    def explained_variance_analysis(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze explained variance of different numbers of components.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input factor matrix
            
        Returns:
        --------
        dict
            Explained variance ratios for different n_components
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        max_components = min(X.shape[1], 20)
        explained_variance = {}
        
        for n_comp in range(1, max_components + 1):
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            pca.fit(X_scaled)
            explained_variance[n_comp] = pca.explained_variance_ratio_.sum()
            
        return explained_variance
    
    def generate_factor_selection_report(self, 
                                       factor_data: pd.DataFrame,
                                       return_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive factor selection report.
        
        Parameters:
        -----------
        factor_data : pd.DataFrame
            Factor exposures
        return_data : pd.DataFrame
            Return data
            
        Returns:
        --------
        dict
            Comprehensive factor selection analysis
        """
        report = {}
        
        # Sparse PCA analysis
        self.fit_sparse_pca(factor_data)
        sparse_components = self.transform_sparse_pca(factor_data)
        report['sparse_pca_components'] = sparse_components.shape[1]
        report['sparse_pca_loadings'] = self.get_sparse_pca_loadings(factor_data.columns.tolist())
        
        # Lasso selection analysis
        stable_factors = self.select_stable_factors(factor_data, return_data)
        report['selected_factors'] = stable_factors
        report['n_selected_factors'] = len(stable_factors)
        
        # Explained variance analysis
        report['explained_variance'] = self.explained_variance_analysis(factor_data)
        
        # Factor importance from latest Lasso fit
        if self.feature_importance_ is not None:
            report['factor_importance'] = self.feature_importance_
            
        return report