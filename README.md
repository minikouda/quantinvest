# quantinvest

`quantinvest` is a research framework for CNE5 factor-based portfolio construction.
It provides building blocks for:

- Factor computation + preprocessing 
- Factor selection 
- Risk modeling 
- Portfolio optimization 
- Backtesting + performance metrics + stress testing

## Installation

From the repository root (the folder containing `pyproject.toml`):

```bash
pip install -e .
```

## Quickstart

### 1) Estimate covariance, optimize weights, backtest

```python
import numpy as np
import pandas as pd

from quantinvest.risk.covariance_estimators import LedoitWolfEstimator
from quantinvest.optimization.mean_variance import RegularizedMeanVarianceOptimizer
from quantinvest.backtesting.engine import BacktestEngine

# returns: DataFrame (dates x assets) of simple returns
# Example: returns = prices.pct_change().dropna()

X = returns.to_numpy()
cov = LedoitWolfEstimator().fit(X).predict()

mu = returns.mean(axis=0).to_numpy()  # simple expected return proxy

opt = RegularizedMeanVarianceOptimizer(
		risk_aversion=1.0,
		l2_reg=1e-4,
		turnover_penalty=0.0,
)

w = opt.optimize(mu, cov)  # numpy array, length = n_assets

# BacktestEngine expects weights over time.
weights = pd.DataFrame([w], index=[returns.index[0]], columns=returns.columns)
bt = BacktestEngine(commission=0.001, risk_free_rate=0.0001)
results = bt.run_backtest(weights=weights, returns=returns)

print(results["metrics"])
```

### 2) Compute style factors (beta/momentum/liquidity/residual-vol)

```python
from quantinvest.factors.style_factors import StyleFactors

# Provide a dict of DataFrames keyed by expected names.
# All inputs should be (dates x assets).
data = {
		"close": close_prices,                 # optional if returns provided
		"returns": returns,                   # optional if close provided
		"market_returns": market_returns,     # required for beta
		"volume": volume,
		"float_shares": float_shares,
		"market_cap": market_cap,             # optional if close*float_shares available
		"turnover": turnover,                 # optional if volume/float_shares available
		"risk_free": risk_free_rate_series,   # optional (scalar/Series/DataFrame)
}

sf = StyleFactors()
style_factor_panel = sf.compute(data)

# The output is a wide DataFrame with a column MultiIndex:
# level 0 = factor name, level 1 = asset
print(style_factor_panel.columns.levels[0])
```

## Configuration

`quantinvest.utils.config.Config` looks for a `config.yaml` in common locations (including the repo root).# quantinvest
