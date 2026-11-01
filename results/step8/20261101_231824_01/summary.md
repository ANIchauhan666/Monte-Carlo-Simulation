# Step 8 — Robustness and sensitivity summary

Baseline: S0=100.0, K=100.0, T=1.0 years, r=0.05, q=0.02, sigma=0.2, pairs=10000, seed=42.

European estimates inside their 95% intervals: 80/84.
Maximum absolute MC/BS error across European scenarios: 0.0830501.
Finite European prices, benchmarks, and standard errors: True.

European estimates outside their 95% intervals: 4; this can occur from ordinary sampling variation, especially at smaller pair counts.
## Relative sensitivity

Ranked by price range divided by baseline Black-Scholes price, averaged across calls and puts:
- Strike price: 375.76%
- Volatility: 200.77%
- Maturity: 106.19%
- Risk-free rate: 56.14%
- Dividend yield: 31.40%

## Directional checks
- Volatility, call: consistent (increasing).
- Volatility, put: consistent (increasing).
- Strike price, call: consistent (decreasing).
- Strike price, put: consistent (increasing).
- Risk-free rate, call: consistent (increasing).
- Risk-free rate, put: consistent (decreasing).
- Dividend yield, call: consistent (decreasing).
- Dividend yield, put: consistent (increasing).
- Maturity is reported without a monotonicity assertion; dividends and rates can affect the direction.

## Simulation and limitations

- European options use one exact terminal GBM step; intermediate steps do not change a terminal-only payoff.
- Asian and barrier estimates are discretely monitored; their sensitivity to 1, 12, 52, and 252 steps is in the path-resolution CSV and plot.
- Monte Carlo uncertainty remains; confidence intervals use the existing antithetic pair averaging and cross-fitted control-variate estimator.
- The analysis assumes constant GBM parameters and continuous dividend yield; it does not model jumps, stochastic volatility, or discrete dividends.
- A barrier monitored discretely can miss between-step crossings; a finer grid reduces but does not eliminate this bias.


## Estimates outside the 95% interval

- strike=120 call: absolute error 0.0176926; 95% half-width 0.0164173. This is a finite-sample deviation, not by itself evidence of model bias.
- strike=120 put: absolute error 0.0176926; 95% half-width 0.0164173. This is a finite-sample deviation, not by itself evidence of model bias.
- pairs=100000 call: absolute error 0.0134418; 95% half-width 0.0126537. This is a finite-sample deviation, not by itself evidence of model bias.
- pairs=100000 put: absolute error 0.0134418; 95% half-width 0.0126537. This is a finite-sample deviation, not by itself evidence of model bias.
