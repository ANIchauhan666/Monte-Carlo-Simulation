"""
Automated Monte Carlo option pricer built on real-world data and market-standard equations.

Give it a symbol; everything else is fetched or calculated, and charts are drawn automatically:

  Spot            live quote (Yahoo chart API), with an explicitly reported latest-close fallback
  Dividends       trailing-12-month dividends / spot (a continuous-yield proxy)
  Risk-free rate  FRED quoted-yield proxies, interpolated and converted approximately to continuous
                  rates; USD CMT quotes are not zero-coupon rates and INR is an interbank proxy
  Expiry time     calendar time to the exchange close on expiry day, ACT/365
  Volatility      implied vol solved by bisection from each contract's bid/ask mid (OTM options),
                  giving one vol per strike (the smile); EWMA (RiskMetrics, 0.94) if no chain
  Pricing         risk-neutral GBM with dividend yield, antithetic variates, cross-fitted control
                  variates; European (checked against Black-Scholes) and American (Longstaff-
                  Schwartz trained/evaluated on separate paths); Monte Carlo Greeks; ATM exotics
  Charts          saved as PNG in ./plots and shown on screen when a display is available:
                    <SYMBOL>_simulation.png        volatility smile, sample paths, terminal-price
                                                   histogram with the theoretical lognormal curve
                    <SYMBOL>_<expiry>_results.png  prices vs strike, MC vs Black-Scholes Greeks,
                                                   Asian/barrier prices (one file per expiry)

Usage:
  python monte_carlo_option_pricer.py AAPL                       # ladder of expiries, prices + charts
  python monte_carlo_option_pricer.py AAPL --days 45             # YOU choose the time: 45 days
  python monte_carlo_option_pricer.py AAPL --days 2026-12-18     # ...or an exact expiry date
  python monte_carlo_option_pricer.py AAPL --days 2w,3m,1y       # ...or shorthand / a list
  python monte_carlo_option_pricer.py AAPL --list-expiries       # see which expiry dates exist
  python monte_carlo_option_pricer.py AAPL TSLA --days pick --csv results.csv
  python monte_carlo_option_pricer.py AAPL --days 45 --strikes 200,210,220 --pairs 20000
  python monte_carlo_option_pricer.py AAPL --no-show             # save the PNG charts, open no windows
  python monte_carlo_option_pricer.py AAPL --no-plot             # numbers only
  python monte_carlo_option_pricer.py                            # prompts: symbol, then YOUR expiry choice
Overrides (all optional): --r --q --sigma --strikes --days --pairs --seed --plot-dir

Scheduling (weekdays 16:30):  30 16 * * 1-5  cd /path && python3 monte_carlo_option_pricer.py AAPL TSLA --csv results.csv --no-show
(On a machine with no display, such as cron or a server, charts are saved only; no window is opened.)
"""
import argparse
import csv
import math
import os
import re
import sys
import warnings
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bounded_path_cache import BoundedPathCache

try:
    from zoneinfo import ZoneInfo
except ImportError:                      # very old Python
    ZoneInfo = None


# ========== 1. Configuration ==========
BASE_URLS = (
    "https://query1.finance.yahoo.co.in/v8/finance/chart/",
    "https://query1.finance.yahoo.com/v8/finance/chart/",
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124 Safari/537.36"
    )
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_MAX_AGE_DAYS = {"USD": 14, "INR": 60}

# Quoted-yield proxies per currency: (tenor in years, FRED series id). USD CMT quotes are
# par-yield proxies, not zero-coupon rates; "m" is the assumed annual compounding frequency.
# INR uses one 3-month interbank-rate proxy flat across tenors, not a sovereign yield curve.
CURVES = {
    "USD": {"m": 2, "points": [(1 / 12, "DGS1MO"), (3 / 12, "DGS3MO"), (6 / 12, "DGS6MO"),
                                (1.0, "DGS1"), (2.0, "DGS2")]},
    "INR": {"m": 1, "points": [(0.25, "IR3TIB01INM156N")]},
}
# Exchange close used to measure time to expiry: currency -> (timezone, hour, minute)
MARKET_CLOSE = {"USD": ("America/New_York", 16, 0), "INR": ("Asia/Kolkata", 15, 30)}

AUTO_DAYS = (7, 30, 90, 180)   # 'ladder': 1 week, 1 month, 3 months, 6 months
PICK_DAYS = 30                 # 'pick': the listed expiry closest to 1 month


# ========== 2. Market data ==========
def get_history(symbol: str, period: str = "1y") -> dict:
    """Fetch a quote and adjusted daily closes; disclose when a fallback is required."""
    params = {"range": period, "interval": "1d"}
    response = None
    status_codes = []
    for base_url in BASE_URLS:
        try:
            candidate = requests.get(base_url + symbol, params=params, headers=HEADERS, timeout=10)
        except requests.RequestException:
            continue
        status_codes.append(candidate.status_code)
        if candidate.status_code == 200:
            response = candidate
            break
    if response is None:
        detail = f" HTTP statuses: {status_codes}." if status_codes else ""
        raise ValueError(f"No Yahoo data found for '{symbol}'. Check the symbol or network.{detail}")

    try:
        chart = response.json()["chart"]
        if chart.get("error"):
            raise ValueError(chart["error"])
        result = chart["result"][0]
        raw_closes = [
            float(value) for value in result["indicators"]["quote"][0]["close"]
            if value is not None and math.isfinite(float(value)) and float(value) > 0
        ]
        if len(raw_closes) < 30:
            raise ValueError("Not enough usable observations")
        meta = result["meta"]
        live = meta.get("regularMarketPrice")
        if isinstance(live, (int, float)) and math.isfinite(float(live)) and live > 0:
            spot = float(live)
            spot_source = "Yahoo regular-market quote"
        else:
            spot = float(raw_closes[-1])
            spot_source = "Yahoo latest close (live quote unavailable)"
            warnings.warn(f"{symbol}: {spot_source} used as spot.", RuntimeWarning, stacklevel=2)

        currency = meta.get("currency")
        if not isinstance(currency, str) or not currency.strip():
            raise ValueError("Yahoo did not provide the instrument currency")

        adjusted_data = result.get("indicators", {}).get("adjclose") or []
        adjusted = adjusted_data[0].get("adjclose") if adjusted_data else None
        adjusted_closes = []
        if adjusted:
            adjusted_closes = [
                float(value) for value in adjusted
                if value is not None and math.isfinite(float(value)) and float(value) > 0
            ]
        if len(adjusted_closes) >= 30:
            history_closes = adjusted_closes
            close_source = "Yahoo adjusted close"
        else:
            history_closes = raw_closes
            close_source = "Yahoo unadjusted close"
            warnings.warn(
                f"{symbol}: adjusted closes unavailable; historical volatility uses unadjusted closes.",
                RuntimeWarning, stacklevel=2,
            )
        return {
            "symbol": meta.get("symbol", symbol), "currency": currency,
            "close": np.array(history_closes, dtype=float), "spot": spot,
            "spot_source": spot_source, "close_source": close_source,
        }
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Yahoo returned no usable price data for '{symbol}'.") from error


def get_dividend_yield(symbol: str, spot: float) -> float:
    """Trailing-12-month dividends / spot, a proxy for a continuous dividend yield."""
    spot = _finite_value("Spot price", spot)
    if spot <= 0:
        raise ValueError("Spot price must be finite and greater than zero.")
    try:
        divs = yf.Ticker(symbol).dividends
    except Exception as error:
        raise ValueError(f"Could not retrieve Yahoo dividend history for '{symbol}'.") from error
    if divs is None or len(divs) == 0:
        warnings.warn(
            f"{symbol}: Yahoo returned no dividend history; assuming zero dividend yield.",
            RuntimeWarning, stacklevel=2,
        )
        return 0.0
    try:
        cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
        amount = float(divs[divs.index >= cutoff].sum())
        if not math.isfinite(amount):
            raise ValueError("Yahoo dividend history contains a non-finite total")
        return amount / float(spot)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Yahoo returned unusable dividend history for '{symbol}'.") from error


def ewma_vol(close, lam: float = 0.94) -> float:
    """RiskMetrics EWMA volatility, initialized once from the first 20 daily returns."""
    prices = np.asarray(close, dtype=float)
    if prices.ndim != 1 or prices.size < 21 or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("EWMA volatility requires at least 21 finite, positive closing prices.")
    if not math.isfinite(float(lam)) or not 0 <= lam < 1:
        raise ValueError("EWMA lambda must be finite and in the interval [0, 1).")
    ret = np.diff(np.log(prices))
    var = float(np.var(ret[:20], ddof=1))
    for x in ret[20:]:
        var = lam * var + (1 - lam) * x * x
    if not math.isfinite(var) or var < 0:
        raise ValueError("EWMA variance calculation was not finite and non-negative.")
    return math.sqrt(var * 252)


# ========== 3. Risk-free rate (yield curve) ==========
def default_rate(currency: str) -> float:
    """Educational fallback levels, never represented as current market observations."""
    if currency == "INR":
        return 0.065
    if currency == "USD":
        return 0.045
    raise ValueError(
        f"No configured risk-free proxy for {currency!r}; provide --r explicitly."
    )


def fetch_fred(series_id: str):
    """Latest valid observation of a FRED series -> (date string, percent)."""
    resp = requests.get(FRED_URL, params={"id": series_id}, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    latest = None
    for parts in csv.reader(resp.text.strip().splitlines()):
        if len(parts) >= 2 and parts[0] != "observation_date":
            try:
                value = float(parts[1])
            except ValueError:
                continue
            if math.isfinite(value):
                latest = (parts[0], value)
    if latest is None:
        raise ValueError(f"No data in FRED series {series_id}")
    return latest


_curve_cache = {}


def get_rate_curve(currency: str) -> dict:
    """Build approximate continuously compounded rates from quoted yield proxies."""
    if currency in _curve_cache:
        return _curve_cache[currency]
    spec, points, dates, failures = CURVES.get(currency), [], [], []
    if spec is None:
        raise ValueError(
            f"No configured risk-free curve for {currency!r}; provide --r explicitly."
        )
    m = spec["m"]
    for tenor, series in spec["points"]:
        try:
            asof, pct = fetch_fred(series)
            observation_date = date.fromisoformat(asof)
            age_days = (date.today() - observation_date).days
            max_age = FRED_MAX_AGE_DAYS[currency]
            if age_days < 0 or age_days > max_age:
                raise ValueError(
                    f"observation is {age_days} days old; maximum accepted age is {max_age}"
                )
            y = pct / 100.0
            if not math.isfinite(y) or 1 + y / m <= 0:
                raise ValueError(f"invalid quoted yield {pct!r}")
            points.append((tenor, m * math.log1p(y / m)))
            dates.append(asof)
        except Exception as error:
            failures.append(f"{series}: {error}")
    if points:
        if failures:
            warnings.warn(
                f"FRED returned only a partial {currency} yield proxy; missing points: "
                + "; ".join(failures),
                RuntimeWarning, stacklevel=2,
            )
        curve = {
            "points": sorted(points),
            "source": (
                f"FRED {currency} quoted-yield proxy, continuously compounded "
                f"(latest observation {max(dates)})"
            ),
        }
    else:
        fallback = default_rate(currency)
        warnings.warn(
            f"FRED rate fetch failed for {currency}; using the educational placeholder "
            f"{fallback:.2%}. This is not a live market rate; pass --r to override.",
            RuntimeWarning, stacklevel=2,
        )
        curve = {
            "points": [(1.0, fallback)],
            "source": f"EDUCATIONAL {currency} placeholder {fallback:.2%} (FRED unavailable)",
        }
    _curve_cache[currency] = curve
    return curve


def rate_at(curve: dict, T: float) -> float:
    if not math.isfinite(float(T)) or T < 0:
        raise ValueError("Rate-curve maturity must be finite and non-negative.")
    if not curve.get("points"):
        raise ValueError("Rate curve must contain at least one tenor/rate point.")
    xs, ys = zip(*curve["points"])
    return float(np.interp(T, xs, ys))          # flat extrapolation outside the quoted tenors


# ========== 4. Option chain ==========
def years_to_expiry(expiry: str, currency: str) -> float:
    """ACT/365 time from now to the exchange close on expiry day."""
    exp = date.fromisoformat(expiry)
    spec = MARKET_CLOSE.get(currency)
    if spec and ZoneInfo:
        try:
            tz = ZoneInfo(spec[0])
        except (KeyError, OSError):
            tz = None
        if tz is not None:
            close = datetime(exp.year, exp.month, exp.day, spec[1], spec[2], tzinfo=tz)
            now = datetime.now(tz)
            secs = (
                close.astimezone(timezone.utc) - now.astimezone(timezone.utc)
            ).total_seconds()
            if secs <= 0:
                raise ValueError(f"Expiry {expiry} is not in the future at the exchange close.")
            return secs / (365 * 86400)
    days = (exp - date.today()).days
    if days <= 0:
        raise ValueError(f"Expiry {expiry} is not in the future.")
    warnings.warn(
        "Exchange timezone data unavailable; expiry time uses date-based ACT/365 approximation.",
        RuntimeWarning, stacklevel=2,
    )
    return days / 365


def get_option_chain(symbol: str, target_days: int, currency: str):
    """Listed expiry (at least 1 day out) closest to target_days, or None."""
    if not isinstance(target_days, (int, np.integer)) or target_days < 1:
        raise ValueError("Target expiry must be a positive number of days.")
    try:
        ticker = yf.Ticker(symbol)
        spec = MARKET_CLOSE.get(currency)
        today = (
            datetime.now(ZoneInfo(spec[0])).date()
            if spec and ZoneInfo else date.today()
        )
        candidates = []
        for exp in ticker.options or ():
            d = (date.fromisoformat(exp) - today).days
            if d >= 1:
                candidates.append((abs(d - target_days), exp, d))
        if not candidates:
            return None
        _, exp, d = min(candidates)
        oc = ticker.option_chain(exp)
        return {"expiry": exp, "days": d, "T": years_to_expiry(exp, currency),
                "calls": oc.calls.set_index("strike"), "puts": oc.puts.set_index("strike")}
    except Exception as error:
        warnings.warn(
            f"{symbol}: Yahoo option-chain retrieval failed ({error}); using model-only pricing.",
            RuntimeWarning, stacklevel=2,
        )
        return None


def _num(row, col) -> float:
    try:
        x = float(row[col])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def quote_price(row):
    """Bid/ask mid if the quote is two-sided and not too wide; else last trade; else None."""
    bid, ask, last = _num(row, "bid"), _num(row, "ask"), _num(row, "lastPrice")
    if bid > 0 and ask >= bid:
        mid = (bid + ask) / 2
        if (ask - bid) / mid <= 0.5:
            return mid, "mid"
        return (last, "last") if last > 0 else (None, "wide")
    return (last, "last") if last > 0 else (None, "none")


# ========== 5. Black-Scholes-Merton, Greeks, implied volatility ==========
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _finite_value(name, value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _validate_market_inputs(S, r, q, sigma, T) -> None:
    S = _finite_value("Stock price", S)
    r = _finite_value("Risk-free rate", r)
    q = _finite_value("Dividend yield", q)
    sigma = _finite_value("Volatility", sigma)
    T = _finite_value("Time to maturity", T)
    if S <= 0:
        raise ValueError("Stock price must be greater than zero.")
    if sigma < 0:
        raise ValueError("Volatility cannot be negative.")
    if T < 0:
        raise ValueError("Time to maturity cannot be negative.")


def _validate_option_inputs(S, K, r, q, sigma, T, kind) -> None:
    _validate_market_inputs(S, r, q, sigma, T)
    K = _finite_value("Strike price", K)
    if K <= 0:
        raise ValueError("Strike price must be greater than zero.")
    if kind not in ("call", "put"):
        raise ValueError("Option kind must be 'call' or 'put'.")


def bs_price(S, K, r, q, sigma, T, kind):
    _validate_option_inputs(S, K, r, q, sigma, T, kind)
    if sigma <= 0 or T <= 0:
        fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
        return max(fwd if kind == "call" else -fwd, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    if kind == "call":
        return S * dq * norm_cdf(d1) - K * dr * norm_cdf(d2)
    return K * dr * norm_cdf(-d2) - S * dq * norm_cdf(-d1)


def bs_greeks(S, K, r, q, sigma, T, kind) -> dict:
    """Delta, gamma, vega (per vol point), theta (per calendar day), rho (per 1% rate)."""
    _validate_option_inputs(S, K, r, q, sigma, T, kind)
    if sigma <= 0 or T <= 0:
        raise ValueError("Black-Scholes Greeks require positive volatility and time to maturity.")
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    dq, dr, pdf = math.exp(-q * T), math.exp(-r * T), norm_pdf(d1)
    gamma = dq * pdf / (S * sq)
    vega = S * dq * pdf * math.sqrt(T) / 100
    decay = -S * dq * pdf * sigma / (2 * math.sqrt(T))
    if kind == "call":
        delta = dq * norm_cdf(d1)
        theta = decay - r * K * dr * norm_cdf(d2) + q * S * dq * norm_cdf(d1)
        rho = K * T * dr * norm_cdf(d2) / 100
    else:
        delta = -dq * norm_cdf(-d1)
        theta = decay + r * K * dr * norm_cdf(-d2) - q * S * dq * norm_cdf(-d1)
        rho = -K * T * dr * norm_cdf(-d2) / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta / 365, "rho": rho}


def implied_vol(price, S, K, r, q, T, kind):
    """Bisection on the (monotone) Black-Scholes price. None if the quote is arbitrage-violating."""
    _validate_option_inputs(S, K, r, q, sigma=0.0, T=T, kind=kind)
    price = _finite_value("Option price", price)
    if price < 0 or T <= 0:
        return None
    fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
    lower = max(fwd if kind == "call" else -fwd, 0.0)
    upper = S * math.exp(-q * T) if kind == "call" else K * math.exp(-r * T)
    if not (lower + 1e-8 < price < upper - 1e-8):
        return None
    lo, hi = 1e-4, 5.0
    if bs_price(S, K, r, q, hi, T, kind) < price:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, r, q, mid, T, kind) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def build_smile(chain, S0, r, q, T) -> list:
    """Implied vols from OTM options only (puts below spot, calls at/above), the market standard."""
    pts = []
    for kind, df in (("call", chain["calls"]), ("put", chain["puts"])):
        for K, row in df.iterrows():
            if (kind == "put") != (K < S0):
                continue
            price, src = quote_price(row)
            if price is None or price < 0.01:
                continue
            iv = implied_vol(price, S0, float(K), r, q, T, kind)
            if iv is None or not 0.02 < iv < 4.0:
                continue
            pts.append({"strike": float(K), "iv": iv, "price": price, "kind": kind, "source": src})
    return sorted(pts, key=lambda p: p["strike"])


def smile_iv(pts, K) -> float:
    return float(np.interp(K, [p["strike"] for p in pts], [p["iv"] for p in pts]))


# ========== 6. Monte Carlo engine ==========
def simulate_paths(S0, r, q, sigma, T, steps, n_pairs, seed):
    """Risk-neutral GBM with exact log steps and memory-bounded antithetic paths."""
    _validate_market_inputs(S0, r, q, sigma, T)
    if T <= 0:
        raise ValueError("Monte Carlo time to maturity must be greater than zero.")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("Number of time steps must be a positive integer.")
    if not isinstance(n_pairs, (int, np.integer)) or n_pairs < 4:
        raise ValueError("Number of antithetic pairs must be an integer of at least four.")
    if seed is not None and (not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("Random seed must be a non-negative integer or None.")
    rng = np.random.default_rng(seed)
    dt = T / steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt)
    paths = np.empty((steps + 1, 2 * n_pairs), dtype=float)
    paths[0] = S0
    log_spot = np.full(2 * n_pairs, math.log(S0))
    for step in range(steps):
        z = rng.standard_normal(n_pairs)
        log_spot[:n_pairs] += drift + diffusion * z
        log_spot[n_pairs:] += drift - diffusion * z
        with np.errstate(over="ignore", invalid="ignore"):
            paths[step + 1] = np.exp(log_spot)
    if not np.all(np.isfinite(paths)):
        raise ValueError("GBM simulation overflowed; review volatility, rates, or maturity.")
    return paths


def pair_mean_se(x):
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size < 4 or values.size % 2:
        raise ValueError("Antithetic estimates require an even 1D array with at least two pairs.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Antithetic estimates contain non-finite values.")
    half = values.size // 2
    p = 0.5 * (values[:half] + values[half:])
    return float(p.mean()), float(p.std(ddof=1) / math.sqrt(half))


def cv_estimate(values, control, control_mean):
    """Cross-fitted control variate after antithetic pair averaging."""
    values = np.asarray(values, dtype=float)
    control = np.asarray(control, dtype=float)
    control_mean = _finite_value("Control expectation", control_mean)
    if values.ndim != 1 or control.ndim != 1 or values.size != control.size:
        raise ValueError("Payoff and control arrays must be 1D arrays of equal length.")
    if values.size < 8 or values.size % 2:
        raise ValueError("Control-variate estimates require at least four antithetic pairs.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(control)):
        raise ValueError("Payoff and control arrays must contain only finite values.")
    half = values.size // 2
    v = 0.5 * (values[:half] + values[half:])
    c = 0.5 * (control[:half] + control[half:])
    if v.size < 4:
        raise ValueError("Control-variate estimates require at least four antithetic pairs.")

    split = v.size // 2

    def estimate_beta(v_fit, c_fit):
        var_c = float(np.var(c_fit, ddof=1))
        return float(np.cov(v_fit, c_fit, ddof=1)[0, 1]) / var_c if var_c > 0 else 0.0

    adj = np.empty_like(v)
    adj[:split] = v[:split] - estimate_beta(v[split:], c[split:]) * (
        c[:split] - control_mean
    )
    adj[split:] = v[split:] - estimate_beta(v[:split], c[:split]) * (
        c[split:] - control_mean
    )
    return float(adj.mean()), float(adj.std(ddof=1) / math.sqrt(half))


def payoff_fn(K, kind):
    if not math.isfinite(float(K)) or K <= 0:
        raise ValueError("Strike price must be finite and greater than zero.")
    if kind == "call":
        return lambda s: np.maximum(s - K, 0.0)
    if kind == "put":
        return lambda s: np.maximum(K - s, 0.0)
    raise ValueError("Option kind must be 'call' or 'put'.")


def european_mc(paths, S0, K, r, q, T, kind):
    _validate_option_inputs(S0, K, r, q, sigma=0.0, T=T, kind=kind)
    if T <= 0:
        raise ValueError("Monte Carlo time to maturity must be greater than zero.")
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 8 or paths.shape[1] % 2:
        raise ValueError("Paths must be a 2D array with an even number of at least eight paths.")
    if not np.all(np.isfinite(paths)) or np.any(paths <= 0):
        raise ValueError("Simulated paths must contain only finite, positive prices.")
    if not np.allclose(paths[0], S0, rtol=0, atol=1e-10):
        raise ValueError("Every path must start at the supplied stock price.")
    ST, disc = paths[-1], math.exp(-r * T)
    values = disc * payoff_fn(K, kind)(ST)
    price, se = cv_estimate(values, disc * ST, S0 * math.exp(-q * T))
    indicator = (ST > K) if kind == "call" else -(ST < K).astype(float)
    delta, _ = pair_mean_se(disc * ST / S0 * indicator)                   # pathwise delta
    return price, se, values, delta


def mc_greeks(paths, S0, K, r, q, sigma, T, kind):
    """Estimate European Greeks using pathwise delta and common-random-number differences."""
    _validate_option_inputs(S0, K, r, q, sigma, T, kind)
    if sigma <= 0 or T <= 0:
        raise ValueError("Monte Carlo Greeks require positive volatility and time to maturity.")
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 8 or paths.shape[1] % 2:
        raise ValueError("Greek estimation requires an even 2D path array with at least eight paths.")
    if not np.all(np.isfinite(paths)) or np.any(paths <= 0):
        raise ValueError("Greek estimation paths must contain only finite, positive prices.")
    if not np.allclose(paths[0], S0, rtol=0, atol=1e-10):
        raise ValueError("Every path must start at the supplied stock price.")

    log_return = np.log(paths[-1] / S0)
    z = (log_return - (r - q - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    def price_at(spot, vol, maturity):
        terminal = spot * np.exp(
            (r - q - 0.5 * vol**2) * maturity + vol * math.sqrt(maturity) * z
        )
        discount = math.exp(-r * maturity)
        payoff = discount * payoff_fn(K, kind)(terminal)
        control = discount * terminal
        return cv_estimate(payoff, control, spot * math.exp(-q * maturity))[0]

    _, _, _, delta = european_mc(paths, S0, K, r, q, T, kind)
    spot_bump = max(S0 * 0.005, 1e-4)
    base_price = price_at(S0, sigma, T)
    gamma = (
        price_at(S0 + spot_bump, sigma, T) - 2 * base_price
        + price_at(S0 - spot_bump, sigma, T)
    ) / spot_bump**2

    vol_bump = min(0.01, sigma * 0.49)
    vega = (
        price_at(S0, sigma + vol_bump, T) - price_at(S0, sigma - vol_bump, T)
    ) / (2 * vol_bump) * 0.01

    time_bump = min(1 / 365, T / 2)
    theta = -(
        price_at(S0, sigma, T + time_bump) - price_at(S0, sigma, T - time_bump)
    ) / (2 * time_bump * 365)
    return {"delta": delta, "gamma": float(gamma), "vega": float(vega), "theta": float(theta)}


def _fit_continuation(moneyness, cashflows, degree):
    degree = min(degree, len(cashflows) - 1)
    scaled = (moneyness - 1.0) / (moneyness + 1.0)
    basis = np.polynomial.legendre.legvander(scaled, degree)
    coefficients, _, _, _ = np.linalg.lstsq(basis, cashflows, rcond=None)
    return coefficients, degree


def _predict_continuation(moneyness, model):
    coefficients, degree = model
    scaled = (moneyness - 1.0) / (moneyness + 1.0)
    return np.polynomial.legendre.legval(scaled, coefficients[:degree + 1])


def lsm_values(paths, K, r, T, kind, degree=3, evaluation_paths=None, q=0.0):
    """Train a Longstaff-Schwartz exercise policy and value it on separate paths."""
    _validate_option_inputs(1.0, K, r, q, sigma=0.0, T=T, kind=kind)
    if T <= 0:
        raise ValueError("Longstaff-Schwartz time to maturity must be greater than zero.")
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("Regression degree must be a non-negative integer.")
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 2 or not np.all(np.isfinite(paths)) or np.any(paths <= 0):
        raise ValueError("Training paths must be a finite, positive 2D price array.")
    if evaluation_paths is None:
        warnings.warn(
            "LSM is using its training paths for valuation; an independent evaluation_paths "
            "array is preferable to reduce look-ahead bias.",
            RuntimeWarning, stacklevel=2,
        )
        evaluation_paths = paths
    else:
        evaluation_paths = np.asarray(evaluation_paths, dtype=float)
    if (
        evaluation_paths.ndim != 2
        or evaluation_paths.shape[0] != paths.shape[0]
        or evaluation_paths.shape[1] < 4
        or evaluation_paths.shape[1] % 2
        or not np.all(np.isfinite(evaluation_paths))
        or np.any(evaluation_paths <= 0)
    ):
        raise ValueError("Evaluation paths must match training steps and have an even, finite path set.")
    if not np.allclose(paths[0], paths[0, 0], rtol=0, atol=1e-10):
        raise ValueError("Every Longstaff-Schwartz training path must share its initial spot.")
    if not np.allclose(evaluation_paths[0], paths[0, 0], rtol=0, atol=1e-10):
        raise ValueError("Training and evaluation paths must share the same initial spot.")

    steps = paths.shape[0] - 1
    disc = math.exp(-r * T / steps)
    pay = payoff_fn(K, kind)
    if kind == "call" and q <= 0 and r >= 0:
        return math.exp(-r * T) * pay(evaluation_paths[-1])

    cash = pay(paths[-1])
    models = {}
    for t in range(steps - 1, 0, -1):
        cash *= disc
        exercise = pay(paths[t])
        itm = np.flatnonzero(exercise > 0)
        if itm.size < 2:
            continue
        model = _fit_continuation(paths[t, itm] / K, cash[itm], degree)
        models[t] = model
        continuation = _predict_continuation(paths[t, itm] / K, model)
        exercise_now = exercise[itm] > continuation
        cash[itm[exercise_now]] = exercise[itm[exercise_now]]

    spot = float(paths[0, 0])
    continuation_at_zero = float(np.mean(cash * disc))
    intrinsic_at_zero = float(pay(np.array([spot]))[0])
    if intrinsic_at_zero > continuation_at_zero:
        return np.full(evaluation_paths.shape[1], intrinsic_at_zero, dtype=float)

    discounted_values = np.zeros(evaluation_paths.shape[1], dtype=float)
    alive = np.ones(evaluation_paths.shape[1], dtype=bool)
    for t in range(1, steps):
        model = models.get(t)
        if model is None:
            continue
        spots = evaluation_paths[t]
        exercise = pay(spots)
        continuation = _predict_continuation(spots / K, model)
        exercise_now = alive & (exercise > 0) & (exercise > continuation)
        discounted_values[exercise_now] = exercise[exercise_now] * math.exp(-r * T * t / steps)
        alive[exercise_now] = False
    discounted_values[alive] = pay(evaluation_paths[-1, alive]) * math.exp(-r * T)
    return discounted_values


def american_cv(v_amer, v_euro, bs_euro):
    """American = BS European + E[V_amer - V_euro]; the European leg is an excellent control variate."""
    v_amer, v_euro = np.asarray(v_amer, dtype=float), np.asarray(v_euro, dtype=float)
    if v_amer.shape != v_euro.shape:
        raise ValueError("American and European payoff arrays must have matching shapes.")
    prem, prem_se = pair_mean_se(v_amer - v_euro)
    bs_euro = _finite_value("European control price", bs_euro)
    prem = max(prem, 0.0)                       # early-exercise premium cannot be negative
    return bs_euro + prem, prem_se, prem


def price_exotics(S0, r, q, T, paths):
    """ATM arithmetic Asians and a 90% down-and-out call monitored at every time step."""
    _validate_market_inputs(S0, r, q, sigma=0.0, T=T)
    if T <= 0:
        raise ValueError("Exotic-option time to maturity must be greater than zero.")
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 8 or paths.shape[1] % 2:
        raise ValueError("Exotic pricing requires an even 2D path array with at least eight paths.")
    if not np.all(np.isfinite(paths)) or np.any(paths <= 0):
        raise ValueError("Exotic-option paths must contain only finite, positive prices.")
    disc, ST = math.exp(-r * T), paths[-1]
    avg, low = paths[1:].mean(axis=0), paths[1:].min(axis=0)
    barrier = 0.9 * S0
    alive = low > barrier                      # discretely monitored once per step
    control, cmean = disc * ST, S0 * math.exp(-q * T)
    out = {
        "Asian call (arithmetic, ATM)": np.maximum(avg - S0, 0.0),
        "Asian put (arithmetic, ATM)": np.maximum(S0 - avg, 0.0),
        f"Down-and-out call (ATM, B={barrier:,.2f})": np.maximum(ST - S0, 0.0) * alive,
    }
    return {name: cv_estimate(disc * v, control, cmean) for name, v in out.items()}


# ========== 7. Pipeline ==========
_UNIT_DAYS = {"d": 1, "w": 7, "m": 30, "y": 365}
_DAYS_HELP = ("Use a number of days (30), shorthand (2w, 3m, 1y), a date (2026-12-18), "
              "a list (7,30,90), 'pick' or 'ladder'.")


def _parse_day_token(token: str) -> int:
    token = token.strip().lower()
    if re.fullmatch(r"\d+", token):
        days = int(token)
    elif re.fullmatch(r"\d+[dwmy]", token):
        days = int(token[:-1]) * _UNIT_DAYS[token[-1]]
    else:
        try:
            days = (date.fromisoformat(token) - date.today()).days
        except ValueError:
            raise ValueError(f"Cannot read '{token}'. {_DAYS_HELP}") from None
    if days < 1:
        raise ValueError(f"'{token}' is not in the future. {_DAYS_HELP}")
    return days


def parse_days(text, default="ladder") -> list:
    """
    The time(s) to expiry YOU choose, as calendar days:
      '30' -> [30]   '2w' -> [14]   '3m' -> [90]   '1y' -> [365]   '2026-12-18' -> days until then
      '7,30,90' or '1w,1m,2026-12-18' -> a list      'pick' -> ~1 month      'ladder' -> 1w,1m,3m,6m
    """
    text = str(text).strip().lower() or default
    if text in ("ladder", "auto"):
        return list(AUTO_DAYS)
    if text == "pick":
        return [PICK_DAYS]
    tokens = text.replace(" ", "").split(",")
    if any(not token for token in tokens):
        raise ValueError(f"Empty entry in the list. {_DAYS_HELP}")
    return [_parse_day_token(token) for token in tokens]


def is_explicit_days(text) -> bool:
    """True when the user named their own time (so we tell them if it had to be snapped)."""
    return str(text).strip().lower() not in ("", "pick", "ladder", "auto")


def list_expiries(symbol: str) -> list:
    """Listed option expiries as [(date string, days away)], soonest first."""
    try:
        today = date.today()
        out = [(e, (date.fromisoformat(e) - today).days) for e in (yf.Ticker(symbol).options or ())]
        return [(e, d) for e, d in out if d >= 1]
    except Exception:
        return []


def print_expiries(symbol: str, expiries: list) -> None:
    if not expiries:
        print(f"\nNo listed option expiries found for {symbol}; a time you type is priced "
              "on a model-only strike grid.")
        return
    print(f"\nListed expiries for {symbol} (date: days away):")
    for i in range(0, len(expiries), 4):
        print("  " + "   ".join(f"{e}: {d:>3}d" for e, d in expiries[i:i + 4]))


def ask_days(symbol: str):
    """Show the listed expiries, then let the user choose. Returns (days_list, raw_text)."""
    print_expiries(symbol, list_expiries(symbol))
    while True:
        raw = input("Choose time to expiry: days (30), 2w / 3m / 1y, a date (2026-12-18), "
                    "a list (7,30,90), 'pick' or 'ladder' [pick]: ")
        try:
            return parse_days(raw, default="pick"), raw
        except ValueError as error:
            print(" ", error)


def monitoring_steps(T: float) -> int:
    """Use one step per calendar day where practical, capped at 252 steps per expiry."""
    T = _finite_value("Time to maturity", T)
    if T <= 0:
        raise ValueError("Time to maturity must be greater than zero.")
    return min(252, max(1, math.ceil(T * 252)))


def price_expiry(ctx, chain, days, T, curve, opt) -> dict:
    S0, q, hv = ctx["spot"], ctx["q"], ctx["hist_vol"]
    _validate_market_inputs(S0, 0.0, q, hv, T)
    if T <= 0:
        raise ValueError("Time to maturity must be greater than zero.")
    r = rate_at(curve, T)
    pts = build_smile(chain, S0, r, q, T) if chain else []
    use_chain = len(pts) >= 3
    steps = monitoring_steps(T)
    if opt.pairs < 4:
        raise ValueError("Number of antithetic pairs must be at least four.")
    if opt.nstrikes < 1:
        raise ValueError("Number of displayed strikes must be at least one.")
    if opt.sigma is not None and (
        not math.isfinite(float(opt.sigma)) or opt.sigma <= 0
    ):
        raise ValueError("Volatility override must be finite and greater than zero.")

    if opt.sigma is not None:
        vol_source, sigma_of = "user flat vol", (lambda _: opt.sigma)
    elif use_chain:
        vol_source, sigma_of = "market implied-vol smile", (lambda K: smile_iv(pts, K))
    else:
        vol_source, sigma_of = "EWMA historical", (lambda _: hv)

    if opt.strikes is not None:
        strikes = [_finite_value("Strike price", strike) for strike in opt.strikes]
        if not strikes or any(strike <= 0 for strike in strikes):
            raise ValueError("Strike overrides must be a non-empty list of positive values.")
        if len(set(strikes)) != len(strikes):
            raise ValueError("Strike overrides cannot contain duplicates.")
    elif use_chain:
        near = [p for p in pts if 0.8 <= p["strike"] / S0 <= 1.2] or pts
        near = sorted(near, key=lambda p: abs(p["strike"] - S0))[:opt.nstrikes]
        strikes = sorted(p["strike"] for p in near)
    else:
        moneyness = [1.0] if opt.nstrikes == 1 else np.linspace(0.90, 1.10, opt.nstrikes)
        strikes = [float(S0 * m) for m in moneyness]
    market = {p["strike"]: p for p in pts}

    path_cache = BoundedPathCache()

    def get_cached_paths(vol, seed_val):
        key = (round(float(vol), 12), seed_val)
        return path_cache.get_or_create(
            key,
            lambda: simulate_paths(S0, r, q, vol, T, steps, opt.pairs, seed_val),
            cache=seed_val is not None,
        )

    rows = []
    for K in strikes:
        kind = "put" if K < S0 else "call"
        sigma = _finite_value("Volatility", sigma_of(K))
        if sigma <= 0:
            raise ValueError("Pricing volatility must be greater than zero.")
        paths = get_cached_paths(sigma, opt.seed)                                 # common random numbers
        mc, se, v_eu, mc_delta = european_mc(paths, S0, K, r, q, T, kind)
        bs = bs_price(S0, K, r, q, sigma, T, kind)
        training_paths = paths
        if not (kind == "call" and q <= 0 and r >= 0):
            training_seed = None if opt.seed is None else opt.seed + 1
            training_paths = get_cached_paths(sigma, training_seed)
        amer_values = lsm_values(
            training_paths, K, r, T, kind, evaluation_paths=paths, q=q
        )
        am, am_se, eep = american_cv(amer_values, v_eu, bs)
        mc_greek_values = mc_greeks(paths, S0, K, r, q, sigma, T, kind)
        analytic_greeks = bs_greeks(S0, K, r, q, sigma, T, kind)
        mk = market.get(float(K))
        row = {"symbol": ctx["symbol"], "expiry": chain["expiry"] if chain else "n/a", "days": days,
               "T_years": round(T, 5), "spot": round(S0, 4), "type": kind, "strike": K,
               "mkt_price": round(mk["price"], 4) if mk else None,
               "mkt_source": mk["source"] if mk else None, "iv": round(sigma, 4),
               "mc_euro": round(mc, 4), "mc_euro_ci95": round(1.96 * se, 4), "bs_euro": round(bs, 4),
               "mc_american": round(am, 4), "mc_american_ci95": round(1.96 * am_se, 4),
               "early_ex_premium": round(eep, 4), "mc_delta": round(mc_delta, 5),
               "mc_gamma": round(mc_greek_values["gamma"], 5),
               "mc_vega": round(mc_greek_values["vega"], 5),
               "mc_theta": round(mc_greek_values["theta"], 5),
               **{f"bs_{k}": round(v, 5) for k, v in analytic_greeks.items()},
               "r": round(r, 5), "q": round(q, 5), "vol_source": vol_source}
        rows.append(row)

    sigma_atm = sigma_of(S0)
    exotics = price_exotics(S0, r, q, T, get_cached_paths(sigma_atm, opt.seed))
    meta = {"expiry": chain["expiry"] if chain else "n/a (no chain)", "days": days, "T": T, "r": r,
            "steps": steps, "vol_source": vol_source, "sigma_atm": sigma_atm,
            "n_quotes": len(pts), "model_only": not use_chain, "paths": 2 * opt.pairs}
    return {"meta": meta, "rows": rows, "exotics": exotics}


def analyse_symbol(symbol, days_list, opt, announce=False):
    hist = get_history(symbol)
    S0, cur = hist["spot"], hist["currency"]
    q = (
        _finite_value("Dividend-yield override", opt.q)
        if opt.q is not None else get_dividend_yield(symbol, S0)
    )
    q_source = (
        "user override" if opt.q is not None
        else "Yahoo trailing-12-month dividends / spot (continuous-yield proxy)"
    )
    risk_free = _finite_value("Risk-free-rate override", opt.r) if opt.r is not None else None
    curve = ({"points": [(1.0, risk_free)], "source": "user override"}
             if risk_free is not None
             else get_rate_curve(cur))
    ctx = {"symbol": hist["symbol"], "currency": cur, "spot": S0, "q": q,
           "spot_source": hist["spot_source"], "close_source": hist["close_source"],
           "q_source": q_source, "hist_vol": ewma_vol(hist["close"]),
           "curve_source": curve["source"]}

    results, seen = [], set()
    for target in days_list:
        chain = get_option_chain(symbol, target, cur)
        if announce and chain and chain["days"] != target:
            print(f"[{symbol}] you asked for {target}d; the nearest listed expiry is "
                  f"{chain['expiry']} ({chain['days']}d), so that is what is priced.")
        elif announce and not chain:
            print(f"[{symbol}] no option chain; pricing exactly {target}d on a model-only strike grid.")
        key = chain["expiry"] if chain else ("grid", target)
        if key in seen:
            continue
        seen.add(key)
        days = chain["days"] if chain else target
        T = chain["T"] if chain else target / 365
        try:
            results.append(price_expiry(ctx, chain, days, T, curve, opt))
        except Exception as error:
            print(f"[{symbol}] {days}d skipped: {error}")
    if not results:
        raise ValueError(f"No expiries could be priced for '{symbol}'.")
    return ctx, results


# ========== 8. Output ==========
def fmt(x, nd=2):
    return "-" if x is None else f"{x:,.{nd}f}"


def print_symbol(ctx, results):
    print("\n" + "=" * 146)
    print(f"{ctx['symbol']} | spot {ctx['spot']:,.2f} {ctx['currency']} "
          f"({ctx['spot_source']}) | dividend yield proxy {ctx['q']:.2%} "
          f"({ctx['q_source']}) | EWMA hist vol {ctx['hist_vol']:.1%} "
          f"({ctx['close_source']})")
    print(f"Risk-free curve: {ctx['curve_source']}")
    for res in results:
        m = res["meta"]
        print("\n" + "-" * 146)
        print(f"Expiry {m['expiry']} ({m['days']}d, T={m['T']:.4f}y) | r={m['r']:.2%} | "
              f"vol: {m['vol_source']}" + ("" if m["model_only"] else f" ({m['n_quotes']} quotes)")
              + f" | {m['paths']:,} paths x {m['steps']} steps")
        if m["model_only"]:
            print("No usable option chain: model-only prices on a strike grid, no market comparison.")
        header = (f"{'Type':<5}{'Strike':>9}{'Mkt':>9}{'IV':>8}{'MC Euro':>10}{'±95%':>8}"
                  f"{'BS Euro':>9}{'MC Amer':>10}{'EEP':>8}{'MC Δ':>8}{'BS Δ':>8}"
                  f"{'MC Γ':>8}{'BS Γ':>8}{'MC Vega':>9}{'BS Vega':>9}"
                  f"{'MC θ/day':>10}{'BS θ/day':>10}")
        print(header)
        for r in res["rows"]:
            print(f"{r['type']:<5}{r['strike']:>9,.2f}{fmt(r['mkt_price']):>9}{r['iv']:>8.1%}"
                  f"{r['mc_euro']:>10,.3f}{r['mc_euro_ci95']:>8.3f}{r['bs_euro']:>9,.3f}"
                  f"{r['mc_american']:>10,.3f}{r['early_ex_premium']:>8.3f}"
                  f"{r['mc_delta']:>8.3f}{r['bs_delta']:>8.3f}"
                  f"{r['mc_gamma']:>8.4f}{r['bs_gamma']:>8.4f}"
                  f"{r['mc_vega']:>9.3f}{r['bs_vega']:>9.3f}"
                  f"{r['mc_theta']:>10.4f}{r['bs_theta']:>10.4f}")
        print(f"Exotics at ATM vol {m['sigma_atm']:.1%}: " + "; ".join(
            f"{name} {p:,.3f} ±{1.96 * se:.3f}" for name, (p, se) in res["exotics"].items()))
    print("\nNotes: IV uses a quoted bid/ask mid when available; Monte Carlo and Black-Scholes "
          "prices should agree within simulation error, not by construction. Greeks show MC / "
          "Black-Scholes values; vega is per vol point and theta per calendar day. USD FRED "
          "constant-maturity yields are quoted-yield proxies, not zero-coupon Treasury rates; the "
          "INR input is a 3-month interbank proxy, not an Indian government-bond curve. Asian "
          "averages include each simulated time step but not S0; the barrier is monitored at each "
          "step (approximately each trading day for maturities up to one year), with no rebate. "
          "American-style market contracts use European Black-Scholes IV inference, which can "
          "bias the fitted volatility. Yahoo data may be delayed.")


def save_csv(rows, path):
    if not rows:
        return
    fields = ["run_at"] + list(rows[0].keys())
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="", encoding="utf-8") as f:
            existing_fields = next(csv.reader(f), [])
        if existing_fields != fields:
            root, ext = os.path.splitext(path)
            version = 2
            while True:
                candidate = f"{root}_v{version}{ext}"
                if not os.path.exists(candidate):
                    path = candidate
                    break
                with open(candidate, newline="", encoding="utf-8") as f:
                    candidate_fields = next(csv.reader(f), [])
                if candidate_fields == fields:
                    path = candidate
                    break
                version += 1
            print(f"Existing CSV has different columns; writing to {path} instead.")
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    stamp = datetime.now().isoformat(timespec="seconds")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({"run_at": stamp, **row})
    print(f"Saved {len(rows)} rows to {path}")


# ========== 9. Charts ==========
def _prepare_matplotlib(show: bool):
    """
    Choose a safe backend. With no display (cron, servers, SSH) or show=False we use the file-only
    'Agg' backend, so charts are saved but no window opens and nothing blocks.
    Returns (pyplot, can_show_windows).
    """
    import matplotlib
    headless = (sys.platform.startswith("linux")
                and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"))
    if (headless or not show) and not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    file_only = {"agg", "pdf", "svg", "ps", "cairo", "template", "pgf"}
    return plt, show and plt.get_backend().lower() not in file_only


def _safe_name(value) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def plot_results(ctx, results, pairs=2500, seed=42, output_dir="plots", show=True):
    """
    Draw and save the charts for one symbol. Returns the list of PNG files written.

    Figure 1  <SYMBOL>_simulation.png: volatility used by strike | sample risk-neutral paths |
              terminal-price histogram with the theoretical lognormal density overlaid.
    Figure 2+ <SYMBOL>_<expiry>_results.png (one per expiry): prices by strike (MC European with CI,
              Black-Scholes, MC American, market quotes), delta, gamma, vega, theta (MC vs
              Black-Scholes), and Asian / barrier prices.
    """
    if not results:
        raise ValueError("At least one expiry result is required to plot.")
    if not isinstance(pairs, (int, np.integer)) or pairs < 4:
        raise ValueError("Plot simulation requires at least four antithetic pairs.")
    if not output_dir:
        raise ValueError("Plot output directory cannot be empty.")
    os.makedirs(output_dir, exist_ok=True)
    plt, can_show = _prepare_matplotlib(show)
    saved, sym = [], _safe_name(ctx["symbol"])

    try:
        first = results[0]["meta"]
        sigma, r, T, S0, q = first["sigma_atm"], first["r"], first["T"], ctx["spot"], ctx["q"]
        paths = simulate_paths(S0, r, q, sigma, T, first["steps"], min(int(pairs), 2500), seed)
        days_axis = np.linspace(0, T * 365, first["steps"] + 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for res in results:
            rows = res["rows"]
            axes[0].plot([x["strike"] for x in rows], [x["iv"] * 100 for x in rows],
                         marker="o", label=f"{res['meta']['days']}d")
        axes[0].axvline(S0, color="grey", linestyle="--", label="Spot")
        axes[0].set(title=f"{ctx['symbol']}: volatility used by strike", xlabel="Strike", ylabel="Volatility (%)")
        axes[0].legend(); axes[0].grid(True)

        shown = min(paths.shape[1], 30)
        for i in range(shown):
            axes[1].plot(days_axis, paths[:, i], linewidth=0.8, alpha=0.6)
        axes[1].axhline(S0, color="black", linestyle="--", linewidth=1, label="Initial spot")
        axes[1].set(title=f"Sample simulated paths ({shown} of {paths.shape[1]:,}), {first['days']}d",
                    xlabel="Days", ylabel=f"Price ({ctx['currency']})")
        axes[1].legend(); axes[1].grid(True)

        terminal = paths[-1]
        axes[2].hist(terminal, bins="auto", density=True, alpha=0.7, edgecolor="white",
                     label="Simulated")
        grid = np.linspace(max(float(terminal.min()), 1e-9), float(terminal.max()), 400)
        mu, s = math.log(S0) + (r - q - 0.5 * sigma**2) * T, sigma * math.sqrt(T)
        axes[2].plot(grid, np.exp(-(np.log(grid) - mu) ** 2 / (2 * s * s)) / (grid * s * math.sqrt(2 * math.pi)),
                     color="tab:red", linewidth=2, label="Lognormal theory")
        axes[2].axvline(S0, color="black", linestyle="--", label="Initial spot")
        forward = S0 * math.exp((r - q) * T)
        axes[2].axvline(forward, color="tab:green", linestyle="-.", label=f"Risk-neutral mean {forward:,.2f}")
        axes[2].set(title=f"Terminal price distribution ({first['days']}d)",
                    xlabel=f"Price at expiry ({ctx['currency']})", ylabel="Density")
        axes[2].legend(); axes[2].grid(True)
        fig.tight_layout()
        path = os.path.join(output_dir, f"{sym}_simulation.png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        saved.append(path)

        for result in results:
            rows, meta = result["rows"], result["meta"]
            if not rows:
                continue
            strikes = [x["strike"] for x in rows]
            fig, ax = plt.subplots(2, 3, figsize=(16, 9))
            ax = ax.ravel()
            ax[0].errorbar(strikes, [x["mc_euro"] for x in rows], yerr=[x["mc_euro_ci95"] for x in rows],
                           marker="o", capsize=3, label="Monte Carlo European (95% CI)")
            ax[0].plot(strikes, [x["bs_euro"] for x in rows], marker="s", label="Black-Scholes European")
            ax[0].plot(strikes, [x["mc_american"] for x in rows], marker="^", label="Monte Carlo American (LSM)")
            quoted = [x for x in rows if x["mkt_price"] is not None]
            if quoted:
                ax[0].scatter([x["strike"] for x in quoted], [x["mkt_price"] for x in quoted],
                              marker="x", color="black", zorder=5, label="Market quote")
            ax[0].axvline(ctx["spot"], color="grey", linestyle="--")
            ax[0].set(title="Option prices by strike (puts left of spot, calls right)", xlabel="Strike",
                      ylabel=f"Price ({ctx['currency']})")
            ax[0].legend(fontsize=8, loc="center right"); ax[0].grid(True)

            for axis, (mc_key, bs_key, title) in zip(ax[1:5], (
                    ("mc_delta", "bs_delta", "Delta"), ("mc_gamma", "bs_gamma", "Gamma"),
                    ("mc_vega", "bs_vega", "Vega (per vol point)"),
                    ("mc_theta", "bs_theta", "Theta (per calendar day)"))):
                axis.plot(strikes, [x[mc_key] for x in rows], marker="o", label="Monte Carlo")
                axis.plot(strikes, [x[bs_key] for x in rows], marker="s", linestyle="--", label="Black-Scholes")
                axis.axvline(ctx["spot"], color="grey", linestyle=":", label="Spot (put | call)")
                axis.set(title=title, xlabel="Strike"); axis.legend(fontsize=8); axis.grid(True)

            names = list(result["exotics"])
            ax[5].bar(range(len(names)), [result["exotics"][n][0] for n in names],
                      yerr=[1.96 * result["exotics"][n][1] for n in names], capsize=3, alpha=0.75)
            ax[5].set(title="Asian and barrier prices (95% CI)", ylabel=f"Price ({ctx['currency']})",
                      xticks=range(len(names)), xticklabels=names)
            ax[5].tick_params(axis="x", labelrotation=20, labelsize=8)
            ax[5].grid(True, axis="y")
            fig.suptitle(f"{ctx['symbol']} pricing results | expiry {meta['expiry']} ({meta['days']}d) | "
                         f"r={meta['r']:.2%} | {meta['paths']:,} paths")
            fig.tight_layout()
            tag = f"{meta['days']}d" if str(meta["expiry"]).startswith("n/a") else meta["expiry"]
            path = os.path.join(output_dir, f"{sym}_{_safe_name(tag)}_results.png")
            fig.savefig(path, dpi=160, bbox_inches="tight")
            saved.append(path)

        for p in saved:
            print(f"Saved chart: {p}")
        if can_show:
            plt.show()                       # blocks until the windows are closed
    finally:
        plt.close("all")                     # never leak figures between symbols
    return saved


# ========== 10. Command line ==========
def run(symbols, days_list, opt, csv_path=None, plot=True, plot_dir="plots", show=True,
        announce=False):
    all_rows = []
    for sym in symbols:
        try:
            ctx, results = analyse_symbol(sym, days_list, opt, announce)
        except Exception as error:
            print(f"\n[{sym}] skipped: {error}")
            continue
        print_symbol(ctx, results)
        for res in results:
            all_rows.extend(res["rows"])
        if plot:
            try:                              # a chart problem must never lose the numbers
                plot_results(ctx, results, pairs=opt.pairs, seed=opt.seed,
                             output_dir=plot_dir, show=show)
            except Exception as error:
                print(f"[{sym}] charts skipped: {error}")
    if csv_path:
        save_csv(all_rows, csv_path)
    return all_rows


def validate_cli_args(args) -> None:
    if args.pairs < 4:
        raise ValueError("--pairs must be at least 4.")
    if args.nstrikes < 1:
        raise ValueError("--nstrikes must be at least 1.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    for name, value in (("--r", args.r), ("--q", args.q)):
        if value is not None:
            _finite_value(name, value)
    if args.sigma is not None and (
        not math.isfinite(float(args.sigma)) or args.sigma <= 0
    ):
        raise ValueError("--sigma must be finite and greater than zero.")
    if args.strikes is not None:
        if not args.strikes or any(not math.isfinite(float(k)) or float(k) <= 0 for k in args.strikes):
            raise ValueError("--strikes must contain only finite, positive values.")
        if len(set(args.strikes)) != len(args.strikes):
            raise ValueError("--strikes cannot contain duplicates.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Automated Monte Carlo option pricer (real-world data).")
    ap.add_argument("symbols", nargs="*", help="e.g. AAPL TSLA RELIANCE.NS")
    ap.add_argument("--symbols", dest="symbols_csv", help="comma-separated alternative to positional")
    ap.add_argument("--days", default=None,
                    help="YOUR time to expiry: days (30), 2w/3m/1y, a date (2026-12-18), a list "
                         "(7,30,90), 'pick' (~1 month) or 'ladder' (1w/1m/3m/6m). If omitted: the "
                         "prompt mode asks you; command-line runs use 'ladder'")
    ap.add_argument("--list-expiries", action="store_true",
                    help="print the listed expiry dates for the symbol(s) and exit")
    ap.add_argument("--pairs", type=int, default=10_000,
                    help="antithetic path pairs, minimum 4 (default 10000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nstrikes", type=int, default=9, help="strikes per expiry (default 9)")
    ap.add_argument("--strikes", help="override strikes, e.g. 200,210,220")
    ap.add_argument("--r", type=float, default=None, help="override risk-free rate (continuous, decimal)")
    ap.add_argument("--q", type=float, default=None, help="override dividend yield (decimal)")
    ap.add_argument("--sigma", type=float, default=None, help="override with one flat volatility")
    ap.add_argument("--csv", default=None, help="append results to this CSV")
    ap.add_argument("--no-plot", action="store_true", help="skip the charts (numbers only)")
    ap.add_argument("--no-show", action="store_true",
                    help="save the PNG charts but do not open windows (also automatic with no display)")
    ap.add_argument("--plot", action="store_true", help=argparse.SUPPRESS)   # kept: charts are now the default
    ap.add_argument("--plot-dir", default="plots", help="folder for the PNG charts (default: plots)")
    args = ap.parse_args(argv)

    try:
        days_list = parse_days(args.days) if args.days else None
        if args.strikes:
            strike_tokens = [x.strip() for x in args.strikes.split(",")]
            if any(not x for x in strike_tokens):
                raise ValueError("Strike list cannot contain empty entries.")
            args.strikes = [float(x) for x in strike_tokens]
        else:
            args.strikes = None
        validate_cli_args(args)
    except ValueError as error:
        ap.error(str(error))

    plot, show = not args.no_plot, not args.no_show
    symbols = [s.upper() for s in args.symbols]
    if args.symbols_csv:
        symbols += [s.strip().upper() for s in args.symbols_csv.split(",") if s.strip()]

    if args.list_expiries:
        for sym in symbols:
            print_expiries(sym, list_expiries(sym))
        return

    if symbols:
        explicit = is_explicit_days(args.days) if args.days else False
        if not run(symbols, days_list or parse_days("ladder"), args, args.csv, plot,
                   args.plot_dir, show, explicit):
            raise SystemExit(1)
        return

    print("Automated Monte Carlo Option Pricer. Examples: AAPL, TSLA, RELIANCE.NS")
    while True:
        raw = input("\nEnter symbol(s), comma-separated (or 'q' to quit): ").strip().upper()
        if raw == "Q":
            print("Goodbye!")
            return
        if raw:
            chosen = [s.strip() for s in raw.split(",") if s.strip()]
            if days_list:                                   # --days was given on the command line
                dl, explicit = days_list, is_explicit_days(args.days)
            else:                                           # otherwise YOU pick the time here
                dl, text = ask_days(chosen[0])
                explicit = is_explicit_days(text)
            run(chosen, dl, args, args.csv, plot, args.plot_dir, show, explicit)


if __name__ == "__main__":
    main()
