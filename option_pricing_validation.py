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
  Charts          saved as PNG in ./charts and shown on screen when a display is available:
                    <SYMBOL>_simulation.png        volatility smile, sample paths, terminal-price
                                                   histogram with the theoretical lognormal curve
                    <SYMBOL>_<expiry>_results.png  prices vs strike, MC vs Black-Scholes Greeks,
                                                   Asian/barrier prices (one file per expiry)

Usage:
  python option_pricing_validation.py AAPL                       # ladder of expiries, prices + charts
  python option_pricing_validation.py AAPL --days 45             # YOU choose the time: 45 days
  python option_pricing_validation.py AAPL --days 2026-12-18     # ...or an exact expiry date
  python option_pricing_validation.py AAPL --days 2w,3m,1y       # ...or shorthand / a list
  python option_pricing_validation.py AAPL --list-expiries       # see which expiry dates exist
  python option_pricing_validation.py AAPL TSLA --days pick --csv results.csv
  python option_pricing_validation.py AAPL --days 45 --strikes 200,210,220 --pairs 20000
  python option_pricing_validation.py AAPL --no-show             # save the PNG charts, open no windows
  python option_pricing_validation.py AAPL --no-plot             # numbers only
  python option_pricing_validation.py                            # prompts: symbol, then YOUR expiry choice
Overrides (all optional): --r --q --sigma --strikes --days --pairs --seed --plot-dir

Scheduling (weekdays 16:30):  30 16 * * 1-5  cd /path && python3 option_pricing_validation.py AAPL TSLA --csv results.csv --no-show
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
except ImportError:
    ZoneInfo = None


# ========== 1. Configuration ==========
BASE_URLS = (
    "https://query1.finance.yahoo.com/v8/finance/chart/",
    "https://query2.finance.yahoo.com/v8/finance/chart/",
    "https://query1.finance.yahoo.co.in/v8/finance/chart/",
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
    "USD": {
        "m": 2,
        "points": [
            (1 / 12, "DGS1MO"),
            (3 / 12, "DGS3MO"),
            (6 / 12, "DGS6MO"),
            (1.0, "DGS1"),
            (2.0, "DGS2"),
        ],
    },
    "INR": {"m": 1, "points": [(0.25, "IR3TIB01INM156N")]},
}
# Exchange close used to measure time to expiry: currency -> (timezone, hour, minute)
MARKET_CLOSE = {"USD": ("America/New_York", 16, 0), "INR": ("Asia/Kolkata", 15, 30)}

AUTO_DAYS = (7, 30, 90, 180)   # 'ladder': 1 week, 1 month, 3 months, 6 months
PICK_DAYS = 30                 # 'pick': the listed expiry closest to 1 month

_HTTP_SESSION = requests.Session()
_HTTP_SESSION.headers.update(HEADERS)


def _get_market_tz(currency: str):
    """Return the exchange ZoneInfo for currency, or None if unavailable."""
    spec = MARKET_CLOSE.get(currency)
    if spec and ZoneInfo is not None:
        try:
            return ZoneInfo(spec[0])
        except (KeyError, OSError):
            return None
    return None


def _market_today(currency: str = None) -> date:
    """Current date in the exchange timezone when available, else local date."""
    tz = _get_market_tz(currency) if currency else None
    return datetime.now(tz).date() if tz is not None else date.today()


# ========== 2. Market data ==========
def get_history(symbol: str, period: str = "1y") -> dict:
    """Fetch a quote and adjusted daily closes; disclose when a fallback is required."""
    params = {"range": period, "interval": "1d"}
    response = None
    status_codes = []
    for base_url in BASE_URLS:
        try:
            candidate = _HTTP_SESSION.get(base_url + symbol, params=params, timeout=10)
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
            float(value)
            for value in result["indicators"]["quote"][0]["close"]
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
                float(value)
                for value in adjusted
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
                RuntimeWarning,
                stacklevel=2,
            )
        return {
            "symbol": meta.get("symbol", symbol),
            "currency": currency,
            "close": np.array(history_closes, dtype=float),
            "spot": spot,
            "spot_source": spot_source,
            "close_source": close_source,
        }
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Yahoo returned no usable price data for '{symbol}'.") from error


def get_dividend_yield(symbol: str, spot: float, ticker=None) -> float:
    """Trailing-12-month dividends / spot, a proxy for a continuous dividend yield."""
    spot = _finite_value("Spot price", spot)
    if spot <= 0:
        raise ValueError("Spot price must be finite and greater than zero.")
    try:
        tk = ticker if ticker is not None else yf.Ticker(symbol)
        divs = tk.dividends
    except Exception as error:
        raise ValueError(f"Could not retrieve Yahoo dividend history for '{symbol}'.") from error
    if divs is None or len(divs) == 0:
        warnings.warn(
            f"{symbol}: Yahoo returned no dividend history; assuming zero dividend yield.",
            RuntimeWarning,
            stacklevel=2,
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
    one_minus_lam = 1.0 - lam
    for x in ret[20:]:
        var = lam * var + one_minus_lam * (x * x)
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
    resp = _HTTP_SESSION.get(FRED_URL, params={"id": series_id}, timeout=10)
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
    today = _market_today(currency)
    for tenor, series in spec["points"]:
        try:
            asof, pct = fetch_fred(series)
            observation_date = date.fromisoformat(asof)
            age_days = (today - observation_date).days
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
                RuntimeWarning,
                stacklevel=2,
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
            RuntimeWarning,
            stacklevel=2,
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
    tz = _get_market_tz(currency)
    if spec and tz is not None:
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
        RuntimeWarning,
        stacklevel=2,
    )
    return days / 365


def get_option_chain(symbol: str, target_days: int, currency: str, ticker=None, available_expiries=None):
    """Listed expiry (at least 1 day out) closest to target_days, or None."""
    if not isinstance(target_days, (int, np.integer)) or target_days < 1:
        raise ValueError("Target expiry must be a positive number of days.")
    try:
        tk = ticker if ticker is not None else yf.Ticker(symbol)
        today = _market_today(currency)
        expiries = available_expiries if available_expiries is not None else (tk.options or ())
        candidates = []
        for exp in expiries:
            d = (date.fromisoformat(exp) - today).days
            if d >= 1:
                candidates.append((abs(d - target_days), exp, d))
        if not candidates:
            return None
        _, exp, d = min(candidates)
        oc = tk.option_chain(exp)
        return {
            "expiry": exp,
            "days": d,
            "T": years_to_expiry(exp, currency),
            "calls": oc.calls.set_index("strike"),
            "puts": oc.puts.set_index("strike"),
        }
    except Exception as error:
        warnings.warn(
            f"{symbol}: Yahoo option-chain retrieval failed ({error}); using model-only pricing.",
            RuntimeWarning,
            stacklevel=2,
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
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
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
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    fwd = S * dq - K * dr
    lower = max(fwd if kind == "call" else -fwd, 0.0)
    upper = S * dq if kind == "call" else K * dr
    if not (lower + 1e-8 < price < upper - 1e-8):
        return None
    lo, hi = 1e-4, 5.0
    if bs_price(S, K, r, q, hi, T, kind) < price:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, r, q, mid, T, kind) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-7:
            break
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
    log_diff = np.zeros(n_pairs, dtype=float)
    for step in range(steps):
        z = rng.standard_normal(n_pairs)
        log_diff += diffusion * z
        base = S0 * math.exp(drift * (step + 1))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            exp_diff = np.exp(log_diff)
            paths[step + 1, :n_pairs] = base * exp_diff
            paths[step + 1, n_pairs:] = base / exp_diff
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
        dc = c_fit - c_fit.mean()
        denom = float(np.dot(dc, dc))
        if denom <= 0:
            return 0.0
        dv = v_fit - v_fit.mean()
        return float(np.dot(dv, dc)) / denom

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
    indicator = (ST > K).astype(float) if kind == "call" else -(ST < K).astype(float)
    delta, _ = pair_mean_se(disc * (ST / S0) * indicator)                  # pathwise delta
    return price, se, values, delta


def mc_greeks(paths, S0, K, r, q, sigma, T, kind, delta=None):
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

    ST = paths[-1]
    log_return = np.log(ST / S0)
    z = (log_return - (r - q - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    pay = payoff_fn(K, kind)

    def price_at(spot, vol, maturity):
        terminal = spot * np.exp(
            (r - q - 0.5 * vol**2) * maturity + vol * math.sqrt(maturity) * z
        )
        discount = math.exp(-r * maturity)
        payoff = discount * pay(terminal)
        control = discount * terminal
        return cv_estimate(payoff, control, spot * math.exp(-q * maturity))[0]

    if delta is None:
        _, _, _, delta = european_mc(paths, S0, K, r, q, T, kind)

    # Second central difference for Gamma: the linear control variate S_T has zero second
    # derivative w.r.t. S0, so direct antithetic pair averaging avoids spurious beta-shift noise.
    spot_bump = max(S0 * 0.005, 1e-4)
    disc_T = math.exp(-r * T)
    unit_terminal = ST / S0
    gamma_diff = disc_T * (
        pay((S0 + spot_bump) * unit_terminal)
        - 2.0 * pay(ST)
        + pay((S0 - spot_bump) * unit_terminal)
    ) / (spot_bump**2)
    gamma, _ = pair_mean_se(gamma_diff)

    vol_bump = min(0.01, sigma * 0.49)
    vega = (
        price_at(S0, sigma + vol_bump, T) - price_at(S0, sigma - vol_bump, T)
    ) / (2 * vol_bump) * 0.01

    time_bump = min(1 / 365, T / 2)
    theta = -(
        price_at(S0, sigma, T + time_bump) - price_at(S0, sigma, T - time_bump)
    ) / (2 * time_bump * 365)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta)}


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
            RuntimeWarning,
            stacklevel=2,
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
        moneyness = paths[t, itm] / K
        model = _fit_continuation(moneyness, cash[itm], degree)
        models[t] = model
        continuation = _predict_continuation(moneyness, model)
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
        candidates = np.flatnonzero(alive & (exercise > 0))
        if candidates.size == 0:
            continue
        continuation = _predict_continuation(spots[candidates] / K, model)
        exercise_now = candidates[exercise[candidates] > continuation]
        if exercise_now.size:
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
    prem = max(prem, 0.0)                    # early-exercise premium cannot be negative
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
    alive = low > barrier                    # discretely monitored once per step
    control, cmean = disc * ST, S0 * math.exp(-q * T)
    out = {
        "Asian call (arithmetic, ATM)": np.maximum(avg - S0, 0.0),
        "Asian put (arithmetic, ATM)": np.maximum(S0 - avg, 0.0),
        f"Down-and-out call (ATM, B={barrier:,.2f})": np.maximum(ST - S0, 0.0) * alive,
    }
    return {name: cv_estimate(disc * v, control, cmean) for name, v in out.items()}


# ========== 7. Pipeline ==========
_UNIT_DAYS = {"d": 1, "w": 7, "m": 30, "y": 365}
_DAYS_HELP = (
    "Use a number of days (30), shorthand (2w, 3m, 1y), a date (2026-12-18), "
    "a list (7,30,90), 'pick' or 'ladder'."
)


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
      '7,30,90' or '1w,1m,2026-12-18' -> a list       'pick' -> ~1 month       'ladder' -> 1w,1m,3m,6m
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


def list_expiries(symbol: str, currency: str = None, ticker=None) -> list:
    """Listed option expiries as [(date string, days away)], soonest first."""
    try:
        tk = ticker if ticker is not None else yf.Ticker(symbol)
        today = _market_today(currency)
        out = [(e, (date.fromisoformat(e) - today).days) for e in (tk.options or ())]
        return [(e, d) for e, d in out if d >= 1]
    except Exception:
        return []


def print_expiries(symbol: str, expiries: list) -> None:
    if not expiries:
        print(
            f"\nNo listed option expiries found for {symbol}; a time you type is priced "
            "on a model-only strike grid."
        )
        return
    print(f"\nListed expiries for {symbol} (date: days away):")
    for i in range(0, len(expiries), 4):
        print("  " + "   ".join(f"{e}: {d:>3}d" for e, d in expiries[i : i + 4]))


def ask_days(symbol: str):
    """Show the listed expiries, then let the user choose. Returns (days_list, raw_text)."""
    print_expiries(symbol, list_expiries(symbol))
    while True:
        raw = input(
            "Choose time to expiry: days (30), 2w / 3m / 1y, a date (2026-12-18), "
            "a list (7,30,90), 'pick' or 'ladder' [pick]: "
        )
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
        near = sorted(near, key=lambda p: abs(p["strike"] - S0))[: opt.nstrikes]
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
        )

    rows = []
    for K in strikes:
        kind = "put" if K < S0 else "call"
        sigma = _finite_value("Volatility", sigma_of(K))
        if sigma <= 0:
            raise ValueError("Pricing volatility must be greater than zero.")
        paths = get_cached_paths(sigma, opt.seed)                           # common random numbers
        mc, se, v_eu, mc_delta = european_mc(paths, S0, K, r, q, T, kind)
        bs = bs_price(S0, K, r, q, sigma, T, kind)
        training_paths = paths
        if not (kind == "call" and q <= 0 and r >= 0):
            training_seed = None if opt.seed is None else opt.seed + 1
            training_paths = (
                simulate_paths(S0, r, q, sigma, T, steps, opt.pairs, None)
                if training_seed is None
                else get_cached_paths(sigma, training_seed)
            )
        amer_values = lsm_values(
            training_paths, K, r, T, kind, evaluation_paths=paths, q=q
        )
        am, am_se, eep = american_cv(amer_values, v_eu, bs)
        mc_greek_values = mc_greeks(paths, S0, K, r, q, sigma, T, kind, delta=mc_delta)
        analytic_greeks = bs_greeks(S0, K, r, q, sigma, T, kind)
        mk = market.get(float(K))
        row = {
            "symbol": ctx["symbol"],
            "expiry": chain["expiry"] if chain else "n/a",
            "days": days,
            "T_years": round(T, 5),
            "spot": round(S0, 4),
            "type": kind,
            "strike": K,
            "mkt_price": round(mk["price"], 4) if mk else None,
            "mkt_source": mk["source"] if mk else None,
            "iv": round(sigma, 4),
            "mc_euro": round(mc, 4),
            "mc_euro_ci95": round(1.96 * se, 4),
            "bs_euro": round(bs, 4),
            "mc_american": round(am, 4),
            "mc_american_ci95": round(1.96 * am_se, 4),
            "early_ex_premium": round(eep, 4),
            "mc_delta": round(mc_delta, 5),
            "mc_gamma": round(mc_greek_values["gamma"], 5),
            "mc_vega": round(mc_greek_values["vega"], 5),
            "mc_theta": round(mc_greek_values["theta"], 5),
            **{f"bs_{k}": round(v, 5) for k, v in analytic_greeks.items()},
            "r": round(r, 5),
            "q": round(q, 5),
            "vol_source": vol_source,
        }
        rows.append(row)

    sigma_atm = sigma_of(S0)
    exotic_paths = get_cached_paths(sigma_atm, opt.seed)
    exotics = price_exotics(S0, r, q, T, exotic_paths)
    meta = {
        "expiry": chain["expiry"] if chain else "n/a (no chain)",
        "days": days,
        "T": T,
        "r": r,
        "steps": steps,
        "vol_source": vol_source,
        "sigma_atm": sigma_atm,
        "n_quotes": len(pts),
        "model_only": not use_chain,
        "paths": 2 * opt.pairs,
    }
    return {"meta": meta, "rows": rows, "exotics": exotics}


def analyse_symbol(symbol, days_list, opt, announce=False):
    hist = get_history(symbol)
    S0, cur = hist["spot"], hist["currency"]
    ticker = yf.Ticker(symbol)
    q = (
        _finite_value("Dividend-yield override", opt.q)
        if opt.q is not None
        else get_dividend_yield(symbol, S0, ticker=ticker)
    )
    q_source = (
        "user override"
        if opt.q is not None
        else "Yahoo trailing-12-month dividends / spot (continuous-yield proxy)"
    )
    risk_free = _finite_value("Risk-free-rate override", opt.r) if opt.r is not None else None
    curve = (
        {"points": [(1.0, risk_free)], "source": "user override"}
        if risk_free is not None
        else get_rate_curve(cur)
    )
    ctx = {
        "symbol": hist["symbol"],
        "currency": cur,
        "spot": S0,
        "q": q,
        "spot_source": hist["spot_source"],
        "close_source": hist["close_source"],
        "q_source": q_source,
        "hist_vol": ewma_vol(hist["close"]),
        "curve_source": curve["source"],
    }

    try:
        available_expiries = ticker.options or ()
    except Exception:
        available_expiries = None

    results, seen = [], set()
    for target in days_list:
        chain = get_option_chain(
            symbol, target, cur, ticker=ticker, available_expiries=available_expiries
        )
        if announce and chain and chain["days"] != target:
            print(
                f"[{symbol}] you asked for {target}d; the nearest listed expiry is "
                f"{chain['expiry']} ({chain['days']}d), so that is what is priced."
            )
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
    print(
        f"{ctx['symbol']} | spot {ctx['spot']:,.2f} {ctx['currency']} "
        f"({ctx['spot_source']}) | dividend yield proxy {ctx['q']:.2%} "
        f"({ctx['q_source']}) | EWMA hist vol {ctx['hist_vol']:.1%} "
        f"({ctx['close_source']})"
    )
    print(f"Risk-free curve: {ctx['curve_source']}")
    for res in results:
        m = res["meta"]
        print("\n" + "-" * 146)
        print(
            f"Expiry {m['expiry']} ({m['days']}d, T={m['T']:.4f}y) | r={m['r']:.2%} | "
            f"vol: {m['vol_source']}"
            + ("" if m["model_only"] else f" ({m['n_quotes']} quotes)")
            + f" | {m['paths']:,} paths x {m['steps']} steps"
        )
        if m["model_only"]:
            print("No usable option chain: model-only prices on a strike grid, no market comparison.")
        header = (
            f"{'Type':<5}{'Strike':>9}{'Mkt':>9}{'IV':>8}{'MC Euro':>10}{'±95%':>8}"
            f"{'BS Euro':>9}{'MC Amer':>10}{'EEP':>8}{'MC Δ':>8}{'BS Δ':>8}"
            f"{'MC Γ':>8}{'BS Γ':>8}{'MC Vega':>9}{'BS Vega':>9}"
            f"{'MC θ/day':>10}{'BS θ/day':>10}"
        )
        print(header)
        for r in res["rows"]:
            print(
                f"{r['type']:<5}{r['strike']:>9,.2f}{fmt(r['mkt_price']):>9}{r['iv']:>8.1%}"
                f"{r['mc_euro']:>10,.3f}{r['mc_euro_ci95']:>8.3f}{r['bs_euro']:>9,.3f}"
                f"{r['mc_american']:>10,.3f}{r['early_ex_premium']:>8.3f}"
                f"{r['mc_delta']:>8.3f}{r['bs_delta']:>8.3f}"
                f"{r['mc_gamma']:>8.4f}{r['bs_gamma']:>8.4f}"
                f"{r['mc_vega']:>9.3f}{r['bs_vega']:>9.3f}"
                f"{r['mc_theta']:>10.4f}{r['bs_theta']:>10.4f}"
            )
        print(
            f"Exotics at ATM vol {m['sigma_atm']:.1%}: "
            + "; ".join(f"{name} {p:,.3f} ±{1.96 * se:.3f}" for name, (p, se) in res["exotics"].items())
        )
    print(
        "\nNotes: IV uses a quoted bid/ask mid when available; Monte Carlo and Black-Scholes "
        "prices should agree within simulation error, not by construction. Greeks show MC / "
        "Black-Scholes values; vega is per vol point and theta per calendar day. USD FRED "
        "constant-maturity yields are quoted-yield proxies, not zero-coupon Treasury rates; the "
        "INR input is a 3-month interbank proxy, not an Indian government-bond curve. Asian "
        "averages include each simulated time step but not S0; the barrier is monitored at each "
        "step (approximately each trading day for maturities up to one year), with no rebate. "
        "American-style market contracts use European Black-Scholes IV inference, which can "
        "bias the fitted volatility. Yahoo data may be delayed."
    )


def save_csv(rows, path):
    if not rows:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
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

    headless = (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    )
    if (headless or not show) and not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    file_only = {"agg", "pdf", "svg", "ps", "cairo", "template", "pgf"}
    return plt, show and plt.get_backend().lower() not in file_only


def _safe_name(value) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def plot_results(ctx, results, pairs=2500, seed=42, output_dir="charts", show=True):
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
            axes[0].plot(
                [x["strike"] for x in rows],
                [x["iv"] * 100 for x in rows],
                marker="o",
                label=f"{res['meta']['days']}d",
            )
        axes[0].axvline(S0, color="grey", linestyle="--", label="Spot")
        axes[0].set(
            title=f"{ctx['symbol']}: volatility used by strike",
            xlabel="Strike",
            ylabel="Volatility (%)",
        )
        axes[0].legend()
        axes[0].grid(True)

        shown = min(paths.shape[1], 30)
        for i in range(shown):
            axes[1].plot(days_axis, paths[:, i], linewidth=0.8, alpha=0.6)
        axes[1].axhline(S0, color="black", linestyle="--", linewidth=1, label="Initial spot")
        axes[1].set(
            title=f"Sample simulated paths ({shown} of {paths.shape[1]:,}), {first['days']}d",
            xlabel="Days",
            ylabel=f"Price ({ctx['currency']})",
        )
        axes[1].legend()
        axes[1].grid(True)

        terminal = paths[-1]
        axes[2].hist(
            terminal,
            bins="auto",
            density=True,
            alpha=0.7,
            edgecolor="white",
            label="Simulated",
        )
        grid = np.linspace(max(float(terminal.min()), 1e-9), float(terminal.max()), 400)
        mu, s = math.log(S0) + (r - q - 0.5 * sigma**2) * T, sigma * math.sqrt(T)
        axes[2].plot(
            grid,
            np.exp(-((np.log(grid) - mu) ** 2) / (2 * s * s)) / (grid * s * math.sqrt(2 * math.pi)),
            color="tab:red",
            linewidth=2,
            label="Lognormal theory",
        )
        axes[2].axvline(S0, color="black", linestyle="--", label="Initial spot")
        forward = S0 * math.exp((r - q) * T)
        axes[2].axvline(
            forward,
            color="tab:green",
            linestyle="-.",
            label=f"Risk-neutral mean {forward:,.2f}",
        )
        axes[2].set(
            title=f"Terminal price distribution ({first['days']}d)",
            xlabel=f"Price at expiry ({ctx['currency']})",
            ylabel="Density",
        )
        axes[2].legend()
        axes[2].grid(True)
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
            ax[0].errorbar(
                strikes,
                [x["mc_euro"] for x in rows],
                yerr=[x["mc_euro_ci95"] for x in rows],
                marker="o",
                capsize=3,
                label="Monte Carlo European (95% CI)",
            )
            ax[0].plot(
                strikes,
                [x["bs_euro"] for x in rows],
                marker="s",
                label="Black-Scholes European",
            )
            ax[0].plot(
                strikes,
                [x["mc_american"] for x in rows],
                marker="^",
                label="Monte Carlo American (LSM)",
            )
            quoted = [x for x in rows if x["mkt_price"] is not None]
            if quoted:
                ax[0].scatter(
                    [x["strike"] for x in quoted],
                    [x["mkt_price"] for x in quoted],
                    marker="x",
                    color="black",
                    zorder=5,
                    label="Market quote",
                )
            ax[0].axvline(ctx["spot"], color="grey", linestyle="--")
            ax[0].set(
                title="Option prices by strike (puts left of spot, calls right)",
                xlabel="Strike",
                ylabel=f"Price ({ctx['currency']})",
            )
            ax[0].legend(fontsize=8, loc="center right")
            ax[0].grid(True)

            for axis, (mc_key, bs_key, title) in zip(
                ax[1:5],
                (
                    ("mc_delta", "bs_delta", "Delta"),
                    ("mc_gamma", "bs_gamma", "Gamma"),
                    ("mc_vega", "bs_vega", "Vega (per vol point)"),
                    ("mc_theta", "bs_theta", "Theta (per calendar day)"),
                ),
            ):
                axis.plot(strikes, [x[mc_key] for x in rows], marker="o", label="Monte Carlo")
                axis.plot(
                    strikes,
                    [x[bs_key] for x in rows],
                    marker="s",
                    linestyle="--",
                    label="Black-Scholes",
                )
                axis.axvline(ctx["spot"], color="grey", linestyle=":", label="Spot (put | call)")
                axis.set(title=title, xlabel="Strike")
                axis.legend(fontsize=8)
                axis.grid(True)

            names = list(result["exotics"])
            ax[5].bar(
                range(len(names)),
                [result["exotics"][n][0] for n in names],
                yerr=[1.96 * result["exotics"][n][1] for n in names],
                capsize=3,
                alpha=0.75,
            )
            ax[5].set(
                title="Asian and barrier prices (95% CI)",
                ylabel=f"Price ({ctx['currency']})",
                xticks=range(len(names)),
                xticklabels=names,
            )
            ax[5].tick_params(axis="x", labelrotation=20, labelsize=8)
            ax[5].grid(True, axis="y")
            fig.suptitle(
                f"{ctx['symbol']} pricing results | expiry {meta['expiry']} ({meta['days']}d) | "
                f"r={meta['r']:.2%} | {meta['paths']:,} paths"
            )
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


# ========== 10. Validation: Monte Carlo vs Black-Scholes ==========
def _percentage_error(estimate, benchmark):
    error = abs(estimate - benchmark)
    if benchmark != 0:
        return error / abs(benchmark) * 100
    return 0.0 if error == 0 else math.inf


def validate_european_prices(S0=100, K=100, r=0.05, q=0.02, sigma=0.20, T=1.0, pairs=25000, seed=42):
    """
    Compare Monte Carlo European option prices against analytical Black-Scholes prices.
    Uses identical parameters for both methods to validate the Monte Carlo implementation.

    Returns a dictionary with MC prices, BS prices, absolute errors, and percentage errors.
    """
    print("\n" + "=" * 80)
    print("VALIDATION: Monte Carlo vs Black-Scholes European Option Prices")
    print("=" * 80)
    print("\nParameters:")
    print(f"  Stock Price (S0):     {S0:.2f}")
    print(f"  Strike Price (K):     {K:.2f}")
    print(f"  Risk-free Rate (r):   {r:.4f} ({r*100:.2f}%)")
    print(f"  Dividend Yield (q):   {q:.4f} ({q*100:.2f}%)")
    print(f"  Volatility (σ):       {sigma:.4f} ({sigma*100:.2f}%)")
    print(f"  Time to Maturity (T): {T:.4f} years")
    print(f"  Simulation Pairs:     {pairs:,}")
    print(f"  Random Seed:          {seed}")

    bs_call = bs_price(S0, K, r, q, sigma, T, "call")
    bs_put = bs_price(S0, K, r, q, sigma, T, "put")

    # European options depend strictly on terminal price S_T; 1 exact log-step is distributionally identical
    paths = simulate_paths(S0, r, q, sigma, T, 1, pairs, seed)

    mc_call, se_call, _, _ = european_mc(paths, S0, K, r, q, T, "call")
    mc_put, se_put, _, _ = european_mc(paths, S0, K, r, q, T, "put")

    abs_error_call = abs(mc_call - bs_call)
    abs_error_put = abs(mc_put - bs_put)
    pct_error_call = _percentage_error(mc_call, bs_call)
    pct_error_put = _percentage_error(mc_put, bs_put)

    print("\n" + "-" * 80)
    print(f"{'Option Type':<15} {'Monte Carlo':>15} {'Black-Scholes':>15} {'Abs Error':>12} {'% Error':>10}")
    print("-" * 80)
    print(f"{'Call':<15} {mc_call:>15.4f} {bs_call:>15.4f} {abs_error_call:>12.4f} {pct_error_call:>9.3f}%")
    print(f"{'Put':<15} {mc_put:>15.4f} {bs_put:>15.4f} {abs_error_put:>12.4f} {pct_error_put:>9.3f}%")
    print("-" * 80)

    print("\nStandard Errors:")
    print(f"  Call: {se_call:.4f}  |  95% CI: [{mc_call - 1.96*se_call:.4f}, {mc_call + 1.96*se_call:.4f}]")
    print(f"  Put:  {se_put:.4f}  |  95% CI: [{mc_put - 1.96*se_put:.4f}, {mc_put + 1.96*se_put:.4f}]")

    print("\nValidation:")
    if pct_error_call < 2.0 and pct_error_put < 2.0:
        print("  ✓ PASS: Monte Carlo prices are within 2% of Black-Scholes analytical prices")
    else:
        print("  ⚠ WARNING: Monte Carlo prices deviate more than 2% from Black-Scholes")
        print("    Consider increasing the number of simulation pairs")

    return {
        "mc_call": mc_call,
        "bs_call": bs_call,
        "abs_error_call": abs_error_call,
        "pct_error_call": pct_error_call,
        "mc_put": mc_put,
        "bs_put": bs_put,
        "abs_error_put": abs_error_put,
        "pct_error_put": pct_error_put,
        "se_call": se_call,
        "se_put": se_put,
    }


def convergence_analysis(S0=100, K=100, r=0.05, q=0.02, sigma=0.20, T=1.0, seed=42, output_dir="charts"):
    """
    Analyze Monte Carlo convergence by testing different simulation counts.
    Generates plots showing how MC prices converge to Black-Scholes prices as simulations increase.
    """
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS: Monte Carlo Price Stability")
    print("=" * 80)

    sim_counts = [1000, 5000, 10000, 25000, 50000]

    bs_call = bs_price(S0, K, r, q, sigma, T, "call")
    bs_put = bs_price(S0, K, r, q, sigma, T, "put")

    print("\nBlack-Scholes Benchmark Prices:")
    print(f"  Call: {bs_call:.4f}")
    print(f"  Put:  {bs_put:.4f}")

    results = []

    print(
        f"\n{'Sim Pairs':<12} {'MC Call':>12} {'Call SE':>10} {'Call %Err':>12} "
        f"{'MC Put':>12} {'Put SE':>10} {'Put %Err':>12}"
    )
    print("-" * 90)

    for pairs in sim_counts:
        paths = simulate_paths(S0, r, q, sigma, T, 1, pairs, seed)

        mc_call, se_call, _, _ = european_mc(paths, S0, K, r, q, T, "call")
        mc_put, se_put, _, _ = european_mc(paths, S0, K, r, q, T, "put")

        pct_err_call = abs(mc_call - bs_call) / bs_call * 100 if bs_call > 0 else 0.0
        pct_err_put = abs(mc_put - bs_put) / bs_put * 100 if bs_put > 0 else 0.0

        results.append(
            {
                "pairs": pairs,
                "mc_call": mc_call,
                "se_call": se_call,
                "pct_err_call": pct_err_call,
                "mc_put": mc_put,
                "se_put": se_put,
                "pct_err_put": pct_err_put,
            }
        )

        print(
            f"{pairs:<12,} {mc_call:>12.4f} {se_call:>10.4f} {pct_err_call:>11.3f}% "
            f"{mc_put:>12.4f} {se_put:>10.4f} {pct_err_put:>11.3f}%"
        )

    print("-" * 90)

    try:
        plt, can_show = _prepare_matplotlib(True)
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        pairs_list = [r["pairs"] for r in results]
        mc_calls = [r["mc_call"] for r in results]
        mc_puts = [r["mc_put"] for r in results]
        se_calls = [r["se_call"] for r in results]
        se_puts = [r["se_put"] for r in results]

        axes[0].errorbar(
            pairs_list,
            mc_calls,
            yerr=[1.96 * se for se in se_calls],
            marker="o",
            capsize=5,
            linewidth=2,
            markersize=8,
            label="Monte Carlo Call (95% CI)",
        )
        axes[0].axhline(
            bs_call, color="red", linestyle="--", linewidth=2, label=f"Black-Scholes Call ({bs_call:.4f})"
        )
        axes[0].set_xlabel("Number of Simulation Pairs", fontsize=11)
        axes[0].set_ylabel("Call Option Price", fontsize=11)
        axes[0].set_title("Call Option Price Convergence", fontsize=13, fontweight="bold")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale("log")

        axes[1].errorbar(
            pairs_list,
            mc_puts,
            yerr=[1.96 * se for se in se_puts],
            marker="s",
            capsize=5,
            linewidth=2,
            markersize=8,
            color="green",
            label="Monte Carlo Put (95% CI)",
        )
        axes[1].axhline(
            bs_put, color="red", linestyle="--", linewidth=2, label=f"Black-Scholes Put ({bs_put:.4f})"
        )
        axes[1].set_xlabel("Number of Simulation Pairs", fontsize=11)
        axes[1].set_ylabel("Put Option Price", fontsize=11)
        axes[1].set_title("Put Option Price Convergence", fontsize=13, fontweight="bold")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale("log")

        fig.suptitle(
            f"Monte Carlo Convergence Analysis\nS₀={S0}, K={K}, r={r}, σ={sigma}, T={T}y",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()

        path = os.path.join(output_dir, "convergence_analysis.png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"\n✓ Convergence plot saved: {path}")

        if can_show:
            plt.show()
        plt.close("all")

    except Exception as error:
        print(f"\n⚠ Chart generation skipped: {error}")

    return results


def confidence_interval_analysis(S0=100, K=100, r=0.05, q=0.02, sigma=0.20, T=1.0, pairs=25000, seed=42):
    """
    Calculate and display 95% confidence intervals for Monte Carlo option prices.
    Demonstrates the statistical reliability of the Monte Carlo estimates.
    """
    print("\n" + "=" * 80)
    print("95% CONFIDENCE INTERVAL ANALYSIS")
    print("=" * 80)
    print(f"\nParameters: S₀={S0}, K={K}, r={r}, σ={sigma}, T={T}y, Pairs={pairs:,}")

    bs_call = bs_price(S0, K, r, q, sigma, T, "call")
    bs_put = bs_price(S0, K, r, q, sigma, T, "put")

    paths = simulate_paths(S0, r, q, sigma, T, 1, pairs, seed)

    mc_call, se_call, _, _ = european_mc(paths, S0, K, r, q, T, "call")
    mc_put, se_put, _, _ = european_mc(paths, S0, K, r, q, T, "put")

    ci_call_lower = mc_call - 1.96 * se_call
    ci_call_upper = mc_call + 1.96 * se_call
    ci_put_lower = mc_put - 1.96 * se_put
    ci_put_upper = mc_put + 1.96 * se_put

    bs_in_ci_call = ci_call_lower <= bs_call <= ci_call_upper
    bs_in_ci_put = ci_put_lower <= bs_put <= ci_put_upper

    print("\n" + "-" * 80)
    print("CALL OPTION")
    print("-" * 80)
    print(f"  Monte Carlo Price:       {mc_call:.6f}")
    print(f"  Standard Error:          {se_call:.6f}")
    print(f"  95% Confidence Interval: [{ci_call_lower:.6f}, {ci_call_upper:.6f}]")
    print(f"  Interval Width:          {ci_call_upper - ci_call_lower:.6f}")
    print(f"  Black-Scholes Price:     {bs_call:.6f}")
    print(f"  BS in 95% CI?            {'✓ YES' if bs_in_ci_call else '✗ NO'}")

    print("\n" + "-" * 80)
    print("PUT OPTION")
    print("-" * 80)
    print(f"  Monte Carlo Price:       {mc_put:.6f}")
    print(f"  Standard Error:          {se_put:.6f}")
    print(f"  95% Confidence Interval: [{ci_put_lower:.6f}, {ci_put_upper:.6f}]")
    print(f"  Interval Width:          {ci_put_upper - ci_put_lower:.6f}")
    print(f"  Black-Scholes Price:     {bs_put:.6f}")
    print(f"  BS in 95% CI?            {'✓ YES' if bs_in_ci_put else '✗ NO'}")

    print("\n" + "-" * 80)
    print("INTERPRETATION")
    print("-" * 80)
    if bs_in_ci_call and bs_in_ci_put:
        print("  ✓ Both Black-Scholes prices fall within the 95% confidence intervals.")
        print("    This confirms the Monte Carlo implementation is statistically consistent")
        print("    with the analytical Black-Scholes model.")
    else:
        print("  ⚠ One or more Black-Scholes prices fall outside the 95% confidence intervals.")
        print("    This may indicate:")
        print("    - Insufficient simulation pairs (increase for narrower intervals)")
        print("    - Random seed effects (try different seeds)")
        print("    - Implementation issues (review the Monte Carlo engine)")

    print("\n  Note: The 95% confidence interval means that if we repeated the Monte Carlo")
    print("  simulation many times, approximately 95% of the calculated intervals would")
    print("  contain the true option price.")

    return {
        "call": {
            "price": mc_call,
            "se": se_call,
            "ci_lower": ci_call_lower,
            "ci_upper": ci_call_upper,
            "bs_price": bs_call,
            "bs_in_ci": bs_in_ci_call,
        },
        "put": {
            "price": mc_put,
            "se": se_put,
            "ci_lower": ci_put_lower,
            "ci_upper": ci_put_upper,
            "bs_price": bs_put,
            "bs_in_ci": bs_in_ci_put,
        },
    }


def put_call_parity_validation(S0=100, K=100, r=0.05, q=0.02, sigma=0.20, T=1.0, pairs=25000, seed=42):
    """
    Validate put-call parity for European options with continuous dividend yield.

    Put-Call Parity Formula: C - P = S0 × exp(-q × T) - K × exp(-r × T)

    This test verifies that the Monte Carlo prices satisfy the no-arbitrage relationship
    between call and put options with the same strike and maturity.
    """
    print("\n" + "=" * 80)
    print("PUT-CALL PARITY VALIDATION")
    print("=" * 80)
    print(f"\nParameters: S₀={S0}, K={K}, r={r}, q={q}, σ={sigma}, T={T}y, Pairs={pairs:,}")

    paths = simulate_paths(S0, r, q, sigma, T, 1, pairs, seed)

    mc_call, se_call, values_call, _ = european_mc(paths, S0, K, r, q, T, "call")
    mc_put, se_put, values_put, _ = european_mc(paths, S0, K, r, q, T, "put")

    bs_call = bs_price(S0, K, r, q, sigma, T, "call")
    bs_put = bs_price(S0, K, r, q, sigma, T, "put")

    mc_lhs = mc_call - mc_put
    _, mc_lhs_se = cv_estimate(
        values_call - values_put,
        math.exp(-r * T) * paths[-1],
        S0 * math.exp(-q * T),
    )

    bs_lhs = bs_call - bs_put
    rhs = S0 * math.exp(-q * T) - K * math.exp(-r * T)

    mc_parity_error = abs(mc_lhs - rhs)
    bs_parity_error = abs(bs_lhs - rhs)

    tolerance = max(1.96 * mc_lhs_se, 1e-10)
    print("\n" + "-" * 80)
    print("PUT-CALL PARITY EQUATION: C - P = S₀×e^(-qT) - K×e^(-rT)")
    print("-" * 80)
    print("\n  Left-Hand Side (C - P):")
    print(f"    Monte Carlo:          {mc_lhs:>12.6f}  (SE: {mc_lhs_se:.6f})")
    print(f"    Black-Scholes:        {bs_lhs:>12.6f}")

    print("\n  Right-Hand Side:")
    print(f"    S₀×e^(-qT) - K×e^(-rT) = {rhs:>12.6f}")
    print(f"    where: S₀×e^(-qT) = {S0 * math.exp(-q * T):.6f}")
    print(f"           K×e^(-rT)    = {K * math.exp(-r * T):.6f}")

    print("\n  Parity Error (|LHS - RHS|):")
    print(f"    Monte Carlo:          {mc_parity_error:>12.6f}")
    print(f"    Black-Scholes:        {bs_parity_error:>12.6f}")

    print(f"\n  Tolerance (95% CI):     {tolerance:>12.6f}")

    print("\n" + "-" * 80)
    print("VALIDATION RESULT:")
    print("-" * 80)

    if mc_parity_error <= tolerance:
        print("  ✓ PASS: Monte Carlo put-call parity satisfied within numerical tolerance")
        print(f"    Error ({mc_parity_error:.6f}) <= Tolerance ({tolerance:.6f})")
        status = "PASS"
    elif mc_parity_error < 0.10:
        print("  ⚠ WARNING: Put-call parity error is small but exceeds statistical tolerance")
        print(f"    Error ({mc_parity_error:.6f}) > Tolerance ({tolerance:.6f})")
        print("    Consider increasing simulation pairs for tighter confidence intervals")
        status = "WARNING"
    else:
        print("  ✗ FAIL: Put-call parity violation detected")
        print(f"    Error ({mc_parity_error:.6f}) significantly exceeds tolerance ({tolerance:.6f})")
        print("    Review Monte Carlo implementation for potential bugs")
        status = "FAIL"

    return {
        "mc_call": mc_call,
        "mc_put": mc_put,
        "mc_lhs": mc_lhs,
        "mc_lhs_se": mc_lhs_se,
        "bs_call": bs_call,
        "bs_put": bs_put,
        "bs_lhs": bs_lhs,
        "rhs": rhs,
        "mc_parity_error": mc_parity_error,
        "bs_parity_error": bs_parity_error,
        "tolerance": tolerance,
        "status": status,
    }


def no_arbitrage_bounds_validation(S0=100, K=100, r=0.05, q=0.02, sigma=0.20, T=1.0, pairs=25000, seed=42):
    """
    Validate that Monte Carlo option prices satisfy theoretical no-arbitrage bounds.

    European Call Bounds:
      Lower: max(0, S0 × exp(-q × T) - K × exp(-r × T))
      Upper: S0 × exp(-q × T)

    European Put Bounds:
      Lower: max(0, K × exp(-r × T) - S0 × exp(-q × T))
      Upper: K × exp(-r × T)
    """
    print("\n" + "=" * 80)
    print("NO-ARBITRAGE BOUNDS VALIDATION")
    print("=" * 80)
    print(f"\nParameters: S₀={S0}, K={K}, r={r}, q={q}, σ={sigma}, T={T}y, Pairs={pairs:,}")

    discount_q = math.exp(-q * T)
    discount_r = math.exp(-r * T)

    call_lower = max(0.0, S0 * discount_q - K * discount_r)
    call_upper = S0 * discount_q

    put_lower = max(0.0, K * discount_r - S0 * discount_q)
    put_upper = K * discount_r

    paths = simulate_paths(S0, r, q, sigma, T, 1, pairs, seed)
    mc_call, se_call, _, _ = european_mc(paths, S0, K, r, q, T, "call")
    mc_put, se_put, _, _ = european_mc(paths, S0, K, r, q, T, "put")

    bs_call = bs_price(S0, K, r, q, sigma, T, "call")
    bs_put = bs_price(S0, K, r, q, sigma, T, "put")

    call_tolerance = 1.96 * se_call
    put_tolerance = 1.96 * se_put

    print("\n" + "-" * 80)
    print("EUROPEAN CALL OPTION BOUNDS")
    print("-" * 80)
    print(f"  Theoretical Lower Bound:  {call_lower:>12.6f}")
    print(f"  Theoretical Upper Bound:  {call_upper:>12.6f}")
    print(f"\n  Monte Carlo Price:        {mc_call:>12.6f}  (±{call_tolerance:.6f} at 95% CI)")
    print(f"  Black-Scholes Price:      {bs_call:>12.6f}")

    call_lower_ok = mc_call >= call_lower - call_tolerance
    call_upper_ok = mc_call <= call_upper + call_tolerance
    call_status = "PASS" if (call_lower_ok and call_upper_ok) else "FAIL"

    print(f"\n  Lower Bound Check:        {'✓ PASS' if call_lower_ok else '✗ FAIL'}")
    print(f"  Upper Bound Check:        {'✓ PASS' if call_upper_ok else '✗ FAIL'}")

    print("\n" + "-" * 80)
    print("EUROPEAN PUT OPTION BOUNDS")
    print("-" * 80)
    print(f"  Theoretical Lower Bound:  {put_lower:>12.6f}")
    print(f"  Theoretical Upper Bound:  {put_upper:>12.6f}")
    print(f"\n  Monte Carlo Price:        {mc_put:>12.6f}  (±{put_tolerance:.6f} at 95% CI)")
    print(f"  Black-Scholes Price:      {bs_put:>12.6f}")

    put_lower_ok = mc_put >= put_lower - put_tolerance
    put_upper_ok = mc_put <= put_upper + put_tolerance
    put_status = "PASS" if (put_lower_ok and put_upper_ok) else "FAIL"

    print(f"\n  Lower Bound Check:        {'✓ PASS' if put_lower_ok else '✗ FAIL'}")
    print(f"  Upper Bound Check:        {'✓ PASS' if put_upper_ok else '✗ FAIL'}")

    print("\n" + "-" * 80)
    print("OVERALL VALIDATION RESULT:")
    print("-" * 80)

    if call_status == "PASS" and put_status == "PASS":
        print("  ✓ PASS: All no-arbitrage bounds satisfied")
        overall_status = "PASS"
    else:
        print("  ✗ FAIL: One or more bounds violated")
        overall_status = "FAIL"

    return {
        "call": {
            "price": mc_call,
            "lower": call_lower,
            "upper": call_upper,
            "se": se_call,
            "status": call_status,
        },
        "put": {
            "price": mc_put,
            "lower": put_lower,
            "upper": put_upper,
            "se": se_put,
            "status": put_status,
        },
        "overall_status": overall_status,
    }


def multiple_scenario_validation(base_seed=42, pairs=25000):
    """
    Test Monte Carlo pricing robustness across multiple market scenarios.

    Scenarios:
    1. At-the-money: S0=100, K=100
    2. In-the-money call: S0=120, K=100
    3. Out-of-the-money call: S0=80, K=100
    4. Low volatility: σ=0.10
    5. High volatility: σ=0.40
    6. Short maturity: T=0.25
    7. Long maturity: T=2.0
    """
    print("\n" + "=" * 80)
    print("MULTIPLE SCENARIO VALIDATION")
    print("=" * 80)
    print(f"\nTesting 7 market scenarios with {pairs:,} simulation pairs each")

    base_params = {"S0": 100, "K": 100, "r": 0.05, "q": 0.02, "sigma": 0.20, "T": 1.0}

    scenarios = [
        ("1. ATM Option", {"S0": 100, "K": 100}),
        ("2. ITM Call", {"S0": 120, "K": 100}),
        ("3. OTM Call", {"S0": 80, "K": 100}),
        ("4. Low Vol", {"sigma": 0.10}),
        ("5. High Vol", {"sigma": 0.40}),
        ("6. Short Maturity", {"T": 0.25}),
        ("7. Long Maturity", {"T": 2.0}),
    ]

    results = []

    print("\n" + "-" * 120)
    print(
        f"{'Scenario':<20} {'MC Call':>10} {'BS Call':>10} {'Call Err%':>10} "
        f"{'MC Put':>10} {'BS Put':>10} {'Put Err%':>10} {'Status':>8}"
    )
    print("-" * 120)

    for scenario_name, overrides in scenarios:
        params = base_params.copy()
        params.update(overrides)

        S0, K = params["S0"], params["K"]
        r, q = params["r"], params["q"]
        sigma, T = params["sigma"], params["T"]

        bs_call = bs_price(S0, K, r, q, sigma, T, "call")
        bs_put = bs_price(S0, K, r, q, sigma, T, "put")

        paths = simulate_paths(S0, r, q, sigma, T, 1, pairs, base_seed)
        mc_call, se_call, _, _ = european_mc(paths, S0, K, r, q, T, "call")
        mc_put, se_put, _, _ = european_mc(paths, S0, K, r, q, T, "put")

        call_err_pct = _percentage_error(mc_call, bs_call)
        put_err_pct = _percentage_error(mc_put, bs_put)

        call_in_ci = abs(mc_call - bs_call) <= 1.96 * se_call
        put_in_ci = abs(mc_put - bs_put) <= 1.96 * se_put

        if call_in_ci and put_in_ci:
            status = "✓ PASS"
        elif call_err_pct < 5.0 and put_err_pct < 5.0:
            status = "⚠ WARN"
        else:
            status = "✗ FAIL"

        results.append(
            {
                "scenario": scenario_name,
                "params": params,
                "mc_call": mc_call,
                "bs_call": bs_call,
                "call_err_pct": call_err_pct,
                "mc_put": mc_put,
                "bs_put": bs_put,
                "put_err_pct": put_err_pct,
                "se_call": se_call,
                "se_put": se_put,
                "call_in_ci": call_in_ci,
                "put_in_ci": put_in_ci,
                "status": status,
            }
        )

        print(
            f"{scenario_name:<20} {mc_call:>10.4f} {bs_call:>10.4f} {call_err_pct:>9.3f}% "
            f"{mc_put:>10.4f} {bs_put:>10.4f} {put_err_pct:>9.3f}% {status:>8}"
        )

    print("-" * 120)

    pass_count = sum(1 for r in results if "PASS" in r["status"])
    warn_count = sum(1 for r in results if "WARN" in r["status"])
    fail_count = sum(1 for r in results if "FAIL" in r["status"])

    print(f"\nSummary: {pass_count} PASS, {warn_count} WARNING, {fail_count} FAIL")

    if fail_count == 0:
        print("✓ All scenarios produce statistically reasonable results")
    else:
        print(f"⚠ {fail_count} scenario(s) require investigation")

    return results


def empirical_ci_coverage(
    n_experiments=500,
    pairs_per_experiment=10000,
    S0=100,
    K=100,
    r=0.05,
    q=0.02,
    sigma=0.20,
    T=1.0,
    base_seed=42,
    output_dir="charts",
):
    """
    Empirically test whether 95% confidence intervals actually contain the true BS price 95% of the time.

    Runs multiple independent Monte Carlo experiments with different random seeds and checks
    how often the analytical Black-Scholes price falls within the calculated 95% CI.
    """
    print("\n" + "=" * 80)
    print("EMPIRICAL CONFIDENCE INTERVAL COVERAGE TEST")
    print("=" * 80)
    print(f"\nRunning {n_experiments} independent Monte Carlo experiments")
    print(f"Parameters: S₀={S0}, K={K}, r={r}, σ={sigma}, T={T}y")
    print(f"Simulations per experiment: {pairs_per_experiment:,}")

    bs_call = bs_price(S0, K, r, q, sigma, T, "call")
    bs_put = bs_price(S0, K, r, q, sigma, T, "put")

    print("\nTrue Black-Scholes Prices:")
    print(f"  Call: {bs_call:.6f}")
    print(f"  Put:  {bs_put:.6f}")

    call_coverage_count = 0
    put_coverage_count = 0

    call_results = []
    put_results = []

    print(
        f"\n{'Experiment':<12} {'MC Call':>10} {'Call SE':>10} {'Call CI Contains BS?':>20} "
        f"{'MC Put':>10} {'Put SE':>10} {'Put CI Contains BS?':>20}"
    )
    print("-" * 100)

    for exp in range(n_experiments):
        seed = base_seed + exp * 1000

        paths = simulate_paths(S0, r, q, sigma, T, 1, pairs_per_experiment, seed)
        mc_call, se_call, _, _ = european_mc(paths, S0, K, r, q, T, "call")
        mc_put, se_put, _, _ = european_mc(paths, S0, K, r, q, T, "put")

        call_ci_lower = mc_call - 1.96 * se_call
        call_ci_upper = mc_call + 1.96 * se_call
        put_ci_lower = mc_put - 1.96 * se_put
        put_ci_upper = mc_put + 1.96 * se_put

        call_contains = call_ci_lower <= bs_call <= call_ci_upper
        put_contains = put_ci_lower <= bs_put <= put_ci_upper

        if call_contains:
            call_coverage_count += 1
        if put_contains:
            put_coverage_count += 1

        call_results.append({"mc": mc_call, "se": se_call, "contains": call_contains})
        put_results.append({"mc": mc_put, "se": se_put, "contains": put_contains})

        if (exp + 1) % 100 == 0 or exp == 0:
            print(
                f"{exp+1:<12} {mc_call:>10.4f} {se_call:>10.4f} {'✓ YES' if call_contains else '✗ NO':>20} "
                f"{mc_put:>10.4f} {se_put:>10.4f} {'✓ YES' if put_contains else '✗ NO':>20}"
            )

    print("-" * 100)

    call_coverage_pct = (call_coverage_count / n_experiments) * 100
    put_coverage_pct = (put_coverage_count / n_experiments) * 100

    print(f"\n{'='*80}")
    print("EMPIRICAL COVERAGE RESULTS")
    print(f"{'='*80}")
    print(f"\n  Total Experiments:               {n_experiments}")
    print("\n  CALL OPTION:")
    print(f"    Intervals containing BS price: {call_coverage_count}")
    print(f"    Intervals NOT containing BS:   {n_experiments - call_coverage_count}")
    print(f"    Empirical Coverage:            {call_coverage_pct:.2f}%")
    print("    Expected Coverage:             95.00%")
    print(f"    Difference:                    {call_coverage_pct - 95.0:+.2f}%")

    print("\n  PUT OPTION:")
    print(f"    Intervals containing BS price: {put_coverage_count}")
    print(f"    Intervals NOT containing BS:   {n_experiments - put_coverage_count}")
    print(f"    Empirical Coverage:            {put_coverage_pct:.2f}%")
    print("    Expected Coverage:             95.00%")
    print(f"    Difference:                    {put_coverage_pct - 95.0:+.2f}%")

    print(f"\n{'='*80}")
    print("STATISTICAL INTERPRETATION")
    print(f"{'='*80}")

    tolerance = 1.96 * 100 * math.sqrt(0.95 * 0.05 / n_experiments)
    call_status = "PASS" if abs(call_coverage_pct - 95.0) <= tolerance else "WARNING"
    put_status = "PASS" if abs(put_coverage_pct - 95.0) <= tolerance else "WARNING"

    print("\n  Note: Empirical coverage is subject to sampling variation.")
    print(
        f"  With {n_experiments} experiments, the approximate 95% sampling tolerance "
        f"is ±{tolerance:.2f} percentage points around 95%."
    )

    if call_status == "PASS" and put_status == "PASS":
        print("\n  ✓ PASS: Both call and put coverage are within expected range")
        print("    The 95% confidence intervals are statistically valid")
    else:
        print("\n  ⚠ WARNING: Coverage deviates from expected 95% level")
        if call_status != "PASS":
            print(f"    Call coverage ({call_coverage_pct:.2f}%) is outside [{95-tolerance:.2f}%, {95+tolerance:.2f}%]")
        if put_status != "PASS":
            print(f"    Put coverage ({put_coverage_pct:.2f}%) is outside [{95-tolerance:.2f}%, {95+tolerance:.2f}%]")
        print("    Investigate standard error calculation or payoff distribution")

    try:
        plt, can_show = _prepare_matplotlib(True)
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        experiments = np.arange(1, n_experiments + 1)
        call_cumulative = np.cumsum([r["contains"] for r in call_results]) / experiments * 100
        axes[0].plot(experiments, call_cumulative, linewidth=2, label="Empirical Coverage")
        axes[0].axhline(95, color="red", linestyle="--", linewidth=2, label="Expected 95%")
        axes[0].axhline(
            95 + tolerance, color="orange", linestyle=":", linewidth=1, alpha=0.5,
            label=f"±{tolerance:.2f} percentage points",
        )
        axes[0].axhline(95 - tolerance, color="orange", linestyle=":", linewidth=1, alpha=0.5)
        axes[0].set_xlabel("Number of Experiments", fontsize=11)
        axes[0].set_ylabel("Coverage Percentage (%)", fontsize=11)
        axes[0].set_title(
            f"Call Option CI Coverage\nFinal: {call_coverage_pct:.2f}%", fontsize=13, fontweight="bold"
        )
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([80, 100])

        put_cumulative = np.cumsum([r["contains"] for r in put_results]) / experiments * 100
        axes[1].plot(experiments, put_cumulative, linewidth=2, color="green", label="Empirical Coverage")
        axes[1].axhline(95, color="red", linestyle="--", linewidth=2, label="Expected 95%")
        axes[1].axhline(95 + tolerance, color="orange", linestyle=":", linewidth=1, alpha=0.5)
        axes[1].axhline(95 - tolerance, color="orange", linestyle=":", linewidth=1, alpha=0.5)
        axes[1].set_xlabel("Number of Experiments", fontsize=11)
        axes[1].set_ylabel("Coverage Percentage (%)", fontsize=11)
        axes[1].set_title(
            f"Put Option CI Coverage\nFinal: {put_coverage_pct:.2f}%", fontsize=13, fontweight="bold"
        )
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([80, 100])

        fig.suptitle(
            f"Empirical Confidence Interval Coverage Test\n{n_experiments} experiments, {pairs_per_experiment:,} simulations each",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()

        path = os.path.join(output_dir, "empirical_ci_coverage.png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"\n✓ Coverage plot saved: {path}")

        if can_show:
            plt.show()
        plt.close("all")

    except Exception as error:
        print(f"\n⚠ Chart generation skipped: {error}")

    return {
        "call_coverage": call_coverage_pct,
        "put_coverage": put_coverage_pct,
        "call_count": call_coverage_count,
        "put_count": put_coverage_count,
        "n_experiments": n_experiments,
        "tolerance": tolerance,
        "call_status": call_status,
        "put_status": put_status,
    }


def greeks_validation(S0=100, K=100, r=0.05, q=0.02, sigma=0.20, T=1.0, pairs=25000, seed=42):
    """
    Validate Monte Carlo Greeks against analytical Black-Scholes Greeks.

    Tests: Delta, Gamma, Vega (per 1.0 vol change), Theta (per year)

    Both mc_greeks() and bs_greeks() use internal conventions:
      - Vega: per 0.01 change in volatility (per vol point)
      - Theta: per calendar day
    This function converts both to standard display conventions:
      - Vega: per 1.00 change in volatility (×100)
      - Theta: per year (×365)
    """
    print("\n" + "=" * 80)
    print("GREEKS VALIDATION: Monte Carlo vs Black-Scholes")
    print("=" * 80)
    print(f"\nParameters: S₀={S0}, K={K}, r={r}, q={q}, σ={sigma}, T={T}y, Pairs={pairs:,}")

    bs_greeks_call = bs_greeks(S0, K, r, q, sigma, T, "call")
    bs_vega_call_display = bs_greeks_call["vega"] * 100   # per vol point -> per 1.0 vol change
    bs_theta_call_display = bs_greeks_call["theta"] * 365  # per day -> per year

    bs_greeks_put = bs_greeks(S0, K, r, q, sigma, T, "put")
    bs_vega_put_display = bs_greeks_put["vega"] * 100
    bs_theta_put_display = bs_greeks_put["theta"] * 365

    paths = simulate_paths(S0, r, q, sigma, T, 1, pairs, seed)
    mc_greeks_call = mc_greeks(paths, S0, K, r, q, sigma, T, "call")
    mc_vega_call_display = mc_greeks_call["vega"] * 100
    mc_theta_call_display = mc_greeks_call["theta"] * 365

    mc_greeks_put = mc_greeks(paths, S0, K, r, q, sigma, T, "put")
    mc_vega_put_display = mc_greeks_put["vega"] * 100
    mc_theta_put_display = mc_greeks_put["theta"] * 365

    print("\n" + "-" * 100)
    print("CALL OPTION GREEKS")
    print("-" * 100)
    print(f"{'Greek':<14} {'Monte Carlo':>15} {'Black-Scholes':>15} {'Abs Error':>12} {'% Error':>10} {'Status':>8}")
    print("-" * 100)

    mc_delta = mc_greeks_call["delta"]
    bs_delta = bs_greeks_call["delta"]
    delta_err = abs(mc_delta - bs_delta)
    delta_pct = _percentage_error(mc_delta, bs_delta)
    delta_status = "PASS" if delta_pct < 5.0 else "WARNING"
    print(f"{'Delta':<14} {mc_delta:>15.6f} {bs_delta:>15.6f} {delta_err:>12.6f} {delta_pct:>9.3f}% {delta_status:>8}")

    mc_gamma = mc_greeks_call["gamma"]
    bs_gamma = bs_greeks_call["gamma"]
    gamma_err = abs(mc_gamma - bs_gamma)
    gamma_pct = _percentage_error(mc_gamma, bs_gamma)
    gamma_status = "PASS" if gamma_pct < 10.0 else "WARNING"
    print(f"{'Gamma':<14} {mc_gamma:>15.6f} {bs_gamma:>15.6f} {gamma_err:>12.6f} {gamma_pct:>9.3f}% {gamma_status:>8}")

    mc_vega = mc_vega_call_display
    bs_vega = bs_vega_call_display
    vega_err = abs(mc_vega - bs_vega)
    vega_pct = _percentage_error(mc_vega, bs_vega)
    vega_status = "PASS" if vega_pct < 10.0 else "WARNING"
    print(
        f"{'Vega (per 1.0)':<14} {mc_vega:>15.6f} {bs_vega:>15.6f} {vega_err:>12.6f} {vega_pct:>9.3f}% {vega_status:>8}"
    )

    mc_theta = mc_theta_call_display
    bs_theta = bs_theta_call_display
    theta_err = abs(mc_theta - bs_theta)
    theta_pct = _percentage_error(mc_theta, bs_theta)
    theta_status = "PASS" if theta_pct < 15.0 else "WARNING"
    print(
        f"{'Theta (per yr)':<14} {mc_theta:>15.6f} {bs_theta:>15.6f} {theta_err:>12.6f} {theta_pct:>9.3f}% {theta_status:>8}"
    )

    mc_theta_daily = mc_greeks_call["theta"]
    bs_theta_daily = bs_greeks_call["theta"]
    print(f"  (Theta per day:  MC={mc_theta_daily:.6f}, BS={bs_theta_daily:.6f})")

    print("\n" + "-" * 100)
    print("PUT OPTION GREEKS")
    print("-" * 100)
    print(f"{'Greek':<14} {'Monte Carlo':>15} {'Black-Scholes':>15} {'Abs Error':>12} {'% Error':>10} {'Status':>8}")
    print("-" * 100)

    mc_delta_put = mc_greeks_put["delta"]
    bs_delta_put = bs_greeks_put["delta"]
    delta_err_put = abs(mc_delta_put - bs_delta_put)
    delta_pct_put = _percentage_error(mc_delta_put, bs_delta_put)
    delta_status_put = "PASS" if delta_pct_put < 5.0 else "WARNING"
    print(
        f"{'Delta':<14} {mc_delta_put:>15.6f} {bs_delta_put:>15.6f} {delta_err_put:>12.6f} {delta_pct_put:>9.3f}% {delta_status_put:>8}"
    )

    mc_gamma_put = mc_greeks_put["gamma"]
    bs_gamma_put = bs_greeks_put["gamma"]
    gamma_err_put = abs(mc_gamma_put - bs_gamma_put)
    gamma_pct_put = _percentage_error(mc_gamma_put, bs_gamma_put)
    gamma_status_put = "PASS" if gamma_pct_put < 10.0 else "WARNING"
    print(
        f"{'Gamma':<14} {mc_gamma_put:>15.6f} {bs_gamma_put:>15.6f} {gamma_err_put:>12.6f} {gamma_pct_put:>9.3f}% {gamma_status_put:>8}"
    )

    mc_vega_put = mc_vega_put_display
    bs_vega_put = bs_vega_put_display
    vega_err_put = abs(mc_vega_put - bs_vega_put)
    vega_pct_put = _percentage_error(mc_vega_put, bs_vega_put)
    vega_status_put = "PASS" if vega_pct_put < 10.0 else "WARNING"
    print(
        f"{'Vega (per 1.0)':<14} {mc_vega_put:>15.6f} {bs_vega_put:>15.6f} {vega_err_put:>12.6f} {vega_pct_put:>9.3f}% {vega_status_put:>8}"
    )

    mc_theta_put = mc_theta_put_display
    bs_theta_put = bs_theta_put_display
    theta_err_put = abs(mc_theta_put - bs_theta_put)
    theta_pct_put = _percentage_error(mc_theta_put, bs_theta_put)
    theta_status_put = "PASS" if theta_pct_put < 15.0 else "WARNING"
    print(
        f"{'Theta (per yr)':<14} {mc_theta_put:>15.6f} {bs_theta_put:>15.6f} {theta_err_put:>12.6f} {theta_pct_put:>9.3f}% {theta_status_put:>8}"
    )

    mc_theta_put_daily = mc_greeks_put["theta"]
    bs_theta_put_daily = bs_greeks_put["theta"]
    print(f"  (Theta per day:  MC={mc_theta_put_daily:.6f}, BS={bs_theta_put_daily:.6f})")

    print("\n" + "-" * 100)
    print("NOTES ON CONVENTIONS")
    print("-" * 100)
    print("  • Vega is reported per 1.0 (100%) change in volatility")
    print("  • Theta is reported per year (annualized); daily theta also shown for reference")
    print("  • Theta sign convention: negative for long options (time decay reduces value)")
    print("  • Delta for calls: 0 to 1; Delta for puts: -1 to 0")
    print("  • Gamma is identical for calls and puts (put-call parity)")
    print("  • Vega is identical for calls and puts (put-call parity)")
    print("  • MC Greeks use finite-difference with common random numbers to reduce noise")

    all_statuses = [
        delta_status,
        gamma_status,
        vega_status,
        theta_status,
        delta_status_put,
        gamma_status_put,
        vega_status_put,
        theta_status_put,
    ]
    warn_count = sum(1 for s in all_statuses if s == "WARNING")
    if warn_count == 0:
        overall = "PASS: All Greeks within acceptable tolerance"
    elif warn_count <= 2:
        overall = "WARNING: Some Greeks show larger-than-expected errors (normal for MC)"
        overall += "\n    Monte Carlo Greeks use finite differences and are inherently noisier than prices"
    else:
        overall = "FAIL: Significant Greek calculation errors detected"

    print(f"\n  Overall: {overall}")

    return {
        "call": {
            "delta": (mc_delta, bs_delta),
            "gamma": (mc_gamma, bs_gamma),
            "vega": (mc_vega, bs_vega),
            "theta": (mc_theta, bs_theta),
        },
        "put": {
            "delta": (mc_delta_put, bs_delta_put),
            "gamma": (mc_gamma_put, bs_gamma_put),
            "vega": (mc_vega_put, bs_vega_put),
            "theta": (mc_theta_put, bs_theta_put),
        },
        "overall_status": overall,
    }


def final_validation_dashboard():
    """
    Execute all validation tests and display a consolidated dashboard with PASS/WARNING/FAIL status.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION DASHBOARD")
    print("=" * 80)

    S0, K, r, q, sigma, T = 100, 100, 0.05, 0.02, 0.20, 1.0
    pairs, seed = 25000, 42

    results = {}

    print("\n[1/8] Running Monte Carlo vs Black-Scholes price comparison...")
    try:
        test1 = validate_european_prices(S0, K, r, q, sigma, T, pairs, seed)
        results["MC vs BS Prices"] = (
            "PASS" if test1["pct_error_call"] < 2.0 and test1["pct_error_put"] < 2.0 else "WARNING"
        )
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["MC vs BS Prices"] = "FAIL"

    print("\n[2/8] Running convergence analysis...")
    try:
        test2 = convergence_analysis(S0, K, r, q, sigma, T, seed)
        call_errors = [r["pct_err_call"] for r in test2]
        put_errors = [r["pct_err_put"] for r in test2]
        if (
            call_errors[-1] < call_errors[0]
            and put_errors[-1] < put_errors[0]
            and max(call_errors[-1], put_errors[-1]) < 0.5
        ):
            results["Convergence"] = "PASS"
        else:
            results["Convergence"] = "WARNING"
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["Convergence"] = "FAIL"

    print("\n[3/8] Running confidence interval analysis...")
    try:
        test3 = confidence_interval_analysis(S0, K, r, q, sigma, T, pairs, seed)
        if test3["call"]["bs_in_ci"] and test3["put"]["bs_in_ci"]:
            results["Confidence Intervals"] = "PASS"
        else:
            results["Confidence Intervals"] = "WARNING"
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["Confidence Intervals"] = "FAIL"

    print("\n[4/8] Running put-call parity validation...")
    try:
        test4 = put_call_parity_validation(S0, K, r, q, sigma, T, pairs, seed)
        results["Put-Call Parity"] = test4["status"]
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["Put-Call Parity"] = "FAIL"

    print("\n[5/8] Running no-arbitrage bounds validation...")
    try:
        test5 = no_arbitrage_bounds_validation(S0, K, r, q, sigma, T, pairs, seed)
        results["No-Arbitrage Bounds"] = test5["overall_status"]
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["No-Arbitrage Bounds"] = "FAIL"

    print("\n[6/8] Running multiple scenario validation...")
    try:
        test6 = multiple_scenario_validation(seed, pairs)
        fail_count = sum(1 for r in test6 if "FAIL" in r["status"])
        warn_count = sum(1 for r in test6 if "WARN" in r["status"])
        if fail_count > 1:
            results["Multiple Scenarios"] = "FAIL"
        elif fail_count or warn_count:
            results["Multiple Scenarios"] = "WARNING"
        else:
            results["Multiple Scenarios"] = "PASS"
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["Multiple Scenarios"] = "FAIL"

    print("\n[7/8] Running empirical CI coverage test (500 experiments)...")
    try:
        test7 = empirical_ci_coverage(
            n_experiments=500,
            pairs_per_experiment=10000,
            S0=S0,
            K=K,
            r=r,
            q=q,
            sigma=sigma,
            T=T,
            base_seed=seed,
        )
        if test7["call_status"] == "PASS" and test7["put_status"] == "PASS":
            results["Empirical CI Coverage"] = "PASS"
        else:
            results["Empirical CI Coverage"] = "WARNING"
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["Empirical CI Coverage"] = "FAIL"

    print("\n[8/8] Running Greeks validation...")
    try:
        test8 = greeks_validation(S0, K, r, q, sigma, T, pairs, seed)
        if "PASS" in test8["overall_status"]:
            results["Greeks Validation"] = "PASS"
        elif "WARNING" in test8["overall_status"]:
            results["Greeks Validation"] = "WARNING"
        else:
            results["Greeks Validation"] = "FAIL"
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["Greeks Validation"] = "FAIL"

    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Validation Test':<30} {'Expected Result':<35} {'Status':>10}")
    print("-" * 80)

    expected_results = {
        "MC vs BS Prices": "Statistically consistent",
        "Convergence": "Estimates stabilize",
        "Confidence Intervals": "Contain BS benchmark",
        "Put-Call Parity": "Satisfied within error",
        "No-Arbitrage Bounds": "Prices within bounds",
        "Multiple Scenarios": "Reasonable across markets",
        "Empirical CI Coverage": "Approximately 95%",
        "Greeks Validation": "Consistent with BS Greeks",
    }

    for test_name, status in results.items():
        expected = expected_results.get(test_name, "")
        status_symbol = "✓" if status == "PASS" else ("⚠" if status == "WARNING" else "✗")
        print(f"{test_name:<30} {expected:<35} {status_symbol} {status:>7}")

    print("-" * 80)

    pass_count = sum(1 for s in results.values() if s == "PASS")
    warn_count = sum(1 for s in results.values() if s == "WARNING")
    fail_count = sum(1 for s in results.values() if s == "FAIL")

    print(f"\nOverall: {pass_count} PASS, {warn_count} WARNING, {fail_count} FAIL")

    if fail_count == 0 and warn_count <= 2:
        print("\n✓ VALIDATION SUCCESSFUL")
        print("  The Monte Carlo implementation is statistically sound and ready for production use.")
    elif fail_count == 0:
        print("\n⚠ VALIDATION PASSED WITH WARNINGS")
        print("  The implementation is functional but some tests show marginal results.")
        print("  Consider increasing simulation pairs or reviewing specific tests.")
    else:
        print("\n✗ VALIDATION FAILED")
        print("  Critical issues detected. Review the failing tests before using in production.")

    return results


def run_full_validation():
    """
    Execute all validation steps with the specified parameters.
    This is the main entry point for validating the Monte Carlo implementation.
    """
    print("\n" + "=" * 80)
    print("MONTE CARLO OPTION PRICING - COMPREHENSIVE VALIDATION SUITE")
    print("=" * 80)

    final_validation_dashboard()

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


# ========== 11. Command line ==========
def run(symbols, days_list, opt, csv_path=None, plot=True, plot_dir="charts", show=True, announce=False):
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
            try:                                  # a chart problem must never lose the numbers
                plot_results(
                    ctx, results, pairs=opt.pairs, seed=opt.seed, output_dir=plot_dir, show=show
                )
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
    ap.add_argument(
        "--days",
        default=None,
        help="YOUR time to expiry: days (30), 2w/3m/1y, a date (2026-12-18), a list "
        "(7,30,90), 'pick' (~1 month) or 'ladder' (1w/1m/3m/6m). If omitted: the "
        "prompt mode asks you; command-line runs use 'ladder'",
    )
    ap.add_argument(
        "--list-expiries",
        action="store_true",
        help="print the listed expiry dates for the symbol(s) and exit",
    )
    ap.add_argument(
        "--pairs",
        type=int,
        default=10_000,
        help="antithetic path pairs, minimum 4 (default 10000)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nstrikes", type=int, default=9, help="strikes per expiry (default 9)")
    ap.add_argument("--strikes", help="override strikes, e.g. 200,210,220")
    ap.add_argument("--r", type=float, default=None, help="override risk-free rate (continuous, decimal)")
    ap.add_argument("--q", type=float, default=None, help="override dividend yield (decimal)")
    ap.add_argument("--sigma", type=float, default=None, help="override with one flat volatility")
    ap.add_argument("--csv", default=None, help="append results to this CSV")
    ap.add_argument("--no-plot", action="store_true", help="skip the charts (numbers only)")
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="save the PNG charts but do not open windows (also automatic with no display)",
    )
    ap.add_argument("--plot", action="store_true", help=argparse.SUPPRESS)   # kept: charts are now the default
    ap.add_argument("--plot-dir", default="charts", help="folder for the PNG charts (default: charts)")
    ap.add_argument(
        "--validate",
        action="store_true",
        help="run the full validation suite (MC vs Black-Scholes, convergence, confidence intervals)",
    )
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

    if args.validate:
        run_full_validation()
        return

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
        if not run(
            symbols,
            days_list or parse_days("ladder"),
            args,
            args.csv,
            plot,
            args.plot_dir,
            show,
            explicit,
        ):
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