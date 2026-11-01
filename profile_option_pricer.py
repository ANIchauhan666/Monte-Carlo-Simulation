"""Reproducible synthetic profiling/benchmark for one option expiry.

Examples:
    python3 profile_option_pricer.py
    python3 profile_option_pricer.py --pairs 10000 --repeats 5 --profile
"""

import argparse
import cProfile
import pstats
import resource
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

import test_european_pricing as pricer
from bounded_path_cache import BoundedPathCache


def synthetic_chain(spot=100.0, T=0.5, r=0.05, q=0.02):
    strikes = np.arange(80.0, 121.0, 5.0)
    calls, puts = [], []
    for index, strike in enumerate(strikes):
        sigma = 0.18 + 0.004 * index
        for kind, rows in (("call", calls), ("put", puts)):
            mid = pricer.bs_price(spot, strike, r, q, sigma, T, kind)
            rows.append({
                "strike": strike,
                "bid": max(mid - 0.02, 0.001),
                "ask": mid + 0.02,
                "lastPrice": mid,
            })
    return {
        "expiry": "synthetic",
        "days": round(T * 365),
        "calls": pd.DataFrame(calls).set_index("strike"),
        "puts": pd.DataFrame(puts).set_index("strike"),
    }


def peak_rss_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=5_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-mib", type=float, default=96,
                        help="path-cache budget used for this benchmark run (default: 96 MiB)")
    parser.add_argument("--profile", action="store_true",
                        help="print the top cumulative-time functions for the last run")
    args = parser.parse_args()
    if args.pairs < 4 or args.repeats < 1 or args.cache_mib < 0:
        parser.error("--pairs must be at least 4, --repeats at least 1, and --cache-mib non-negative")

    cache_bytes = int(args.cache_mib * 1024 * 1024)
    pricer.BoundedPathCache = lambda: BoundedPathCache(cache_bytes)

    spot, T, r, q, hist_vol = 100.0, 0.5, 0.05, 0.02, 0.20
    chain = synthetic_chain(spot, T, r, q)
    ctx = {"spot": spot, "q": q, "hist_vol": hist_vol, "symbol": "SYNTHETIC"}
    opt = SimpleNamespace(
        pairs=args.pairs, nstrikes=9, sigma=None, strikes=None, seed=42,
    )
    curve = {"points": [(1.0, r)]}
    timings = []
    profiler = cProfile.Profile() if args.profile else None

    for _ in range(args.repeats):
        start = time.perf_counter()
        if profiler is None:
            result = pricer.price_expiry(ctx, chain, round(T * 365), T, curve, opt)
        else:
            result = profiler.runcall(
                pricer.price_expiry, ctx, chain, round(T * 365), T, curve, opt,
            )
        timings.append(time.perf_counter() - start)

    print(f"Synthetic expiry: {len(result['rows'])} strikes, {args.pairs:,} pairs, seed=42")
    print(f"Path-cache budget: {args.cache_mib:g} MiB")
    print(f"Run times (s): {', '.join(f'{value:.3f}' for value in timings)}")
    print(f"Median (s): {statistics.median(timings):.3f}")
    print(f"Process peak RSS (MiB): {peak_rss_mib():.1f}")
    print(f"Path matrix size per scenario (MiB): "
          f"{(pricer.monitoring_steps(T) + 1) * 2 * args.pairs * 8 / (1024**2):.1f}")
    if profiler is not None:
        print("\nTop cumulative-time functions:")
        pstats.Stats(profiler).strip_dirs().sort_stats("cumtime").print_stats(15)


if __name__ == "__main__":
    main()
