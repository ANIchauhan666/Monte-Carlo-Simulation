# ========== 1. Imports ==========
import math
from datetime import datetime, timezone

import numpy as np
import requests
import yfinance as yf

import matplotlib.pyplot as plt


# ========== 2. API configuration ==========
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


# ========== 3. Data retrieval ==========
def get_ohlc(symbol: str, period: str = "1y") -> dict:
    params = {"range": period, "interval": "1d"}
    response = None

    for base_url in BASE_URLS:
        try:
            candidate = requests.get(
                base_url + symbol,
                params=params,
                headers=HEADERS,
                timeout=10,
            )
        except requests.RequestException:
            continue

        if candidate.status_code == 200:
            response = candidate
            break

    if response is None:
        raise ValueError(
            f"No data found for '{symbol}'. Check the symbol."
        )

    try:
        chart = response.json()["chart"]

        if chart.get("error"):
            raise ValueError(chart["error"])

        results = chart["result"]
        if not results:
            raise ValueError("Empty result")

        result = results[0]
        quote = result["indicators"]["quote"][0]
        timestamps = result.get("timestamp", [])

        valid_rows = [
            (o, h, l, t)
            for o, h, l, t in zip(
                quote["open"],
                quote["high"],
                quote["low"],
                timestamps,
            )
            if o is not None
            and h is not None
            and l is not None
            and t is not None
            and all(math.isfinite(x) for x in (o, h, l))
            and o > 0
            and h >= l > 0
        ]

        if len(valid_rows) < 20:
            raise ValueError("At least 20 usable daily OHLC observations are required")

        return {
            "symbol": result["meta"].get("symbol", symbol),
            "currency": result["meta"].get("currency", ""),
            "open": valid_rows[-1][0],
            "log_high_low": np.array(
                [math.log(row[1] / row[2]) for row in valid_rows]
            ),
            "from": datetime.fromtimestamp(
                valid_rows[0][3], tz=timezone.utc
            ).strftime("%Y-%m-%d"),
            "to": datetime.fromtimestamp(
                valid_rows[-1][3], tz=timezone.utc
            ).strftime("%Y-%m-%d"),
        }

    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(
            f"Yahoo returned no usable price data for '{symbol}'."
        ) from error


# ========== 4. Fundamental data helpers ==========
def get_fundamentals(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)

    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    def valid_number(value):
        if isinstance(value, (int, float)) and math.isfinite(value):
            return value
        return None

    return {
        "current_price": valid_number(
            info.get("regularMarketPrice")
            or info.get("currentPrice")
        ),
        "dividend_per_share": valid_number(
            info.get("dividendRate")
        ),
        "dividend_yield": valid_number(
            info.get("dividendYield")
        ),
        "pe_ratio": valid_number(
            info.get("trailingPE")
        ),
        "eps": valid_number(
            info.get("trailingEps")
        ),
    }


# ========== 5. Input and formatting utilities ==========
def get_input(prompt, default, cast=float, min_value=None):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()

        if not raw:
            return default

        try:
            value = cast(raw)
        except ValueError:
            print("  Invalid input, please enter a number.")
            continue

        if min_value is not None and value <= min_value:
            print(f"  Value must be greater than {min_value}.")
            continue

        return value


def display_value(value, suffix=""):
    if value is None:
        return "Unavailable"
    return f"{value:,.2f}{suffix}"


# ========== 6. Monte Carlo simulation and reporting ==========
def main() -> None:
    print("Examples: RELIANCE.NS (NSE), RELIANCE.BO (BSE), AAPL, TSLA")

    while True:
        symbol = input(
            "\nEnter stock symbol (or 'q' to quit): "
        ).strip().upper()

        if symbol == "Q":
            print("Goodbye!")
            return

        if not symbol:
            print("Please enter a symbol.")
            continue

        try:
            stock = get_ohlc(symbol)
        except ValueError as error:
            print(error)
            continue

        fundamentals = get_fundamentals(symbol)

        initial_price = (
            fundamentals["current_price"]
            or stock["open"]
        )
        sigma = math.sqrt(
            252 * np.mean(stock["log_high_low"] ** 2) / (4 * math.log(2))
        )

        years = 1
        steps = 360
        simulations = 10_000
        seed = 42

        expected_return = 0.05
        if (
            fundamentals["dividend_per_share"] is not None
            and fundamentals["eps"] is not None
            and fundamentals["pe_ratio"] is not None
            and fundamentals["current_price"] not in (None, 0)
        ):
            expected_return = (
                (
                    fundamentals["dividend_per_share"]
                    + fundamentals["eps"] * fundamentals["pe_ratio"]
                )
                / fundamentals["current_price"]
            ) - 1

        dt = years / steps
        rng = np.random.default_rng(seed)
        shocks = rng.standard_normal((steps, simulations))

        prices = np.empty((steps + 1, simulations))
        prices[0] = initial_price

        for step in range(steps):
            prices[step + 1] = prices[step] * np.exp(
                (expected_return - 0.5 * sigma**2) * dt
                + sigma * math.sqrt(dt) * shocks[step]
            )

        final_prices = prices[-1]

        print("\n========== STOCK INFORMATION ==========")
        print("Symbol:", stock["symbol"])
        print("Currency:", stock["currency"])
        print("Historical range:", stock["from"], "to", stock["to"])
        print(
            "Current market price:",
            display_value(fundamentals["current_price"])
        )
        print(
            "Annual dividend per share:",
            display_value(fundamentals["dividend_per_share"])
        )

        dividend_yield = fundamentals["dividend_yield"]
        print(
            "Dividend yield:",
            f"{dividend_yield:.2%}"
            if dividend_yield is not None
            else "Unavailable",
        )

        print(
            "Trailing P/E ratio:",
            display_value(fundamentals["pe_ratio"])
        )
        print(
            "Trailing EPS:",
            display_value(fundamentals["eps"])
        )

        print("\n========== MONTE CARLO RESULTS ==========")
        print("Initial simulation price:", display_value(initial_price))
        print(f"Estimated annual volatility: {sigma:.2%}")
        print("Number of simulations:", simulations)
        print("Simulation horizon:", years, "year")

        print(
            "Mean final price:",
            display_value(np.mean(final_prices))
        )
        print(
            "Median final price:",
            display_value(np.median(final_prices))
        )
        print(
            "Minimum final price:",
            display_value(np.min(final_prices))
        )
        print(
            "Maximum final price:",
            display_value(np.max(final_prices))
        )
        print(
            "Probability of finishing below initial price:",
            f"{np.mean(final_prices < initial_price):.2%}",
        )

        # ========== 7.Graphical Representation ==========
        time_points = np.linspace(0, years, steps + 1)
        plt.figure(figsize=(12, 6))

        for simulation in range(min(50, simulations)):
            plt.plot(
                time_points,
                prices[:, simulation],
                linewidth=0.8,
                label="Simulated paths" if simulation == 0 else None,
            )

        plt.axhline(
            initial_price,
            linestyle="--",
            label="Initial price",
        )
        plt.title("Monte Carlo Stock Price Paths")
        plt.xlabel("Time (years)")
        plt.ylabel(f"Stock price ({stock['currency']})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.figure(figsize=(12, 6))
        plt.hist(final_prices, bins=50, edgecolor="black", alpha=0.75)
        plt.axvline(
            initial_price,
            color="red",
            linestyle="--",
            label="Initial price",
        )
        plt.title("Distribution of Final Simulated Prices")
        plt.xlabel(f"Final stock price ({stock['currency']})")
        plt.ylabel("Number of simulations")
        plt.legend()
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()