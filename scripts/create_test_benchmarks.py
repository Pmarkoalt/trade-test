#!/usr/bin/env python3
"""Create test benchmark files with 250+ days of data."""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)


def create_spy_benchmark():
    """Create SPY benchmark with 252 days of data."""
    # Create 252 trading days
    dates = pd.date_range("2022-10-01", "2023-12-31", freq="D")
    dates = dates[dates.weekday < 5]  # Weekdays only
    dates = dates[:252]

    # Generate synthetic SPY data
    base_price = 380.0
    prices = []
    for i in range(len(dates)):
        change = np.random.normal(0.1, 2.0)
        if i == 0:
            price = base_price
        else:
            price = prices[-1] + change
        prices.append(max(price, 100.0))

    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close + abs(np.random.normal(0, 1))
        low = close - abs(np.random.normal(0, 1))
        open_price = prices[i - 1] if i > 0 else close
        volume = int(np.random.normal(80000000, 10000000))
        volume = max(volume, 1000000)

        data.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    output_path = Path("tests/fixtures/benchmarks/SPY.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} days of data")
    return df


def create_btc_benchmark():
    """Create BTC benchmark with 252 days of data."""
    # Create 252 days (crypto trades 24/7)
    dates = pd.date_range("2022-10-01", periods=252, freq="D")

    # Generate synthetic BTC data
    base_price = 30000.0
    prices = []
    for i in range(len(dates)):
        change = np.random.normal(50, 500)
        if i == 0:
            price = base_price
        else:
            price = prices[-1] + change
        prices.append(max(price, 10000.0))

    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close + abs(np.random.normal(0, 200))
        low = close - abs(np.random.normal(0, 200))
        open_price = prices[i - 1] if i > 0 else close
        volume = np.random.normal(1000000, 200000)
        volume = max(volume, 100000)

        data.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": int(volume),
            }
        )

    df = pd.DataFrame(data)
    output_path = Path("tests/fixtures/benchmarks/BTC.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} days of data")
    return df


if __name__ == "__main__":
    create_spy_benchmark()
    create_btc_benchmark()
    print("Benchmark files created successfully!")
