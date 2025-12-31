#!/usr/bin/env python3
"""Download real historical market data for testing.

This script downloads actual historical market data from yfinance to test
the system with real-world data quality issues and different market conditions.

Usage:
    python scripts/download_real_market_data.py --output data/real_market_data/
    python scripts/download_real_market_data.py --symbols AAPL MSFT GOOGL --start 2020-01-01 --end 2024-01-01
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance is required. Install with: pip install yfinance")
    sys.exit(1)


def download_equity_data(
    symbols: List[str],
    output_dir: Path,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> None:
    """Download equity OHLCV data from yfinance.

    Args:
        symbols: List of equity symbols to download
        output_dir: Directory to save CSV files
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading equity data for {len(symbols)} symbols...")
    print(f"Date range: {start_date} to {end_date or 'today'}")

    for symbol in symbols:
        try:
            print(f"  Downloading {symbol}...", end=" ", flush=True)

            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                print(f"  WARNING: No data for {symbol}")
                continue

            # Convert to standard format
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["Date"]).dt.date
            df = df[["date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]

            # Compute dollar volume
            df["dollar_volume"] = df["close"] * df["volume"]

            # Sort by date
            df = df.sort_values("date")

            # Save to CSV
            output_file = output_dir / f"{symbol}.csv"
            df.to_csv(output_file, index=False)
            print(f"✓ ({len(df)} rows)")

        except Exception as e:
            print(f"✗ ERROR: {e}")


def download_crypto_data(
    symbols: List[str],
    output_dir: Path,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> None:
    """Download crypto OHLCV data from yfinance.

    Args:
        symbols: List of crypto symbols (e.g., BTC-USD, ETH-USD)
        output_dir: Directory to save CSV files
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading crypto data for {len(symbols)} symbols...")
    print(f"Date range: {start_date} to {end_date or 'today'}")

    for symbol in symbols:
        try:
            # yfinance uses format like BTC-USD for crypto
            yf_symbol = symbol if "-" in symbol else f"{symbol}-USD"
            print(f"  Downloading {yf_symbol}...", end=" ", flush=True)

            # Download data
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                print(f"  WARNING: No data for {yf_symbol}")
                continue

            # Convert to standard format
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["Date"]).dt.date
            df = df[["date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]

            # Compute dollar volume
            df["dollar_volume"] = df["close"] * df["volume"]

            # Sort by date
            df = df.sort_values("date")

            # Save to CSV (use base symbol name, not yf_symbol)
            base_symbol = symbol.split("-")[0] if "-" in symbol else symbol
            output_file = output_dir / f"{base_symbol}.csv"
            df.to_csv(output_file, index=False)
            print(f"✓ ({len(df)} rows)")

        except Exception as e:
            print(f"✗ ERROR: {e}")


def download_benchmark_data(
    symbols: List[str],
    output_dir: Path,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> None:
    """Download benchmark OHLCV data (SPY, BTC).

    Args:
        symbols: List of benchmark symbols
        output_dir: Directory to save CSV files
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading benchmark data for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            # For BTC, use BTC-USD format
            yf_symbol = symbol if symbol != "BTC" else "BTC-USD"
            print(f"  Downloading {yf_symbol}...", end=" ", flush=True)

            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                print(f"  WARNING: No data for {yf_symbol}")
                continue

            # Convert to standard format
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["Date"]).dt.date
            df = df[["date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]

            # Compute dollar volume
            df["dollar_volume"] = df["close"] * df["volume"]

            # Sort by date
            df = df.sort_values("date")

            # Save to CSV
            output_file = output_dir / f"{symbol}.csv"
            df.to_csv(output_file, index=False)
            print(f"✓ ({len(df)} rows)")

        except Exception as e:
            print(f"✗ ERROR: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download real market data for testing")
    parser.add_argument(
        "--output",
        type=str,
        default="data/real_market_data",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--equity-symbols",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        help="Equity symbols to download",
    )
    parser.add_argument(
        "--crypto-symbols",
        nargs="+",
        default=["BTC", "ETH", "SOL"],
        help="Crypto symbols to download (without -USD suffix)",
    )
    parser.add_argument(
        "--benchmark-symbols",
        nargs="+",
        default=["SPY", "BTC"],
        help="Benchmark symbols to download",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--equity-only",
        action="store_true",
        help="Download only equity data",
    )
    parser.add_argument(
        "--crypto-only",
        action="store_true",
        help="Download only crypto data",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    # Download equity data
    if not args.crypto_only:
        equity_dir = output_path / "equity" / "ohlcv"
        download_equity_data(
            args.equity_symbols,
            equity_dir,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    # Download crypto data
    if not args.equity_only:
        crypto_dir = output_path / "crypto" / "ohlcv"
        download_crypto_data(
            args.crypto_symbols,
            crypto_dir,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    # Download benchmark data
    if not args.equity_only and not args.crypto_only:
        benchmark_dir = output_path / "benchmarks"
        download_benchmark_data(
            args.benchmark_symbols,
            benchmark_dir,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    print("\n✓ Download complete!")
    print(f"Data saved to: {output_path}")


if __name__ == "__main__":
    main()


