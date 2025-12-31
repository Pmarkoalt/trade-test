#!/usr/bin/env python3
"""Download market data from Alpha Vantage or Massive (formerly Polygon.io) APIs.

This script downloads OHLCV data for equities and crypto, saving to CSV format
compatible with the trading system's data loader.

Usage:
    # Alpha Vantage (free tier: 25 calls/day, compact data only)
    python scripts/download_data.py --source alphavantage --preset small --compact

    # Massive (free tier: 5 calls/min, full historical data)
    python scripts/download_data.py --source massive --preset small
    python scripts/download_data.py --source massive --preset large --years 2

    # Download specific symbols
    python scripts/download_data.py --source massive --symbols AAPL MSFT GOOGL

    # Download crypto
    python scripts/download_data.py --source massive --preset crypto

Rate limits:
    - Alpha Vantage free: 25 calls/day, 5 calls/min (full data is premium)
    - Massive free: 5 calls/min (full historical data available)
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests

# Try to load dotenv if available (for local development)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # Running in Docker with env_file, dotenv not needed

# Directory structure
DATA_DIR = Path(__file__).parent.parent / "data"
EQUITY_DIR = DATA_DIR / "equity" / "daily"
CRYPTO_DIR = DATA_DIR / "crypto" / "daily"
UNIVERSE_DIR = DATA_DIR / "equity" / "universe"
BENCHMARK_DIR = DATA_DIR / "test_benchmarks"  # Use test_ prefix for reduced min days

# Preset symbol lists
PRESETS = {
    "small": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "medium": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "COST"],
    "large": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "AMD",
        "NFLX",
        "COST",
        "ADBE",
        "CRM",
        "INTC",
        "QCOM",
        "AVGO",
        "TXN",
        "SBUX",
        "PYPL",
        "GILD",
        "MRNA",
    ],
    "crypto": ["BTC", "ETH", "SOL"],
}


# =============================================================================
# Alpha Vantage Functions
# =============================================================================


def fetch_alphavantage_equity(symbol: str, api_key: str, outputsize: str = "compact") -> Optional[pd.DataFrame]:
    """Fetch daily equity data from Alpha Vantage."""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": outputsize,
        "datatype": "csv",
    }

    try:
        print(f"  Fetching {symbol}...", end=" ", flush=True)
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()

        text = response.text
        if "Error Message" in text or text.strip().startswith("{"):
            print(f"API Error: {text[:150]}")
            return None

        df = pd.read_csv(StringIO(text))
        if df.empty or "timestamp" not in df.columns:
            print("Invalid response")
            return None

        df = df.rename(columns={"timestamp": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df[["date", "open", "high", "low", "close", "volume"]]

        print(f"OK ({len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()})")
        return df

    except Exception as e:
        print(f"Error: {e}")
        return None


def fetch_alphavantage_crypto(symbol: str, api_key: str) -> Optional[pd.DataFrame]:
    """Fetch daily crypto data from Alpha Vantage."""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": symbol,
        "market": "USD",
        "apikey": api_key,
        "datatype": "csv",
    }

    try:
        print(f"  Fetching {symbol} (crypto)...", end=" ", flush=True)
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()

        text = response.text
        if "Error Message" in text or text.strip().startswith("{"):
            print(f"API Error: {text[:150]}")
            return None

        df = pd.read_csv(StringIO(text))
        if df.empty or "timestamp" not in df.columns:
            print("Invalid response")
            return None

        df = df.rename(
            columns={
                "timestamp": "date",
                "open (USD)": "open",
                "high (USD)": "high",
                "low (USD)": "low",
                "close (USD)": "close",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df[["date", "open", "high", "low", "close", "volume"]]

        print(f"OK ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"Error: {e}")
        return None


# =============================================================================
# Massive (formerly Polygon.io) Functions
# =============================================================================


def fetch_massive_equity(symbol: str, api_key: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch daily equity data from Massive (Polygon.io) API.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        api_key: Massive API key
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with OHLCV data or None on error
    """
    # Massive uses the same API structure as Polygon.io
    base_url = "https://api.polygon.io/v2"
    url = f"{base_url}/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"

    params = {
        "apiKey": api_key,
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
    }

    try:
        print(f"  Fetching {symbol}...", end=" ", flush=True)
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Accept both "OK" and "DELAYED" status (free tier returns DELAYED)
        if data.get("status") not in ("OK", "DELAYED"):
            error_msg = data.get("error", data.get("message", f"Status: {data.get('status')}"))
            print(f"API Error: {error_msg}")
            return None

        results = data.get("results", [])
        if not results:
            print("No data returned")
            return None

        # Convert to DataFrame
        records = []
        for bar in results:
            # Convert timestamp to date only (strip time component)
            ts = pd.Timestamp(bar["t"], unit="ms")
            records.append(
                {
                    "date": ts.normalize(),  # Normalize to midnight
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "volume": bar["v"],
                }
            )

        df = pd.DataFrame(records)
        # Convert to date strings without time
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df = df.sort_values("date").reset_index(drop=True)

        print(f"OK ({len(df)} rows, {df['date'].min()} to {df['date'].max()})")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def fetch_massive_crypto(symbol: str, api_key: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch daily crypto data from Massive (Polygon.io) API.

    Args:
        symbol: Crypto symbol (e.g., 'BTC' - will be converted to X:BTCUSD)
        api_key: Massive API key
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with OHLCV data or None on error
    """
    # Massive/Polygon uses X:BTCUSD format for crypto
    ticker = f"X:{symbol}USD"
    base_url = "https://api.polygon.io/v2"
    url = f"{base_url}/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"

    params = {
        "apiKey": api_key,
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
    }

    try:
        print(f"  Fetching {symbol} (crypto)...", end=" ", flush=True)
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Accept both "OK" and "DELAYED" status (free tier returns DELAYED)
        if data.get("status") not in ("OK", "DELAYED"):
            error_msg = data.get("error", data.get("message", f"Status: {data.get('status')}"))
            print(f"API Error: {error_msg}")
            return None

        results = data.get("results", [])
        if not results:
            print("No data returned")
            return None

        # Convert to DataFrame
        records = []
        for bar in results:
            # Convert timestamp to date only (strip time component)
            ts = pd.Timestamp(bar["t"], unit="ms")
            records.append(
                {
                    "date": ts.normalize(),  # Normalize to midnight
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "volume": bar["v"],
                }
            )

        df = pd.DataFrame(records)
        # Convert to date strings without time
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df = df.sort_values("date").reset_index(drop=True)

        print(f"OK ({len(df)} rows, {df['date'].min()} to {df['date'].max()})")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


# =============================================================================
# Common Functions
# =============================================================================


def save_data(df: pd.DataFrame, filepath: Path) -> bool:
    """Save DataFrame to CSV."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        return True
    except Exception as e:
        print(f"  Error saving {filepath}: {e}")
        return False


def create_universe_file(symbols: List[str], name: str = "custom") -> Path:
    """Create universe CSV file."""
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    filepath = UNIVERSE_DIR / f"{name}.csv"
    df = pd.DataFrame({"symbol": symbols})
    df.to_csv(filepath, index=False)
    print(f"Created universe file: {filepath}")
    return filepath


def download_symbols(
    symbols: List[str],
    source: str,
    api_key: str,
    output_dir: Path,
    is_crypto: bool = False,
    rate_limit_delay: float = 12.5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    outputsize: str = "compact",
) -> Tuple[List[str], List[str]]:
    """Download data for multiple symbols."""
    successful = []
    failed = []

    for i, symbol in enumerate(symbols):
        if i > 0:
            print(f"  Rate limit delay ({rate_limit_delay}s)...")
            time.sleep(rate_limit_delay)

        df = None
        if source == "alphavantage":
            if is_crypto:
                df = fetch_alphavantage_crypto(symbol, api_key)
            else:
                df = fetch_alphavantage_equity(symbol, api_key, outputsize)
        elif source == "massive":
            if is_crypto:
                df = fetch_massive_crypto(symbol, api_key, start_date or "", end_date or "")
            else:
                df = fetch_massive_equity(symbol, api_key, start_date or "", end_date or "")

        if df is not None and not df.empty:
            filepath = output_dir / f"{symbol}.csv"
            if save_data(df, filepath):
                successful.append(symbol)
            else:
                failed.append(symbol)
        else:
            failed.append(symbol)

    return successful, failed


def main():
    parser = argparse.ArgumentParser(description="Download market data from Alpha Vantage or Massive APIs")
    parser.add_argument(
        "--source", choices=["alphavantage", "massive"], default="massive", help="Data source (default: massive)"
    )
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to download")
    parser.add_argument("--preset", choices=["small", "medium", "large", "crypto"], help="Use preset symbol list")
    parser.add_argument("--benchmark", action="store_true", help="Download SPY/BTC benchmarks only")
    parser.add_argument("--compact", action="store_true", help="Download compact data (Alpha Vantage: 100 days)")
    parser.add_argument("--years", type=int, default=2, help="Years of historical data to download (Massive only, default: 2)")
    parser.add_argument("--universe-name", default="custom", help="Name for universe file")

    args = parser.parse_args()

    # Get API key based on source
    if args.source == "alphavantage":
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            print("Error: ALPHA_VANTAGE_API_KEY not found in environment")
            print("Add it to .env file: ALPHA_VANTAGE_API_KEY=your_key_here")
            sys.exit(1)
        rate_limit_delay = 12.5  # 5 calls/min
        source_name = "Alpha Vantage"
    else:  # massive
        api_key = os.getenv("MASSIVE_API_KEY")
        if not api_key:
            print("Error: MASSIVE_API_KEY not found in environment")
            print("Add it to .env file: MASSIVE_API_KEY=your_key_here")
            sys.exit(1)
        rate_limit_delay = 12.5  # 5 calls/min on free tier
        source_name = "Massive (Polygon.io)"

    # Calculate date range for Massive
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.years * 365)).strftime("%Y-%m-%d")

    print(f"{source_name} Data Downloader")
    print(f"API Key: {api_key[:8]}...")
    print(f"Rate limit: {rate_limit_delay}s between calls")
    if args.source == "massive":
        print(f"Date range: {start_date} to {end_date} ({args.years} years)")
    print()

    outputsize = "compact" if args.compact else "full"

    # Download benchmarks only
    if args.benchmark:
        print("Downloading benchmarks (SPY, BTC)...")
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

        if args.source == "alphavantage":
            df = fetch_alphavantage_equity("SPY", api_key, outputsize)
        else:
            df = fetch_massive_equity("SPY", api_key, start_date, end_date)

        if df is not None:
            save_data(df, BENCHMARK_DIR / "SPY.csv")

        time.sleep(rate_limit_delay)

        if args.source == "alphavantage":
            df = fetch_alphavantage_crypto("BTC", api_key)
        else:
            df = fetch_massive_crypto("BTC", api_key, start_date, end_date)

        if df is not None:
            save_data(df, BENCHMARK_DIR / "BTC.csv")

        print("Benchmark download complete!")
        return

    # Determine symbols to download
    if args.symbols:
        symbols = args.symbols
        universe_name = args.universe_name
    elif args.preset:
        symbols = PRESETS[args.preset]
        universe_name = args.preset
    else:
        symbols = PRESETS["small"]
        universe_name = "small"

    is_crypto = args.preset == "crypto"
    output_dir = CRYPTO_DIR if is_crypto else EQUITY_DIR

    print(f"Symbols to download: {symbols}")
    print(f"Output directory: {output_dir}")
    print(f"This will use {len(symbols) + 2} API calls (including SPY + BTC benchmarks)")
    print()

    # Download equity/crypto data
    print(f"Downloading {'crypto' if is_crypto else 'equity'} data...")
    successful, failed = download_symbols(
        symbols=symbols,
        source=args.source,
        api_key=api_key,
        output_dir=output_dir,
        is_crypto=is_crypto,
        rate_limit_delay=rate_limit_delay,
        start_date=start_date,
        end_date=end_date,
        outputsize=outputsize,
    )

    # Download benchmarks
    if successful:
        print()
        print("Downloading benchmarks (SPY, BTC)...")
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

        time.sleep(rate_limit_delay)
        if args.source == "alphavantage":
            df = fetch_alphavantage_equity("SPY", api_key, outputsize)
        else:
            df = fetch_massive_equity("SPY", api_key, start_date, end_date)
        if df is not None:
            save_data(df, BENCHMARK_DIR / "SPY.csv")

        time.sleep(rate_limit_delay)
        if args.source == "alphavantage":
            df = fetch_alphavantage_crypto("BTC", api_key)
        else:
            df = fetch_massive_crypto("BTC", api_key, start_date, end_date)
        if df is not None:
            save_data(df, BENCHMARK_DIR / "BTC.csv")

    # Create universe file
    if successful:
        print()
        create_universe_file(successful, universe_name)

    # Summary
    print()
    print("=" * 50)
    print("Download Summary")
    print("=" * 50)
    print(f"Source: {source_name}")
    print(f"Successful: {len(successful)} symbols")
    if successful:
        print(f"  {', '.join(successful)}")
    print(f"Failed: {len(failed)} symbols")
    if failed:
        print(f"  {', '.join(failed)}")
    print()
    print(f"Data saved to: {output_dir}")
    print(f"Universe file: {UNIVERSE_DIR / f'{universe_name}.csv'}")
    print(f"Benchmark files: {BENCHMARK_DIR}")


if __name__ == "__main__":
    main()
