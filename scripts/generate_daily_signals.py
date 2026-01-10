#!/usr/bin/env python3
"""Daily signal generation script for Bucket A (Safe S&P) and Bucket B (Top-Cap Crypto).

This script:
1. Loads data for both equity and crypto universes
2. Calculates features/indicators
3. Generates signals for both buckets
4. Outputs signals with rationale tags for newsletter consumption
5. Saves signals to JSON for downstream processing

Usage:
    python scripts/generate_daily_signals.py --date 2024-01-15
    python scripts/generate_daily_signals.py  # Uses latest available date
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from trading_system.configs.strategy_config import load_strategy_config
from trading_system.data import load_all_data, load_benchmark, select_equity_universe, select_top_crypto_by_volume
from trading_system.features.calculator import FeatureCalculator
from trading_system.models.signals import Signal
from trading_system.strategies import create_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_bucket_configs() -> tuple[dict, dict]:
    """Load configuration files for both buckets.

    Returns:
        Tuple of (bucket_a_config, bucket_b_config)
    """
    config_dir = Path(__file__).parent.parent / "configs"

    bucket_a_path = config_dir / "bucket_a_safe_sp.yaml"
    bucket_b_path = config_dir / "bucket_b_topcat_crypto.yaml"

    if not bucket_a_path.exists():
        raise FileNotFoundError(f"Bucket A config not found: {bucket_a_path}")
    if not bucket_b_path.exists():
        raise FileNotFoundError(f"Bucket B config not found: {bucket_b_path}")

    bucket_a_config = load_strategy_config(str(bucket_a_path))
    bucket_b_config = load_strategy_config(str(bucket_b_path))

    return bucket_a_config, bucket_b_config


def generate_signals_for_bucket(
    strategy_config: dict,
    universe_data: Dict[str, pd.DataFrame],
    benchmark_data: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> List[Signal]:
    """Generate signals for a single bucket.

    Args:
        strategy_config: Strategy configuration
        universe_data: Dictionary of symbol -> OHLCV DataFrame
        benchmark_data: Benchmark OHLCV DataFrame
        reference_date: Date to generate signals for (uses latest if None)

    Returns:
        List of Signal objects
    """
    # Create strategy instance
    strategy = create_strategy(strategy_config)

    # Calculate features for all symbols
    feature_calc = FeatureCalculator(strategy_config)

    signals = []

    for symbol, ohlcv_data in universe_data.items():
        if ohlcv_data.empty:
            logger.debug(f"Skipping {symbol}: empty data")
            continue

        # Calculate features
        features_df = feature_calc.calculate_features(
            ohlcv_data=ohlcv_data,
            benchmark_data=benchmark_data,
            symbol=symbol,
        )

        if features_df.empty:
            logger.debug(f"Skipping {symbol}: no features calculated")
            continue

        # Get latest features (or features at reference_date)
        if reference_date is not None:
            available_dates = features_df.index[features_df.index <= reference_date]
            if len(available_dates) == 0:
                logger.debug(f"Skipping {symbol}: no data at reference_date {reference_date}")
                continue
            latest_date = available_dates[-1]
        else:
            latest_date = features_df.index[-1]

        features = features_df.loc[latest_date]

        # Estimate order notional (placeholder - should come from position sizer)
        # For now, assume $10k per position
        order_notional = 10000.0

        # Generate signal
        signal = strategy.generate_signal(
            symbol=symbol,
            features=features,
            order_notional=order_notional,
            diversification_bonus=0.0,
        )

        if signal is not None:
            signals.append(signal)
            logger.info(f"Generated signal for {symbol}: {signal.trigger_reason}")

    return signals


def format_signals_for_output(signals: List[Signal]) -> List[dict]:
    """Format signals for JSON output.

    Args:
        signals: List of Signal objects

    Returns:
        List of signal dictionaries
    """
    output = []

    for signal in signals:
        signal_dict = {
            "symbol": signal.symbol,
            "asset_class": signal.asset_class,
            "date": signal.date.isoformat() if hasattr(signal.date, "isoformat") else str(signal.date),
            "side": signal.side.value if hasattr(signal.side, "value") else str(signal.side),
            "trigger_reason": signal.trigger_reason,
            "entry_price": float(signal.entry_price) if signal.entry_price is not None else None,
            "stop_price": float(signal.stop_price) if signal.stop_price is not None else None,
            "score": float(signal.score) if signal.score is not None else 0.0,
            "urgency": float(signal.urgency) if signal.urgency is not None else 0.5,
            "bucket": signal.metadata.get("bucket", "unknown") if signal.metadata else "unknown",
            "rationale_tags": signal.metadata.get("rationale_tags", []) if signal.metadata else [],
            "breakout_type": signal.metadata.get("breakout_type") if signal.metadata else None,
            "breakout_clearance": signal.metadata.get("breakout_clearance") if signal.metadata else None,
            "momentum_strength": signal.metadata.get("momentum_strength") if signal.metadata else None,
            "volatility_adjustment": signal.metadata.get("volatility_adjustment") if signal.metadata else None,
            "capacity_passed": signal.capacity_passed,
            "passed_eligibility": signal.passed_eligibility,
        }

        output.append(signal_dict)

    return output


def main():
    parser = argparse.ArgumentParser(description="Generate daily signals for both buckets")
    parser.add_argument(
        "--date",
        type=str,
        help="Date to generate signals for (YYYY-MM-DD). Uses latest if not specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/daily_signals",
        help="Output directory for signal files",
    )
    parser.add_argument(
        "--equity-data-dir",
        type=str,
        default="data/equity/daily",
        help="Directory containing equity OHLCV data",
    )
    parser.add_argument(
        "--crypto-data-dir",
        type=str,
        default="data/crypto/daily",
        help="Directory containing crypto OHLCV data",
    )

    args = parser.parse_args()

    # Parse reference date
    reference_date = None
    if args.date:
        try:
            reference_date = pd.Timestamp(args.date)
            logger.info(f"Generating signals for date: {reference_date.date()}")
        except Exception as e:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD. Error: {e}")
            return 1
    else:
        logger.info("Generating signals for latest available date")

    # Load bucket configurations
    logger.info("Loading bucket configurations...")
    try:
        bucket_a_config, bucket_b_config = load_bucket_configs()
    except Exception as e:
        logger.error(f"Failed to load bucket configurations: {e}")
        return 1

    # ===== BUCKET A: Safe S&P =====
    logger.info("\n" + "=" * 60)
    logger.info("BUCKET A: Safe S&P")
    logger.info("=" * 60)

    try:
        # Load equity data
        logger.info("Loading equity data...")
        equity_data_path = Path(args.equity_data_dir)
        if not equity_data_path.exists():
            logger.warning(f"Equity data directory not found: {equity_data_path}")
            logger.warning("Skipping Bucket A (Safe S&P)")
            bucket_a_signals = []
        else:
            equity_data = load_all_data(str(equity_data_path), asset_class="equity")

            # Select SP500 universe
            sp500_universe = select_equity_universe(
                universe_type="SP500",
                available_data=equity_data,
                min_bars=200,
            )
            logger.info(f"Selected {len(sp500_universe)} symbols from SP500 universe")

            # Filter data to universe
            universe_data = {sym: equity_data[sym] for sym in sp500_universe if sym in equity_data}

            # Load SPY benchmark
            spy_data = load_benchmark("SPY", str(equity_data_path))

            # Generate signals
            bucket_a_signals = generate_signals_for_bucket(
                strategy_config=bucket_a_config,
                universe_data=universe_data,
                benchmark_data=spy_data,
                reference_date=reference_date,
            )

            logger.info(f"Generated {len(bucket_a_signals)} signals for Bucket A")

    except Exception as e:
        logger.error(f"Error generating Bucket A signals: {e}", exc_info=True)
        bucket_a_signals = []

    # ===== BUCKET B: Top-Cap Crypto =====
    logger.info("\n" + "=" * 60)
    logger.info("BUCKET B: Top-Cap Crypto")
    logger.info("=" * 60)

    try:
        # Load crypto data
        logger.info("Loading crypto data...")
        crypto_data_path = Path(args.crypto_data_dir)
        if not crypto_data_path.exists():
            logger.warning(f"Crypto data directory not found: {crypto_data_path}")
            logger.warning("Skipping Bucket B (Top-Cap Crypto)")
            bucket_b_signals = []
        else:
            crypto_data = load_all_data(str(crypto_data_path), asset_class="crypto")

            # Select top 10 crypto by volume
            top_crypto = select_top_crypto_by_volume(
                available_data=crypto_data,
                top_n=10,
                lookback_days=30,
                reference_date=reference_date,
            )
            logger.info(f"Selected top {len(top_crypto)} crypto: {top_crypto}")

            # Filter data to universe
            universe_data = {sym: crypto_data[sym] for sym in top_crypto if sym in crypto_data}

            # Load BTC benchmark
            btc_data = load_benchmark("BTC", str(crypto_data_path))

            # Generate signals
            bucket_b_signals = generate_signals_for_bucket(
                strategy_config=bucket_b_config,
                universe_data=universe_data,
                benchmark_data=btc_data,
                reference_date=reference_date,
            )

            logger.info(f"Generated {len(bucket_b_signals)} signals for Bucket B")

    except Exception as e:
        logger.error(f"Error generating Bucket B signals: {e}", exc_info=True)
        bucket_b_signals = []

    # ===== OUTPUT RESULTS =====
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    total_signals = len(bucket_a_signals) + len(bucket_b_signals)
    logger.info(f"Total signals generated: {total_signals}")
    logger.info(f"  - Bucket A (Safe S&P): {len(bucket_a_signals)}")
    logger.info(f"  - Bucket B (Top-Cap Crypto): {len(bucket_b_signals)}")

    # Format signals for output
    all_signals = {
        "bucket_a": format_signals_for_output(bucket_a_signals),
        "bucket_b": format_signals_for_output(bucket_b_signals),
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "reference_date": reference_date.isoformat() if reference_date else None,
            "total_signals": total_signals,
        },
    }

    # Save to file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"daily_signals_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(all_signals, f, indent=2)

    logger.info(f"\nSignals saved to: {output_file}")

    # Print summary
    if bucket_a_signals:
        logger.info("\nBucket A signals:")
        for signal in bucket_a_signals[:5]:  # Show first 5
            tags = signal.metadata.get("rationale_tags", []) if signal.metadata else []
            logger.info(f"  {signal.symbol}: {signal.trigger_reason} | Tags: {', '.join(tags)}")
        if len(bucket_a_signals) > 5:
            logger.info(f"  ... and {len(bucket_a_signals) - 5} more")

    if bucket_b_signals:
        logger.info("\nBucket B signals:")
        for signal in bucket_b_signals[:5]:  # Show first 5
            tags = signal.metadata.get("rationale_tags", []) if signal.metadata else []
            logger.info(f"  {signal.symbol}: {signal.trigger_reason} | Tags: {', '.join(tags)}")
        if len(bucket_b_signals) > 5:
            logger.info(f"  ... and {len(bucket_b_signals) - 5} more")

    return 0


if __name__ == "__main__":
    exit(main())
