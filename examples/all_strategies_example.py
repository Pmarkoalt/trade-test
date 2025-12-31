"""
All Strategies Example

This example demonstrates how to use all available strategy types in the trading system:
1. Momentum (Equity)
2. Momentum (Crypto)
3. Mean Reversion
4. Multi-Timeframe
5. Factor-Based
6. Pairs Trading

Each strategy type has different entry/exit logic and is suited for different market conditions.

Usage:
    python examples/all_strategies_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import trading_system
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.configs.strategy_config import StrategyConfig  # noqa: E402
from trading_system.strategies.strategy_loader import load_strategy_from_config  # noqa: E402
from trading_system.strategies.strategy_registry import list_available_strategies  # noqa: E402


def demonstrate_all_strategies():
    """Demonstrate loading and configuring all strategy types."""
    print("=" * 70)
    print("All Strategies Example")
    print("=" * 70)
    print()
    print("This example shows how to load and configure all available strategy types.")
    print()

    # Show available strategies
    print("=" * 70)
    print("Available Strategy Types")
    print("=" * 70)
    print()
    available = list_available_strategies()
    for strategy_type, asset_class in available:
        print(f"  - {strategy_type.capitalize()} ({asset_class})")
    print()

    # Strategy configurations
    strategy_configs = {
        "Momentum (Equity)": {
            "path": "EXAMPLE_CONFIGS/equity_config.yaml",
            "description": "Breakout-based entries with trend filters (MA50)",
            "best_for": "Trending equities, NASDAQ-100, S&P 500",
            "key_features": [
                "Entry: 20D or 55D breakouts with clearance",
                "Exit: MA20 cross or hard stop (2.5x ATR)",
                "Trend filter: Must be above MA50 with positive slope",
            ],
        },
        "Momentum (Crypto)": {
            "path": "EXAMPLE_CONFIGS/crypto_config.yaml",
            "description": "Crypto momentum with staged exits and strict trend filter",
            "best_for": "Cryptocurrencies with clear trends",
            "key_features": [
                "Entry: 20D or 55D breakouts",
                "Exit: Staged (MA20 warning → tighten stop → MA50 exit)",
                "Trend filter: Must be above MA200 (strict)",
            ],
        },
        "Mean Reversion": {
            "path": "EXAMPLE_CONFIGS/mean_reversion_config.yaml",
            "description": "Z-score based entries for oversold conditions",
            "best_for": "Liquid ETFs (SPY, QQQ, etc.) with mean-reverting behavior",
            "key_features": [
                "Entry: Z-score < -2.0 (oversold)",
                "Exit: Z-score >= 0.0 (reverted to mean) or time stop (5 days)",
                "Stop: 2.0x ATR (fixed, no trailing)",
            ],
        },
        "Multi-Timeframe": {
            "path": "EXAMPLE_CONFIGS/multi_timeframe_config.yaml",
            "description": "Higher timeframe trend filter with lower timeframe entries",
            "best_for": "Trending equities with clear higher timeframe trends",
            "key_features": [
                "Entry: Price above MA50 (daily) AND weekly breakout",
                "Exit: Price breaks below MA50 (trend break)",
                "Hold period: Up to 60 days (longer for trend following)",
            ],
        },
        "Factor-Based": {
            "path": "EXAMPLE_CONFIGS/factor_config.yaml",
            "description": "Multi-factor ranking strategy (momentum, value, quality)",
            "best_for": "Large-cap equities with sufficient history",
            "key_features": [
                "Entry: Top 20% by composite factor score on rebalance day",
                "Exit: Not in top decile on rebalance or time stop (90 days)",
                "Rebalance: Monthly or quarterly",
            ],
        },
        "Pairs Trading": {
            "path": "EXAMPLE_CONFIGS/pairs_config.yaml",
            "description": "Spread-based strategy for correlated pairs",
            "best_for": "Highly correlated pairs (sector ETFs, index ETFs)",
            "key_features": [
                "Entry: Spread z-score > 2.0 (divergence)",
                "Exit: Spread z-score < 0.5 (convergence) or time stop (10 days)",
                "Pairs: Defined explicitly (e.g., XLE/XLK, GLD/TLT)",
            ],
        },
    }

    print("=" * 70)
    print("Strategy Configurations")
    print("=" * 70)
    print()

    for strategy_name, config_info in strategy_configs.items():
        print(f"Strategy: {strategy_name}")
        print(f"  Config: {config_info['path']}")
        print(f"  Description: {config_info['description']}")
        print(f"  Best for: {config_info['best_for']}")
        print("  Key features:")
        for feature in config_info["key_features"]:
            print(f"    - {feature}")
        print()

    print("=" * 70)
    print("Loading Strategy Configurations")
    print("=" * 70)
    print()

    loaded_strategies = {}

    for strategy_name, config_info in strategy_configs.items():
        config_path = config_info["path"]
        try:
            print(f"Loading {strategy_name}...")
            print(f"  Config file: {config_path}")

            # Load strategy config
            strategy_config = StrategyConfig.from_yaml(config_path)

            # Create strategy instance
            strategy = load_strategy_from_config(strategy_config)

            loaded_strategies[strategy_name] = {"config": strategy_config, "strategy": strategy}

            print("  ✓ Loaded successfully")
            print(f"    Name: {strategy_config.name}")
            print(f"    Asset class: {strategy_config.asset_class}")
            print(f"    Universe: {strategy_config.universe}")
            print(f"    Benchmark: {strategy_config.benchmark}")
            print()

        except FileNotFoundError:
            print(f"  ⚠️  Config file not found: {config_path}")
            print("    (This is OK if you haven't set up example configs)")
            print()
        except Exception as e:
            print(f"  ❌ Error loading strategy: {e}")
            print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Successfully loaded {len(loaded_strategies)} strategy configurations.")
    print()

    if loaded_strategies:
        print("Loaded strategies:")
        for name in loaded_strategies.keys():
            print(f"  ✓ {name}")
        print()

    print("=" * 70)
    print("How to Use These Strategies")
    print("=" * 70)
    print()
    print("1. **Single Strategy Backtest**:")
    print("   Create a run_config.yaml that references one strategy config:")
    print("   ```yaml")
    print("   strategies:")
    print("     equity:")
    print('       config_path: "EXAMPLE_CONFIGS/equity_config.yaml"')
    print("       enabled: true")
    print("   ```")
    print()
    print("2. **Multi-Strategy Backtest**:")
    print("   Reference multiple strategy configs in your run_config.yaml:")
    print("   ```yaml")
    print("   strategies:")
    print("     equity:")
    print('       config_path: "EXAMPLE_CONFIGS/equity_config.yaml"')
    print("       enabled: true")
    print("     mean_reversion:")
    print('       config_path: "EXAMPLE_CONFIGS/mean_reversion_config.yaml"')
    print("       enabled: true")
    print("   ```")
    print()
    print("3. **Programmatic Usage**:")
    print("   ```python")
    print("   from trading_system.strategies import load_strategy_from_config")
    print("   from trading_system.configs import StrategyConfig")
    print("   ")
    print("   config = StrategyConfig.from_yaml('EXAMPLE_CONFIGS/equity_config.yaml')")
    print("   strategy = load_strategy_from_config(config)")
    print("   ```")
    print()

    print("=" * 70)
    print("Example Configuration Files")
    print("=" * 70)
    print()
    print("All example configurations are in EXAMPLE_CONFIGS/:")
    print("  - equity_config.yaml          : Equity momentum strategy")
    print("  - crypto_config.yaml          : Crypto momentum strategy")
    print("  - mean_reversion_config.yaml  : Mean reversion strategy")
    print("  - multi_timeframe_config.yaml : Multi-timeframe strategy")
    print("  - factor_config.yaml          : Factor-based strategy")
    print("  - pairs_config.yaml           : Pairs trading strategy")
    print("  - run_config.yaml             : Main backtest run configuration")
    print()
    print("See EXAMPLE_CONFIGS/README.md for detailed documentation.")
    print()

    print("=" * 70)
    print("Next Steps")
    print("=" * 70)
    print()
    print("1. Review the configuration files to understand each strategy's parameters")
    print("2. Run backtests with different strategies to compare performance")
    print("3. Modify parameters in config files to optimize for your use case")
    print("4. Combine multiple strategies in a single backtest for diversification")
    print("5. Use the validation suite to test strategy robustness:")
    print("   python -m trading_system validate --config EXAMPLE_CONFIGS/run_config.yaml")
    print()

    return loaded_strategies


if __name__ == "__main__":
    try:
        demonstrate_all_strategies()
        print("✅ All strategies example completed successfully!")
        print()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
