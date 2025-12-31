"""
Custom Strategy Example

This example demonstrates how to create a custom trading strategy by
extending the StrategyInterface base class.

Usage:
    python examples/custom_strategy.py
"""

from pathlib import Path
import sys
from typing import List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.strategies.base.strategy_interface import StrategyInterface
from trading_system.configs.strategy_config import StrategyConfig
from trading_system.models.signals import Signal, SignalSide, SignalType, BreakoutType
from trading_system.models.features import FeatureRow
from trading_system.models.positions import Position, ExitReason
from trading_system.strategies.strategy_registry import register_strategy, create_strategy


class SimpleMovingAverageStrategy(StrategyInterface):
    """
    Example custom strategy: Simple Moving Average Crossover
    
    Entry: When price crosses above 20-day MA and 20-day MA > 50-day MA
    Exit: When price crosses below 20-day MA
    Stop: 2x ATR below entry price
    """
    
    def check_eligibility(self, features: FeatureRow) -> Tuple[bool, List[str]]:
        """Check if symbol is eligible for entry.
        
        For this simple strategy, we require:
        - Price above 50-day MA (uptrend)
        - Sufficient volume (ADV > 0)
        """
        failures = []
        
        # Check if we have required indicators
        if features.ma_50 is None or features.close is None:
            failures.append("Missing required indicators")
            return False, failures
        
        # Price must be above 50-day MA (uptrend)
        if features.close <= features.ma_50:
            failures.append("Price not above 50-day MA")
        
        # Check volume
        if features.adv_20 is None or features.adv_20 <= 0:
            failures.append("Insufficient volume")
        
        is_eligible = len(failures) == 0
        return is_eligible, failures
    
    def check_entry_triggers(
        self, features: FeatureRow
    ) -> Tuple[Optional[BreakoutType], float]:
        """Check if entry triggers are met.
        
        Entry trigger: Price crosses above 20-day MA AND 20-day MA > 50-day MA
        """
        # Check if we have required indicators
        if (features.ma_20 is None or features.ma_50 is None or 
            features.close is None or features.prev_close is None):
            return None, 0.0
        
        # Check MA crossover: price crosses above 20-day MA
        price_crossed_above = (features.prev_close <= features.ma_20 and 
                              features.close > features.ma_20)
        
        # Check trend: 20-day MA must be above 50-day MA
        ma_trend_ok = features.ma_20 > features.ma_50
        
        if price_crossed_above and ma_trend_ok:
            # Return a custom breakout type (or use existing)
            # For simplicity, we'll use BreakoutType.FAST
            return BreakoutType.FAST, 0.0  # No clearance needed for MA crossover
        
        return None, 0.0
    
    def check_exit_signals(
        self, position: Position, features: FeatureRow
    ) -> Optional[ExitReason]:
        """Check if exit signals are met.
        
        Exit: Price crosses below 20-day MA
        """
        if features.ma_20 is None or features.close is None or features.prev_close is None:
            return None
        
        # Check MA cross below
        if features.prev_close >= features.ma_20 and features.close < features.ma_20:
            return ExitReason.MA_CROSS
        
        # Check stop loss (handled by portfolio manager, but we can check here too)
        if position.stop_price is not None and features.close <= position.stop_price:
            return ExitReason.STOP_LOSS
        
        return None
    
    def compute_stop_price(
        self, entry_price: float, entry_date, features: FeatureRow
    ) -> Optional[float]:
        """Compute stop price for new position.
        
        Stop: 2x ATR below entry price
        """
        if features.atr_14 is None:
            return None
        
        stop_price = entry_price - (2.0 * features.atr_14)
        return max(stop_price, 0.0)  # Ensure non-negative
    
    def compute_target_price(
        self, entry_price: float, entry_date, features: FeatureRow
    ) -> Optional[float]:
        """Compute target price (optional - for profit targets).
        
        This strategy doesn't use profit targets, so return None.
        """
        return None


def example_register_custom_strategy():
    """Example: Register a custom strategy."""
    print("=" * 60)
    print("Example 1: Registering Custom Strategy")
    print("=" * 60)
    
    # Register the custom strategy
    print("\nRegistering SimpleMovingAverageStrategy...")
    register_strategy(
        strategy_type="sma_crossover",
        asset_class="equity",
        strategy_class=SimpleMovingAverageStrategy
    )
    
    print("Strategy registered successfully!")
    print("  Type: sma_crossover")
    print("  Asset class: equity")
    print("  Class: SimpleMovingAverageStrategy")
    
    return True


def example_create_custom_strategy_from_config():
    """Example: Create custom strategy instance from config."""
    print("\n" + "=" * 60)
    print("Example 2: Creating Custom Strategy from Config")
    print("=" * 60)
    
    # Create a minimal strategy config
    print("\nCreating strategy config...")
    config_dict = {
        "name": "sma_crossover_equity",
        "asset_class": "equity",
        "universe": ["AAPL", "MSFT", "GOOGL"],  # Small universe for testing
        "benchmark": "SPY",
        "indicators": {
            "ma_periods": [20, 50],
            "atr_period": 14,
            "adv_lookback": 20
        },
        "entry": {
            "fast_clearance": 0.0  # Not used for MA crossover
        },
        "exit": {
            "mode": "ma_cross",
            "exit_ma": 20
        },
        "risk": {
            "risk_per_trade": 0.01,  # 1% risk per trade
            "max_positions": 5
        }
    }
    
    # Create config object
    config = StrategyConfig(**config_dict)
    
    # Create strategy instance
    print("Creating strategy instance...")
    strategy = SimpleMovingAverageStrategy(config)
    
    print(f"Strategy created:")
    print(f"  Name: {strategy.name}")
    print(f"  Asset class: {strategy.asset_class}")
    print(f"  Universe: {strategy.universe}")
    
    return strategy


def example_use_custom_strategy_in_backtest():
    """Example: Use custom strategy in backtest (conceptual)."""
    print("\n" + "=" * 60)
    print("Example 3: Using Custom Strategy in Backtest")
    print("=" * 60)
    
    print("""
To use your custom strategy in a backtest:

1. Register your strategy (as shown in Example 1):
   
   register_strategy(
       strategy_type="sma_crossover",
       asset_class="equity",
       strategy_class=SimpleMovingAverageStrategy
   )

2. Create a strategy config YAML file:
   
   name: "sma_crossover_equity"
   asset_class: "equity"
   universe: ["AAPL", "MSFT", "GOOGL"]
   benchmark: "SPY"
   indicators:
     ma_periods: [20, 50]
     atr_period: 14
   entry:
     fast_clearance: 0.0
   exit:
     mode: "ma_cross"
     exit_ma: 20
   risk:
     risk_per_trade: 0.01
     max_positions: 5

3. Reference it in your run_config.yaml:
   
   strategies:
     equity:
       config_path: "configs/sma_crossover_config.yaml"
       enabled: true

4. Run backtest:
   
   python -m trading_system backtest --config run_config.yaml --period train

Note: The strategy loader will automatically detect and use your custom strategy
      if it's registered and matches the config's asset_class.
    """)


def example_strategy_interface_methods():
    """Example: Show all required methods for StrategyInterface."""
    print("\n" + "=" * 60)
    print("Example 4: StrategyInterface Required Methods")
    print("=" * 60)
    
    print("""
All custom strategies must implement these methods:

1. check_eligibility(features: FeatureRow) -> Tuple[bool, List[str]]
   - Check if symbol is eligible for entry
   - Returns: (is_eligible, list_of_failure_reasons)

2. check_entry_triggers(features: FeatureRow) -> Tuple[Optional[BreakoutType], float]
   - Check if entry triggers are met
   - Returns: (breakout_type, clearance) or (None, 0.0)

3. check_exit_signals(position: Position, features: FeatureRow) -> Optional[ExitReason]
   - Check if exit signals are met
   - Returns: ExitReason or None

4. compute_stop_price(entry_price, entry_date, features) -> Optional[float]
   - Compute stop loss price for new position
   - Returns: stop_price or None

5. compute_target_price(entry_price, entry_date, features) -> Optional[float]
   - Compute profit target (optional)
   - Returns: target_price or None

Optional methods (for advanced strategies):
- compute_signal_score(signal: Signal, features: FeatureRow) -> float
- update_position_management(position: Position, features: FeatureRow) -> None
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Custom Strategy Examples")
    print("=" * 60)
    
    try:
        # Example 1: Register strategy
        example_register_custom_strategy()
        
        # Example 2: Create from config
        strategy = example_create_custom_strategy_from_config()
        
        # Example 3: Use in backtest
        example_use_custom_strategy_in_backtest()
        
        # Example 4: Interface methods
        example_strategy_interface_methods()
        
        print("\n" + "=" * 60)
        print("Custom Strategy Examples Completed!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Implement your custom strategy logic")
        print("  2. Register it with register_strategy()")
        print("  3. Create a strategy config YAML")
        print("  4. Run backtests to test your strategy")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

