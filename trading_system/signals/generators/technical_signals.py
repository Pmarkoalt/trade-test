"""Technical signal generator using existing strategy logic."""

from datetime import date
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import pandas as pd

from ...models.features import FeatureRow
from ...models.signals import Signal
from ...indicators.feature_computer import compute_features, compute_features_for_date

if TYPE_CHECKING:
    from ...portfolio.portfolio import Portfolio
    from ...strategies.base.strategy_interface import StrategyInterface


class TechnicalSignalGenerator:
    """Generate signals using existing strategy logic."""

    def __init__(
        self,
        strategies: List["StrategyInterface"],
        feature_computer: Optional[Callable] = None,
    ):
        """Initialize technical signal generator.

        Args:
            strategies: List of strategy instances to use for signal generation
            feature_computer: Optional custom feature computation function.
                           If None, uses the default compute_features function.
                           Should have signature: (df_ohlc, symbol, asset_class) -> pd.DataFrame
        """
        self.strategies = strategies
        self.feature_computer = feature_computer or compute_features

    def _get_asset_class(self, symbol: str) -> str:
        """Get asset class for a symbol from strategies.

        Args:
            symbol: Symbol name

        Returns:
            "equity" or "crypto" (defaults to "equity" if not found)
        """
        for strategy in self.strategies:
            if symbol in strategy.universe:
                return strategy.asset_class
        return "equity"  # Default

    def _estimate_order_notional(
        self,
        strategy: "StrategyInterface",
        features: FeatureRow,
        portfolio_state: Optional["Portfolio"] = None,
    ) -> float:
        """Estimate order notional for capacity check.

        Args:
            strategy: Strategy instance
            features: FeatureRow with current indicators
            portfolio_state: Optional portfolio state for accurate sizing

        Returns:
            Estimated order notional value
        """
        if portfolio_state is not None:
            # Use portfolio-based estimation
            from ...portfolio.position_sizing import estimate_position_size

            estimated_qty = estimate_position_size(
                equity=portfolio_state.equity,
                risk_pct=strategy.config.risk.risk_per_trade * portfolio_state.risk_multiplier,
                entry_price=features.close,
                stop_price=(
                    features.close - (strategy.config.exit.hard_stop_atr_mult * (features.atr14 or 0.0))
                    if features.atr14 is not None
                    else features.close * 0.98
                ),
                max_position_notional=portfolio_state.equity * strategy.config.risk.max_position_notional,
                risk_multiplier=portfolio_state.risk_multiplier,
            )
            return features.close * estimated_qty
        else:
            # Use simple default estimation (assume $100k equity, 0.75% risk)
            default_equity = 100000.0
            default_risk_pct = strategy.config.risk.risk_per_trade
            atr_mult = strategy.config.exit.hard_stop_atr_mult
            stop_price = features.close - (atr_mult * features.atr14) if features.atr14 else features.close * 0.95

            # Simple quantity estimate: risk_amount / (entry - stop)
            risk_amount = default_equity * default_risk_pct
            price_diff = abs(features.close - stop_price)
            if price_diff > 0:
                estimated_qty = int(risk_amount / price_diff)
            else:
                estimated_qty = 0

            return features.close * estimated_qty

    def generate_signals(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        current_date: date,
        portfolio_state: Optional["Portfolio"] = None,
    ) -> List[Signal]:
        """Generate signals for current date.

        Args:
            ohlcv_data: OHLCV data keyed by symbol. Each DataFrame should have
                       columns ['open', 'high', 'low', 'close', 'volume'] with
                       datetime index.
            current_date: The date to generate signals for (date or pd.Timestamp)
            portfolio_state: Current portfolio (for exit signals and accurate sizing)

        Returns:
            List of Signal objects
        """
        signals = []

        # Convert date to pd.Timestamp if needed
        if isinstance(current_date, date):
            current_date = pd.Timestamp(current_date)

        for symbol, data in ohlcv_data.items():
            # Determine asset class
            asset_class = self._get_asset_class(symbol)

            # Compute features
            try:
                features_df = self.feature_computer(
                    data,
                    symbol=symbol,
                    asset_class=asset_class,
                    use_cache=False,  # Don't cache in live mode
                    optimize_memory=False,  # Don't optimize in live mode
                )
            except Exception as e:
                # Skip symbols with computation errors
                continue

            # Get latest feature row for current_date
            feature_row = compute_features_for_date(features_df, current_date)
            if feature_row is None:
                continue

            # Check if features are valid for entry
            if not feature_row.is_valid_for_entry():
                continue

            # Check each strategy
            for strategy in self.strategies:
                # Skip if asset class doesn't match
                if strategy.asset_class != asset_class:
                    continue

                # Skip if symbol not in strategy universe
                if symbol not in strategy.universe:
                    continue

                # Estimate order notional for capacity check
                order_notional = self._estimate_order_notional(strategy, feature_row, portfolio_state)

                # Generate signal using strategy's generate_signal method
                signal = strategy.generate_signal(
                    symbol=symbol,
                    features=feature_row,
                    order_notional=order_notional,
                    diversification_bonus=0.0,  # Will be computed during scoring if needed
                )

                if signal is not None and signal.is_valid():
                    signals.append(signal)

        return signals
