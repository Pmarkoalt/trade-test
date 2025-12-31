"""Portfolio state management with position tracking and risk metrics."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..models.positions import Position, ExitReason
from ..models.orders import Fill
from ..models.signals import BreakoutType
from .risk_scaling import compute_volatility_scaling
from .correlation import compute_average_pairwise_correlation, compute_correlation_to_portfolio


@dataclass
class Portfolio:
    """Portfolio state at a specific date."""
    
    date: pd.Timestamp
    
    # Cash and equity
    cash: float  # Available cash
    starting_equity: float  # Initial equity (100,000)
    equity: float  # Current equity = cash + sum(position_values)
    
    # Positions
    positions: Dict[str, Position] = field(default_factory=dict)  # symbol -> Position
    
    # Equity curve
    equity_curve: List[float] = field(default_factory=list)  # Historical equity values
    daily_returns: List[float] = field(default_factory=list)  # Daily portfolio returns
    
    # Exposure
    gross_exposure: float = 0.0  # Sum of all position notional values
    gross_exposure_pct: float = 0.0  # gross_exposure / equity
    per_position_exposure: Dict[str, float] = field(default_factory=dict)  # symbol -> pct
    
    # P&L
    realized_pnl: float = 0.0  # Cumulative realized P&L
    unrealized_pnl: float = 0.0  # Sum of all position unrealized P&L
    
    # Risk metrics
    portfolio_vol_20d: Optional[float] = None  # 20-day rolling portfolio volatility (annualized)
    median_vol_252d: Optional[float] = None  # Median vol over last 252 days
    risk_multiplier: float = 1.0  # Volatility scaling multiplier (0.33 to 1.0)
    
    # Correlation metrics
    avg_pairwise_corr: Optional[float] = None  # Average pairwise correlation (if >= 4 positions)
    correlation_matrix: Optional[np.ndarray] = None  # Full correlation matrix
    
    # Trade statistics
    total_trades: int = 0  # Total trades closed
    open_trades: int = 0  # Current open positions
    
    def __post_init__(self):
        """Initialize portfolio with starting equity."""
        if self.equity == 0:
            self.equity = self.starting_equity
        if self.cash == 0:
            self.cash = self.starting_equity
        # Initialize equity curve with starting equity
        if not self.equity_curve:
            self.equity_curve.append(self.equity)
    
    def add_position(self, position: Position) -> None:
        """Add a new position to the portfolio.
        
        Args:
            position: Position to add
        """
        if position.symbol in self.positions:
            raise ValueError(f"Position for {position.symbol} already exists")
        
        self.positions[position.symbol] = position
        self.open_trades = len(self.positions)
    
    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove a position from the portfolio.
        
        Args:
            symbol: Symbol of position to remove
        
        Returns:
            Removed position, or None if not found
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions.pop(symbol)
        self.open_trades = len(self.positions)
        return position
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol.
        
        Args:
            symbol: Symbol to look up
        
        Returns:
            Position if found, None otherwise
        """
        return self.positions.get(symbol)
    
    def update_equity(self, current_prices: Dict[str, float]) -> None:
        """Update equity based on current market prices.
        
        Updates unrealized P&L for all positions, calculates exposure,
        and updates total equity.
        
        Args:
            current_prices: Dictionary mapping symbol to current market price
        """
        total_unrealized = 0.0
        total_exposure = 0.0
        per_position_exposure = {}
        
        for symbol, position in self.positions.items():
            if not position.is_open():
                continue
            
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.update_unrealized_pnl(current_price)
                total_unrealized += position.unrealized_pnl
                
                position_notional = current_price * position.quantity
                total_exposure += position_notional
                per_position_exposure[symbol] = position_notional / self.equity if self.equity > 0 else 0.0
        
        self.unrealized_pnl = total_unrealized
        self.gross_exposure = total_exposure
        self.gross_exposure_pct = total_exposure / self.equity if self.equity > 0 else 0.0
        self.per_position_exposure = per_position_exposure
        self.equity = self.cash + total_exposure
        self.open_trades = len([p for p in self.positions.values() if p.is_open()])
    
    def compute_portfolio_returns(self, lookback: Optional[int] = None) -> List[float]:
        """Compute portfolio returns for volatility calculation.
        
        Args:
            lookback: Optional lookback period (returns last N returns if specified)
        
        Returns:
            List of daily portfolio returns (as decimals)
        """
        if len(self.equity_curve) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] / self.equity_curve[i-1]) - 1
            returns.append(ret)
        
        if lookback is not None:
            return returns[-lookback:] if len(returns) >= lookback else returns
        
        return returns
    
    def update_volatility_scaling(self) -> None:
        """Update risk multiplier based on portfolio volatility.
        
        Computes 20D rolling volatility and compares to median over 252D.
        Updates risk_multiplier, portfolio_vol_20d, and median_vol_252d.
        """
        returns = self.compute_portfolio_returns()
        
        risk_multiplier, vol_20d, median_vol_252d = compute_volatility_scaling(
            returns,
            lookback_20d=20,
            lookback_252d=252
        )
        
        self.risk_multiplier = risk_multiplier
        self.portfolio_vol_20d = vol_20d
        self.median_vol_252d = median_vol_252d
    
    def update_correlation_metrics(
        self,
        returns_data: Dict[str, List[float]],
        lookback: int = 20
    ) -> None:
        """Update correlation metrics for existing positions.
        
        Computes average pairwise correlation if >= 4 positions with sufficient data.
        
        Args:
            returns_data: Dictionary mapping symbol to list of daily returns
            lookback: Number of days to use for correlation (default: 20)
        """
        if len(self.positions) < 4:
            self.avg_pairwise_corr = None
            self.correlation_matrix = None
            return
        
        # Get returns for open positions only
        position_returns = {}
        for symbol, position in self.positions.items():
            if position.is_open() and symbol in returns_data:
                if len(returns_data[symbol]) >= lookback:
                    position_returns[symbol] = returns_data[symbol]
        
        if len(position_returns) < 4:
            self.avg_pairwise_corr = None
            self.correlation_matrix = None
            return
        
        avg_corr, corr_matrix = compute_average_pairwise_correlation(
            position_returns,
            lookback=lookback,
            min_positions=4
        )
        
        self.avg_pairwise_corr = avg_corr
        self.correlation_matrix = corr_matrix
    
    def process_fill(self, fill: Fill, stop_price: float, atr_mult: float, 
                     triggered_on: BreakoutType, adv20_at_entry: float) -> Position:
        """Process a fill and create a new position.
        
        Args:
            fill: Fill object from order execution
            stop_price: Stop loss price for the position
            atr_mult: ATR multiplier used for stop calculation
            triggered_on: Breakout type that triggered entry
            adv20_at_entry: ADV20 at entry time
        
        Returns:
            Created Position object
        """
        # Deduct cash (fill price * quantity + fees)
        total_cost = fill.fill_price * fill.quantity + fill.total_cost
        self.cash -= total_cost
        
        # Create position
        position = Position(
            symbol=fill.symbol,
            asset_class=fill.asset_class,
            entry_date=fill.date,
            entry_price=fill.fill_price,
            entry_fill_id=fill.fill_id,
            quantity=fill.quantity,
            stop_price=stop_price,
            initial_stop_price=stop_price,
            hard_stop_atr_mult=atr_mult,
            entry_slippage_bps=fill.slippage_bps,
            entry_fee_bps=fill.fee_bps,
            entry_total_cost=fill.total_cost,
            triggered_on=triggered_on,
            adv20_at_entry=adv20_at_entry
        )
        
        # Add to portfolio
        self.add_position(position)
        
        return position
    
    def close_position(
        self,
        symbol: str,
        exit_fill: Fill,
        exit_reason: ExitReason
    ) -> Optional[Position]:
        """Close a position and update portfolio state.
        
        Args:
            symbol: Symbol of position to close
            exit_fill: Fill object from exit execution
            exit_reason: Reason for exit
        
        Returns:
            Closed position, or None if not found
        """
        position = self.get_position(symbol)
        if position is None or not position.is_open():
            return None
        
        # Calculate realized P&L
        price_pnl = (exit_fill.fill_price - position.entry_price) * position.quantity
        total_costs = position.entry_total_cost + exit_fill.total_cost
        realized_pnl = price_pnl - total_costs
        
        # Update position
        position.exit_date = exit_fill.date
        position.exit_price = exit_fill.fill_price
        position.exit_fill_id = exit_fill.fill_id
        position.exit_reason = exit_reason
        position.exit_slippage_bps = exit_fill.slippage_bps
        position.exit_fee_bps = exit_fill.fee_bps
        position.exit_total_cost = exit_fill.total_cost
        position.realized_pnl = realized_pnl
        
        # Update cash (receive proceeds minus costs)
        proceeds = exit_fill.fill_price * exit_fill.quantity
        self.cash += proceeds - exit_fill.total_cost
        
        # Update portfolio realized P&L
        self.realized_pnl += realized_pnl
        
        # Remove from positions
        self.remove_position(symbol)
        self.total_trades += 1
        
        return position
    
    def update_stops(
        self,
        current_prices: Dict[str, float],
        features_data: Dict[str, Dict],  # symbol -> {ma20, ma50, atr14}
        exit_mode: str = "ma_cross",
        exit_ma: int = 20
    ) -> List[tuple[str, ExitReason]]:
        """Update trailing stops and check exit signals.
        
        For each open position:
        1. Update trailing stop (can only move up for longs)
        2. Check crypto staged exit (tighten stop after MA20 break)
        3. Check exit signals (hard stop, trailing MA cross)
        
        Args:
            current_prices: Dictionary mapping symbol to current market price
            features_data: Dictionary mapping symbol to features dict (ma20, ma50, atr14)
            exit_mode: Exit mode ("ma_cross" or "staged")
            exit_ma: MA period for exit (20 or 50)
        
        Returns:
            List of (symbol, exit_reason) tuples for positions that should exit
        """
        exit_signals = []
        
        for symbol, position in self.positions.items():
            if not position.is_open():
                continue
            
            if symbol not in current_prices:
                continue  # Missing price data
            
            current_price = current_prices[symbol]
            features = features_data.get(symbol, {})
            
            # Update trailing stop (can only move up for longs)
            # For now, we don't automatically trail - stops are updated explicitly
            # This method checks for exits only
            
            # Crypto staged exit logic
            if exit_mode == "staged" and position.asset_class == "crypto":
                ma20 = features.get("ma20")
                ma50 = features.get("ma50")
                atr14 = features.get("atr14")
                
                if ma20 is not None and atr14 is not None and not position.tightened_stop:
                    if current_price < ma20:
                        # Stage 1: Tighten stop
                        tightened_stop = position.entry_price - (2.0 * atr14)
                        if tightened_stop > position.stop_price:
                            position.update_stop(tightened_stop, reason="tighten")
                            position.tightened_stop_atr_mult = 2.0
            
            # Check exit signals (priority: hard stop > trailing MA)
            exit_reason = None
            
            # Priority 1: Hard stop
            if current_price <= position.stop_price:
                exit_reason = ExitReason.HARD_STOP
            
            # Priority 2: Trailing MA cross
            elif exit_mode == "ma_cross":
                ma_level = features.get(f"ma{exit_ma}")
                if ma_level is not None and current_price < ma_level:
                    exit_reason = ExitReason.TRAILING_MA_CROSS
            
            elif exit_mode == "staged" and position.asset_class == "crypto":
                ma50 = features.get("ma50")
                if ma50 is not None:
                    # Stage 2: MA50 cross OR tightened stop hit
                    if current_price < ma50 or (position.tightened_stop and current_price <= position.stop_price):
                        exit_reason = ExitReason.TRAILING_MA_CROSS
            
            if exit_reason:
                exit_signals.append((symbol, exit_reason))
        
        return exit_signals
    
    def append_daily_metrics(self) -> None:
        """Append current equity to equity curve and compute daily return.
        
        Should be called at end of each trading day after all updates.
        """
        # Append current equity to curve
        self.equity_curve.append(self.equity)
        
        # Compute daily return
        if len(self.equity_curve) > 1:
            daily_return = (self.equity_curve[-1] / self.equity_curve[-2]) - 1
            self.daily_returns.append(daily_return)
    
    def get_exposure_summary(self) -> Dict[str, float]:
        """Get exposure summary.
        
        Returns:
            Dictionary with exposure metrics
        """
        return {
            "gross_exposure": self.gross_exposure,
            "gross_exposure_pct": self.gross_exposure_pct,
            "per_position_exposure": self.per_position_exposure.copy(),
            "cash": self.cash,
            "equity": self.equity,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl
        }

