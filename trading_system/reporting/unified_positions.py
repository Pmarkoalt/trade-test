"""Unified positions view merging system/paper/manual trades."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from ..adapters.base_adapter import BaseAdapter
from ..models.positions import Position
from ..storage.manual_trades import ManualTradeDatabase

logger = logging.getLogger(__name__)


class PositionSource(str, Enum):
    """Source of position data."""

    BACKTEST = "backtest"  # From backtest results
    PAPER = "paper"  # From paper trading broker
    MANUAL = "manual"  # User-managed manual trades


@dataclass
class UnifiedPosition:
    """Unified position record with source tracking."""

    position: Position
    source: PositionSource
    source_id: str  # Original ID from source (trade_id, fill_id, etc.)

    def to_dict(self) -> dict:
        """Convert to dictionary for reporting.

        Returns:
            Dictionary representation
        """
        return {
            "source": self.source.value,
            "source_id": self.source_id,
            "symbol": self.position.symbol,
            "asset_class": self.position.asset_class,
            "side": self.position.side.value,
            "quantity": self.position.quantity,
            "entry_date": self.position.entry_date.isoformat(),
            "entry_price": self.position.entry_price,
            "stop_price": self.position.stop_price,
            "initial_stop_price": self.position.initial_stop_price,
            "exit_date": self.position.exit_date.isoformat() if self.position.exit_date else None,
            "exit_price": self.position.exit_price,
            "exit_reason": self.position.exit_reason.value if self.position.exit_reason else None,
            "realized_pnl": self.position.realized_pnl,
            "unrealized_pnl": self.position.unrealized_pnl,
            "is_open": self.position.is_open(),
        }


class UnifiedPositionView:
    """Unified view of positions from multiple sources.

    Merges positions from:
    - Backtest results (historical)
    - Paper trading (live broker positions)
    - Manual trades (user-managed)
    """

    def __init__(
        self,
        manual_db: Optional[ManualTradeDatabase] = None,
        paper_adapter: Optional[BaseAdapter] = None,
    ):
        """Initialize unified position view.

        Args:
            manual_db: Manual trade database (optional)
            paper_adapter: Paper trading adapter (optional, must be connected)
        """
        self.manual_db = manual_db or ManualTradeDatabase()
        self.paper_adapter = paper_adapter

    def get_all_positions(
        self,
        include_backtest: bool = False,
        include_paper: bool = True,
        include_manual: bool = True,
        open_only: bool = False,
    ) -> List[UnifiedPosition]:
        """Get all positions from enabled sources.

        Args:
            include_backtest: Include backtest positions (not implemented yet)
            include_paper: Include paper trading positions
            include_manual: Include manual trades
            open_only: Only return open positions

        Returns:
            List of UnifiedPosition objects
        """
        positions = []

        # Get manual trades
        if include_manual:
            manual_positions = self._get_manual_positions(open_only=open_only)
            positions.extend(manual_positions)

        # Get paper trading positions
        if include_paper and self.paper_adapter:
            paper_positions = self._get_paper_positions(open_only=open_only)
            positions.extend(paper_positions)

        # TODO: Get backtest positions (would need to query ResultsDatabase)
        if include_backtest:
            logger.warning("Backtest position retrieval not yet implemented")

        return positions

    def _get_manual_positions(self, open_only: bool = False) -> List[UnifiedPosition]:
        """Get positions from manual trades database.

        Args:
            open_only: Only return open positions

        Returns:
            List of UnifiedPosition objects from manual trades
        """
        if open_only:
            manual_trades = self.manual_db.get_open_trades()
        else:
            manual_trades = self.manual_db.get_all_trades()

        positions = []
        for trade in manual_trades:
            unified = UnifiedPosition(
                position=trade.to_position(),
                source=PositionSource.MANUAL,
                source_id=trade.trade_id,
            )
            positions.append(unified)

        logger.info(f"Retrieved {len(positions)} manual positions (open_only={open_only})")
        return positions

    def _get_paper_positions(self, open_only: bool = False) -> List[UnifiedPosition]:
        """Get positions from paper trading broker.

        Args:
            open_only: Only return open positions (paper positions are always open)

        Returns:
            List of UnifiedPosition objects from paper trading
        """
        if not self.paper_adapter:
            logger.warning("Paper adapter not configured")
            return []

        if not self.paper_adapter.is_connected():
            logger.warning("Paper adapter not connected")
            return []

        try:
            broker_positions = self.paper_adapter.get_positions()
            positions = []

            for symbol, position in broker_positions.items():
                # Paper positions are always open
                if open_only or position.is_open():
                    unified = UnifiedPosition(
                        position=position,
                        source=PositionSource.PAPER,
                        source_id=position.entry_fill_id,
                    )
                    positions.append(unified)

            logger.info(f"Retrieved {len(positions)} paper positions")
            return positions

        except Exception as e:
            logger.error(f"Error retrieving paper positions: {e}", exc_info=True)
            return []

    def get_open_positions_by_symbol(self) -> Dict[str, List[UnifiedPosition]]:
        """Get all open positions grouped by symbol.

        Returns:
            Dictionary mapping symbol to list of UnifiedPosition objects
        """
        positions = self.get_all_positions(open_only=True)
        by_symbol: Dict[str, List[UnifiedPosition]] = {}

        for pos in positions:
            symbol = pos.position.symbol
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(pos)

        return by_symbol

    def get_exposure_summary(self) -> Dict[str, float]:
        """Calculate exposure summary across all sources.

        Returns:
            Dictionary with exposure metrics
        """
        positions = self.get_all_positions(open_only=True)

        total_long_notional = 0.0
        total_short_notional = 0.0
        equity_notional = 0.0
        crypto_notional = 0.0

        for unified_pos in positions:
            pos = unified_pos.position
            notional = pos.entry_price * pos.quantity

            if pos.side.value == "LONG":
                total_long_notional += notional
            else:
                total_short_notional += notional

            if pos.asset_class == "equity":
                equity_notional += notional
            else:
                crypto_notional += notional

        return {
            "total_long_notional": total_long_notional,
            "total_short_notional": total_short_notional,
            "net_exposure": total_long_notional - total_short_notional,
            "gross_exposure": total_long_notional + total_short_notional,
            "equity_notional": equity_notional,
            "crypto_notional": crypto_notional,
            "total_positions": len(positions),
        }

    def get_positions_dataframe(self, open_only: bool = False) -> pd.DataFrame:
        """Get all positions as a pandas DataFrame.

        Args:
            open_only: Only return open positions

        Returns:
            DataFrame with position data
        """
        positions = self.get_all_positions(open_only=open_only)

        if not positions:
            return pd.DataFrame()

        data = [pos.to_dict() for pos in positions]
        df = pd.DataFrame(data)

        # Sort by entry date (most recent first)
        df = df.sort_values("entry_date", ascending=False)

        return df

    def export_to_csv(self, output_path: str, open_only: bool = False) -> None:
        """Export positions to CSV file.

        Args:
            output_path: Path to output CSV file
            open_only: Only export open positions
        """
        df = self.get_positions_dataframe(open_only=open_only)

        if df.empty:
            logger.warning("No positions to export")
            return

        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} positions to {output_path}")

    def print_summary(self) -> None:
        """Print a summary of all positions to console."""
        positions = self.get_all_positions(open_only=True)
        exposure = self.get_exposure_summary()

        print("\n" + "=" * 80)
        print("UNIFIED POSITIONS SUMMARY")
        print("=" * 80)

        print(f"\nTotal Open Positions: {exposure['total_positions']}")
        print(f"Gross Exposure: ${exposure['gross_exposure']:,.2f}")
        print(f"Net Exposure: ${exposure['net_exposure']:,.2f}")
        print(f"  Long Notional: ${exposure['total_long_notional']:,.2f}")
        print(f"  Short Notional: ${exposure['total_short_notional']:,.2f}")
        print(f"\nBy Asset Class:")
        print(f"  Equity: ${exposure['equity_notional']:,.2f}")
        print(f"  Crypto: ${exposure['crypto_notional']:,.2f}")

        # Group by source
        by_source = {}
        for pos in positions:
            source = pos.source.value
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(pos)

        print(f"\nBy Source:")
        for source, source_positions in by_source.items():
            print(f"  {source.capitalize()}: {len(source_positions)} positions")

        # Show individual positions
        if positions:
            print("\n" + "-" * 80)
            print("OPEN POSITIONS")
            print("-" * 80)
            print(
                f"{'Source':<10} {'Symbol':<10} {'Side':<6} {'Qty':<8} {'Entry':<10} {'Stop':<10} {'Unrealized P&L':<15}"
            )
            print("-" * 80)

            for unified_pos in positions:
                pos = unified_pos.position
                print(
                    f"{unified_pos.source.value:<10} {pos.symbol:<10} {pos.side.value:<6} "
                    f"{pos.quantity:<8} ${pos.entry_price:<9.2f} ${pos.stop_price:<9.2f} "
                    f"${pos.unrealized_pnl:<14.2f}"
                )

        print("=" * 80 + "\n")
