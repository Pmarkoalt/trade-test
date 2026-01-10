"""CLI commands for unified positions view."""

import logging
import os
from typing import Optional

from ...adapters.alpaca_adapter import AlpacaAdapter
from ...adapters.base_adapter import AdapterConfig
from ...reporting.unified_positions import UnifiedPositionView
from ...storage.manual_trades import ManualTradeDatabase

logger = logging.getLogger(__name__)


def add_positions_parser(subparsers) -> None:
    """Add positions command parser.

    Args:
        subparsers: Subparsers from main argument parser
    """
    positions_parser = subparsers.add_parser(
        "positions",
        help="View unified positions across all sources",
        description="View positions from backtest, paper trading, and manual trades",
    )

    positions_parser.add_argument("--open-only", action="store_true", help="Show only open positions")
    positions_parser.add_argument("--include-backtest", action="store_true", help="Include backtest positions")
    positions_parser.add_argument("--include-paper", action="store_true", default=True, help="Include paper trading positions")
    positions_parser.add_argument("--include-manual", action="store_true", default=True, help="Include manual trades")
    positions_parser.add_argument("--export", help="Export positions to CSV file")
    positions_parser.add_argument("--api-key", help="Alpaca API key (or set ALPACA_API_KEY env var)")
    positions_parser.add_argument("--api-secret", help="Alpaca API secret (or set ALPACA_API_SECRET env var)")
    positions_parser.add_argument("--base-url", help="Alpaca base URL (default: paper trading URL)")


def get_paper_adapter(args) -> Optional[AlpacaAdapter]:
    """Get paper trading adapter if credentials are available.

    Args:
        args: Parsed command-line arguments

    Returns:
        AlpacaAdapter instance or None
    """
    api_key = args.api_key or os.getenv("ALPACA_API_KEY")
    api_secret = args.api_secret or os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        logger.info("Alpaca credentials not provided, skipping paper trading positions")
        return None

    base_url = args.base_url or os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    config = AdapterConfig(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        paper_trading=True,
    )

    adapter = AlpacaAdapter(config)
    try:
        adapter.connect()
        return adapter
    except Exception as e:
        logger.warning(f"Failed to connect to Alpaca: {e}")
        return None


def handle_positions_command(args) -> int:
    """Handle positions command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        # Setup databases and adapters
        manual_db = ManualTradeDatabase()
        paper_adapter = None

        if args.include_paper:
            paper_adapter = get_paper_adapter(args)

        # Create unified view
        view = UnifiedPositionView(manual_db=manual_db, paper_adapter=paper_adapter)

        # Get positions
        positions = view.get_all_positions(
            include_backtest=args.include_backtest,
            include_paper=args.include_paper and paper_adapter is not None,
            include_manual=args.include_manual,
            open_only=args.open_only,
        )

        if not positions:
            print("\nNo positions found.\n")
            if paper_adapter:
                paper_adapter.disconnect()
            return 0

        # Get exposure summary
        exposure = view.get_exposure_summary()

        # Print summary
        print("\n" + "=" * 80)
        print("UNIFIED POSITIONS VIEW")
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
        print("\n" + "-" * 80)
        print("POSITIONS")
        print("-" * 80)
        print(f"{'Source':<10} {'Symbol':<10} {'Side':<6} {'Qty':<8} {'Entry':<10} " f"{'Stop':<10} {'Unrealized P&L':<15}")
        print("-" * 80)

        for unified_pos in positions:
            pos = unified_pos.position
            print(
                f"{unified_pos.source.value:<10} {pos.symbol:<10} {pos.side.value:<6} "
                f"{pos.quantity:<8} ${pos.entry_price:<9.2f} ${pos.stop_price:<9.2f} "
                f"${pos.unrealized_pnl:<14.2f}"
            )

        print("=" * 80 + "\n")

        # Export if requested
        if args.export:
            view.export_to_csv(args.export, open_only=args.open_only)
            print(f"Exported {len(positions)} positions to {args.export}\n")

        # Cleanup
        if paper_adapter:
            paper_adapter.disconnect()

        return 0

    except Exception as e:
        logger.error(f"Error viewing positions: {e}", exc_info=True)
        print(f"\nError: {e}\n")
        return 1
