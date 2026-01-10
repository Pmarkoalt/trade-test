"""CLI commands for paper trading execution."""

import logging
import os

from ...adapters.alpaca_adapter import AlpacaAdapter
from ...adapters.base_adapter import AdapterConfig
from ...execution.paper_trading import PaperTradingConfig, PaperTradingRunner

logger = logging.getLogger(__name__)


def add_paper_trading_parser(subparsers) -> None:
    """Add paper trading command parser.

    Args:
        subparsers: Subparsers from main argument parser
    """
    paper_parser = subparsers.add_parser(
        "paper",
        help="Paper trading execution and monitoring",
        description="Execute and monitor paper trading orders",
    )

    paper_subparsers = paper_parser.add_subparsers(dest="paper_command", help="Paper trading commands")

    # paper status - show account and positions
    status_parser = paper_subparsers.add_parser("status", help="Show paper trading account status and positions")
    status_parser.add_argument("--api-key", help="Alpaca API key (or set ALPACA_API_KEY env var)")
    status_parser.add_argument("--api-secret", help="Alpaca API secret (or set ALPACA_API_SECRET env var)")
    status_parser.add_argument("--base-url", help="Alpaca base URL (default: paper trading URL)")

    # paper positions - show current positions
    positions_parser = paper_subparsers.add_parser("positions", help="Show current paper trading positions")
    positions_parser.add_argument("--api-key", help="Alpaca API key (or set ALPACA_API_KEY env var)")
    positions_parser.add_argument("--api-secret", help="Alpaca API secret (or set ALPACA_API_SECRET env var)")
    positions_parser.add_argument("--base-url", help="Alpaca base URL (default: paper trading URL)")
    positions_parser.add_argument("--export", help="Export positions to CSV file")

    # paper reconcile - reconcile positions with broker
    reconcile_parser = paper_subparsers.add_parser("reconcile", help="Reconcile positions with broker")
    reconcile_parser.add_argument("--api-key", help="Alpaca API key (or set ALPACA_API_KEY env var)")
    reconcile_parser.add_argument("--api-secret", help="Alpaca API secret (or set ALPACA_API_SECRET env var)")
    reconcile_parser.add_argument("--base-url", help="Alpaca base URL (default: paper trading URL)")


def get_adapter_config(args) -> AdapterConfig:
    """Get adapter configuration from args and environment.

    Args:
        args: Parsed command-line arguments

    Returns:
        AdapterConfig instance
    """
    api_key = args.api_key or os.getenv("ALPACA_API_KEY")
    api_secret = args.api_secret or os.getenv("ALPACA_API_SECRET")
    base_url = args.base_url or os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not api_secret:
        raise ValueError(
            "Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_API_SECRET "
            "environment variables or use --api-key and --api-secret flags."
        )

    return AdapterConfig(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        paper_trading=True,
    )


def handle_paper_status(args) -> int:
    """Handle paper status command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        config = get_adapter_config(args)
        adapter = AlpacaAdapter(config)

        print("\n" + "=" * 80)
        print("PAPER TRADING ACCOUNT STATUS")
        print("=" * 80)

        with adapter:
            # Get account info
            account = adapter.get_account_info()
            print(f"\nAccount ID: {account.broker_account_id}")
            print(f"Equity: ${account.equity:,.2f}")
            print(f"Cash: ${account.cash:,.2f}")
            print(f"Buying Power: ${account.buying_power:,.2f}")
            print(f"Margin Used: ${account.margin_used:,.2f}")

            # Get positions
            positions = adapter.get_positions()
            print(f"\nOpen Positions: {len(positions)}")

            if positions:
                print("\n" + "-" * 80)
                print(f"{'Symbol':<10} {'Side':<6} {'Qty':<8} {'Entry':<10} {'Stop':<10} {'Current':<10}")
                print("-" * 80)

                for symbol, position in positions.items():
                    current_price = adapter.get_current_price(symbol) or position.entry_price
                    print(
                        f"{symbol:<10} {position.side.value:<6} {position.quantity:<8} "
                        f"${position.entry_price:<9.2f} ${position.stop_price:<9.2f} ${current_price:<9.2f}"
                    )

        print("=" * 80 + "\n")
        return 0

    except Exception as e:
        logger.error(f"Error getting paper trading status: {e}", exc_info=True)
        print(f"\nError: {e}\n")
        return 1


def handle_paper_positions(args) -> int:
    """Handle paper positions command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        config = get_adapter_config(args)
        adapter = AlpacaAdapter(config)

        with adapter:
            positions = adapter.get_positions()

            if not positions:
                print("\nNo open positions.\n")
                return 0

            print("\n" + "=" * 80)
            print("PAPER TRADING POSITIONS")
            print("=" * 80)
            print(f"\nTotal Positions: {len(positions)}\n")

            # Calculate totals
            total_notional = sum(p.entry_price * p.quantity for p in positions.values())
            print(f"Total Notional: ${total_notional:,.2f}\n")

            print("-" * 80)
            print(
                f"{'Symbol':<10} {'Side':<6} {'Qty':<8} {'Entry':<10} {'Stop':<10} " f"{'Entry Date':<12} {'Asset Class':<12}"
            )
            print("-" * 80)

            for symbol, position in positions.items():
                entry_date = position.entry_date.strftime("%Y-%m-%d") if position.entry_date else "N/A"
                print(
                    f"{symbol:<10} {position.side.value:<6} {position.quantity:<8} "
                    f"${position.entry_price:<9.2f} ${position.stop_price:<9.2f} "
                    f"{entry_date:<12} {position.asset_class:<12}"
                )

            print("=" * 80 + "\n")

            # Export if requested
            if args.export:
                import pandas as pd

                data = []
                for symbol, position in positions.items():
                    data.append(
                        {
                            "symbol": symbol,
                            "side": position.side.value,
                            "quantity": position.quantity,
                            "entry_price": position.entry_price,
                            "stop_price": position.stop_price,
                            "entry_date": position.entry_date.isoformat() if position.entry_date else None,
                            "asset_class": position.asset_class,
                        }
                    )

                df = pd.DataFrame(data)
                df.to_csv(args.export, index=False)
                print(f"Exported {len(positions)} positions to {args.export}\n")

        return 0

    except Exception as e:
        logger.error(f"Error getting paper trading positions: {e}", exc_info=True)
        print(f"\nError: {e}\n")
        return 1


def handle_paper_reconcile(args) -> int:
    """Handle paper reconcile command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        config = get_adapter_config(args)
        adapter = AlpacaAdapter(config)

        print("\n" + "=" * 80)
        print("RECONCILING PAPER TRADING POSITIONS")
        print("=" * 80 + "\n")

        with adapter:
            # Create runner for reconciliation
            paper_config = PaperTradingConfig(adapter_config=config)
            runner = PaperTradingRunner(config=paper_config, adapter=adapter)

            # Reconcile
            positions = runner.reconcile_positions()

            print(f"\nReconciliation complete: {len(positions)} positions found\n")

        return 0

    except Exception as e:
        logger.error(f"Error reconciling paper trading positions: {e}", exc_info=True)
        print(f"\nError: {e}\n")
        return 1


def handle_paper_trading_command(args) -> int:
    """Handle paper trading commands.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    if args.paper_command == "status":
        return handle_paper_status(args)
    elif args.paper_command == "positions":
        return handle_paper_positions(args)
    elif args.paper_command == "reconcile":
        return handle_paper_reconcile(args)
    else:
        print("Error: No paper trading command specified. Use --help for usage.")
        return 1
