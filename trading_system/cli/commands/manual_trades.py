"""CLI commands for manual trade management."""

import logging
import uuid
from datetime import datetime

from ...models.positions import PositionSide
from ...storage.manual_trades import ManualTrade, ManualTradeDatabase

logger = logging.getLogger(__name__)


def add_manual_trades_parser(subparsers) -> None:
    """Add manual trades command parser.

    Args:
        subparsers: Subparsers from main argument parser
    """
    manual_parser = subparsers.add_parser(
        "manual",
        help="Manual trade tracking and management",
        description="Track and manage manually executed trades",
    )

    manual_subparsers = manual_parser.add_subparsers(dest="manual_command", help="Manual trade commands")

    # manual add - add a new manual trade
    add_parser = manual_subparsers.add_parser("add", help="Add a new manual trade")
    add_parser.add_argument("symbol", help="Symbol (e.g., AAPL, BTC)")
    add_parser.add_argument("side", choices=["LONG", "SHORT"], help="Position side")
    add_parser.add_argument("quantity", type=int, help="Quantity (shares/units)")
    add_parser.add_argument("entry_price", type=float, help="Entry price")
    add_parser.add_argument("stop_price", type=float, help="Stop price")
    add_parser.add_argument("--asset-class", choices=["equity", "crypto"], default="equity", help="Asset class")
    add_parser.add_argument("--entry-date", help="Entry date (YYYY-MM-DD, default: today)")
    add_parser.add_argument("--notes", help="Notes about this trade")
    add_parser.add_argument("--tags", help="Comma-separated tags")

    # manual close - close an existing manual trade
    close_parser = manual_subparsers.add_parser("close", help="Close a manual trade")
    close_parser.add_argument("trade_id", help="Trade ID to close")
    close_parser.add_argument("exit_price", type=float, help="Exit price")
    close_parser.add_argument("--exit-date", help="Exit date (YYYY-MM-DD, default: today)")
    close_parser.add_argument("--reason", default="manual", help="Exit reason")

    # manual update - update an existing manual trade
    update_parser = manual_subparsers.add_parser("update", help="Update a manual trade")
    update_parser.add_argument("trade_id", help="Trade ID to update")
    update_parser.add_argument("--stop-price", type=float, help="New stop price")
    update_parser.add_argument("--notes", help="Updated notes")
    update_parser.add_argument("--tags", help="Updated tags (comma-separated)")

    # manual list - list manual trades
    list_parser = manual_subparsers.add_parser("list", help="List manual trades")
    list_parser.add_argument("--open-only", action="store_true", help="Show only open trades")
    list_parser.add_argument("--closed-only", action="store_true", help="Show only closed trades")
    list_parser.add_argument("--symbol", help="Filter by symbol")
    list_parser.add_argument("--export", help="Export to CSV file")

    # manual show - show details of a specific trade
    show_parser = manual_subparsers.add_parser("show", help="Show details of a specific trade")
    show_parser.add_argument("trade_id", help="Trade ID to show")

    # manual delete - delete a manual trade
    delete_parser = manual_subparsers.add_parser("delete", help="Delete a manual trade")
    delete_parser.add_argument("trade_id", help="Trade ID to delete")
    delete_parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")


def handle_manual_add(args) -> int:
    """Handle manual add command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        db = ManualTradeDatabase()

        # Parse entry date
        if args.entry_date:
            entry_date = datetime.strptime(args.entry_date, "%Y-%m-%d")
        else:
            entry_date = datetime.now()

        # Create trade
        trade = ManualTrade(
            trade_id=str(uuid.uuid4()),
            symbol=args.symbol.upper(),
            asset_class=args.asset_class,
            side=PositionSide(args.side),
            entry_date=entry_date,
            entry_price=args.entry_price,
            quantity=args.quantity,
            stop_price=args.stop_price,
            initial_stop_price=args.stop_price,
            notes=args.notes,
            tags=args.tags,
        )

        # Validate stop price
        if trade.side == PositionSide.LONG and trade.stop_price >= trade.entry_price:
            print(f"\nError: Stop price must be below entry price for LONG positions\n")
            return 1
        elif trade.side == PositionSide.SHORT and trade.stop_price <= trade.entry_price:
            print(f"\nError: Stop price must be above entry price for SHORT positions\n")
            return 1

        # Save to database
        trade_id = db.create_trade(trade)

        print("\n" + "=" * 80)
        print("MANUAL TRADE ADDED")
        print("=" * 80)
        print(f"\nTrade ID: {trade_id}")
        print(f"Symbol: {trade.symbol}")
        print(f"Side: {trade.side.value}")
        print(f"Quantity: {trade.quantity}")
        print(f"Entry Price: ${trade.entry_price:.2f}")
        print(f"Stop Price: ${trade.stop_price:.2f}")
        print(f"Entry Date: {trade.entry_date.strftime('%Y-%m-%d')}")
        if trade.notes:
            print(f"Notes: {trade.notes}")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Error adding manual trade: {e}", exc_info=True)
        print(f"\nError: {e}\n")
        return 1


def handle_manual_close(args) -> int:
    """Handle manual close command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        db = ManualTradeDatabase()

        # Parse exit date
        if args.exit_date:
            exit_date = datetime.strptime(args.exit_date, "%Y-%m-%d")
        else:
            exit_date = datetime.now()

        # Close trade
        db.close_trade(
            trade_id=args.trade_id,
            exit_date=exit_date,
            exit_price=args.exit_price,
            exit_reason=args.reason,
        )

        # Get updated trade
        trade = db.get_trade(args.trade_id)
        if not trade:
            print(f"\nError: Trade {args.trade_id} not found\n")
            return 1

        print("\n" + "=" * 80)
        print("MANUAL TRADE CLOSED")
        print("=" * 80)
        print(f"\nTrade ID: {trade.trade_id}")
        print(f"Symbol: {trade.symbol}")
        print(f"Side: {trade.side.value}")
        print(f"Quantity: {trade.quantity}")
        print(f"Entry Price: ${trade.entry_price:.2f}")
        print(f"Exit Price: ${trade.exit_price:.2f}")
        print(f"Realized P&L: ${trade.realized_pnl:,.2f}")
        print(f"Exit Date: {trade.exit_date.strftime('%Y-%m-%d')}")
        print(f"Exit Reason: {trade.exit_reason}")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Error closing manual trade: {e}", exc_info=True)
        print(f"\nError: {e}\n")
        return 1


def handle_manual_update(args) -> int:
    """Handle manual update command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        db = ManualTradeDatabase()

        # Get existing trade
        trade = db.get_trade(args.trade_id)
        if not trade:
            print(f"\nError: Trade {args.trade_id} not found\n")
            return 1

        # Update fields
        if args.stop_price is not None:
            trade.stop_price = args.stop_price
        if args.notes is not None:
            trade.notes = args.notes
        if args.tags is not None:
            trade.tags = args.tags

        # Save updates
        db.update_trade(trade)

        print("\n" + "=" * 80)
        print("MANUAL TRADE UPDATED")
        print("=" * 80)
        print(f"\nTrade ID: {trade.trade_id}")
        print(f"Symbol: {trade.symbol}")
        print(f"Stop Price: ${trade.stop_price:.2f}")
        if trade.notes:
            print(f"Notes: {trade.notes}")
        if trade.tags:
            print(f"Tags: {trade.tags}")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Error updating manual trade: {e}", exc_info=True)
        print(f"\nError: {e}\n")
        return 1


def handle_manual_list(args) -> int:
    """Handle manual list command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        db = ManualTradeDatabase()

        # Get trades based on filters
        if args.open_only:
            trades = db.get_open_trades()
        elif args.closed_only:
            trades = db.get_closed_trades()
        else:
            trades = db.get_all_trades()

        # Filter by symbol if requested
        if args.symbol:
            trades = [t for t in trades if t.symbol.upper() == args.symbol.upper()]

        if not trades:
            print("\nNo trades found.\n")
            return 0

        print("\n" + "=" * 80)
        print("MANUAL TRADES")
        print("=" * 80)
        print(f"\nTotal Trades: {len(trades)}\n")

        # Calculate summary
        open_trades = [t for t in trades if t.is_open()]
        closed_trades = [t for t in trades if not t.is_open()]
        total_realized_pnl = sum(t.realized_pnl for t in closed_trades)

        print(f"Open: {len(open_trades)}")
        print(f"Closed: {len(closed_trades)}")
        if closed_trades:
            print(f"Total Realized P&L: ${total_realized_pnl:,.2f}\n")

        print("-" * 80)
        print(f"{'Trade ID':<38} {'Symbol':<10} {'Side':<6} {'Qty':<8} {'Entry':<10} " f"{'Status':<8} {'P&L':<12}")
        print("-" * 80)

        for trade in trades:
            status = "OPEN" if trade.is_open() else "CLOSED"
            pnl = trade.unrealized_pnl if trade.is_open() else trade.realized_pnl
            pnl_str = f"${pnl:,.2f}"

            print(
                f"{trade.trade_id:<38} {trade.symbol:<10} {trade.side.value:<6} "
                f"{trade.quantity:<8} ${trade.entry_price:<9.2f} {status:<8} {pnl_str:<12}"
            )

        print("=" * 80 + "\n")

        # Export if requested
        if args.export:
            import pandas as pd

            data = []
            for trade in trades:
                data.append(
                    {
                        "trade_id": trade.trade_id,
                        "symbol": trade.symbol,
                        "asset_class": trade.asset_class,
                        "side": trade.side.value,
                        "quantity": trade.quantity,
                        "entry_price": trade.entry_price,
                        "entry_date": trade.entry_date.isoformat(),
                        "stop_price": trade.stop_price,
                        "exit_price": trade.exit_price,
                        "exit_date": trade.exit_date.isoformat() if trade.exit_date else None,
                        "exit_reason": trade.exit_reason,
                        "realized_pnl": trade.realized_pnl,
                        "unrealized_pnl": trade.unrealized_pnl,
                        "notes": trade.notes,
                        "tags": trade.tags,
                    }
                )

            df = pd.DataFrame(data)
            df.to_csv(args.export, index=False)
            print(f"Exported {len(trades)} trades to {args.export}\n")

        return 0

    except Exception as e:
        logger.error(f"Error listing manual trades: {e}", exc_info=True)
        print(f"\nError: {e}\n")
        return 1


def handle_manual_show(args) -> int:
    """Handle manual show command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        db = ManualTradeDatabase()
        trade = db.get_trade(args.trade_id)

        if not trade:
            print(f"\nError: Trade {args.trade_id} not found\n")
            return 1

        print("\n" + "=" * 80)
        print("MANUAL TRADE DETAILS")
        print("=" * 80)
        print(f"\nTrade ID: {trade.trade_id}")
        print(f"Symbol: {trade.symbol}")
        print(f"Asset Class: {trade.asset_class}")
        print(f"Side: {trade.side.value}")
        print(f"Quantity: {trade.quantity}")
        print(f"\nEntry:")
        print(f"  Date: {trade.entry_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Price: ${trade.entry_price:.2f}")
        print(f"  Stop: ${trade.stop_price:.2f}")
        print(f"  Initial Stop: ${trade.initial_stop_price:.2f}")

        if trade.exit_date:
            print(f"\nExit:")
            print(f"  Date: {trade.exit_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Price: ${trade.exit_price:.2f}")
            print(f"  Reason: {trade.exit_reason}")
            print(f"\nRealized P&L: ${trade.realized_pnl:,.2f}")
        else:
            print(f"\nStatus: OPEN")
            print(f"Unrealized P&L: ${trade.unrealized_pnl:,.2f}")

        if trade.notes:
            print(f"\nNotes: {trade.notes}")
        if trade.tags:
            print(f"Tags: {trade.tags}")

        print(f"\nMetadata:")
        print(f"  Created: {trade.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Updated: {trade.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Error showing manual trade: {e}", exc_info=True)
        print(f"\nError: {e}\n")
        return 1


def handle_manual_delete(args) -> int:
    """Handle manual delete command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        db = ManualTradeDatabase()

        # Get trade to confirm
        trade = db.get_trade(args.trade_id)
        if not trade:
            print(f"\nError: Trade {args.trade_id} not found\n")
            return 1

        # Confirm deletion
        if not args.confirm:
            print(f"\nAre you sure you want to delete trade {args.trade_id} ({trade.symbol})? [y/N] ", end="")
            response = input().strip().lower()
            if response not in ["y", "yes"]:
                print("Deletion cancelled.\n")
                return 0

        # Delete
        db.delete_trade(args.trade_id)

        print(f"\nTrade {args.trade_id} deleted successfully.\n")
        return 0

    except Exception as e:
        logger.error(f"Error deleting manual trade: {e}", exc_info=True)
        print(f"\nError: {e}\n")
        return 1


def handle_manual_trades_command(args) -> int:
    """Handle manual trades commands.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    if args.manual_command == "add":
        return handle_manual_add(args)
    elif args.manual_command == "close":
        return handle_manual_close(args)
    elif args.manual_command == "update":
        return handle_manual_update(args)
    elif args.manual_command == "list":
        return handle_manual_list(args)
    elif args.manual_command == "show":
        return handle_manual_show(args)
    elif args.manual_command == "delete":
        return handle_manual_delete(args)
    else:
        print("Error: No manual trades command specified. Use --help for usage.")
        return 1
