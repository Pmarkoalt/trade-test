#!/usr/bin/env python
"""Daily trading workflow automation.

This script orchestrates the complete daily workflow:
1. Generate signals (Agent 1 + Agent 2)
2. Send newsletter (Agent 3)
3. Execute paper trading (Agent 4)
4. Update unified positions view (Agent 4)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.adapters.alpaca_adapter import AlpacaAdapter
from trading_system.adapters.base_adapter import AdapterConfig
from trading_system.execution.paper_trading import PaperTradingConfig, PaperTradingRunner
from trading_system.reporting.unified_positions import UnifiedPositionView
from trading_system.storage.manual_trades import ManualTradeDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_daily_workflow():
    """Run complete daily workflow."""

    print(f"\n{'=' * 80}")
    print(f"DAILY TRADING WORKFLOW - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")

    # Step 1: Generate signals (Agent 1 + Agent 2)
    print("Step 1: Generating daily signals...")
    print("  → This would call the signal generation service from Agent 1 + Agent 2")
    print("  → Signals would be generated for equity and crypto buckets")
    print("  ✓ Skipped (integrate with Agent 1/2 signal generation)\n")

    # Step 2: Send newsletter (Agent 3)
    print("Step 2: Sending newsletter...")
    print("  → This would call the newsletter service from Agent 3")
    print("  → Newsletter would include signals and current positions")
    print("  ✓ Skipped (integrate with Agent 3 newsletter service)\n")

    # Step 3: Execute paper trading (Agent 4)
    print("Step 3: Executing paper trading orders...")

    # Check if Alpaca credentials are configured
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        print("  ⚠ Alpaca credentials not configured")
        print("  → Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        print("  ✓ Skipped paper trading execution\n")
    else:
        try:
            adapter_config = AdapterConfig(
                api_key=api_key,
                api_secret=api_secret,
                paper_trading=True,
            )

            paper_config = PaperTradingConfig(adapter_config=adapter_config)
            adapter = AlpacaAdapter(adapter_config)

            with adapter:
                runner = PaperTradingRunner(config=paper_config, adapter=adapter)

                # Get account status
                account = runner.get_account_info()
                print(f"  ✓ Connected to Alpaca paper account")
                print(f"  ✓ Account equity: ${account.equity:,.2f}")
                print(f"  ✓ Cash available: ${account.cash:,.2f}")

                # Reconcile positions
                positions = runner.reconcile_positions()
                print(f"  ✓ Reconciled {len(positions)} paper positions\n")

                # Note: Order submission would happen here
                # results = runner.submit_orders(orders)

        except Exception as e:
            logger.error(f"Error in paper trading: {e}", exc_info=True)
            print(f"  ✗ Error in paper trading: {e}\n")
            return 1

    # Step 4: Update unified positions view
    print("Step 4: Updating unified positions view...")

    try:
        manual_db = ManualTradeDatabase()

        # Create unified view (with or without paper adapter)
        if api_key and api_secret:
            adapter_config = AdapterConfig(
                api_key=api_key,
                api_secret=api_secret,
                paper_trading=True,
            )
            adapter = AlpacaAdapter(adapter_config)
            adapter.connect()
            view = UnifiedPositionView(manual_db=manual_db, paper_adapter=adapter)
        else:
            view = UnifiedPositionView(manual_db=manual_db, paper_adapter=None)

        # Get exposure summary
        exposure = view.get_exposure_summary()
        print(f"  ✓ Total positions: {exposure['total_positions']}")
        print(f"  ✓ Gross exposure: ${exposure['gross_exposure']:,.2f}")
        print(f"  ✓ Net exposure: ${exposure['net_exposure']:,.2f}")
        print(f"  ✓ Equity exposure: ${exposure['equity_notional']:,.2f}")
        print(f"  ✓ Crypto exposure: ${exposure['crypto_notional']:,.2f}")

        # Export daily snapshot
        output_dir = Path("results/daily_snapshots")
        output_dir.mkdir(parents=True, exist_ok=True)

        snapshot_file = output_dir / f"positions_{datetime.now().strftime('%Y%m%d')}.csv"
        view.export_to_csv(str(snapshot_file), open_only=True)
        print(f"  ✓ Exported snapshot to {snapshot_file}\n")

        # Print summary
        if exposure["total_positions"] > 0:
            print("Current Positions Summary:")
            view.print_summary()

        if api_key and api_secret:
            adapter.disconnect()

    except Exception as e:
        logger.error(f"Error updating positions view: {e}", exc_info=True)
        print(f"  ✗ Error updating positions view: {e}\n")
        return 1

    print(f"{'=' * 80}")
    print("DAILY WORKFLOW COMPLETE")
    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(run_daily_workflow())
