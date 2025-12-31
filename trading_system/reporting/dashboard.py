"""Interactive dashboard for backtest results using Streamlit.

To run the dashboard:
    streamlit run trading_system/reporting/dashboard.py -- --base_path results --run_id <run_id>

Or use the CLI command:
    python -m trading_system.cli dashboard --base_path results --run_id <run_id>
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

from .report_generator import ReportGenerator
from .visualization import BacktestVisualizer

logger = logging.getLogger(__name__)


def check_streamlit():
    """Check if Streamlit is available."""
    if not STREAMLIT_AVAILABLE:
        st.error("Streamlit is not installed. Install it with: pip install streamlit")
        st.stop()


def load_period_selector(report_generator: ReportGenerator) -> Optional[str]:
    """Create period selector widget.

    Args:
        report_generator: ReportGenerator instance

    Returns:
        Selected period name or None
    """
    periods = []
    for period in ["train", "validation", "holdout"]:
        data = report_generator.load_period_data(period)
        if data is not None:
            periods.append(period)

    if len(periods) == 0:
        st.error("No period data found")
        return None

    selected_period = st.sidebar.selectbox("Select Period", periods, index=0)

    return selected_period


def render_metrics_summary(metrics: Dict[str, float]):
    """Render metrics summary cards.

    Args:
        metrics: Dictionary of metric values
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")

    with col2:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")

    with col3:
        st.metric("Total Trades", f"{int(metrics.get('total_trades', 0))}")

    with col4:
        st.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.2f}%")

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}")

    with col6:
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

    with col7:
        st.metric("Expectancy (R)", f"{metrics.get('expectancy', 0):.3f}")

    with col8:
        st.metric("Recovery Factor", f"{metrics.get('recovery_factor', 0):.2f}")


def render_equity_curve(data: Dict[str, Any], period: str):
    """Render equity curve chart.

    Args:
        data: Period data dictionary
        period: Period name
    """
    st.subheader(f"Equity Curve - {period.upper()}")

    equity_df = data["equity_curve_df"]

    # Create chart using Streamlit's native charting
    chart_data = equity_df[["date", "equity"]].copy()
    chart_data = chart_data.set_index("date")

    st.line_chart(chart_data)

    # Display summary stats
    col1, col2, col3 = st.columns(3)
    start_equity = equity_df["equity"].iloc[0]
    end_equity = equity_df["equity"].iloc[-1]
    total_return = (end_equity / start_equity - 1) * 100

    with col1:
        st.write(f"**Start Equity:** ${start_equity:,.2f}")
    with col2:
        st.write(f"**End Equity:** ${end_equity:,.2f}")
    with col3:
        st.write(f"**Total Return:** {total_return:.2f}%")


def render_drawdown(data: Dict[str, Any], period: str):
    """Render drawdown chart.

    Args:
        data: Period data dictionary
        period: Period name
    """
    st.subheader(f"Drawdown Analysis - {period.upper()}")

    equity_df = data["equity_curve_df"]
    equity = equity_df["equity"].values

    # Calculate drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = ((equity - running_max) / running_max) * 100

    # Create DataFrame for charting
    drawdown_df = pd.DataFrame({"date": equity_df["date"], "drawdown": drawdown}).set_index("date")

    st.area_chart(drawdown_df)

    # Display max drawdown
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    max_dd_date = equity_df["date"].iloc[max_dd_idx]

    st.write(f"**Maximum Drawdown:** {max_dd:.2f}% on {max_dd_date.strftime('%Y-%m-%d')}")


def render_trade_analysis(data: Dict[str, Any], period: str):
    """Render trade analysis section.

    Args:
        data: Period data dictionary
        period: Period name
    """
    st.subheader(f"Trade Analysis - {period.upper()}")

    trade_df = data["trade_log_df"]

    if trade_df.empty:
        st.warning("No trades found for this period")
        return

    # Trade statistics
    col1, col2, col3, col4 = st.columns(4)

    total_trades = len(trade_df)
    winning_trades = len(trade_df[trade_df["realized_pnl"] > 0])
    losing_trades = len(trade_df[trade_df["realized_pnl"] < 0])
    total_pnl = trade_df["realized_pnl"].sum()

    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Winning Trades", winning_trades)
    with col3:
        st.metric("Losing Trades", losing_trades)
    with col4:
        st.metric("Total P&L", f"${total_pnl:,.2f}")

    # P&L Distribution
    st.write("**P&L Distribution**")
    pnl_hist = pd.DataFrame({"P&L": trade_df["realized_pnl"].values})
    st.hist_chart(pnl_hist, x="P&L", bins=50)

    # Trade table
    st.write("**Recent Trades**")
    display_cols = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "quantity", "realized_pnl"]
    available_cols = [col for col in display_cols if col in trade_df.columns]

    st.dataframe(trade_df[available_cols].tail(20), use_container_width=True)


def render_monthly_returns(data: Dict[str, Any], period: str):
    """Render monthly returns heatmap.

    Args:
        data: Period data dictionary
        period: Period name
    """
    st.subheader(f"Monthly Returns - {period.upper()}")

    equity_df = data["equity_curve_df"]

    # Calculate daily returns
    equity_df = equity_df.copy()
    equity_df["return"] = equity_df["equity"].pct_change()

    # Extract year and month
    equity_df["year"] = equity_df["date"].dt.year
    equity_df["month"] = equity_df["date"].dt.month

    # Calculate monthly returns
    monthly_returns = equity_df.groupby(["year", "month"])["return"].apply(lambda x: (1 + x).prod() - 1) * 100

    # Reshape for display
    monthly_pivot = monthly_returns.unstack(level="month")
    monthly_pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Display as table with color coding
    st.dataframe(monthly_pivot.style.background_gradient(cmap="RdYlGn", vmin=-10, vmax=10), use_container_width=True)


def main():
    """Main dashboard function."""
    check_streamlit()

    st.set_page_config(page_title="Backtest Results Dashboard", page_icon="📊", layout="wide")

    st.title("📊 Backtest Results Dashboard")

    # Parse command line arguments or use sidebar inputs
    if len(sys.argv) > 1:
        # Parse from command line
        args = {}
        i = 1
        while i < len(sys.argv):
            if sys.argv[i].startswith("--"):
                key = sys.argv[i][2:]
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                    args[key] = sys.argv[i + 1]
                    i += 2
                else:
                    args[key] = True
                    i += 1
            else:
                i += 1

        base_path = args.get("base_path", "results")
        run_id = args.get("run_id", None)
    else:
        # Use sidebar inputs
        base_path = st.sidebar.text_input("Base Path", value="results")
        run_id = st.sidebar.text_input("Run ID", value="")

    if not run_id:
        st.error("Please provide a Run ID")
        st.info("Usage: streamlit run dashboard.py -- --base_path results --run_id <run_id>")
        st.stop()

    try:
        report_generator = ReportGenerator(base_path, run_id)
        visualizer = BacktestVisualizer(base_path, run_id)
    except FileNotFoundError as e:
        st.error(f"Run not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Period selector
    selected_period = load_period_selector(report_generator)
    if selected_period is None:
        st.stop()

    # Load period data
    data = report_generator.load_period_data(selected_period)
    if data is None:
        st.error(f"Could not load data for period: {selected_period}")
        st.stop()

    # Compute metrics
    try:
        metrics = report_generator.compute_metrics_from_data(data)
    except Exception as e:
        st.error(f"Error computing metrics: {e}")
        metrics = {}

    # Render dashboard sections
    render_metrics_summary(metrics)

    st.divider()

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Equity Curve", "Drawdown", "Trade Analysis", "Monthly Returns"])

    with tab1:
        render_equity_curve(data, selected_period)

    with tab2:
        render_drawdown(data, selected_period)

    with tab3:
        render_trade_analysis(data, selected_period)

    with tab4:
        render_monthly_returns(data, selected_period)

    # Sidebar info
    st.sidebar.divider()
    st.sidebar.write(f"**Run ID:** {run_id}")
    st.sidebar.write(f"**Period:** {selected_period.upper()}")

    if st.sidebar.button("Generate Static Plots"):
        with st.spinner("Generating plots..."):
            try:
                plots = visualizer.plot_all(period=selected_period)
                st.sidebar.success(f"Generated {len(plots)} plots")
                for name, path in plots.items():
                    st.sidebar.write(f"- {name}: {path}")
            except Exception as e:
                st.sidebar.error(f"Error generating plots: {e}")


if __name__ == "__main__":
    main()
