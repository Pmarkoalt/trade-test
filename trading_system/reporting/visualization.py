"""Visualization module for backtest results.

Provides static plotting functions for equity curves, drawdowns, trade distributions,
monthly returns heatmaps, and parameter sensitivity analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """Generate visualizations from backtest results."""
    
    def __init__(self, base_path: str, run_id: str):
        """Initialize visualizer.
        
        Args:
            base_path: Base path for results (e.g., "results/")
            run_id: Run ID to visualize
        """
        self.report_generator = ReportGenerator(base_path, run_id)
        self.base_path = base_path
        self.run_id = run_id
    
    def plot_equity_curve(
        self,
        period: str = "train",
        save_path: Optional[Path] = None,
        show_benchmark: bool = False,
        benchmark_symbol: str = "SPY"
    ) -> Figure:
        """Plot equity curve over time.
        
        Args:
            period: Period to plot ("train", "validation", "holdout")
            save_path: Optional path to save figure
            show_benchmark: Whether to overlay benchmark performance
            benchmark_symbol: Benchmark symbol to use (SPY, BTC, etc.)
        
        Returns:
            matplotlib Figure object
        """
        data = self.report_generator.load_period_data(period)
        if data is None:
            raise ValueError(f"No data found for period: {period}")
        
        equity_df = data['equity_curve_df']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot equity curve
        ax.plot(equity_df['date'], equity_df['equity'], 
                linewidth=2, label='Portfolio Equity', color='#2E86AB')
        
        # Add benchmark if requested
        if show_benchmark:
            # Try to load benchmark data from market data
            # For now, we'll skip this as it requires market data access
            # This could be enhanced to load from a benchmark CSV
            pass
        
        # Calculate and display key metrics
        start_equity = equity_df['equity'].iloc[0]
        end_equity = equity_df['equity'].iloc[-1]
        total_return = (end_equity / start_equity - 1) * 100
        
        # Add text annotation
        ax.text(0.02, 0.98, 
                f'Total Return: {total_return:.2f}%\n'
                f'Start: ${start_equity:,.0f}\n'
                f'End: ${end_equity:,.0f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.set_title(f'Equity Curve - {period.upper()} Period', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve saved to {save_path}")
        
        return fig
    
    def plot_drawdown(
        self,
        period: str = "train",
        save_path: Optional[Path] = None
    ) -> Figure:
        """Plot drawdown over time.
        
        Args:
            period: Period to plot ("train", "validation", "holdout")
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure object
        """
        data = self.report_generator.load_period_data(period)
        if data is None:
            raise ValueError(f"No data found for period: {period}")
        
        equity_df = data['equity_curve_df']
        equity = equity_df['equity'].values
        dates = equity_df['date'].values
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Calculate drawdown as percentage
        drawdown = ((equity - running_max) / running_max) * 100
        
        # Calculate max drawdown
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        max_dd_date = dates[max_dd_idx]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Fill area under drawdown curve
        ax.fill_between(dates, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        ax.plot(dates, drawdown, linewidth=1.5, color='darkred')
        
        # Mark maximum drawdown
        ax.plot(max_dd_date, max_dd, 'ro', markersize=10, label=f'Max DD: {max_dd:.2f}%')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title(f'Drawdown Analysis - {period.upper()} Period', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=min(drawdown) * 1.1, top=1)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drawdown plot saved to {save_path}")
        
        return fig
    
    def plot_trade_distribution(
        self,
        period: str = "train",
        save_path: Optional[Path] = None
    ) -> Figure:
        """Plot trade distribution charts (P&L histogram, R-multiple distribution, etc.).
        
        Args:
            period: Period to plot ("train", "validation", "holdout")
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure object
        """
        data = self.report_generator.load_period_data(period)
        if data is None:
            raise ValueError(f"No data found for period: {period}")
        
        trade_df = data['trade_log_df']
        
        if trade_df.empty:
            raise ValueError(f"No trades found for period: {period}")
        
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. P&L Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        pnl_values = trade_df['realized_pnl'].values
        ax1.hist(pnl_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Break Even')
        ax1.axvline(np.mean(pnl_values), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: ${np.mean(pnl_values):.2f}')
        ax1.set_xlabel('Realized P&L ($)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('P&L Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. R-Multiple Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if 'r_multiple' in trade_df.columns:
            r_multiples = trade_df['r_multiple'].dropna().values
            if len(r_multiples) > 0:
                ax2.hist(r_multiples, bins=50, alpha=0.7, color='orange', edgecolor='black')
                ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Break Even')
                ax2.axvline(np.mean(r_multiples), color='green', linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(r_multiples):.2f}R')
                ax2.set_xlabel('R-Multiple', fontsize=11)
                ax2.set_ylabel('Frequency', fontsize=11)
                ax2.set_title('R-Multiple Distribution', fontsize=12, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No R-multiple data', ha='center', va='center', 
                        transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'R-multiple not available', ha='center', va='center',
                    transform=ax2.transAxes)
        
        # 3. Win/Loss Pie Chart
        ax3 = fig.add_subplot(gs[1, 0])
        winning = len(trade_df[trade_df['realized_pnl'] > 0])
        losing = len(trade_df[trade_df['realized_pnl'] < 0])
        breakeven = len(trade_df[trade_df['realized_pnl'] == 0])
        
        # Build pie chart data, excluding zero values
        sizes = []
        labels = []
        colors = []
        explode = []
        
        if winning > 0:
            sizes.append(winning)
            labels.append('Winning')
            colors.append('green')
            explode.append(0.05)
        
        if losing > 0:
            sizes.append(losing)
            labels.append('Losing')
            colors.append('red')
            explode.append(0.05)
        
        if breakeven > 0:
            sizes.append(breakeven)
            labels.append('Breakeven')
            colors.append('gray')
            explode.append(0)
        
        if len(sizes) > 0:
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, explode=explode if len(explode) > 0 else None)
        else:
            ax3.text(0.5, 0.5, 'No trade data', ha='center', va='center',
                    transform=ax3.transAxes)
        
        ax3.set_title('Win/Loss Distribution', fontsize=12, fontweight='bold')
        
        # 4. Holding Period Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        if 'holding_days' in trade_df.columns:
            holding_days = trade_df['holding_days'].dropna().values
            if len(holding_days) > 0:
                ax4.hist(holding_days, bins=30, alpha=0.7, color='purple', edgecolor='black')
                ax4.axvline(np.mean(holding_days), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(holding_days):.1f} days')
                ax4.set_xlabel('Holding Period (days)', fontsize=11)
                ax4.set_ylabel('Frequency', fontsize=11)
                ax4.set_title('Holding Period Distribution', fontsize=12, fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No holding period data', ha='center', va='center',
                        transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'Holding period not available', ha='center', va='center',
                    transform=ax4.transAxes)
        
        fig.suptitle(f'Trade Distribution Analysis - {period.upper()} Period', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trade distribution plot saved to {save_path}")
        
        return fig
    
    def plot_monthly_returns_heatmap(
        self,
        period: str = "train",
        save_path: Optional[Path] = None
    ) -> Figure:
        """Plot monthly returns heatmap.
        
        Args:
            period: Period to plot ("train", "validation", "holdout")
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure object
        """
        data = self.report_generator.load_period_data(period)
        if data is None:
            raise ValueError(f"No data found for period: {period}")
        
        equity_df = data['equity_curve_df']
        
        # Calculate daily returns
        equity_df['return'] = equity_df['equity'].pct_change()
        
        # Extract year and month
        equity_df['year'] = equity_df['date'].dt.year
        equity_df['month'] = equity_df['date'].dt.month
        
        # Calculate monthly returns
        monthly_returns = equity_df.groupby(['year', 'month'])['return'].apply(
            lambda x: (1 + x).prod() - 1
        ) * 100  # Convert to percentage
        
        # Reshape for heatmap
        monthly_pivot = monthly_returns.unstack(level='month')
        monthly_pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(monthly_pivot) * 0.5)))
        
        # Create heatmap
        im = ax.imshow(monthly_pivot.values, cmap='RdYlGn', aspect='auto', 
                      vmin=-10, vmax=10)  # Adjust vmin/vmax as needed
        
        # Set ticks
        ax.set_xticks(np.arange(len(monthly_pivot.columns)))
        ax.set_yticks(np.arange(len(monthly_pivot.index)))
        ax.set_xticklabels(monthly_pivot.columns)
        ax.set_yticklabels(monthly_pivot.index)
        
        # Add text annotations
        for i in range(len(monthly_pivot.index)):
            for j in range(len(monthly_pivot.columns)):
                value = monthly_pivot.iloc[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.1f}%',
                                 ha="center", va="center", color="black", fontsize=9)
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        ax.set_title(f'Monthly Returns Heatmap - {period.upper()} Period', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Return (%)', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Monthly returns heatmap saved to {save_path}")
        
        return fig
    
    def plot_parameter_sensitivity_heatmap(
        self,
        parameter_name: str,
        parameter_values: List[float],
        metric_name: str = "sharpe_ratio",
        save_path: Optional[Path] = None
    ) -> Optional[Figure]:
        """Plot parameter sensitivity heatmap.
        
        This requires multiple backtest runs with different parameter values.
        For now, this is a placeholder that can be extended when parameter
        sweep functionality is available.
        
        Args:
            parameter_name: Name of parameter being varied
            parameter_values: List of parameter values tested
            metric_name: Metric to visualize (e.g., "sharpe_ratio", "max_drawdown")
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure object, or None if data not available
        """
        # This would require loading multiple run results
        # For now, return None as this is a placeholder
        logger.warning("Parameter sensitivity heatmap requires multiple runs. Not yet implemented.")
        return None
    
    def plot_all(
        self,
        period: str = "train",
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """Generate all visualizations for a period.
        
        Args:
            period: Period to visualize ("train", "validation", "holdout")
            output_dir: Directory to save plots. Defaults to run_dir/plots/{period}/
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        if output_dir is None:
            run_dir = Path(self.base_path) / self.run_id
            output_dir = run_dir / "plots" / period
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        try:
            fig = self.plot_equity_curve(period=period)
            path = output_dir / "equity_curve.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plots['equity_curve'] = path
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
        
        try:
            fig = self.plot_drawdown(period=period)
            path = output_dir / "drawdown.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plots['drawdown'] = path
        except Exception as e:
            logger.error(f"Error plotting drawdown: {e}")
        
        try:
            fig = self.plot_trade_distribution(period=period)
            path = output_dir / "trade_distribution.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plots['trade_distribution'] = path
        except Exception as e:
            logger.error(f"Error plotting trade distribution: {e}")
        
        try:
            fig = self.plot_monthly_returns_heatmap(period=period)
            path = output_dir / "monthly_returns_heatmap.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plots['monthly_returns_heatmap'] = path
        except Exception as e:
            logger.error(f"Error plotting monthly returns heatmap: {e}")
        
        logger.info(f"Generated {len(plots)} plots in {output_dir}")
        return plots


def plot_equity_curve_from_data(
    equity_curve: List[float],
    dates: List[pd.Timestamp],
    save_path: Optional[Path] = None,
    title: str = "Equity Curve"
) -> Figure:
    """Plot equity curve from raw data (utility function).
    
    Args:
        equity_curve: List of equity values
        dates: List of dates
        save_path: Optional path to save figure
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates, equity_curve, linewidth=2, color='#2E86AB')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

