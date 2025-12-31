"""Correlation stress analysis during drawdowns."""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd


class CorrelationStressAnalysis:
    """Analyze correlation behavior during drawdowns."""
    
    def __init__(
        self,
        portfolio_history: Dict[pd.Timestamp, Dict],
        returns_data: Dict[str, pd.Series],
        lookback: int = 20
    ):
        """Initialize correlation stress analysis.
        
        Args:
            portfolio_history: Dictionary mapping dates to portfolio states
                Each state should have: equity, positions (dict of symbol -> position info)
            returns_data: Dictionary mapping symbols to daily return Series
            lookback: Lookback window for correlation calculation (days)
        """
        self.portfolio_history = portfolio_history
        self.returns_data = returns_data
        self.lookback = lookback
    
    def run(self) -> Dict:
        """Run correlation stress analysis.
        
        Returns:
            Dictionary with correlation statistics
        """
        # Compute drawdown for each date
        dates = sorted(self.portfolio_history.keys())
        equities = [self.portfolio_history[date]['equity'] for date in dates]
        
        # Compute rolling peaks and drawdowns
        peaks = self._compute_rolling_peaks(equities)
        drawdowns = [
            (eq - peak) / peak if peak > 0 else 0.0
            for eq, peak in zip(equities, peaks)
        ]
        
        # Classify periods
        normal_periods = []  # DD >= -5%
        drawdown_periods = []  # DD < -5%
        
        for date, dd in zip(dates, drawdowns):
            portfolio = self.portfolio_history[date]
            positions = portfolio.get('positions', {})
            
            if len(positions) < 2:
                continue  # Need at least 2 positions for correlation
            
            # Compute correlation
            position_symbols = list(positions.keys())
            corr_matrix = self._compute_correlation_matrix(
                position_symbols,
                date,
                self.lookback
            )
            
            if corr_matrix is None:
                continue
            
            avg_pairwise_corr = self._compute_avg_pairwise_corr(corr_matrix)
            
            if dd >= -0.05:
                normal_periods.append(avg_pairwise_corr)
            else:
                drawdown_periods.append(avg_pairwise_corr)
        
        # Compute statistics
        results = {
            'normal_avg_corr': (
                np.mean(normal_periods) if normal_periods else None
            ),
            'normal_std_corr': (
                np.std(normal_periods) if normal_periods else None
            ),
            'normal_count': len(normal_periods),
            'drawdown_avg_corr': (
                np.mean(drawdown_periods) if drawdown_periods else None
            ),
            'drawdown_std_corr': (
                np.std(drawdown_periods) if drawdown_periods else None
            ),
            'drawdown_count': len(drawdown_periods),
            'correlation_increase': None,
            'warning': False
        }
        
        if results['normal_avg_corr'] is not None and results['drawdown_avg_corr'] is not None:
            results['correlation_increase'] = (
                results['drawdown_avg_corr'] - results['normal_avg_corr']
            )
            
            # Warning if correlation increases significantly during drawdowns
            if results['drawdown_avg_corr'] > 0.70:
                results['warning'] = True
        
        return results
    
    def _compute_rolling_peaks(self, equity_curve: List[float]) -> List[float]:
        """Compute rolling peaks for drawdown calculation.
        
        Args:
            equity_curve: List of equity values
        
        Returns:
            List of peak values (rolling maximum)
        """
        peaks = []
        current_peak = equity_curve[0] if equity_curve else 0.0
        
        for equity in equity_curve:
            if equity > current_peak:
                current_peak = equity
            peaks.append(current_peak)
        
        return peaks
    
    def _compute_correlation_matrix(
        self,
        symbols: List[str],
        date: pd.Timestamp,
        lookback: int
    ) -> Optional[np.ndarray]:
        """Compute correlation matrix for symbols at date.
        
        Uses rolling window of daily returns.
        
        Args:
            symbols: List of symbols to compute correlation for
            date: Date to compute correlation at
            lookback: Number of days to look back
        
        Returns:
            Correlation matrix (n x n) or None if insufficient data
        """
        # Get returns for each symbol
        symbol_returns = []
        
        for symbol in symbols:
            if symbol not in self.returns_data:
                return None
            
            returns = self.returns_data[symbol]

            # Get window ending at date
            try:
                # Use get_indexer for nearest match (compatible with newer pandas)
                date_idx = returns.index.get_indexer([date], method='nearest')[0]
                if date_idx == -1:
                    return None
            except (KeyError, IndexError):
                return None
            
            if date_idx < lookback - 1:
                return None  # Insufficient history
            
            window_returns = returns.iloc[date_idx - lookback + 1:date_idx + 1]
            
            if len(window_returns) < lookback:
                return None
            
            symbol_returns.append(window_returns.values)
        
        if len(symbol_returns) < 2:
            return None
        
        # Compute correlation matrix
        try:
            returns_array = np.array(symbol_returns)
            corr_matrix = np.corrcoef(returns_array)
        except Exception:
            return None
        
        return corr_matrix
    
    def _compute_avg_pairwise_corr(self, corr_matrix: np.ndarray) -> float:
        """Compute average pairwise correlation (exclude diagonal).
        
        Args:
            corr_matrix: Correlation matrix (n x n)
        
        Returns:
            Average pairwise correlation
        """
        n = len(corr_matrix)
        if n < 2:
            return 0.0
        
        off_diagonal = []
        for i in range(n):
            for j in range(i + 1, n):
                corr_val = corr_matrix[i, j]
                if not np.isnan(corr_val):
                    off_diagonal.append(corr_val)
        
        return np.mean(off_diagonal) if off_diagonal else 0.0


def run_correlation_stress_analysis(
    portfolio_history: Dict[pd.Timestamp, Dict],
    returns_data: Dict[str, pd.Series],
    lookback: int = 20
) -> Dict:
    """Convenience function to run correlation stress analysis.
    
    Args:
        portfolio_history: Dictionary mapping dates to portfolio states
        returns_data: Dictionary mapping symbols to daily return Series
        lookback: Lookback window for correlation calculation
    
    Returns:
        Correlation stress analysis results
    """
    analysis = CorrelationStressAnalysis(
        portfolio_history,
        returns_data,
        lookback
    )
    return analysis.run()


def check_correlation_warnings(results: Dict) -> Tuple[bool, List[str]]:
    """Check correlation analysis results for warnings.
    
    Args:
        results: Correlation stress analysis results
    
    Returns:
        (passed, warnings)
    """
    warnings = []
    
    if results.get('warning', False):
        drawdown_corr = results.get('drawdown_avg_corr', 0.0)
        warnings.append(
            f"WARNING: Correlation during drawdowns {drawdown_corr:.2f} > 0.70 "
            f"(diversification fails during stress)"
        )
    
    if results.get('correlation_increase') is not None:
        corr_increase = results['correlation_increase']
        if corr_increase > 0.20:  # Correlation increases by more than 0.20
            warnings.append(
                f"WARNING: Correlation increases by {corr_increase:.2f} during drawdowns "
                f"(significant correlation breakdown)"
            )
    
    return (True, warnings)

