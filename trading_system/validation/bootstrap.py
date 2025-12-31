"""Bootstrap resampling test for statistical robustness."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import percentileofscore


def compute_sharpe_from_r_multiples(r_multiples: List[float], trades_per_year: float = 15.0) -> float:
    """Compute Sharpe ratio from R-multiples.

    Assumes:
    - Risk per trade = 0.75% of equity
    - R-multiple = profit/loss in units of risk
    - Convert to equity returns, then annualize

    Args:
        r_multiples: List of R-multiples per trade
        trades_per_year: Average number of trades per year (for annualization)

    Returns:
        Sharpe ratio (annualized)
    """
    if len(r_multiples) == 0:
        return 0.0

    # Handle single value case
    if len(r_multiples) == 1:
        return 0.0

    mean_r = np.mean(r_multiples)
    std_r = np.std(r_multiples)

    # Handle zero or near-zero std (constant returns)
    # Also handle NaN/inf cases
    if not np.isfinite(std_r) or std_r == 0.0 or std_r < 1e-10:
        return 0.0

    # Annualize (assume ~15 trades per year, or use actual frequency)
    # For momentum system: ~10-20 trades/year, so sqrt(15) ≈ 3.87
    sharpe = mean_r / std_r * np.sqrt(trades_per_year)

    # Ensure result is finite
    if not np.isfinite(sharpe):
        return 0.0

    return sharpe


def compute_max_drawdown_from_r_multiples(
    r_multiples: List[float], starting_equity: float = 100000.0, risk_per_trade: float = 0.0075
) -> float:
    """Compute max drawdown from R-multiples.

    Build equity curve from R-multiples, then compute DD.

    Args:
        r_multiples: List of R-multiples per trade
        starting_equity: Starting equity value
        risk_per_trade: Risk per trade as fraction of equity

    Returns:
        Maximum drawdown (negative value, e.g., -0.15 for -15%)
    """
    if len(r_multiples) == 0:
        return 0.0

    # Build equity curve
    equity_curve = [starting_equity]
    for r_mult in r_multiples:
        # Each trade changes equity by: R_multiple * risk_per_trade * current_equity
        equity_change = r_mult * risk_per_trade * equity_curve[-1]
        new_equity = equity_curve[-1] + equity_change
        equity_curve.append(new_equity)

    # Compute drawdown
    peak = equity_curve[0]
    max_dd = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak
        if dd < max_dd:
            max_dd = dd

    return max_dd


class BootstrapTest:
    """Bootstrap resampling test for trade returns."""

    def __init__(
        self,
        r_multiples: List[float],
        n_iterations: int = 1000,
        starting_equity: float = 100000.0,
        risk_per_trade: float = 0.0075,
        trades_per_year: float = 15.0,
        random_seed: Optional[int] = None,
    ):
        """Initialize bootstrap test.

        Args:
            r_multiples: List of R-multiples per trade (closed positions only)
            n_iterations: Number of bootstrap samples
            starting_equity: Starting equity value
            risk_per_trade: Risk per trade as fraction of equity
            trades_per_year: Average number of trades per year
            random_seed: Random seed for reproducibility
        """
        self.r_multiples = np.array(r_multiples)
        self.n_iterations = n_iterations
        self.starting_equity = starting_equity
        self.risk_per_trade = risk_per_trade
        self.trades_per_year = trades_per_year
        self.random_seed = random_seed

    def run(self) -> Dict:
        """Run bootstrap analysis.

        Returns:
            Dictionary with percentile results and statistics
        """
        # Seed here so each run() is reproducible
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        # Initialize storage
        sharpe_samples = []
        max_dd_samples = []
        calmar_samples = []
        expectancy_samples = []

        # Original metrics (for comparison)
        original_sharpe = compute_sharpe_from_r_multiples(self.r_multiples.tolist(), self.trades_per_year)
        original_max_dd = compute_max_drawdown_from_r_multiples(
            self.r_multiples.tolist(), self.starting_equity, self.risk_per_trade
        )
        original_calmar = original_sharpe / abs(original_max_dd) if original_max_dd != 0 else 0.0
        original_expectancy = np.mean(self.r_multiples)

        # Bootstrap loop
        for _ in range(self.n_iterations):
            # Resample with replacement (same size as original)
            sample_returns = np.random.choice(self.r_multiples, size=len(self.r_multiples), replace=True)

            # Compute metrics on resampled data
            sharpe = compute_sharpe_from_r_multiples(sample_returns.tolist(), self.trades_per_year)
            max_dd = compute_max_drawdown_from_r_multiples(sample_returns.tolist(), self.starting_equity, self.risk_per_trade)
            calmar = sharpe / abs(max_dd) if max_dd != 0 else 0.0
            expectancy = np.mean(sample_returns)

            # Store results
            sharpe_samples.append(sharpe)
            max_dd_samples.append(max_dd)
            calmar_samples.append(calmar)
            expectancy_samples.append(expectancy)

        # Compute percentiles
        sharpe_samples = np.array(sharpe_samples)
        max_dd_samples = np.array(max_dd_samples)
        calmar_samples = np.array(calmar_samples)
        expectancy_samples = np.array(expectancy_samples)

        results = {
            # Sharpe ratio
            "sharpe_5th": np.percentile(sharpe_samples, 5),
            "sharpe_25th": np.percentile(sharpe_samples, 25),
            "sharpe_50th": np.percentile(sharpe_samples, 50),
            "sharpe_75th": np.percentile(sharpe_samples, 75),
            "sharpe_95th": np.percentile(sharpe_samples, 95),
            # Max drawdown
            "max_dd_5th": np.percentile(max_dd_samples, 5),
            "max_dd_50th": np.percentile(max_dd_samples, 50),
            "max_dd_95th": np.percentile(max_dd_samples, 95),
            # Calmar ratio
            "calmar_5th": np.percentile(calmar_samples, 5),
            "calmar_50th": np.percentile(calmar_samples, 50),
            "calmar_95th": np.percentile(calmar_samples, 95),
            # Expectancy
            "expectancy_5th": np.percentile(expectancy_samples, 5),
            "expectancy_50th": np.percentile(expectancy_samples, 50),
            "expectancy_95th": np.percentile(expectancy_samples, 95),
            # Original values
            "original_sharpe": original_sharpe,
            "original_max_dd": original_max_dd,
            "original_calmar": original_calmar,
            "original_expectancy": original_expectancy,
            # Percentile ranks (where original falls in distribution)
            "sharpe_percentile_rank": percentileofscore(sharpe_samples, original_sharpe),
            "calmar_percentile_rank": percentileofscore(calmar_samples, original_calmar),
        }

        return results


def run_bootstrap_test(r_multiples: List[float], n_iterations: int = 1000, random_seed: Optional[int] = None) -> Dict:
    """Convenience function to run bootstrap test.

    Args:
        r_multiples: List of R-multiples per trade
        n_iterations: Number of bootstrap samples
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with bootstrap results
    """
    test = BootstrapTest(r_multiples, n_iterations, random_seed=random_seed)
    return test.run()


def check_bootstrap_results(results: Dict) -> Tuple[bool, List[str]]:
    """Check bootstrap results against acceptance criteria.

    Returns:
        (passed, warnings)
    """
    warnings = []

    # Rejection: Sharpe 5th percentile < 0.4
    if results["sharpe_5th"] < 0.4:
        return (False, ["REJECT: Bootstrap Sharpe 5th percentile < 0.4"])

    # Warning: Sharpe 5th percentile < 0.6
    if results["sharpe_5th"] < 0.6:
        warnings.append("WARNING: Bootstrap Sharpe 5th percentile < 0.6 (fragile)")

    # Warning: Max DD 95th percentile > 25%
    if results["max_dd_95th"] > 0.25:
        warnings.append("WARNING: Bootstrap Max DD 95th percentile > 25% (tail risk)")

    # Check percentile rank (original should be near median)
    if results["sharpe_percentile_rank"] < 40 or results["sharpe_percentile_rank"] > 60:
        warnings.append(f"WARNING: Original Sharpe at {results['sharpe_percentile_rank']:.1f}th " f"percentile (unusual)")

    return (True, warnings)
