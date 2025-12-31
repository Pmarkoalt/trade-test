"""Main backtest engine for walk-forward testing."""

import bisect
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Create a no-op tqdm if not available
    def tqdm(iterable, *args, **kwargs):
        return iterable


class BacktestTimeoutError(Exception):
    """Raised when backtest exceeds maximum allowed duration."""

    pass


from ..data.calendar import get_trading_days
from ..data.loader import load_all_data
from ..indicators.feature_computer import compute_features
from ..ml.feature_engineering import MLFeatureEngineer
from ..ml.models import MLModel
from ..ml.predictor import MLPredictor
from ..models.market_data import MarketData
from ..models.positions import Position
from ..portfolio.portfolio import Portfolio
from ..strategies.base.strategy_interface import StrategyInterface
from .event_loop import DailyEventLoop
from .splits import WalkForwardSplit

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Main backtest engine with walk-forward splits and event-driven loop.

    This engine implements:
    - Walk-forward splits (train/validation/holdout)
    - Event-driven daily loop with no lookahead
    - Portfolio state management
    - Trade logging and metrics
    """

    # Default timeout values (in seconds)
    DEFAULT_TIMEOUT_SECONDS = 3600  # 1 hour
    TIMEOUT_CHECK_INTERVAL = 10  # Check timeout every N days

    def __init__(
        self,
        market_data: MarketData,
        strategies: List[StrategyInterface],
        starting_equity: float = 100000.0,
        seed: Optional[int] = None,
        slippage_multiplier: float = 1.0,
        crash_dates: Optional[List[pd.Timestamp]] = None,
        max_duration_seconds: Optional[int] = None,
    ):
        """Initialize backtest engine.

        Args:
            market_data: MarketData container with bars and benchmarks
            strategies: List of strategy objects (equity, crypto)
            starting_equity: Starting equity (default: 100,000)
            seed: Optional random seed for reproducibility
            slippage_multiplier: Multiplier for base slippage (default: 1.0, for stress tests: 2.0, 3.0, 5.0)
            crash_dates: List of dates on which to simulate flash crashes (5x slippage + forced stops)
            max_duration_seconds: Maximum time allowed for backtest (default: 3600 = 1 hour).
                                  Set to None to disable timeout.
        """
        self.market_data = market_data
        self.strategies = strategies
        self.starting_equity = starting_equity
        self.seed = seed
        self.slippage_multiplier = slippage_multiplier
        self.crash_dates = set(crash_dates) if crash_dates else set()
        self.max_duration_seconds = max_duration_seconds if max_duration_seconds is not None else self.DEFAULT_TIMEOUT_SECONDS

        # Initialize random number generator
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # Results storage
        self.results: Dict[str, Any] = {}
        self.daily_events: List[Dict] = []
        self.closed_trades: List[Position] = []

        # Initialize portfolio
        self.portfolio = Portfolio(
            date=pd.Timestamp.now(),  # Will be updated during backtest
            cash=starting_equity,
            starting_equity=starting_equity,
            equity=starting_equity,
        )

    def run(self, split: WalkForwardSplit, period: str = "train", profile: bool = False) -> Dict[str, Any]:
        """Run backtest for a specific split and period.

        Args:
            split: WalkForwardSplit configuration
            period: One of "train", "validation", "holdout"
            profile: If True, enable profiling for performance analysis

        Returns:
            Dictionary with backtest results
        """
        # Get date range for period
        start_date, end_date = split.get_period_dates(period)

        logger.info(f"Running backtest: {split.name} - {period} " f"({start_date.date()} to {end_date.date()})")

        # Get all trading days in range
        all_dates = self._get_all_dates()
        trading_dates = [d for d in all_dates if start_date <= d <= end_date]

        if not trading_dates:
            logger.warning(f"No trading dates found in range {start_date} to {end_date}")
            return self._empty_results()

        logger.info(f"Processing {len(trading_dates)} trading days")

        # Initialize event loop
        event_loop = self._create_event_loop()

        # Reset portfolio for this run
        self.portfolio = Portfolio(
            date=start_date, cash=self.starting_equity, starting_equity=self.starting_equity, equity=self.starting_equity
        )

        # Optional profiling
        profiler = None
        if profile:
            try:
                import cProfile

                profiler = cProfile.Profile()
                profiler.enable()
            except ImportError:
                logger.warning("cProfile not available, profiling disabled")
                profile = False

        # Track start time for timeout checking
        start_time = time.time()
        days_processed = 0

        # Process each day with progress bar
        self.daily_events = []
        progress_bar = tqdm(
            trading_dates,
            desc=f"Backtesting {split.name} - {period}",
            unit="day",
            disable=not TQDM_AVAILABLE or len(trading_dates) < 100,  # Only show for longer backtests
        )

        for date in progress_bar:
            try:
                events = event_loop.process_day(date)
                self.daily_events.append(events)
                days_processed += 1

                # Update progress bar description with current equity
                if TQDM_AVAILABLE and days_processed % 10 == 0:
                    current_equity = self.portfolio.equity
                    elapsed = time.time() - start_time
                    progress_bar.set_postfix(equity=f"${current_equity:,.0f}", elapsed=f"{elapsed:.0f}s")

                # Check timeout periodically (every N days to avoid overhead)
                if self.max_duration_seconds and days_processed % self.TIMEOUT_CHECK_INTERVAL == 0:
                    elapsed_seconds = time.time() - start_time
                    if elapsed_seconds > self.max_duration_seconds:
                        logger.error(
                            f"Backtest timeout: exceeded {self.max_duration_seconds}s "
                            f"(elapsed: {elapsed_seconds:.1f}s, processed {days_processed}/{len(trading_dates)} days)"
                        )
                        if TQDM_AVAILABLE:
                            progress_bar.close()
                        raise BacktestTimeoutError(
                            f"Backtest exceeded maximum duration of {self.max_duration_seconds} seconds. "
                            f"Processed {days_processed}/{len(trading_dates)} days in {elapsed_seconds:.1f}s. "
                            f"Consider reducing date range or increasing max_duration_seconds."
                        )
            except BacktestTimeoutError:
                raise  # Re-raise timeout errors
            except Exception as e:
                logger.error(f"Error processing day {date}: {e}")
                continue

        if TQDM_AVAILABLE:
            progress_bar.close()

        # Log completion time
        total_time = time.time() - start_time
        logger.info(f"Backtest completed: {days_processed} days in {total_time:.1f}s ({total_time/days_processed:.3f}s/day)")

        # Stop profiling if enabled
        if profiler is not None:
            profiler.disable()
            import pstats
            from io import StringIO

            s = StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats("cumulative")
            ps.print_stats(20)  # Top 20 functions
            logger.info(f"Profiling results:\n{s.getvalue()}")

        # Collect closed trades
        self.closed_trades = [p for p in self.portfolio.positions.values() if not p.is_open()]

        # Compute results
        results = self._compute_results(split, period)

        return results

    def _get_all_dates(self) -> List[pd.Timestamp]:
        """Get all available dates from market data.

        Returns:
            Sorted list of all dates present in market data
        """
        all_dates = set()

        # Get dates from all symbols
        for symbol, bars_df in self.market_data.bars.items():
            all_dates.update(bars_df.index)

        # Get dates from benchmarks
        for benchmark_df in self.market_data.benchmarks.values():
            all_dates.update(benchmark_df.index)

        return sorted(list(all_dates))

    def _create_event_loop(self) -> DailyEventLoop:
        """Create event loop instance.

        Returns:
            DailyEventLoop instance
        """

        def compute_features_fn(
            df_ohlc, symbol, asset_class, benchmark_roc60=None, benchmark_returns=None, use_cache=False, parallel=False
        ):
            """Wrapper for compute_features."""
            return compute_features(
                df_ohlc=df_ohlc,
                symbol=symbol,
                asset_class=asset_class,
                benchmark_roc60=benchmark_roc60,
                benchmark_returns=benchmark_returns,
            )

        # Cache all dates once for performance (get_next_trading_day is called many times)
        _cached_all_dates = self._get_all_dates()

        def get_next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
            """Get next trading day.

            For equity: skip weekends
            For crypto: next calendar day
            """
            # Simple implementation: next calendar day
            # In production, use trading calendar
            next_day = date + pd.Timedelta(days=1)

            # Use cached dates for efficient lookup
            # Use binary search to find next available date
            idx = bisect.bisect_right(_cached_all_dates, date)
            if idx < len(_cached_all_dates):
                return _cached_all_dates[idx]
            else:
                return next_day

        # Check if any strategy has ML enabled and create ML predictor
        ml_predictor = self._create_ml_predictor()

        return DailyEventLoop(
            market_data=self.market_data,
            portfolio=self.portfolio,
            strategies=self.strategies,
            compute_features_fn=compute_features_fn,
            get_next_trading_day=get_next_trading_day,
            rng=self.rng,
            slippage_multiplier=self.slippage_multiplier,
            crash_dates=self.crash_dates,
            ml_predictor=ml_predictor,
        )

    def _create_ml_predictor(self) -> Optional[MLPredictor]:
        """Create ML predictor if any strategy has ML enabled.

        Returns:
            MLPredictor instance if ML is enabled, None otherwise
        """
        # Check if any strategy has ML enabled
        ml_strategy = None
        for strategy in self.strategies:
            if strategy.config.ml.enabled:
                ml_strategy = strategy
                break

        if ml_strategy is None:
            return None

        ml_config = ml_strategy.config.ml

        # Load ML model and feature engineer
        try:
            import pickle

            if ml_config.model_path is None:
                logger.warning("ML model path is None. ML integration disabled.")
                return None
            model_path = Path(ml_config.model_path)

            if not model_path.exists():
                logger.warning(f"ML model path does not exist: {model_path}. ML integration disabled.")
                return None

            # Load model
            model = MLModel.load(model_path)

            # Load feature engineer
            feature_engineer_path = model_path / "feature_engineer.pkl"
            if not feature_engineer_path.exists():
                logger.warning(f"Feature engineer not found at {feature_engineer_path}. Creating default.")
                feature_engineer = MLFeatureEngineer()
            else:
                import pickle as pickle_module

                with open(feature_engineer_path, "rb") as f:
                    feature_engineer = pickle_module.load(f)

            # Create ML predictor
            ml_predictor = MLPredictor(
                model=model,
                feature_engineer=feature_engineer,
                prediction_mode=ml_config.prediction_mode,
                confidence_threshold=ml_config.confidence_threshold,
            )

            logger.info(f"ML predictor loaded from {model_path} (mode: {ml_config.prediction_mode})")
            return ml_predictor

        except Exception as e:
            logger.error(f"Failed to load ML predictor: {e}. ML integration disabled.", exc_info=True)
            return None

    def _compute_results(self, split: WalkForwardSplit, period: str) -> Dict[str, Any]:
        """Compute backtest results.

        Args:
            split: WalkForwardSplit used
            period: Period name

        Returns:
            Dictionary with results
        """
        if not self.daily_events:
            return self._empty_results()

        # Extract equity curve
        equity_curve = self.portfolio.equity_curve

        # Compute metrics
        total_return = (equity_curve[-1] / equity_curve[0] - 1) if len(equity_curve) > 1 else 0.0

        # Compute daily returns
        daily_returns = self.portfolio.daily_returns

        # Compute Sharpe ratio (annualized)
        if len(daily_returns) > 1:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Compute max drawdown
        if len(equity_curve) > 1:
            peak = equity_curve[0]
            max_dd = 0.0
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0.0

        # Trade statistics
        total_trades = len(self.closed_trades)
        winning_trades = len([t for t in self.closed_trades if t.realized_pnl > 0])
        losing_trades = len([t for t in self.closed_trades if t.realized_pnl < 0])

        # Average R-multiple
        r_multiples = [t.compute_r_multiple() for t in self.closed_trades if t.exit_price is not None]
        avg_r_multiple = np.mean(r_multiples) if r_multiples else 0.0

        results = {
            "split_name": split.name,
            "period": period,
            "start_date": split.get_period_dates(period)[0],
            "end_date": split.get_period_dates(period)[1],
            "starting_equity": self.starting_equity,
            "ending_equity": equity_curve[-1] if equity_curve else self.starting_equity,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0.0,
            "avg_r_multiple": avg_r_multiple,
            "realized_pnl": self.portfolio.realized_pnl,
            "final_cash": self.portfolio.cash,
            "final_positions": len(self.portfolio.positions),
            "equity_curve": equity_curve,
            "daily_returns": daily_returns,
            "closed_trades": self.closed_trades,
        }

        return results

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results dictionary.

        Returns:
            Dictionary with default/empty values
        """
        return {
            "split_name": "",
            "period": "",
            "start_date": None,
            "end_date": None,
            "starting_equity": self.starting_equity,
            "ending_equity": self.starting_equity,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_r_multiple": 0.0,
            "realized_pnl": 0.0,
            "final_cash": self.starting_equity,
            "final_positions": 0,
            "equity_curve": [self.starting_equity],
            "daily_returns": [],
            "closed_trades": [],
        }

    def export_results(self, output_path: str) -> None:
        """Export backtest results to files.

        Exports:
        - equity_curve.csv
        - trade_log.csv
        - daily_metrics.csv

        Args:
            output_path: Directory path to write output files
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export equity curve
        if self.portfolio.equity_curve:
            equity_df = pd.DataFrame({"date": range(len(self.portfolio.equity_curve)), "equity": self.portfolio.equity_curve})
            equity_df.to_csv(output_dir / "equity_curve.csv", index=False)

        # Export trade log
        if self.closed_trades:
            trades_data = []
            for trade in self.closed_trades:
                trades_data.append(
                    {
                        "symbol": trade.symbol,
                        "asset_class": trade.asset_class,
                        "entry_date": trade.entry_date,
                        "exit_date": trade.exit_date,
                        "entry_price": trade.entry_price,
                        "exit_price": trade.exit_price,
                        "quantity": trade.quantity,
                        "realized_pnl": trade.realized_pnl,
                        "exit_reason": trade.exit_reason.value if trade.exit_reason else None,
                        "r_multiple": trade.compute_r_multiple() if trade.exit_price else None,
                    }
                )
            trades_df = pd.DataFrame(trades_data)
            trades_df.to_csv(output_dir / "trade_log.csv", index=False)

        # Export daily metrics
        if self.daily_events:
            metrics_data = []
            for event in self.daily_events:
                metrics_data.append(
                    {
                        "date": event["date"],
                        "equity": event["portfolio_state"].get("equity", 0),
                        "cash": event["portfolio_state"].get("cash", 0),
                        "open_positions": event["portfolio_state"].get("open_positions", 0),
                        "realized_pnl": event["portfolio_state"].get("realized_pnl", 0),
                        "unrealized_pnl": event["portfolio_state"].get("unrealized_pnl", 0),
                        "gross_exposure": event["portfolio_state"].get("gross_exposure", 0),
                        "risk_multiplier": event["portfolio_state"].get("risk_multiplier", 1.0),
                    }
                )
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(output_dir / "daily_metrics.csv", index=False)

        logger.info(f"Exported results to {output_path}")


def create_backtest_engine_from_config(
    config_path: str, data_paths: Optional[Dict[str, str]] = None, seed: Optional[int] = None
) -> BacktestEngine:
    """Create backtest engine from configuration files.

    This function loads a run configuration file, creates strategies from their
    config files, loads market data, and returns a fully configured engine.

    Args:
        config_path: Path to run_config.yaml file
        data_paths: Optional dictionary with keys: 'equity', 'crypto', 'benchmark'
                    to override paths in config. If None, uses paths from config.
        seed: Optional random seed (overrides config.random_seed if provided)

    Returns:
        BacktestEngine instance ready to run backtests

    Raises:
        FileNotFoundError: If config file or strategy config files not found
        ValueError: If no strategies are enabled or data loading fails
    """
    from ..configs.run_config import RunConfig
    from ..configs.strategy_config import StrategyConfig
    from ..data.loader import load_all_data, load_universe
    from ..strategies.strategy_loader import load_strategies_from_run_config

    # Load run configuration
    logger.info(f"Loading run configuration from {config_path}")
    config = RunConfig.from_yaml(config_path)

    # Use seed from parameter or config
    final_seed = seed if seed is not None else config.random_seed

    # Determine data paths (use provided or from config)
    if data_paths is not None:
        equity_path = data_paths.get("equity", config.dataset.equity_path)
        crypto_path = data_paths.get("crypto", config.dataset.crypto_path)
        benchmark_path = data_paths.get("benchmark", config.dataset.benchmark_path)
    else:
        equity_path = config.dataset.equity_path
        crypto_path = config.dataset.crypto_path
        benchmark_path = config.dataset.benchmark_path

    # Get date range from config
    start_date = pd.Timestamp(config.dataset.start_date)
    end_date = pd.Timestamp(config.dataset.end_date)

    # Load strategies using strategy loader
    equity_config_path = (
        config.strategies.equity.config_path if config.strategies.equity and config.strategies.equity.enabled else None
    )
    crypto_config_path = (
        config.strategies.crypto.config_path if config.strategies.crypto and config.strategies.crypto.enabled else None
    )

    strategies = load_strategies_from_run_config(equity_config_path=equity_config_path, crypto_config_path=crypto_config_path)

    # Determine universes from strategy configs
    equity_universe = []
    crypto_universe = None
    crypto_universe_config = None

    # Get equity universe
    if equity_config_path:
        equity_strategy_config = StrategyConfig.from_yaml(equity_config_path)
        if isinstance(equity_strategy_config.universe, list):
            equity_universe = equity_strategy_config.universe
        else:
            equity_universe = load_universe(equity_strategy_config.universe)
            logger.info(f"Loaded equity universe: {len(equity_universe)} symbols")

    # Get crypto universe
    if crypto_config_path:
        crypto_strategy_config = StrategyConfig.from_yaml(crypto_config_path)
        if isinstance(crypto_strategy_config.universe, list):
            crypto_universe = crypto_strategy_config.universe
        elif crypto_strategy_config.universe_config is not None:
            crypto_universe_config = crypto_strategy_config.universe_config.model_dump()
            logger.info("Crypto universe: dynamic selection enabled")
        else:
            crypto_universe = load_universe(crypto_strategy_config.universe)

        if crypto_universe is not None:
            logger.info(f"Loaded crypto universe: {len(crypto_universe)} symbols")

    if not strategies:
        raise ValueError("No strategies enabled in configuration")

    # Load market data
    logger.info("Loading market data...")
    logger.info(f"  Equity path: {equity_path}")
    logger.info(f"  Crypto path: {crypto_path}")
    logger.info(f"  Benchmark path: {benchmark_path}")
    logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")

    market_data, benchmarks = load_all_data(
        equity_path=equity_path,
        crypto_path=crypto_path,
        benchmark_path=benchmark_path,
        equity_universe=equity_universe,
        crypto_universe=crypto_universe,
        crypto_universe_config=crypto_universe_config,
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(f"Market data loaded: {len(market_data.bars)} symbols, " f"{len(market_data.benchmarks)} benchmarks")

    # Create and return engine
    logger.info("Creating backtest engine...")
    engine = BacktestEngine(
        market_data=market_data, strategies=strategies, starting_equity=config.portfolio.starting_equity, seed=final_seed
    )

    logger.info("Backtest engine created successfully")
    return engine
