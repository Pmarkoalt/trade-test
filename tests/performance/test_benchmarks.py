"""Performance regression tests using pytest-benchmark."""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_system.indicators.atr import atr
from trading_system.indicators.breakouts import highest_close
from trading_system.indicators.feature_computer import compute_features
from trading_system.indicators.ma import ma
from trading_system.indicators.momentum import roc
from trading_system.indicators.volume import adv
from trading_system.models.market_data import MarketData
from trading_system.models.orders import Fill
from trading_system.models.signals import BreakoutType, SignalSide
from trading_system.portfolio.portfolio import Portfolio


@pytest.mark.performance
class TestIndicatorPerformance:
    """Performance benchmarks for indicators."""

    @pytest.fixture
    def large_price_series(self):
        """Create large price series for performance testing."""
        dates = pd.date_range("2020-01-01", periods=10000, freq="D")
        prices = 100.0 + np.cumsum(np.random.randn(10000) * 0.5)
        return pd.Series(prices, index=dates)

    @pytest.fixture
    def large_ohlc_data(self):
        """Create large OHLC DataFrame for performance testing."""
        dates = pd.date_range("2020-01-01", periods=10000, freq="D")
        base_price = 100.0
        returns = np.random.randn(10000) * 0.02

        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(10000) * 0.001),
                "high": prices * (1 + abs(np.random.randn(10000)) * 0.005),
                "low": prices * (1 - abs(np.random.randn(10000)) * 0.005),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, 10000),
            },
            index=dates,
        )

        df["dollar_volume"] = df["close"] * df["volume"]
        return df

    def test_ma_performance(self, benchmark, large_price_series):
        """Benchmark: Moving average calculation on large series."""
        result = benchmark(ma, large_price_series, window=20)
        assert len(result) == len(large_price_series)

    def test_atr_performance(self, benchmark, large_ohlc_data):
        """Benchmark: ATR calculation on large OHLC data."""
        result = benchmark(atr, large_ohlc_data, period=14)
        assert len(result) == len(large_ohlc_data)

    def test_roc_performance(self, benchmark, large_price_series):
        """Benchmark: ROC calculation on large series."""
        result = benchmark(roc, large_price_series, window=60)
        assert len(result) == len(large_price_series)

    def test_highest_close_performance(self, benchmark, large_price_series):
        """Benchmark: Highest close calculation on large series."""
        result = benchmark(highest_close, large_price_series, window=20)
        assert len(result) == len(large_price_series)

    def test_adv_performance(self, benchmark, large_ohlc_data):
        """Benchmark: ADV calculation on large data."""
        result = benchmark(adv, large_ohlc_data["dollar_volume"], window=20)
        assert len(result) == len(large_ohlc_data)

    def test_compute_features_performance(self, benchmark, large_ohlc_data):
        """Benchmark: Full feature computation on large data."""
        result = benchmark(compute_features, large_ohlc_data, symbol="TEST", asset_class="equity")
        assert len(result) == len(large_ohlc_data)


@pytest.mark.performance
class TestPortfolioPerformance:
    """Performance benchmarks for portfolio operations."""

    @pytest.fixture
    def large_portfolio(self):
        """Create portfolio with many positions."""
        portfolio = Portfolio(date=pd.Timestamp("2020-01-01"), starting_equity=1000000.0, cash=1000000.0, equity=1000000.0)

        # Add 50 positions
        for i in range(50):
            symbol = f"SYM{i}"
            price = 100.0
            quantity = 100
            notional = price * quantity

            fill = Fill(
                fill_id=f"fill_{i}",
                order_id=f"order_{i}",
                symbol=symbol,
                asset_class="equity",
                date=pd.Timestamp("2020-01-01"),
                side=SignalSide.BUY,
                quantity=quantity,
                fill_price=price,
                open_price=price,
                slippage_bps=10.0,
                fee_bps=5.0,
                total_cost=notional * 1.0015,
                vol_mult=1.0,
                size_penalty=1.0,
                weekend_penalty=1.0,
                stress_mult=1.0,
                notional=notional,
            )

            portfolio.process_fill(
                fill=fill, stop_price=price * 0.95, atr_mult=2.5, triggered_on=BreakoutType.FAST_20D, adv20_at_entry=notional
            )

        return portfolio

    def test_portfolio_update_equity_performance(self, benchmark, large_portfolio):
        """Benchmark: Portfolio equity update with many positions."""
        # Create current prices for all positions
        current_prices = {symbol: 105.0 for symbol in large_portfolio.positions.keys()}

        result = benchmark(large_portfolio.update_equity, current_prices)
        assert large_portfolio.equity > 0

    def test_portfolio_exposure_calculation_performance(self, benchmark, large_portfolio):
        """Benchmark: Portfolio exposure calculation."""
        current_prices = {symbol: 105.0 for symbol in large_portfolio.positions.keys()}
        large_portfolio.update_equity(current_prices)

        # Benchmark exposure calculation (done internally in update_equity)
        # This is a synthetic benchmark to measure the exposure calculation overhead
        def calc_exposure():
            portfolio = large_portfolio
            gross_exposure = sum(
                pos.quantity * current_prices.get(pos.symbol, pos.entry_price) for pos in portfolio.positions.values()
            )
            return gross_exposure / portfolio.equity

        result = benchmark(calc_exposure)
        assert 0.0 <= result <= 1.0


@pytest.mark.performance
class TestValidationPerformance:
    """Performance benchmarks for validation suite."""

    @pytest.fixture
    def large_r_multiples(self):
        """Create large list of R-multiples."""
        return np.random.randn(10000).tolist()

    def test_bootstrap_performance(self, benchmark, large_r_multiples):
        """Benchmark: Bootstrap resampling."""
        from trading_system.validation.bootstrap import BootstrapTest

        # Reduced iterations for benchmark (performance test, not validation test)
        test = BootstrapTest(large_r_multiples, n_iterations=100, random_seed=42)

        result = benchmark(test.run)
        assert "sharpe_5th" in result

    def test_permutation_performance(self, benchmark):
        """Benchmark: Permutation test."""
        from trading_system.validation.permutation import PermutationTest

        # Create sample trades
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        actual_trades = [
            {"entry_date": dates[i], "exit_date": dates[i + 10], "symbol": f"SYM{i % 10}", "r_multiple": np.random.randn()}
            for i in range(100)
        ]

        period = (dates[0], dates[-1])

        def compute_sharpe_func(trades):
            from trading_system.validation.bootstrap import compute_sharpe_from_r_multiples

            r_multiples = [t.get("r_multiple", 0.0) for t in trades]
            return compute_sharpe_from_r_multiples(r_multiples)

        # Reduced iterations for benchmark (performance test, not validation test)
        test = PermutationTest(actual_trades, period, compute_sharpe_func, n_iterations=100, random_seed=42)

        result = benchmark(test.run)
        assert "actual_sharpe" in result


@pytest.mark.performance
class TestBacktestEnginePerformance:
    """Performance benchmarks for backtest engine."""

    @pytest.fixture
    def large_market_data(self):
        """Create market data with many symbols and long time series."""
        market_data = MarketData()

        # Create 20 symbols with 2 years of daily data (reduced from 5 years for CI/CD speed)
        dates = pd.bdate_range("2022-01-01", "2024-01-01")
        symbols = [f"SYM{i:02d}" for i in range(20)]

        np.random.seed(42)
        for symbol in symbols:
            base_price = 50.0 + np.random.rand() * 150.0
            returns = np.random.randn(len(dates)) * 0.02
            prices = base_price * np.exp(np.cumsum(returns))

            bars = pd.DataFrame(
                {
                    "open": prices * (1 + np.random.randn(len(dates)) * 0.001),
                    "high": prices * (1 + abs(np.random.randn(len(dates)) * 0.005)),
                    "low": prices * (1 - abs(np.random.randn(len(dates)) * 0.005)),
                    "close": prices,
                    "volume": np.random.randint(1000000, 10000000, len(dates)),
                    "dollar_volume": prices * np.random.randint(1000000, 10000000, len(dates)),
                },
                index=dates,
            )

            market_data.bars[symbol] = bars

        # Add benchmark
        spy_prices = 300.0 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
        spy_bars = pd.DataFrame(
            {
                "open": spy_prices,
                "high": spy_prices * 1.01,
                "low": spy_prices * 0.99,
                "close": spy_prices,
                "volume": np.random.randint(50000000, 100000000, len(dates)),
                "dollar_volume": spy_prices * np.random.randint(50000000, 100000000, len(dates)),
            },
            index=dates,
        )
        market_data.benchmarks["SPY"] = spy_bars

        return market_data

    @pytest.fixture
    def strategy_config(self):
        """Create strategy config for backtesting."""
        from trading_system.configs.strategy_config import StrategyConfig

        return StrategyConfig(
            name="test_momentum",
            asset_class="equity",
            universe=[f"SYM{i:02d}" for i in range(20)],
            benchmark="SPY",
            eligibility={"min_price": 1.0, "min_volume": 100000, "trend_ma": 50, "min_slope": 0.0},
            entry={"fast_breakout_days": 20, "slow_breakout_days": 55, "min_clearance_atr": 0.5},
            exit={"mode": "ma_cross", "exit_ma": 20, "hard_stop_atr_mult": 2.5},
            risk={"risk_per_trade": 0.0075, "max_positions": 10, "max_exposure": 0.80, "max_position_notional": 0.15},
        )

    def test_event_loop_process_day_performance(self, benchmark, large_market_data, strategy_config):
        """Benchmark: Event loop processing a single day."""
        from trading_system.backtest.event_loop import DailyEventLoop
        from trading_system.indicators.feature_computer import compute_features
        from trading_system.strategies import EquityMomentumStrategy

        portfolio = Portfolio(date=pd.Timestamp("2020-01-01"), starting_equity=1000000.0, cash=1000000.0, equity=1000000.0)

        strategy = EquityMomentumStrategy(strategy_config)

        def get_next_trading_day(date):
            dates = sorted(large_market_data.bars[list(large_market_data.bars.keys())[0]].index)
            available = [d for d in dates if d > date]
            return min(available) if available else date + pd.Timedelta(days=1)

        event_loop = DailyEventLoop(
            market_data=large_market_data,
            portfolio=portfolio,
            strategies=[strategy],
            compute_features_fn=compute_features,
            get_next_trading_day=get_next_trading_day,
            rng=np.random.default_rng(42),
        )

        test_date = pd.Timestamp("2020-06-15")

        def process_single_day():
            return event_loop.process_day(test_date)

        result = benchmark(process_single_day)
        assert "date" in result

    def test_backtest_engine_full_run_performance(self, benchmark, large_market_data, strategy_config):
        """Benchmark: Full backtest run for 6 months."""
        from trading_system.backtest import BacktestEngine, WalkForwardSplit
        from trading_system.strategies import EquityMomentumStrategy

        # Reduced period from 6 months to 3 months for CI/CD speed
        split = WalkForwardSplit(
            name="perf_test",
            train_start=pd.Timestamp("2022-01-01"),
            train_end=pd.Timestamp("2022-03-31"),
            validation_start=pd.Timestamp("2022-04-01"),
            validation_end=pd.Timestamp("2022-06-30"),
            holdout_start=pd.Timestamp("2022-07-01"),
            holdout_end=pd.Timestamp("2022-09-30"),
        )

        strategy = EquityMomentumStrategy(strategy_config)

        engine = BacktestEngine(market_data=large_market_data, strategies=[strategy], starting_equity=1000000.0, seed=42)

        def run_backtest():
            return engine.run(split, period="train")

        result = benchmark(run_backtest)
        assert "ending_equity" in result


@pytest.mark.performance
class TestDataLoadingPerformance:
    """Performance benchmarks for data loading operations."""

    @pytest.fixture
    def temp_csv_dir(self):
        """Create temporary directory with CSV files."""
        temp_dir = tempfile.mkdtemp()

        # Create 10 CSV files with 2 years of data each (reduced from 5 years for CI/CD speed)
        dates = pd.bdate_range("2022-01-01", "2024-01-01")
        symbols = [f"TEST{i:02d}" for i in range(10)]

        np.random.seed(42)
        for symbol in symbols:
            base_price = 50.0 + np.random.rand() * 150.0
            returns = np.random.randn(len(dates)) * 0.02
            prices = base_price * np.exp(np.cumsum(returns))

            df = pd.DataFrame(
                {
                    "date": dates,
                    "open": prices * (1 + np.random.randn(len(dates)) * 0.001),
                    "high": prices * (1 + abs(np.random.randn(len(dates)) * 0.005)),
                    "low": prices * (1 - abs(np.random.randn(len(dates)) * 0.005)),
                    "close": prices,
                    "volume": np.random.randint(1000000, 10000000, len(dates)),
                }
            )

            csv_path = os.path.join(temp_dir, f"{symbol}.csv")
            df.to_csv(csv_path, index=False)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_csv_loading_performance(self, benchmark, temp_csv_dir):
        """Benchmark: Loading multiple CSV files."""
        from trading_system.data.sources.csv_source import CSVDataSource

        source = CSVDataSource(temp_csv_dir)
        symbols = [f"TEST{i:02d}" for i in range(10)]

        def load_data():
            return source.load_ohlcv(symbols)

        result = benchmark(load_data)
        assert len(result) == len(symbols)

    @pytest.mark.parametrize("n_symbols", [1, 5, 10])
    def test_multi_symbol_scaling(self, benchmark, n_symbols):
        """Benchmark: Feature computation scales with number of symbols."""
        from trading_system.indicators.feature_computer import compute_features

        # Reduced from 3 years to 1 year for CI/CD speed
        dates = pd.bdate_range("2022-01-01", "2023-01-01")
        symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
        test_data = {}

        np.random.seed(42)
        for symbol in symbols:
            base_price = 100.0
            returns = np.random.randn(len(dates)) * 0.02
            prices = base_price * np.exp(np.cumsum(returns))

            df = pd.DataFrame(
                {
                    "open": prices,
                    "high": prices * 1.01,
                    "low": prices * 0.99,
                    "close": prices,
                    "volume": np.random.randint(1000000, 10000000, len(dates)),
                    "dollar_volume": prices * np.random.randint(1000000, 10000000, len(dates)),
                },
                index=dates,
            )
            test_data[symbol] = df

        def compute_all_features():
            for symbol, df in test_data.items():
                compute_features(df, symbol, asset_class="equity")

        result = benchmark(compute_all_features)
        # Verify computation completed
        assert result is None  # compute_features returns None, modifies DataFrame in place


@pytest.mark.performance
class TestSignalScoringPerformance:
    """Performance benchmarks for signal scoring and queue selection."""

    @pytest.fixture
    def large_signal_list(self):
        """Create a large list of signals for scoring."""
        from trading_system.models.signals import Signal, SignalSide, SignalType

        np.random.seed(42)
        signals = []
        dates = pd.bdate_range("2020-01-01", "2020-12-31")

        for i in range(100):
            symbol = f"SYM{i:03d}"
            date = np.random.choice(dates)
            entry_price = 50.0 + np.random.rand() * 150.0

            signal = Signal(
                symbol=symbol,
                asset_class="equity",
                date=date,
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason="test",
                entry_price=entry_price,
                stop_price=entry_price * 0.95,
                breakout_strength=np.random.rand(),
                momentum_strength=np.random.rand(),
                diversification_bonus=np.random.rand(),
            )
            signals.append(signal)

        return signals

    @pytest.fixture
    def portfolio_with_positions(self):
        """Create portfolio with existing positions."""
        portfolio = Portfolio(date=pd.Timestamp("2020-06-15"), starting_equity=1000000.0, cash=500000.0, equity=1000000.0)

        # Add 10 positions
        for i in range(10):
            symbol = f"POS{i:02d}"
            price = 100.0
            quantity = 100

            fill = Fill(
                fill_id=f"fill_{i}",
                order_id=f"order_{i}",
                symbol=symbol,
                asset_class="equity",
                date=pd.Timestamp("2020-06-01"),
                side=SignalSide.BUY,
                quantity=quantity,
                fill_price=price,
                open_price=price,
                slippage_bps=10.0,
                fee_bps=5.0,
                total_cost=price * quantity * 1.0015,
                notional=price * quantity,
            )

            portfolio.process_fill(
                fill=fill,
                stop_price=price * 0.95,
                atr_mult=2.5,
                triggered_on=BreakoutType.FAST_20D,
                adv20_at_entry=price * quantity * 20,
            )

        return portfolio

    def test_signal_scoring_performance(self, benchmark, large_signal_list, portfolio_with_positions):
        """Benchmark: Scoring a large list of signals."""
        from trading_system.models.features import FeatureRow
        from trading_system.strategies.scoring import score_signals

        # Create mock feature rows
        def get_features(signal):
            return FeatureRow(
                date=signal.date,
                symbol=signal.symbol,
                asset_class=signal.asset_class,
                close=signal.entry_price,
                ma20=signal.entry_price * 0.98,
                ma50=signal.entry_price * 0.95,
                atr14=signal.entry_price * 0.02,
                roc60=0.05,
                highest_close_20d=signal.entry_price * 0.98,
                adv20=signal.entry_price * 50000000,
                returns_1d=0.001,
                benchmark_roc60=0.03,
                benchmark_returns_1d=0.0005,
            )

        # Create mock returns data
        candidate_returns = {s.symbol: np.random.randn(60).tolist() for s in large_signal_list}
        portfolio_returns = {f"POS{i:02d}": np.random.randn(60).tolist() for i in range(10)}

        def score_all_signals():
            score_signals(
                large_signal_list, get_features, portfolio_with_positions, candidate_returns, portfolio_returns, lookback=20
            )

        result = benchmark(score_all_signals)
        # Verify signals were scored
        assert any(s.score > 0.0 for s in large_signal_list)

    def test_queue_selection_performance(self, benchmark, large_signal_list, portfolio_with_positions):
        """Benchmark: Selecting signals from queue."""
        from trading_system.strategies.queue import select_signals_from_queue

        # Pre-score signals
        for signal in large_signal_list:
            signal.score = np.random.rand()

        candidate_returns = {s.symbol: np.random.randn(60).tolist() for s in large_signal_list}
        portfolio_returns = {f"POS{i:02d}": np.random.randn(60).tolist() for i in range(10)}

        def select_from_queue():
            return select_signals_from_queue(
                large_signal_list,
                portfolio_with_positions,
                max_positions=20,
                max_exposure=0.80,
                risk_per_trade=0.0075,
                max_position_notional=100000.0,
                candidate_returns=candidate_returns,
                portfolio_returns=portfolio_returns,
                lookback=20,
            )

        result = benchmark(select_from_queue)
        assert isinstance(result, list)


@pytest.mark.performance
class TestStrategyEvaluationPerformance:
    """Performance benchmarks for strategy evaluation."""

    @pytest.fixture
    def large_feature_set(self):
        """Create large set of features for strategy evaluation."""
        from trading_system.models.features import FeatureRow

        # Reduced from 3 years/50 symbols to 1 year/20 symbols for CI/CD speed
        dates = pd.bdate_range("2022-01-01", "2023-01-01")
        symbols = [f"SYM{i:02d}" for i in range(20)]

        features = {}
        np.random.seed(42)
        for symbol in symbols:
            symbol_features = []
            base_price = 100.0 + np.random.rand() * 50.0

            for date in dates:
                price = base_price * (1 + np.random.randn() * 0.02)
                feature = FeatureRow(
                    date=date,
                    symbol=symbol,
                    asset_class="equity",
                    close=price,
                    ma20=price * 0.98,
                    ma50=price * 0.95,
                    ma200=price * 0.90,
                    atr14=price * 0.02,
                    roc60=np.random.randn() * 0.1,
                    highest_close_20d=price * 0.98,
                    highest_close_55d=price * 0.95,
                    adv20=price * 50000000,
                    returns_1d=np.random.randn() * 0.01,
                    benchmark_roc60=0.03,
                    benchmark_returns_1d=0.0005,
                )
                symbol_features.append(feature)

            features[symbol] = symbol_features

        return features

    def test_equity_strategy_evaluation_performance(self, benchmark, large_feature_set):
        """Benchmark: Equity strategy evaluating many symbols."""
        from trading_system.configs.strategy_config import StrategyConfig
        from trading_system.strategies import EquityMomentumStrategy

        config = StrategyConfig(
            name="test_equity",
            asset_class="equity",
            universe=list(large_feature_set.keys()),
            benchmark="SPY",
            eligibility={"min_price": 1.0, "trend_ma": 50},
            entry={"fast_breakout_days": 20},
            exit={"mode": "ma_cross", "exit_ma": 20},
            risk={"risk_per_trade": 0.0075, "max_positions": 20},
        )

        strategy = EquityMomentumStrategy(config)

        def evaluate_all_symbols():
            date = pd.Timestamp("2021-06-15")
            signals = []

            for symbol, feature_list in large_feature_set.items():
                # Get features up to date
                available = [f for f in feature_list if f.date <= date]
                if len(available) < 200:
                    continue

                latest_features = available[-1]
                symbol_signals = strategy.evaluate_entry(latest_features, available)
                signals.extend(symbol_signals)

            return signals

        result = benchmark(evaluate_all_symbols)
        assert isinstance(result, list)


@pytest.mark.performance
class TestReportingPerformance:
    """Performance benchmarks for reporting operations."""

    @pytest.fixture
    def sample_backtest_results(self):
        """Create sample backtest results for reporting."""
        dates = pd.bdate_range("2020-01-01", "2023-01-01")

        # Generate equity curve
        returns = np.random.randn(len(dates)) * 0.01
        equity_curve = 1000000.0 * np.exp(np.cumsum(returns))

        # Generate trades
        trades = []
        for i in range(100):
            entry_idx = np.random.randint(0, len(dates) - 20)
            exit_idx = entry_idx + np.random.randint(5, 20)

            trades.append(
                {
                    "symbol": f"SYM{i % 20:02d}",
                    "entry_date": dates[entry_idx],
                    "exit_date": dates[exit_idx],
                    "entry_price": 100.0 + np.random.rand() * 50.0,
                    "exit_price": 100.0 + np.random.rand() * 50.0,
                    "quantity": 100,
                    "pnl": np.random.randn() * 1000.0,
                    "r_multiple": np.random.randn() * 2.0,
                }
            )

        return {
            "equity_curve": equity_curve,
            "dates": dates,
            "trades": trades,
            "starting_equity": 1000000.0,
            "ending_equity": equity_curve[-1],
        }

    def test_report_generation_performance(self, benchmark, sample_backtest_results):
        """Benchmark: Generating backtest report."""
        from trading_system.reporting.metrics import MetricsCalculator
        from trading_system.reporting.report_generator import ReportGenerator

        calc = MetricsCalculator()

        def generate_full_report():
            # Compute metrics
            metrics = calc.compute_all_metrics(
                equity_curve=sample_backtest_results["equity_curve"], trades=sample_backtest_results["trades"]
            )
            return metrics

        result = benchmark(generate_full_report)
        assert isinstance(result, dict)

    def test_csv_export_performance(self, benchmark, sample_backtest_results):
        """Benchmark: Exporting results to CSV."""

        def export_to_csv():
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                csv_path = f.name

            try:
                # Create DataFrame from trades
                df = pd.DataFrame(sample_backtest_results["trades"])
                df.to_csv(csv_path, index=False)
                return csv_path
            finally:
                if os.path.exists(csv_path):
                    os.unlink(csv_path)

        result = benchmark(export_to_csv)
        assert result is not None
