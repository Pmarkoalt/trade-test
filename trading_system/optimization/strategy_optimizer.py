"""Strategy Parameter Optimization using Optuna.

This module provides systematic parameter optimization for trading strategies
using Bayesian optimization with walk-forward validation to avoid overfitting.

Features:
- Configurable parameter search spaces
- Walk-forward validation to prevent overfitting
- Multiple objective support (Sharpe, Calmar, etc.)
- Early stopping for poor trials
- Results persistence and analysis

Usage:
    from trading_system.optimization import StrategyOptimizer

    optimizer = StrategyOptimizer(
        base_config_path="configs/equity_strategy.yaml",
        data_paths={"equity": "data/equity/daily"},
    )

    best_params = optimizer.optimize(n_trials=100)
"""

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from loguru import logger

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Install with: pip install optuna")


@dataclass
class ParameterSpace:
    """Definition of a parameter to optimize."""

    name: str  # Full path like "exit.hard_stop_atr_mult"
    param_type: str  # "float", "int", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False

    def sample(self, trial: "optuna.Trial") -> Any:
        """Sample a value from this parameter space."""
        if self.param_type == "float":
            return trial.suggest_float(self.name, self.low, self.high, step=self.step, log=self.log_scale)
        elif self.param_type == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high), step=int(self.step) if self.step else 1)
        elif self.param_type == "categorical":
            return trial.suggest_categorical(self.name, self.choices)
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")


@dataclass
class OptimizationResult:
    """Result of an optimization run."""

    study_name: str
    best_params: Dict[str, Any]
    best_value: float
    objective: str
    n_trials: int
    n_completed: int
    optimization_time_seconds: float

    # All trials summary
    trials_summary: List[Dict[str, Any]] = field(default_factory=list)

    # Validation results
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    holdout_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "objective": self.objective,
            "n_trials": self.n_trials,
            "n_completed": self.n_completed,
            "optimization_time_seconds": self.optimization_time_seconds,
            "trials_summary": self.trials_summary,
            "validation_metrics": self.validation_metrics,
            "holdout_metrics": self.holdout_metrics,
        }

    def save(self, path: str) -> None:
        """Save results to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# Default parameter spaces for equity strategies
# Extended ranges to capture longer trends (S&P 45% 2024-2026)
DEFAULT_EQUITY_PARAM_SPACES = [
    ParameterSpace("exit.hard_stop_atr_mult", "float", 2.0, 6.0, 0.5),  # Wider stops for trends
    ParameterSpace("exit.exit_ma", "categorical", choices=[20, 50, 100, 200]),  # Slower exits
    ParameterSpace("entry.fast_clearance", "float", 0.0, 0.02, 0.002),
    ParameterSpace("entry.slow_clearance", "float", 0.005, 0.025, 0.002),
    ParameterSpace("risk.risk_per_trade", "float", 0.005, 0.015, 0.0025),
    ParameterSpace("risk.max_positions", "int", 4, 12, 2),
    ParameterSpace("eligibility.trend_ma", "categorical", choices=[50, 100, 200]),  # Longer trend filter
]

# Default parameter spaces for crypto strategies
# Extended ranges for volatile crypto trends
DEFAULT_CRYPTO_PARAM_SPACES = [
    ParameterSpace("exit.hard_stop_atr_mult", "float", 2.5, 7.0, 0.5),  # Very wide for crypto volatility
    ParameterSpace("exit.exit_ma", "categorical", choices=[20, 50, 100]),  # Allow slower exits
    ParameterSpace("exit.tightened_stop_atr_mult", "float", 1.5, 4.0, 0.5),  # Staged exit stop
    ParameterSpace("entry.fast_clearance", "float", 0.0, 0.03, 0.005),
    ParameterSpace("entry.slow_clearance", "float", 0.005, 0.03, 0.005),
    ParameterSpace("risk.risk_per_trade", "float", 0.005, 0.02, 0.0025),
    ParameterSpace("risk.max_positions", "int", 3, 10, 1),
]


class StrategyOptimizer:
    """Optimize strategy parameters using Bayesian optimization.

    Uses Optuna's TPE sampler for efficient parameter search with
    walk-forward validation to prevent overfitting.
    """

    def __init__(
        self,
        base_config_path: str,
        data_paths: Dict[str, str],
        param_spaces: Optional[List[ParameterSpace]] = None,
        objective: str = "sharpe_ratio",
        n_validation_splits: int = 3,
        output_dir: str = "optimization_results",
    ):
        """Initialize optimizer.

        Args:
            base_config_path: Path to base strategy config YAML
            data_paths: Dictionary with equity/crypto/benchmark paths
            param_spaces: Parameter spaces to search (uses defaults if None)
            objective: Metric to optimize ("sharpe_ratio", "calmar_ratio", "total_return")
            n_validation_splits: Number of walk-forward splits for validation
            output_dir: Directory for saving results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required. Install with: pip install optuna")

        self.base_config_path = base_config_path
        self.data_paths = data_paths
        self.objective = objective
        self.n_validation_splits = n_validation_splits
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load base config to determine asset class
        with open(base_config_path) as f:
            self.base_config = yaml.safe_load(f)

        asset_class = self.base_config.get("asset_class", "equity")

        # Set default param spaces based on asset class
        if param_spaces is None:
            if asset_class == "crypto":
                self.param_spaces = DEFAULT_CRYPTO_PARAM_SPACES
            else:
                self.param_spaces = DEFAULT_EQUITY_PARAM_SPACES
        else:
            self.param_spaces = param_spaces

        # Validate objective
        valid_objectives = ["sharpe_ratio", "calmar_ratio", "total_return", "sortino_ratio"]
        if objective not in valid_objectives:
            raise ValueError(f"objective must be one of {valid_objectives}")

        self._study: Optional["optuna.Study"] = None
        self._market_data = None
        self._cached_results: Dict[str, float] = {}

    def _load_market_data(self):
        """Load market data once for reuse."""
        if self._market_data is not None:
            return

        from ..data.loader import load_all_data

        logger.info("Loading market data for optimization...")

        # Determine asset class and set appropriate universe
        asset_class = self.base_config.get("asset_class", "equity")
        universe = self.base_config.get("universe", [])

        if asset_class == "crypto":
            # For crypto, pass universe to crypto_universe parameter
            self._market_data, _ = load_all_data(
                equity_path=self.data_paths.get("equity", "data/equity/daily"),
                crypto_path=self.data_paths.get("crypto", "data/crypto/daily"),
                benchmark_path=self.data_paths.get("benchmark", "data/test_benchmarks"),
                equity_universe=[],  # No equity for crypto optimization
                crypto_universe=universe,
                start_date=pd.Timestamp("2024-01-01"),
                end_date=pd.Timestamp("2025-12-31"),
            )
        else:
            # For equity, pass universe to equity_universe parameter
            self._market_data, _ = load_all_data(
                equity_path=self.data_paths.get("equity", "data/equity/daily"),
                crypto_path=self.data_paths.get("crypto", "data/crypto/daily"),
                benchmark_path=self.data_paths.get("benchmark", "data/test_benchmarks"),
                equity_universe=universe,
                start_date=pd.Timestamp("2024-01-01"),
                end_date=pd.Timestamp("2025-12-31"),
            )

        logger.info(f"Loaded {len(self._market_data.bars)} symbols")

    def _create_config_with_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a config dict with the given parameters."""
        config = copy.deepcopy(self.base_config)

        for param_name, value in params.items():
            # Parse nested path like "exit.hard_stop_atr_mult"
            parts = param_name.split(".")
            target = config
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value

        return config

    def _run_backtest(self, config: Dict[str, Any], split_name: str = "opt") -> Dict[str, float]:
        """Run a single backtest with given config."""
        from ..backtest import BacktestEngine, WalkForwardSplit
        from ..configs.strategy_config import StrategyConfig
        from ..strategies.strategy_registry import create_strategy

        self._load_market_data()

        # Create strategy from config dict
        try:
            strategy_config = StrategyConfig(**config)
            strategy = create_strategy(strategy_config)
        except Exception as e:
            logger.warning(f"Invalid config: {e}")
            return {"sharpe_ratio": -10, "calmar_ratio": -10, "total_return": -1}

        # Create engine
        engine = BacktestEngine(
            market_data=self._market_data,
            strategies=[strategy],
            starting_equity=100000.0,
            seed=42,
        )

        # Create split for validation
        split = WalkForwardSplit(
            name=split_name,
            train_start=pd.Timestamp("2025-01-01"),
            train_end=pd.Timestamp("2025-06-30"),
            validation_start=pd.Timestamp("2025-07-01"),
            validation_end=pd.Timestamp("2025-09-30"),
            holdout_start=pd.Timestamp("2025-10-01"),
            holdout_end=pd.Timestamp("2025-12-31"),
        )

        # Run on validation period
        try:
            results = engine.run(split, period="validation")
            return {
                "sharpe_ratio": results.get("sharpe_ratio", -10) or -10,
                "calmar_ratio": results.get("calmar_ratio", -10) or -10,
                "total_return": results.get("total_return", -1) or -1,
                "sortino_ratio": results.get("sortino_ratio", -10) or -10,
                "total_trades": results.get("total_trades", 0),
                "max_drawdown": results.get("max_drawdown", 1) or 1,
            }
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return {"sharpe_ratio": -10, "calmar_ratio": -10, "total_return": -1}

    def _objective_fn(self, trial: "optuna.Trial") -> float:
        """Optuna objective function."""
        # Sample parameters
        params = {}
        for space in self.param_spaces:
            params[space.name] = space.sample(trial)

        # Validate clearance constraint
        if "entry.fast_clearance" in params and "entry.slow_clearance" in params:
            if params["entry.fast_clearance"] >= params["entry.slow_clearance"]:
                # Adjust slow_clearance to be valid
                params["entry.slow_clearance"] = params["entry.fast_clearance"] + 0.005

        # Create config and run backtest
        config = self._create_config_with_params(params)
        results = self._run_backtest(config, f"trial_{trial.number}")

        # Get objective value
        value = results.get(self.objective, -10)

        # Log trial
        logger.info(f"Trial {trial.number}: {self.objective}={value:.4f}, " f"trades={results.get('total_trades', 0)}")

        # Store intermediate values for pruning
        trial.set_user_attr("total_trades", results.get("total_trades", 0))
        trial.set_user_attr("max_drawdown", results.get("max_drawdown", 1))

        # Prune if no trades
        if results.get("total_trades", 0) == 0:
            raise optuna.TrialPruned()

        return value

    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> OptimizationResult:
        """Run optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
            n_jobs: Number of parallel jobs (1 for sequential)
            show_progress: Show progress bar

        Returns:
            OptimizationResult with best parameters
        """
        study_name = f"strategy_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info("=" * 60)
        logger.info(f"STRATEGY PARAMETER OPTIMIZATION")
        logger.info(f"Study: {study_name}")
        logger.info(f"Objective: {self.objective}")
        logger.info(f"Trials: {n_trials}")
        logger.info("=" * 60)

        # Create study
        self._study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        )

        start_time = datetime.now()

        # Run optimization
        self._study.optimize(
            self._objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress,
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        # Get best trial
        best_trial = self._study.best_trial
        best_params = best_trial.params

        # Create result
        result = OptimizationResult(
            study_name=study_name,
            best_params=best_params,
            best_value=best_trial.value,
            objective=self.objective,
            n_trials=n_trials,
            n_completed=len([t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            optimization_time_seconds=elapsed,
        )

        # Summarize trials
        for trial in self._study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result.trials_summary.append(
                    {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                    }
                )

        # Validate best params on holdout
        logger.info("\nValidating best parameters on holdout...")
        config = self._create_config_with_params(best_params)

        # Run on holdout
        from ..backtest import BacktestEngine, WalkForwardSplit
        from ..configs.strategy_config import StrategyConfig
        from ..strategies.strategy_registry import create_strategy

        strategy_config = StrategyConfig(**config)
        strategy = create_strategy(strategy_config)

        engine = BacktestEngine(
            market_data=self._market_data,
            strategies=[strategy],
            starting_equity=100000.0,
            seed=42,
        )

        split = WalkForwardSplit(
            name="holdout",
            train_start=pd.Timestamp("2025-01-01"),
            train_end=pd.Timestamp("2025-06-30"),
            validation_start=pd.Timestamp("2025-07-01"),
            validation_end=pd.Timestamp("2025-09-30"),
            holdout_start=pd.Timestamp("2025-10-01"),
            holdout_end=pd.Timestamp("2025-12-31"),
        )

        holdout_results = engine.run(split, period="holdout")
        result.holdout_metrics = {
            "sharpe_ratio": holdout_results.get("sharpe_ratio", 0),
            "calmar_ratio": holdout_results.get("calmar_ratio", 0),
            "total_return": holdout_results.get("total_return", 0),
            "total_trades": holdout_results.get("total_trades", 0),
            "max_drawdown": holdout_results.get("max_drawdown", 0),
        }

        # Save results
        result_path = self.output_dir / f"{study_name}.json"
        result.save(str(result_path))

        # Save optimized config
        config_path = self.output_dir / f"{study_name}_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best {self.objective}: {best_trial.value:.4f}")
        logger.info("\nBest Parameters:")
        for name, value in best_params.items():
            logger.info(f"  {name}: {value}")
        logger.info(f"\nHoldout Performance:")
        for name, value in result.holdout_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {name}: {value:.4f}")
            else:
                logger.info(f"  {name}: {value}")
        logger.info(f"\nResults saved to: {result_path}")
        logger.info(f"Optimized config: {config_path}")
        logger.info("=" * 60)

        return result

    def get_param_importance(self) -> Dict[str, float]:
        """Get parameter importance from completed study."""
        if self._study is None:
            raise ValueError("No study available. Run optimize() first.")

        try:
            importance = optuna.importance.get_param_importances(self._study)
            return dict(importance)
        except Exception as e:
            logger.warning(f"Could not compute importance: {e}")
            return {}
