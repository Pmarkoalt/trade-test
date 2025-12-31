"""Parameter sensitivity grid search and visualization."""

import json
import logging
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..configs.run_config import RunConfig, SensitivityConfig
    from ..configs.strategy_config import StrategyConfig

import numpy as np
import pandas as pd

# Try to import parallel execution libraries
HAS_JOBLIB = False
HAS_MULTIPROCESSING = False

try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except ImportError:
    pass

try:
    from multiprocessing import Pool, cpu_count

    HAS_MULTIPROCESSING = True
except ImportError:
    pass

# Matplotlib for basic visualization
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment]

# Plotly for interactive visualization (optional)
try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None
    px = None

logger = logging.getLogger(__name__)

# Export visualization availability flags
__all__ = [
    "ParameterSensitivityGrid",
    "run_parameter_sensitivity",
    "generate_parameter_grid_from_config",
    "apply_parameters_to_strategy_config",
    "apply_parameters_to_run_config",
    "save_sensitivity_results",
    "generate_all_heatmaps",
    "HAS_MATPLOTLIB",
    "HAS_PLOTLY",
]


class ParameterSensitivityGrid:
    """Parameter sensitivity grid search with heatmap visualization."""

    def __init__(
        self,
        parameter_ranges: Dict[str, List[Any]],
        metric_func: Callable[[Dict[str, Any]], float],
        random_seed: Optional[int] = None,
        n_jobs: Optional[int] = None,
        parallel: bool = True,
    ):
        """Initialize parameter sensitivity grid.

        Args:
            parameter_ranges: Dictionary mapping parameter names to lists of values to test
            metric_func: Function that takes a parameter dict and returns a metric (e.g., Sharpe)
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (default: -1 for all CPUs, None for sequential)
            parallel: If True, use parallel execution when available (default: True)
        """
        self.parameter_ranges = parameter_ranges
        self.metric_func = metric_func
        self.results: List[Dict[str, Any]] = []
        self.n_jobs = n_jobs
        self.parallel = parallel and (HAS_JOBLIB or HAS_MULTIPROCESSING)

        if random_seed is not None:
            np.random.seed(random_seed)

    def run(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """Run grid search over all parameter combinations.

        Args:
            progress_callback: Optional callback function(completed, total) for progress tracking

        Returns:
            Dictionary with results and analysis
        """
        param_names = list(self.parameter_ranges.keys())
        param_values = list(self.parameter_ranges.values())

        # Generate all combinations
        all_combinations = list(product(*param_values))
        total_combinations = len(all_combinations)

        logger.info(f"Running parameter sensitivity: {total_combinations} combinations")

        if self.parallel and total_combinations > 1:
            # Use parallel execution
            results = self._run_parallel(all_combinations, param_names, progress_callback)
        else:
            # Sequential execution
            results = self._run_sequential(all_combinations, param_names, progress_callback)

        self.results = results

        # Analyze results
        metrics = [r["metric"] for r in results] if results else []

        analysis = {
            "results": results,
            "best_params": self._find_best_params(),
            "worst_params": self._find_worst_params(),
            "metric_mean": np.mean(metrics) if metrics else 0.0,
            "metric_std": np.std(metrics) if metrics else 0.0,
            "metric_min": np.min(metrics) if metrics else 0.0,
            "metric_max": np.max(metrics) if metrics else 0.0,
            "has_sharp_peaks": self._check_sharp_peaks(),
            "stable_neighborhoods": self._find_stable_neighborhoods(),
        }

        return analysis

    def _run_sequential(
        self, all_combinations: List[Tuple], param_names: List[str], progress_callback: Optional[Callable[[int, int], None]]
    ) -> List[Dict[str, Any]]:
        """Run grid search sequentially.

        Args:
            all_combinations: List of parameter value tuples
            param_names: List of parameter names
            progress_callback: Optional progress callback

        Returns:
            List of result dictionaries
        """
        results = []
        total = len(all_combinations)

        for i, combo in enumerate(all_combinations):
            params = dict(zip(param_names, combo))
            try:
                metric_value = self.metric_func(params)
                results.append({"params": params, "metric": metric_value})
            except Exception as e:
                logger.debug(f"Skipping invalid parameter combination: {params}, error: {e}")
                continue

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def _run_parallel(
        self, all_combinations: List[Tuple], param_names: List[str], progress_callback: Optional[Callable[[int, int], None]]
    ) -> List[Dict[str, Any]]:
        """Run grid search in parallel.

        Args:
            all_combinations: List of parameter value tuples
            param_names: List of parameter names
            progress_callback: Optional progress callback

        Returns:
            List of result dictionaries
        """

        def evaluate_combo(combo: Tuple) -> Optional[Dict[str, Any]]:
            """Evaluate a single parameter combination."""
            params = dict(zip(param_names, combo))
            try:
                metric_value = self.metric_func(params)
                return {"params": params, "metric": metric_value}
            except Exception as e:
                logger.debug(f"Skipping invalid parameter combination: {params}, error: {e}")
                return None

        # Determine number of jobs
        if self.n_jobs is None:
            n_jobs = -1  # Use all CPUs
        else:
            n_jobs = self.n_jobs

        logger.info(f"Running parallel grid search with {n_jobs if n_jobs > 0 else 'all'} workers")

        if HAS_JOBLIB:
            # Use joblib (preferred, better memory management)
            results = Parallel(n_jobs=n_jobs, verbose=1 if logger.level <= logging.INFO else 0)(
                delayed(evaluate_combo)(combo) for combo in all_combinations
            )
        elif HAS_MULTIPROCESSING:
            # Fallback to multiprocessing
            with Pool(processes=n_jobs if n_jobs > 0 else cpu_count()) as pool:
                results = pool.map(evaluate_combo, all_combinations)
        else:
            # Fallback to sequential
            logger.warning("Parallel execution not available, falling back to sequential")
            return self._run_sequential(all_combinations, param_names, progress_callback)

        # Filter out None results (invalid combinations)
        filtered_results: List[Dict[str, Any]] = [r for r in results if r is not None]

        if progress_callback:
            progress_callback(len(filtered_results), len(all_combinations))

        return filtered_results

    def _find_best_params(self) -> Dict[str, Any]:
        """Find parameters with best metric value."""
        if not self.results:
            return {}

        best_result = max(self.results, key=lambda x: x["metric"])
        params = best_result.get("params", {})
        return dict(params) if params else {}

    def _find_worst_params(self) -> Dict[str, Any]:
        """Find parameters with worst metric value."""
        if not self.results:
            return {}

        worst_result = min(self.results, key=lambda x: x["metric"])
        params = worst_result.get("params", {})
        return dict(params) if params else {}

    def _check_sharp_peaks(self, threshold: float = 2.0) -> bool:
        """Check if there are sharp peaks (overfitting indicator).

        A sharp peak means the best result is much better than nearby results.

        Args:
            threshold: Standard deviations above mean to consider a sharp peak

        Returns:
            True if sharp peaks detected
        """
        if not self.results:
            return False

        metrics = np.array([r["metric"] for r in self.results])
        mean_metric = np.mean(metrics)
        std_metric = np.std(metrics)

        if std_metric == 0:
            return False

        best_metric = float(np.max(metrics))
        z_score = (best_metric - mean_metric) / std_metric

        return bool(z_score > threshold)

    def _find_stable_neighborhoods(self, tolerance: float = 0.1) -> List[Dict[str, Any]]:
        """Find stable neighborhoods (parameters with similar performance).

        Args:
            tolerance: Fraction of metric range to consider "similar"

        Returns:
            List of parameter sets in stable neighborhoods
        """
        if not self.results:
            return []

        metrics = np.array([r["metric"] for r in self.results])
        metric_range = np.max(metrics) - np.min(metrics)
        threshold = tolerance * metric_range

        # Find results within threshold of best
        best_metric = np.max(metrics)
        stable = [r for r in self.results if abs(r["metric"] - best_metric) <= threshold]

        return stable

    def plot_heatmap(self, param_x: str, param_y: str, output_path: Optional[str] = None, use_plotly: bool = False) -> None:
        """Plot 2D heatmap for two parameters.

        Args:
            param_x: Name of parameter for x-axis
            param_y: Name of parameter for y-axis
            output_path: Optional path to save figure
            use_plotly: If True, use plotly for interactive visualization (requires plotly)
        """
        if not self.results:
            raise ValueError("No results to plot. Run grid search first.")

        # Filter results that have both parameters
        valid_results = [r for r in self.results if param_x in r["params"] and param_y in r["params"]]

        if not valid_results:
            raise ValueError(f"No results found with both {param_x} and {param_y}")

        # Extract unique parameter values
        x_values = sorted(set(r["params"][param_x] for r in valid_results))
        y_values = sorted(set(r["params"][param_y] for r in valid_results))

        if len(x_values) < 2 or len(y_values) < 2:
            raise ValueError(f"Insufficient unique values for heatmap: {param_x}={len(x_values)}, {param_y}={len(y_values)}")

        # Create matrix
        matrix = np.full((len(y_values), len(x_values)), np.nan)

        for result in valid_results:
            params = result["params"]
            try:
                x_idx = x_values.index(params[param_x])
                y_idx = y_values.index(params[param_y])
                matrix[y_idx, x_idx] = result["metric"]
            except (ValueError, KeyError):
                continue

        # Check if we have any valid values
        if np.all(np.isnan(matrix)):
            raise ValueError(f"No valid metric values for {param_x} vs {param_y}")

        # Use plotly if requested and available
        if use_plotly and HAS_PLOTLY:
            self._plot_heatmap_plotly(matrix, x_values, y_values, param_x, param_y, output_path)
        elif HAS_MATPLOTLIB:
            self._plot_heatmap_matplotlib(matrix, x_values, y_values, param_x, param_y, output_path)
        else:
            raise ImportError("Neither matplotlib nor plotly is available for plotting")

    def _plot_heatmap_matplotlib(
        self,
        matrix: np.ndarray,
        x_values: List[Any],
        y_values: List[Any],
        param_x: str,
        param_y: str,
        output_path: Optional[str] = None,
    ) -> None:
        """Plot heatmap using matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", interpolation="nearest")

        # Set ticks
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_xticklabels([str(v) for v in x_values], rotation=45, ha="right")
        ax.set_yticklabels([str(v) for v in y_values])

        # Labels
        ax.set_xlabel(param_x.replace(".", " ").title(), fontsize=12)
        ax.set_ylabel(param_y.replace(".", " ").title(), fontsize=12)
        ax.set_title(f"Parameter Sensitivity: {param_x} vs {param_y}", fontsize=14)

        # Colorbar
        plt.colorbar(im, ax=ax, label="Metric Value")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def _plot_heatmap_plotly(
        self,
        matrix: np.ndarray,
        x_values: List[Any],
        y_values: List[Any],
        param_x: str,
        param_y: str,
        output_path: Optional[str] = None,
    ) -> None:
        """Plot heatmap using plotly."""
        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=[str(v) for v in x_values],
                y=[str(v) for v in y_values],
                colorscale="RdYlGn",
                colorbar=dict(title="Metric Value"),
                hovertemplate="%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<br>Metric: %{z:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Parameter Sensitivity: {param_x} vs {param_y}",
            xaxis_title=param_x.replace(".", " ").title(),
            yaxis_title=param_y.replace(".", " ").title(),
            width=800,
            height=600,
        )

        if output_path:
            # Save as HTML for interactive viewing
            html_path = str(output_path).replace(".png", ".html")
            fig.write_html(html_path)
            logger.info(f"Saved interactive heatmap to {html_path}")

            # Also save as static image if possible
            try:
                fig.write_image(output_path)
            except Exception as e:
                logger.warning(f"Could not save static image (requires kaleido): {e}")
        else:
            fig.show()


def run_parameter_sensitivity(
    parameter_ranges: Dict[str, List[Any]],
    metric_func: Callable[[Dict[str, Any]], float],
    random_seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
    parallel: bool = True,
) -> Dict[str, Any]:
    """Convenience function to run parameter sensitivity analysis.

    Args:
        parameter_ranges: Dictionary mapping parameter names to lists of values
        metric_func: Function that takes parameters and returns a metric
        random_seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs (default: -1 for all CPUs, None for sequential)
        parallel: If True, use parallel execution when available (default: True)

    Returns:
        Analysis results dictionary
    """
    grid = ParameterSensitivityGrid(parameter_ranges, metric_func, random_seed, n_jobs, parallel)
    return grid.run()


def generate_parameter_grid_from_config(
    sensitivity_config: "SensitivityConfig", asset_class: str = "equity"
) -> Dict[str, List[Union[int, float, str]]]:
    """Generate parameter grid from SensitivityConfig.

    Maps config parameter ranges to actual strategy parameter names.

    Args:
        sensitivity_config: SensitivityConfig instance from RunConfig
        asset_class: "equity" or "crypto"

    Returns:
        Dictionary mapping parameter names to value lists
    """
    parameter_ranges = {}

    if asset_class == "equity":
        # Map equity parameters
        if hasattr(sensitivity_config, "equity_atr_mult") and sensitivity_config.equity_atr_mult:
            parameter_ranges["exit.hard_stop_atr_mult"] = sensitivity_config.equity_atr_mult

        if hasattr(sensitivity_config, "equity_breakout_clearance") and sensitivity_config.equity_breakout_clearance:
            # This affects both fast_clearance and slow_clearance
            # For simplicity, we'll test fast_clearance (can be extended)
            parameter_ranges["entry.fast_clearance"] = sensitivity_config.equity_breakout_clearance

        if hasattr(sensitivity_config, "equity_exit_ma") and sensitivity_config.equity_exit_ma:
            parameter_ranges["exit.exit_ma"] = [float(x) for x in sensitivity_config.equity_exit_ma]

    elif asset_class == "crypto":
        # Map crypto parameters
        if hasattr(sensitivity_config, "crypto_atr_mult") and sensitivity_config.crypto_atr_mult:
            parameter_ranges["exit.hard_stop_atr_mult"] = sensitivity_config.crypto_atr_mult

        if hasattr(sensitivity_config, "crypto_breakout_clearance") and sensitivity_config.crypto_breakout_clearance:
            parameter_ranges["entry.fast_clearance"] = sensitivity_config.crypto_breakout_clearance

        if hasattr(sensitivity_config, "crypto_exit_mode") and sensitivity_config.crypto_exit_mode:
            # Map exit modes to strategy config format
            # Handle as separate parameters - mode and exit_ma
            exit_modes = []
            exit_mas = []
            for mode in sensitivity_config.crypto_exit_mode:
                if mode == "MA20":
                    exit_modes.append("ma_cross")
                    exit_mas.append(20)
                elif mode == "MA50":
                    exit_modes.append("ma_cross")
                    exit_mas.append(50)
                elif mode == "staged":
                    exit_modes.append("staged")
                    exit_mas.append(50)  # Default for staged

            # Add as separate parameters
            if exit_modes:
                unique_modes = list(set(exit_modes))
                if len(unique_modes) > 1:
                    parameter_ranges["exit.mode"] = unique_modes  # type: ignore[assignment]
                unique_mas = list(set(exit_mas))
                if len(unique_mas) > 1:
                    parameter_ranges["exit.exit_ma"] = [float(x) for x in unique_mas]
                elif len(unique_mas) == 1:
                    # If only one MA value, still add it if mode varies
                    if len(unique_modes) > 1:
                        parameter_ranges["exit.exit_ma"] = [float(x) for x in unique_mas]

    # Portfolio-level parameters
    if hasattr(sensitivity_config, "vol_scaling_mode") and sensitivity_config.vol_scaling_mode:
        parameter_ranges["volatility_scaling.mode"] = sensitivity_config.vol_scaling_mode  # type: ignore[assignment]

    # Convert all values to lists of Union[int, float, str] for type consistency
    converted_ranges: Dict[str, List[Union[int, float, str]]] = {}
    for key, value_list in parameter_ranges.items():
        converted_list: List[Union[int, float, str]] = []
        for v in value_list:
            if isinstance(v, str):
                converted_list.append(v)
            elif isinstance(v, (int, float)):
                converted_list.append(v)
            else:  # pragma: no cover
                # Type narrowing means this is unreachable, but handle for runtime safety
                converted_list.append(str(v))
        converted_ranges[key] = converted_list

    return converted_ranges


def apply_parameters_to_strategy_config(strategy_config: "StrategyConfig", parameters: Dict[str, Any]) -> "StrategyConfig":
    """Apply parameter modifications to a strategy config.

    Args:
        strategy_config: StrategyConfig instance
        parameters: Dictionary of parameter changes (e.g., {'exit.hard_stop_atr_mult': 3.0})

    Returns:
        Modified StrategyConfig instance (copy)
    """
    from ..configs.strategy_config import StrategyConfig

    # Create a deep copy by converting to dict and back
    config_dict = strategy_config.model_dump()

    # Apply parameter changes using dot notation
    for param_path, value in parameters.items():
        parts = param_path.split(".")
        if len(parts) == 2:
            section, param = parts
            if section in config_dict and isinstance(config_dict[section], dict):
                config_dict[section][param] = value

    # Recreate config from modified dict
    return StrategyConfig(**config_dict)


def apply_parameters_to_run_config(run_config: "RunConfig", parameters: Dict[str, Any]) -> "RunConfig":
    """Apply portfolio-level parameter modifications to run config.

    Args:
        run_config: RunConfig instance
        parameters: Dictionary of parameter changes (e.g., {'volatility_scaling.mode': 'regime'})

    Returns:
        Modified RunConfig instance (copy)
    """
    from ..configs.run_config import RunConfig

    # Create a deep copy
    config_dict = run_config.model_dump()

    # Apply parameter changes
    for param_path, value in parameters.items():
        parts = param_path.split(".")
        if len(parts) == 2:
            section, param = parts
            if section in config_dict and isinstance(config_dict[section], dict):
                config_dict[section][param] = value

    # Recreate config from modified dict
    return RunConfig(**config_dict)


def save_sensitivity_results(
    analysis: Dict[str, Any], parameter_ranges: Dict[str, List[Any]], output_dir: Path, metric_name: str = "sharpe_ratio"
) -> None:
    """Save sensitivity analysis results to files.

    Args:
        analysis: Analysis results from ParameterSensitivityGrid.run()
        parameter_ranges: Original parameter ranges used
        output_dir: Directory to save results
        metric_name: Name of the metric used
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON summary
    summary = {
        "metric_name": metric_name,
        "parameter_ranges": {k: [str(v) for v in vals] for k, vals in parameter_ranges.items()},
        "best_params": {k: str(v) for k, v in analysis["best_params"].items()},
        "worst_params": {k: str(v) for k, v in analysis["worst_params"].items()},
        "metric_stats": {
            "mean": float(analysis["metric_mean"]),
            "std": float(analysis["metric_std"]),
            "min": float(analysis["metric_min"]),
            "max": float(analysis["metric_max"]),
        },
        "has_sharp_peaks": analysis["has_sharp_peaks"],
        "stable_neighborhoods_count": len(analysis["stable_neighborhoods"]),
        "total_combinations": len(analysis["results"]),
    }

    with open(output_dir / "sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed results CSV
    results_data = []
    for result in analysis["results"]:
        row = result["params"].copy()
        row[metric_name] = result["metric"]
        results_data.append(row)

    df = pd.DataFrame(results_data)
    df.to_csv(output_dir / "sensitivity_results.csv", index=False)

    logger.info(f"Saved sensitivity results to {output_dir}")


def generate_all_heatmaps(
    grid: ParameterSensitivityGrid, output_dir: Path, metric_name: str = "sharpe_ratio", use_plotly: bool = False
) -> List[Path]:
    """Generate heatmaps for all parameter pairs.

    Args:
        grid: ParameterSensitivityGrid instance with results
        output_dir: Directory to save heatmaps
        metric_name: Name of the metric being visualized
        use_plotly: If True, use plotly for interactive visualization

    Returns:
        List of paths to saved heatmap files
    """
    if not grid.results:
        logger.warning("No results to plot. Run grid search first.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all parameter names
    param_names = list(set(param_name for result in grid.results for param_name in result["params"].keys()))

    if len(param_names) < 2:
        logger.warning("Need at least 2 parameters to generate heatmaps")
        return []

    saved_paths = []

    # Generate heatmaps for all pairs
    for i, param_x in enumerate(param_names):
        for param_y in param_names[i + 1 :]:
            try:
                if use_plotly and HAS_PLOTLY:
                    output_path = output_dir / f'heatmap_{param_x.replace(".", "_")}_vs_{param_y.replace(".", "_")}.html'
                else:
                    output_path = output_dir / f'heatmap_{param_x.replace(".", "_")}_vs_{param_y.replace(".", "_")}.png'

                grid.plot_heatmap(param_x, param_y, output_path=str(output_path), use_plotly=use_plotly)
                saved_paths.append(output_path)
                logger.info(f"Generated heatmap: {output_path}")
            except (ValueError, Exception) as e:
                logger.warning(f"Failed to generate heatmap for {param_x} vs {param_y}: {e}")

    return saved_paths
