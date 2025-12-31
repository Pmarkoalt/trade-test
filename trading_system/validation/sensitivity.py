"""Parameter sensitivity grid search and visualization."""

from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class ParameterSensitivityGrid:
    """Parameter sensitivity grid search with heatmap visualization."""
    
    def __init__(
        self,
        parameter_ranges: Dict[str, List[Any]],
        metric_func: Callable[[Dict[str, Any]], float],
        random_seed: Optional[int] = None
    ):
        """Initialize parameter sensitivity grid.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to lists of values to test
            metric_func: Function that takes a parameter dict and returns a metric (e.g., Sharpe)
            random_seed: Random seed for reproducibility
        """
        self.parameter_ranges = parameter_ranges
        self.metric_func = metric_func
        self.results: List[Dict[str, Any]] = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def run(self) -> Dict[str, Any]:
        """Run grid search over all parameter combinations.
        
        Returns:
            Dictionary with results and analysis
        """
        param_names = list(self.parameter_ranges.keys())
        param_values = list(self.parameter_ranges.values())
        
        # Generate all combinations
        all_combinations = list(product(*param_values))
        
        results = []
        for combo in all_combinations:
            params = dict(zip(param_names, combo))
            try:
                metric_value = self.metric_func(params)
                results.append({
                    'params': params,
                    'metric': metric_value
                })
            except Exception as e:
                # Skip invalid parameter combinations
                continue
        
        self.results = results
        
        # Analyze results
        metrics = [r['metric'] for r in results]
        
        analysis = {
            'results': results,
            'best_params': self._find_best_params(),
            'worst_params': self._find_worst_params(),
            'metric_mean': np.mean(metrics),
            'metric_std': np.std(metrics),
            'metric_min': np.min(metrics),
            'metric_max': np.max(metrics),
            'has_sharp_peaks': self._check_sharp_peaks(),
            'stable_neighborhoods': self._find_stable_neighborhoods(),
        }
        
        return analysis
    
    def _find_best_params(self) -> Dict[str, Any]:
        """Find parameters with best metric value."""
        if not self.results:
            return {}
        
        best_result = max(self.results, key=lambda x: x['metric'])
        return best_result['params']
    
    def _find_worst_params(self) -> Dict[str, Any]:
        """Find parameters with worst metric value."""
        if not self.results:
            return {}
        
        worst_result = min(self.results, key=lambda x: x['metric'])
        return worst_result['params']
    
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
        
        metrics = np.array([r['metric'] for r in self.results])
        mean_metric = np.mean(metrics)
        std_metric = np.std(metrics)
        
        if std_metric == 0:
            return False
        
        best_metric = np.max(metrics)
        z_score = (best_metric - mean_metric) / std_metric
        
        return z_score > threshold
    
    def _find_stable_neighborhoods(self, tolerance: float = 0.1) -> List[Dict[str, Any]]:
        """Find stable neighborhoods (parameters with similar performance).
        
        Args:
            tolerance: Fraction of metric range to consider "similar"
        
        Returns:
            List of parameter sets in stable neighborhoods
        """
        if not self.results:
            return []
        
        metrics = np.array([r['metric'] for r in self.results])
        metric_range = np.max(metrics) - np.min(metrics)
        threshold = tolerance * metric_range
        
        # Find results within threshold of best
        best_metric = np.max(metrics)
        stable = [
            r for r in self.results
            if abs(r['metric'] - best_metric) <= threshold
        ]
        
        return stable
    
    def plot_heatmap(
        self,
        param_x: str,
        param_y: str,
        output_path: Optional[str] = None
    ) -> None:
        """Plot 2D heatmap for two parameters.
        
        Args:
            param_x: Name of parameter for x-axis
            param_y: Name of parameter for y-axis
            output_path: Optional path to save figure
        """
        if not self.results:
            raise ValueError("No results to plot. Run grid search first.")
        
        # Extract unique parameter values
        x_values = sorted(set(r['params'][param_x] for r in self.results))
        y_values = sorted(set(r['params'][param_y] for r in self.results))
        
        # Create matrix
        matrix = np.full((len(y_values), len(x_values)), np.nan)
        
        for result in self.results:
            params = result['params']
            if param_x in params and param_y in params:
                x_idx = x_values.index(params[param_x])
                y_idx = y_values.index(params[param_y])
                matrix[y_idx, x_idx] = result['metric']
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        
        # Set ticks
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_xticklabels([str(v) for v in x_values])
        ax.set_yticklabels([str(v) for v in y_values])
        
        # Labels
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_title(f'Parameter Sensitivity: {param_x} vs {param_y}')
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Metric Value')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def run_parameter_sensitivity(
    parameter_ranges: Dict[str, List[Any]],
    metric_func: Callable[[Dict[str, Any]], float],
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """Convenience function to run parameter sensitivity analysis.
    
    Args:
        parameter_ranges: Dictionary mapping parameter names to lists of values
        metric_func: Function that takes parameters and returns a metric
        random_seed: Random seed for reproducibility
    
    Returns:
        Analysis results dictionary
    """
    grid = ParameterSensitivityGrid(parameter_ranges, metric_func, random_seed)
    return grid.run()

