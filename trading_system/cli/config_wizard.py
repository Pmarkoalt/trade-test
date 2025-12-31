"""Interactive configuration wizard."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

if TYPE_CHECKING:
    from rich.console import Console

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ConfigWizard:
    """Interactive configuration wizard for creating config files."""

    def __init__(self, use_rich: bool = True):
        """Initialize the wizard.

        Args:
            use_rich: Whether to use rich for enhanced output (if available)
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        if self.use_rich:
            self.console: Optional["Console"] = Console()
        else:
            self.console = None

    def print(self, message: str, style: Optional[str] = None):
        """Print a message.

        Args:
            message: Message to print
            style: Optional rich style
        """
        if self.use_rich and style and self.console:
            self.console.print(message, style=style)
        else:
            print(message)

    def prompt(self, question: str, default: Optional[str] = None, required: bool = True) -> str:
        """Prompt for user input.

        Args:
            question: Question to ask
            default: Default value
            required: Whether input is required

        Returns:
            User input string
        """
        if self.use_rich:
            return str(Prompt.ask(question, default=default or ""))
        else:
            prompt_text = question
            if default:
                prompt_text += f" [{default}]"
            prompt_text += ": "

            while True:
                value = input(prompt_text).strip()
                if value or not required:
                    return value or default or ""
                print("This field is required. Please enter a value.")

    def confirm(self, question: str, default: bool = True) -> bool:
        """Ask a yes/no question.

        Args:
            question: Question to ask
            default: Default value

        Returns:
            Boolean answer
        """
        if self.use_rich:
            return bool(Confirm.ask(question, default=default))
        else:
            prompt_text = question
            prompt_text += " (y/n) [y]" if default else " (y/n) [n]"
            prompt_text += ": "

            while True:
                value = input(prompt_text).strip().lower()
                if not value:
                    return default
                if value in ["y", "yes"]:
                    return True
                if value in ["n", "no"]:
                    return False
                print("Please enter 'y' or 'n'")

    def run_wizard(self, config_type: str = "run") -> Dict[str, Any]:
        """Run the configuration wizard.

        Args:
            config_type: Type of config to create ("run" or "strategy")

        Returns:
            Configuration dictionary
        """
        self.print(f"\n{'='*60}", style="bold")
        self.print(f"Configuration Wizard: {config_type.upper()} Configuration", style="bold cyan")
        self.print(f"{'='*60}\n", style="bold")

        if config_type == "run":
            return self._create_run_config()
        elif config_type == "strategy":
            return self._create_strategy_config()
        else:
            raise ValueError(f"Unknown config type: {config_type}")

    def _create_run_config(self) -> Dict[str, Any]:
        """Create a run configuration interactively."""
        config: Dict[str, Any] = {}

        self.print("Dataset Configuration", style="bold yellow")
        self.print("-" * 40)

        config["dataset"] = {
            "equity_path": self.prompt("Equity data path", default="data/equity/ohlcv/"),
            "crypto_path": self.prompt("Crypto data path", default="data/crypto/ohlcv/"),
            "benchmark_path": self.prompt("Benchmark data path", default="data/benchmarks/"),
            "format": self.prompt("Data format", default="csv"),
            "start_date": self.prompt("Start date (YYYY-MM-DD)", required=True),
            "end_date": self.prompt("End date (YYYY-MM-DD)", required=True),
            "min_lookback_days": int(self.prompt("Minimum lookback days", default="250")),
        }

        self.print("\nSplits Configuration", style="bold yellow")
        self.print("-" * 40)

        config["splits"] = {
            "train_start": self.prompt("Train start date (YYYY-MM-DD)", required=True),
            "train_end": self.prompt("Train end date (YYYY-MM-DD)", required=True),
            "validation_start": self.prompt("Validation start date (YYYY-MM-DD)", required=True),
            "validation_end": self.prompt("Validation end date (YYYY-MM-DD)", required=True),
            "holdout_start": self.prompt("Holdout start date (YYYY-MM-DD)", required=True),
            "holdout_end": self.prompt("Holdout end date (YYYY-MM-DD)", required=True),
        }

        self.print("\nStrategies Configuration", style="bold yellow")
        self.print("-" * 40)

        config["strategies"] = {}

        if self.confirm("Enable equity strategy?", default=True):
            config["strategies"]["equity"] = {
                "config_path": self.prompt("Equity strategy config path", default="configs/equity_config.yaml"),
                "enabled": True,
            }

        if self.confirm("Enable crypto strategy?", default=True):
            config["strategies"]["crypto"] = {
                "config_path": self.prompt("Crypto strategy config path", default="configs/crypto_config.yaml"),
                "enabled": True,
            }

        if not config["strategies"]:
            self.print("\nWarning: At least one strategy must be enabled!", style="bold red")
            sys.exit(1)

        self.print("\nPortfolio Configuration", style="bold yellow")
        self.print("-" * 40)

        config["portfolio"] = {"starting_equity": float(self.prompt("Starting equity (USD)", default="100000.0"))}

        # Use defaults for other sections
        config["volatility_scaling"] = {
            "enabled": True,
            "mode": "continuous",
            "lookback": 20,
            "baseline_lookback": 252,
            "min_multiplier": 0.33,
            "max_multiplier": 1.0,
        }

        config["correlation_guard"] = {
            "enabled": True,
            "min_positions": 4,
            "avg_pairwise_threshold": 0.70,
            "candidate_threshold": 0.75,
        }

        config["scoring"] = {"weights": {"breakout": 0.50, "momentum": 0.30, "diversification": 0.20}}

        config["execution"] = {"signal_timing": "close", "execution_timing": "next_open", "slippage_model": "full"}

        config["output"] = {"base_path": "results/", "log_level": "INFO", "log_file": "backtest.log"}

        config["random_seed"] = 42

        return config

    def _create_strategy_config(self) -> Dict[str, Any]:
        """Create a strategy configuration interactively."""
        config: Dict[str, Any] = {}

        config["name"] = self.prompt("Strategy name", default="momentum_strategy")
        config["asset_class"] = self.prompt("Asset class", default="equity")

        if config["asset_class"] not in ["equity", "crypto"]:
            self.print("Invalid asset class. Must be 'equity' or 'crypto'.", style="bold red")
            sys.exit(1)

        universe_input = self.prompt(
            "Universe (symbol list like 'NASDAQ-100', 'SP500', or comma-separated symbols)",
            default="NASDAQ-100" if config["asset_class"] == "equity" else "BTC,ETH,BNB",
        )

        # Try to parse as list if it contains commas
        if "," in universe_input:
            config["universe"] = [s.strip() for s in universe_input.split(",")]
        else:
            config["universe"] = universe_input

        config["benchmark"] = self.prompt("Benchmark symbol", default="SPY" if config["asset_class"] == "equity" else "BTC")

        # Use defaults for other sections (user can edit manually)
        config["indicators"] = {
            "ma_periods": [20, 50, 200],
            "atr_period": 14,
            "roc_period": 60,
            "breakout_fast": 20,
            "breakout_slow": 55,
            "adv_lookback": 20,
            "corr_lookback": 20,
        }

        config["eligibility"] = {
            "trend_ma": 50 if config["asset_class"] == "equity" else 200,
            "ma_slope_lookback": 20,
            "ma_slope_min": 0.005,
            "require_close_above_trend_ma": config["asset_class"] == "equity",
            "require_close_above_ma200": config["asset_class"] == "crypto",
            "relative_strength_enabled": False,
            "relative_strength_min": 0.0,
        }

        config["entry"] = {"fast_clearance": 0.005, "slow_clearance": 0.010}

        if config["asset_class"] == "equity":
            config["exit"] = {"mode": "ma_cross", "exit_ma": 20, "hard_stop_atr_mult": 2.5}
        else:
            config["exit"] = {"mode": "staged", "exit_ma": 50, "hard_stop_atr_mult": 3.0, "tightened_stop_atr_mult": 2.0}

        config["risk"] = {"risk_per_trade": 0.0075, "max_positions": 8, "max_exposure": 0.80, "max_position_notional": 0.15}

        config["capacity"] = {"max_order_pct_adv": 0.005 if config["asset_class"] == "equity" else 0.0025}

        if config["asset_class"] == "equity":
            config["costs"] = {
                "fee_bps": 1,
                "slippage_base_bps": 8,
                "slippage_std_mult": 0.75,
                "weekend_penalty": 1.0,
                "stress_threshold": -0.03,
                "stress_slippage_mult": 2.0,
            }
        else:
            config["costs"] = {
                "fee_bps": 8,
                "slippage_base_bps": 10,
                "slippage_std_mult": 0.75,
                "weekend_penalty": 1.5,
                "stress_threshold": -0.05,
                "stress_slippage_mult": 2.0,
            }

        return config

    def save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            config: Configuration dictionary
            output_path: Path to save the file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.print(f"\n✓ Configuration saved to: {output_path}", style="bold green")


def run_wizard(config_type: str = "run", output_path: Optional[str] = None) -> None:
    """Run the configuration wizard.

    Args:
        config_type: Type of config ("run" or "strategy")
        output_path: Optional path to save the config. If None, prompts for path.
    """
    wizard = ConfigWizard()

    try:
        config = wizard.run_wizard(config_type)

        if output_path is None:
            default_name = "run_config.yaml" if config_type == "run" else "strategy_config.yaml"
            output_path = wizard.prompt(f"\nSave configuration to", default=default_name, required=True)

        wizard.save_config(config, output_path)

        wizard.print("\n✓ Configuration wizard completed successfully!", style="bold green")

    except KeyboardInterrupt:
        wizard.print("\n\nConfiguration wizard cancelled.", style="yellow")
        sys.exit(1)
    except Exception as e:
        wizard.print(f"\n✗ Error: {e}", style="bold red")
        sys.exit(1)
