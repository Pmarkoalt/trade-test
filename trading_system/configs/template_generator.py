"""Configuration template generator."""

from pathlib import Path
from typing import Optional

import yaml

from .run_config import DatasetConfig, RunConfig, SplitsConfig, StrategiesConfig, StrategyConfigRef
from .strategy_config import StrategyConfig


def generate_run_config_template(output_path: Optional[str] = None, include_comments: bool = True) -> str:
    """Generate a template run_config.yaml file.

    Args:
        output_path: Optional path to save the template. If None, returns as string.
        include_comments: Whether to include helpful comments in the template

    Returns:
        Template YAML content as string
    """
    # Create a default RunConfig using model constructors

    config = RunConfig(
        dataset=DatasetConfig(
            equity_path="data/equity/ohlcv/",
            crypto_path="data/crypto/ohlcv/",
            benchmark_path="data/benchmarks/",
            format="csv",
            start_date="2023-01-01",
            end_date="2024-12-31",
            min_lookback_days=250,
        ),
        splits=SplitsConfig(
            train_start="2023-01-01",
            train_end="2024-03-31",
            validation_start="2024-04-01",
            validation_end="2024-06-30",
            holdout_start="2024-07-01",
            holdout_end="2024-12-31",
        ),
        strategies=StrategiesConfig(
            equity=StrategyConfigRef(config_path="configs/equity_config.yaml", enabled=True),
            crypto=StrategyConfigRef(config_path="configs/crypto_config.yaml", enabled=True),
        ),
    )

    # Generate YAML
    yaml_content = str(yaml.dump(config.model_dump(), default_flow_style=False, sort_keys=False))

    # Add comments if requested
    if include_comments:
        yaml_lines = yaml_content.split("\n")
        commented_lines = []

        for line in yaml_lines:
            commented_lines.append(line)

            # Add helpful comments for key sections
            if line.strip() == "dataset:":
                commented_lines.append("# Dataset configuration")
                commented_lines.append("# Paths to data directories and date ranges")
            elif line.strip() == "splits:":
                commented_lines.append("# Walk-forward analysis splits (pre-registered)")
            elif line.strip() == "strategies:":
                commented_lines.append("# Strategy configurations to load")
            elif line.strip() == "portfolio:":
                commented_lines.append("# Portfolio-level settings")
            elif line.strip() == "volatility_scaling:":
                commented_lines.append("# Volatility-based position sizing")
            elif line.strip() == "correlation_guard:":
                commented_lines.append("# Position correlation filtering")
            elif line.strip() == "output:":
                commented_lines.append("# Output files and logging configuration")
            elif line.strip() == "validation:":
                commented_lines.append("# Validation suite settings (optional)")
            elif line.strip() == "metrics:":
                commented_lines.append("# Metrics targets for evaluation (optional)")

        yaml_content = str("\n".join(commented_lines))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(yaml_content)

    return yaml_content


def generate_strategy_config_template(
    asset_class: str = "equity", output_path: Optional[str] = None, include_comments: bool = True
) -> str:
    """Generate a template strategy_config.yaml file.

    Args:
        asset_class: "equity" or "crypto"
        output_path: Optional path to save the template. If None, returns as string.
        include_comments: Whether to include helpful comments in the template

    Returns:
        Template YAML content as string

    Raises:
        ValueError: If asset_class is not "equity" or "crypto"
    """
    if asset_class not in ["equity", "crypto"]:
        raise ValueError(f"asset_class must be 'equity' or 'crypto', got '{asset_class}'")

    # Create default config based on asset class using model constructors
    from .strategy_config import CostsConfig, EligibilityConfig, ExitConfig

    if asset_class == "equity":
        config = StrategyConfig(
            name="equity_momentum",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            exit=ExitConfig(mode="ma_cross", exit_ma=20, hard_stop_atr_mult=2.5),
            costs=CostsConfig(fee_bps=1, slippage_base_bps=8),
        )
    else:  # crypto
        config = StrategyConfig(
            name="crypto_momentum",
            asset_class="crypto",
            universe=["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "MATIC", "LTC", "LINK"],
            benchmark="BTC",
            exit=ExitConfig(mode="staged", exit_ma=50, hard_stop_atr_mult=3.0, tightened_stop_atr_mult=2.0),
            costs=CostsConfig(fee_bps=8, slippage_base_bps=10, weekend_penalty=1.5),
            eligibility=EligibilityConfig(require_close_above_ma200=True),
        )

    # Generate YAML
    yaml_content = str(yaml.dump(config.model_dump(), default_flow_style=False, sort_keys=False))

    # Add comments if requested
    if include_comments:
        yaml_lines = yaml_content.split("\n")
        commented_lines = []

        for line in yaml_lines:
            commented_lines.append(line)

            # Add helpful comments
            if line.strip().startswith("name:"):
                commented_lines.append(f"# Strategy name ({asset_class} momentum)")
            elif line.strip().startswith("asset_class:"):
                commented_lines.append(f"# Asset class: {asset_class}")
            elif line.strip() == "indicators:":
                commented_lines.append("# Technical indicator calculation parameters")
            elif line.strip() == "eligibility:":
                commented_lines.append("# Entry eligibility filters (trend qualification)")
            elif line.strip() == "entry:":
                commented_lines.append("# Entry trigger configuration")
            elif line.strip() == "exit:":
                commented_lines.append("# Exit signal configuration")
            elif line.strip() == "risk:":
                commented_lines.append("# Risk management parameters (FROZEN - do not change)")
            elif line.strip() == "capacity:":
                commented_lines.append("# Capacity constraints (FROZEN - do not change)")
            elif line.strip() == "costs:":
                commented_lines.append("# Execution cost model parameters")

        yaml_content = str("\n".join(commented_lines))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(yaml_content)

    return yaml_content
