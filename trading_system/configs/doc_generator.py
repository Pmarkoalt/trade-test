"""Configuration documentation generator."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

from pydantic import BaseModel


def get_field_info(model_class: type[BaseModel]) -> Dict[str, Any]:
    """Extract field information from a Pydantic model.

    Args:
        model_class: Pydantic model class

    Returns:
        Dictionary with field information
    """
    fields_info = {}
    model_fields = model_class.model_fields

    for field_name, field_info in model_fields.items():
        field_type = field_info.annotation
        default = field_info.default if field_info.default is not ... else None
        description = field_info.description if hasattr(field_info, "description") else None

        # Get default from default_factory if present
        if default is None and hasattr(field_info, "default_factory"):
            default_factory = field_info.default_factory
            if default_factory is not None and callable(default_factory):
                try:
                    # default_factory is a callable that takes no arguments
                    # Cast to tell mypy the signature
                    factory = cast(Callable[[], Any], default_factory)
                    default = factory()
                except Exception:
                    default = "<default_factory>"

        fields_info[field_name] = {
            "type": field_type,
            "default": default,
            "description": description,
            "required": field_info.default is ...,
        }

    return fields_info


def format_type(type_hint: Any) -> str:
    """Format a type hint as a string.

    Args:
        type_hint: Type hint object

    Returns:
        Formatted type string
    """
    if hasattr(type_hint, "__origin__"):
        # Handle generic types like List[int], Optional[str], etc.
        origin = type_hint.__origin__
        args = type_hint.__args__ if hasattr(type_hint, "__args__") else []

        if origin is list or origin is List:
            arg_str = format_type(args[0]) if args else "Any"
            return f"List[{arg_str}]"
        elif origin is dict or origin is Dict:
            key_str = format_type(args[0]) if len(args) > 0 else "Any"
            val_str = format_type(args[1]) if len(args) > 1 else "Any"
            return f"Dict[{key_str}, {val_str}]"
        elif hasattr(type_hint, "__name__"):
            return str(type_hint.__name__)

    # Handle Literal types
    if hasattr(type_hint, "__args__"):
        args = type_hint.__args__
        if all(isinstance(arg, str) for arg in args):
            return f"Literal[{', '.join(repr(arg) for arg in args)}]"

    # Regular type
    if hasattr(type_hint, "__name__"):
        return str(type_hint.__name__)

    return str(type_hint)


def generate_config_docs(output_path: Optional[str] = None) -> str:
    """Generate documentation for all configuration models.

    Args:
        output_path: Optional path to save the documentation. If None, returns as string.

    Returns:
        Markdown documentation as string
    """
    lines = []

    lines.append("# Configuration Documentation")
    lines.append("")
    lines.append("Complete reference for all configuration options in the trading system.")
    lines.append("")

    # Run Config Documentation
    lines.append("## Run Configuration (`run_config.yaml`)")
    lines.append("")
    lines.append("Main configuration file for backtest runs. Used by CLI commands: `backtest`, `validate`, `holdout`.")
    lines.append("")

    # Import config classes
    from .run_config import DatasetConfig, PortfolioConfig, SplitsConfig, VolatilityScalingConfig

    # DatasetConfig
    lines.append("### Dataset Configuration")
    lines.append("")
    lines.append("```yaml")
    lines.append("dataset:")
    fields = get_field_info(DatasetConfig)
    for field_name, info in fields.items():
        default_str = f"  # Default: {info['default']}" if info["default"] is not None and not info["required"] else ""
        req_str = " (required)" if info["required"] else ""
        type_str = format_type(info["type"])
        desc_str = f" - {info['description']}" if info["description"] else ""
        lines.append(f"  {field_name}: <{type_str}>{req_str}{desc_str}{default_str}")
    lines.append("```")
    lines.append("")

    # SplitsConfig
    lines.append("### Splits Configuration")
    lines.append("")
    lines.append("Walk-forward analysis date splits (pre-registered).")
    lines.append("")
    lines.append("```yaml")
    lines.append("splits:")
    fields = get_field_info(SplitsConfig)
    for field_name, info in fields.items():
        type_str = format_type(info["type"])
        lines.append(f"  {field_name}: <{type_str}>  # Required, format: YYYY-MM-DD")
    lines.append("```")
    lines.append("")
    lines.append("**Date Ordering Requirements:**")
    lines.append("- `train_start` < `train_end` < `validation_start`")
    lines.append("- `validation_start` < `validation_end` < `holdout_start`")
    lines.append("- `holdout_start` < `holdout_end`")
    lines.append("")

    # StrategiesConfig
    lines.append("### Strategies Configuration")
    lines.append("")
    lines.append("```yaml")
    lines.append("strategies:")
    lines.append("  equity:")
    lines.append("    config_path: <str>  # Path to equity strategy config (required)")
    lines.append("    enabled: <bool>     # Default: true")
    lines.append("  crypto:")
    lines.append("    config_path: <str>  # Path to crypto strategy config (required)")
    lines.append("    enabled: <bool>     # Default: true")
    lines.append("```")
    lines.append("")
    lines.append("**Note:** At least one strategy must be enabled.")
    lines.append("")

    # PortfolioConfig
    lines.append("### Portfolio Configuration")
    lines.append("")
    lines.append("```yaml")
    lines.append("portfolio:")
    fields = get_field_info(PortfolioConfig)
    for field_name, info in fields.items():
        default_str = f"  # Default: {info['default']}" if info["default"] is not None else ""
        type_str = format_type(info["type"])
        lines.append(f"  {field_name}: <{type_str}>{default_str}")
    lines.append("```")
    lines.append("")

    # VolatilityScalingConfig
    lines.append("### Volatility Scaling Configuration")
    lines.append("")
    lines.append("```yaml")
    lines.append("volatility_scaling:")
    fields = get_field_info(VolatilityScalingConfig)
    for field_name, info in fields.items():
        default_str = f"  # Default: {info['default']}" if info["default"] is not None else ""
        type_str = format_type(info["type"])
        lines.append(f"  {field_name}: <{type_str}>{default_str}")
    lines.append("```")
    lines.append("")

    # Continue with other configs...
    lines.append("### Other Configurations")
    lines.append("")
    lines.append("- `correlation_guard`: Position correlation filtering")
    lines.append("- `scoring`: Position queue ranking weights")
    lines.append("- `execution`: Signal/execution timing and slippage model")
    lines.append("- `output`: Output file paths and logging configuration")
    lines.append("- `validation`: Validation suite settings (optional)")
    lines.append("- `metrics`: Metrics targets for evaluation (optional)")
    lines.append("")

    # Strategy Config Documentation
    lines.append("## Strategy Configuration (`*_config.yaml`)")
    lines.append("")
    lines.append("Strategy-specific configuration files for equity or crypto strategies.")
    lines.append("")

    lines.append("### Required Fields")
    lines.append("")
    lines.append("```yaml")
    lines.append("name: <str>              # Strategy name")
    lines.append("asset_class: equity|crypto  # Asset class type")
    lines.append("universe: <str|List[str]>   # Universe identifier or symbol list")
    lines.append("benchmark: <str>         # Benchmark symbol (e.g., 'SPY' or 'BTC')")
    lines.append("```")
    lines.append("")

    lines.append("### Configuration Sections")
    lines.append("")
    lines.append("- `indicators`: Technical indicator calculation parameters")
    lines.append("- `eligibility`: Entry eligibility filters (trend qualification)")
    lines.append("- `entry`: Entry trigger configuration")
    lines.append("- `exit`: Exit signal configuration")
    lines.append("- `risk`: Risk management parameters (FROZEN - do not change)")
    lines.append("- `capacity`: Capacity constraints (FROZEN - do not change)")
    lines.append("- `costs`: Execution cost model parameters")
    lines.append("")

    lines.append("## Validation Rules")
    lines.append("")
    lines.append("The configuration system validates:")
    lines.append("")
    lines.append("1. **File existence**: Configuration files must exist")
    lines.append("2. **YAML format**: Valid YAML syntax")
    lines.append("3. **Date formats**: All dates must be in YYYY-MM-DD format")
    lines.append("4. **Date ordering**: Date ranges must be valid (start < end)")
    lines.append("5. **Value ranges**: Numeric values must be within allowed ranges")
    lines.append("6. **Required fields**: All required fields must be present")
    lines.append("7. **Enum values**: String fields with limited options must match allowed values")
    lines.append("")

    lines.append("## Getting Help")
    lines.append("")
    lines.append("### Configuration Tools")
    lines.append("")
    lines.append("- **Generate a template**: `python -m trading_system config template --type run`")
    lines.append("  - Creates a template configuration file with default values and helpful comments")
    lines.append("")
    lines.append("- **Validate a config**: `python -m trading_system config validate --path <config.yaml>`")
    lines.append("  - Validates configuration file and provides detailed error messages with hints")
    lines.append("  - Auto-detects config type (run or strategy) if not specified")
    lines.append("")
    lines.append("- **Interactive wizard**: `python -m trading_system config wizard`")
    lines.append("  - Step-by-step interactive configuration creation")
    lines.append("  - Supports both run and strategy configurations")
    lines.append("")
    lines.append("- **Export JSON Schema**: `python -m trading_system config schema --type run`")
    lines.append("  - Exports JSON Schema for configuration models")
    lines.append("  - Useful for IDE autocomplete and external validation tools")
    lines.append("")
    lines.append("- **Generate documentation**: `python -m trading_system config docs`")
    lines.append("  - Generates complete configuration reference documentation")
    lines.append("")

    doc_content = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(doc_content)

    return doc_content
