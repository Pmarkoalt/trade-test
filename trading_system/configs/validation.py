"""Configuration validation helpers with enhanced error messages."""

from typing import Any, Dict, List, Optional
from pathlib import Path
from pydantic import ValidationError, Field, BaseModel
from datetime import datetime
import yaml
import json
import json


class ConfigValidationError(Exception):
    """Enhanced configuration validation error with helpful messages."""
    
    def __init__(self, message: str, errors: Optional[List[Dict[str, Any]]] = None, config_path: Optional[str] = None):
        super().__init__(message)
        self.errors = errors or []
        self.message = message
        self.config_path = config_path
    
    def format_errors(self) -> str:
        """Format validation errors into a user-friendly string."""
        lines = []
        
        if self.config_path:
            lines.append(f"Configuration file: {self.config_path}")
            lines.append("")
        
        lines.append(self.message)
        
        if not self.errors:
            return "\n".join(lines)
        
        lines.append("\n" + "="*70)
        lines.append("Configuration Errors:")
        lines.append("="*70)
        
        for i, error in enumerate(self.errors, 1):
            loc = " -> ".join(str(x) for x in error.get("loc", []))
            msg = error.get("msg", "Unknown error")
            error_type = error.get("type", "value_error")
            input_value = error.get("input", None)
            
            lines.append(f"\n[{i}] Field: {loc}")
            lines.append(f"    Error: {msg}")
            
            if input_value is not None:
                lines.append(f"    Provided value: {repr(input_value)}")
            
            # Add helpful suggestions based on error type and field
            hint = self._get_error_hint(error, loc)
            if hint:
                lines.append(f"    💡 Hint: {hint}")
            
            # Add example values for common fields
            example = self._get_field_example(loc)
            if example:
                lines.append(f"    📝 Example: {example}")
        
        lines.append("\n" + "="*70)
        lines.append("\nTo fix these errors:")
        lines.append("  1. Check the field names and types match the expected format")
        lines.append("  2. Use 'python -m trading_system config template' to generate a template")
        lines.append("  3. Use 'python -m trading_system config validate --path <file>' to validate")
        lines.append("  4. Use 'python -m trading_system config wizard' for interactive setup")
        
        return "\n".join(lines)
    
    def _get_error_hint(self, error: Dict[str, Any], field_path: str) -> Optional[str]:
        """Get a helpful hint based on error type and field."""
        error_type = error.get("type", "").lower()
        field_lower = field_path.lower()
        
        # Missing field errors
        if "missing" in error_type:
            if "date" in field_lower:
                return "Date fields are required and must be in YYYY-MM-DD format (e.g., '2024-01-15')"
            elif "path" in field_lower:
                return "Path fields are required. Use relative or absolute paths to configuration files."
            elif "config_path" in field_lower:
                return "Strategy config path is required. Point to a valid strategy configuration YAML file."
            return "This field is required. Please add it to your configuration file."
        
        # Type errors
        if "type_error" in error_type:
            if "int" in str(error.get("msg", "")):
                return "This field must be an integer (whole number)."
            elif "float" in str(error.get("msg", "")):
                return "This field must be a number (can include decimals)."
            elif "str" in str(error.get("msg", "")):
                return "This field must be a string (text in quotes)."
            elif "bool" in str(error.get("msg", "")):
                return "This field must be a boolean (true or false)."
            elif "list" in str(error.get("msg", "")):
                return "This field must be a list/array (use YAML list syntax: [item1, item2])."
            return "The value type doesn't match what's expected. Check the field type."
        
        # Value errors
        if "value_error" in error_type:
            if "date" in field_lower or "format" in str(error.get("msg", "")).lower():
                return "Date must be in YYYY-MM-DD format (e.g., '2024-01-15')."
            elif "greater_than" in error_type or "less_than" in error_type:
                ctx = error.get("ctx", {})
                if "gt" in ctx:
                    return f"Value must be greater than {ctx['gt']}."
                elif "ge" in ctx:
                    return f"Value must be greater than or equal to {ctx['ge']}."
                elif "lt" in ctx:
                    return f"Value must be less than {ctx['lt']}."
                elif "le" in ctx:
                    return f"Value must be less than or equal to {ctx['le']}."
            elif "literal" in str(error.get("msg", "")).lower():
                return "Value must be one of the allowed options. Check the documentation for valid values."
        
        # Constraint errors
        if "constraint" in error_type or "assertion" in error_type:
            if "date" in field_lower and "range" in str(error.get("msg", "")).lower():
                return "Start date must be before end date. Check your date ordering."
            elif "clearance" in field_lower:
                return "fast_clearance must be less than slow_clearance."
            elif "multiplier" in field_lower:
                return "min_multiplier must be less than or equal to max_multiplier."
            elif "exposure" in field_lower or "position" in field_lower:
                return "max_position_notional should not exceed max_exposure."
            elif "weights" in field_lower:
                return "Scoring weights must sum to 1.0. Adjust the weights accordingly."
        
        return None
    
    def _get_field_example(self, field_path: str) -> Optional[str]:
        """Get an example value for a field based on its path."""
        field_lower = field_path.lower()
        
        # Date fields
        if "start_date" in field_lower or "end_date" in field_lower:
            if "train" in field_lower:
                return "train_start: '2023-01-01'"
            elif "validation" in field_lower:
                return "validation_start: '2024-04-01'"
            elif "holdout" in field_lower:
                return "holdout_start: '2024-07-01'"
            return "start_date: '2024-01-15'"
        
        # Path fields
        if "path" in field_lower:
            if "equity" in field_lower:
                return "equity_path: 'data/equity/ohlcv/'"
            elif "crypto" in field_lower:
                return "crypto_path: 'data/crypto/ohlcv/'"
            elif "benchmark" in field_lower:
                return "benchmark_path: 'data/benchmarks/'"
            elif "config" in field_lower:
                return "config_path: 'configs/equity_config.yaml'"
            return "path: 'data/example/'"
        
        # Numeric fields with common ranges
        if "risk_per_trade" in field_lower:
            return "risk_per_trade: 0.0075  # 0.75%"
        elif "max_positions" in field_lower:
            return "max_positions: 8"
        elif "max_exposure" in field_lower:
            return "max_exposure: 0.80  # 80%"
        elif "atr_mult" in field_lower:
            return "hard_stop_atr_mult: 2.5"
        elif "clearance" in field_lower:
            return "fast_clearance: 0.005  # 0.5%"
        elif "lookback" in field_lower:
            return "lookback: 20  # days"
        
        # Enum/choice fields
        if "format" in field_lower:
            return "format: 'csv'  # or 'parquet', 'database'"
        elif "mode" in field_lower:
            if "exit" in field_lower:
                return "mode: 'ma_cross'  # or 'staged'"
            elif "volatility" in field_lower or "scaling" in field_lower:
                return "mode: 'continuous'  # or 'regime', 'off'"
        elif "timing" in field_lower:
            return "signal_timing: 'close'  # or execution_timing: 'next_open'"
        elif "slippage" in field_lower:
            return "slippage_model: 'full'  # or 'simple', 'none'"
        elif "asset_class" in field_lower:
            return "asset_class: 'equity'  # or 'crypto'"
        elif "universe" in field_lower:
            return "universe: 'NASDAQ-100'  # or ['BTC', 'ETH', ...]"
        elif "benchmark" in field_lower:
            return "benchmark: 'SPY'  # or 'BTC'"
        
        return None


def validate_file_exists(path: str, config_type: str = "config") -> None:
    """Validate that a configuration file exists.
    
    Args:
        path: Path to the file
        config_type: Type of configuration file (for error message)
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"{config_type.capitalize()} file not found: {path}\n"
            f"Please check the path and ensure the file exists.\n"
            f"Use 'python -m trading_system.cli config template' to generate a template."
        )
    
    if not path_obj.is_file():
        raise ValueError(f"Path exists but is not a file: {path}")


def validate_yaml_format(path: str) -> Dict[str, Any]:
    """Validate YAML file format and return parsed data.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Parsed YAML data
        
    Raises:
        ValueError: If YAML is invalid
    """
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Invalid YAML format in {path}:\n{str(e)}\n"
            f"Please check the YAML syntax. Common issues:\n"
            f"  - Missing colons after keys\n"
            f"  - Incorrect indentation (use spaces, not tabs)\n"
            f"  - Unclosed quotes or brackets"
        ) from e
    except Exception as e:
        raise ValueError(f"Error reading YAML file {path}: {str(e)}") from e


def validate_date_format(date_str: str, field_name: str = "date") -> None:
    """Validate date string format (YYYY-MM-DD).
    
    Args:
        date_str: Date string to validate
        field_name: Name of the field (for error message)
        
    Raises:
        ValueError: If date format is invalid
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Invalid {field_name} format: '{date_str}'\n"
            f"Expected format: YYYY-MM-DD (e.g., '2024-01-15')\n"
            f"Please correct the date format in your configuration."
        )


def validate_date_range(start_date: str, end_date: str, field_prefix: str = "") -> None:
    """Validate that start date is before end date.
    
    Args:
        start_date: Start date string
        end_date: End date string
        field_prefix: Prefix for error message (e.g., "train", "validation")
        
    Raises:
        ValueError: If date range is invalid
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start >= end:
            prefix = f"{field_prefix} " if field_prefix else ""
            raise ValueError(
                f"Invalid {prefix}date range: start date ({start_date}) must be before end date ({end_date})\n"
                f"Please correct the date range in your configuration."
            )
    except ValueError as e:
        # Re-raise if it's our custom error, otherwise let it propagate
        if "Invalid" in str(e) and "date range" in str(e):
            raise
        # Otherwise it's a date format error, which validate_date_format will handle


def wrap_validation_error(
    e: Exception, 
    config_type: str = "Configuration",
    config_path: Optional[str] = None
) -> ConfigValidationError:
    """Wrap Pydantic ValidationError with helpful messages.
    
    Args:
        e: Pydantic ValidationError or any Exception
        config_type: Type of configuration (for error message)
        config_path: Optional path to the config file (for error context)
        
    Returns:
        ConfigValidationError with formatted messages
    """
    from pydantic import ValidationError
    
    if isinstance(e, ValidationError):
        errors = e.errors()
        message = f"{config_type} validation failed. Please fix the following errors:"
        return ConfigValidationError(message, errors, config_path=config_path)
    else:
        # Not a ValidationError, just wrap the message
        message = f"{config_type} validation failed: {str(e)}"
        return ConfigValidationError(message, [], config_path=config_path)


def export_json_schema(model_class: type[BaseModel], output_path: Optional[str] = None) -> Dict[str, Any]:
    """Export JSON Schema for a Pydantic model.
    
    Args:
        model_class: Pydantic model class to export schema for
        output_path: Optional path to save the schema JSON file
        
    Returns:
        JSON Schema dictionary
    """
    schema = model_class.model_json_schema()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)
    
    return schema


def validate_against_schema(
    data: Dict[str, Any],
    model_class: type[BaseModel],
    config_path: Optional[str] = None
) -> BaseModel:
    """Validate data against a Pydantic model schema.
    
    Args:
        data: Data dictionary to validate
        model_class: Pydantic model class to validate against
        config_path: Optional path to config file (for error context)
        
    Returns:
        Validated model instance
        
    Raises:
        ConfigValidationError: If validation fails
    """
    try:
        return model_class(**data)
    except ValidationError as e:
        config_type = model_class.__name__.replace("Config", " configuration")
        raise wrap_validation_error(e, config_type, config_path=config_path) from e


def validate_config_file(
    config_path: str,
    config_type: str = "auto"
) -> tuple[bool, Optional[str], Optional[BaseModel]]:
    """Validate a configuration file and return detailed results.
    
    Args:
        config_path: Path to configuration file
        config_type: Type of config ("run", "strategy", or "auto" to detect)
        
    Returns:
        Tuple of (is_valid, error_message, config_instance)
        - is_valid: True if valid, False otherwise
        - error_message: Error message if invalid, None if valid
        - config_instance: Loaded config instance if valid, None otherwise
    """
    from .run_config import RunConfig
    from .strategy_config import StrategyConfig
    
    # Auto-detect config type
    if config_type == "auto":
        path_lower = config_path.lower()
        if "run_config" in path_lower or "run" in path_lower:
            config_type = "run"
        elif "strategy" in path_lower or "equity_config" in path_lower or "crypto_config" in path_lower:
            config_type = "strategy"
        else:
            # Try to load as run config first
            try:
                config = RunConfig.from_yaml(config_path)
                return True, None, config
            except:
                try:
                    config = StrategyConfig.from_yaml(config_path)
                    return True, None, config
                except Exception as e:
                    if isinstance(e, ConfigValidationError):
                        return False, e.format_errors(), None
                    return False, str(e), None
    
    # Validate based on type
    try:
        if config_type == "run":
            config = RunConfig.from_yaml(config_path)
            return True, None, config
        elif config_type == "strategy":
            config = StrategyConfig.from_yaml(config_path)
            return True, None, config
        else:
            return False, f"Unknown config type: {config_type}", None
    except ConfigValidationError as e:
        return False, e.format_errors(), None
    except Exception as e:
        return False, f"Validation failed: {str(e)}", None

