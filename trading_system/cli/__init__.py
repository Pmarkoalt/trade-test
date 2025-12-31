"""CLI utilities and wizards for the trading system."""

# Import wizards
# Import CLI commands from cli.py using importlib to handle naming conflict
# Since we have both cli.py and cli/ directory, we need to load cli.py directly
import importlib.util
import sys
from pathlib import Path

from .config_wizard import run_wizard
from .strategy_wizard import run_strategy_wizard

_parent_dir = Path(__file__).parent.parent
_cli_py_path = _parent_dir / "cli.py"

if _cli_py_path.exists():
    # Load cli.py as a module with a unique name to avoid conflicts
    _cli_spec = importlib.util.spec_from_file_location("trading_system.cli_file", _cli_py_path)
    if _cli_spec and _cli_spec.loader:
        _cli_module = importlib.util.module_from_spec(_cli_spec)
        # Set proper package context for relative imports in cli.py
        _cli_module.__package__ = "trading_system"
        _cli_module.__name__ = "trading_system.cli_file"
        # Store in sys.modules with a unique name
        sys.modules["trading_system.cli_file"] = _cli_module
        _cli_spec.loader.exec_module(_cli_module)

        # Re-export functions
        main = _cli_module.main
        cmd_backtest = _cli_module.cmd_backtest
        cmd_validate = _cli_module.cmd_validate
        cmd_holdout = _cli_module.cmd_holdout
        cmd_report = _cli_module.cmd_report
        cmd_strategy_create = _cli_module.cmd_strategy_create
        cmd_strategy_template = _cli_module.cmd_strategy_template
        setup_logging = _cli_module.setup_logging
    else:
        raise ImportError("Failed to load cli.py")
else:
    raise ImportError("trading_system/cli.py not found")

__all__ = [
    "main",
    "run_wizard",
    "run_strategy_wizard",
    "cmd_backtest",
    "cmd_validate",
    "cmd_holdout",
    "cmd_report",
    "cmd_strategy_create",
    "cmd_strategy_template",
    "setup_logging",
]
