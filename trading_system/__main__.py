"""Entry point for running trading_system as a module: python -m trading_system"""

import importlib.util
import sys
from pathlib import Path

# Load .env file before running CLI
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, environment variables must be set manually

# Import from cli.py directly (not from cli/ package)
# Since we have both cli.py and cli/ directory, we need to load cli.py explicitly
_module_dir = Path(__file__).parent
_cli_py_path = _module_dir / "cli.py"

if _cli_py_path.exists():
    _spec = importlib.util.spec_from_file_location("trading_system.cli_module", _cli_py_path)
    if _spec and _spec.loader:
        _cli_module = importlib.util.module_from_spec(_spec)
        # Set proper package context for relative imports
        _cli_module.__package__ = "trading_system"
        _cli_module.__name__ = "trading_system.cli_module"
        sys.modules["trading_system.cli_module"] = _cli_module
        _spec.loader.exec_module(_cli_module)
        main = _cli_module.main
    else:
        raise ImportError("Failed to load cli.py")
else:
    # Fallback: try importing from cli package
    from .cli import main

if __name__ == "__main__":
    exit(main())
