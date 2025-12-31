# Code Style & Standards

This document outlines the coding standards and conventions for the Trading System.

## Python Style Guide

The project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications.

### General Guidelines

- **Line length**: 100 characters (soft limit)
- **Indentation**: 4 spaces (no tabs)
- **String quotes**: Prefer double quotes for strings, single quotes for characters
- **Imports**: Use absolute imports, group by standard library, third-party, local

### Type Hints

All functions and methods should have type hints:

```python
from typing import List, Dict, Optional
import pandas as pd

def calculate_ma(data: pd.DataFrame, window: int) -> pd.Series:
    """Calculate moving average."""
    return data['close'].rolling(window=window).mean()
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_ma(data: pd.DataFrame, window: int) -> pd.Series:
    """Calculate moving average.

    Args:
        data: DataFrame with 'close' column
        window: Rolling window size

    Returns:
        Series with moving average values

    Raises:
        ValueError: If window is <= 0
    """
    if window <= 0:
        raise ValueError("Window must be positive")
    return data['close'].rolling(window=window).mean()
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `BacktestEngine`)
- **Functions/Methods**: `snake_case` (e.g., `calculate_ma`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_RISK_PCT`)
- **Private methods**: Prefix with `_` (e.g., `_validate_data`)

### Error Handling

Use specific exceptions and provide clear error messages:

```python
if not data.empty:
    raise ValueError("Data cannot be empty")

if window <= 0:
    raise ValueError(f"Window must be positive, got {window}")
```

## Code Organization

### Module Structure

```python
"""Module docstring."""

# Standard library imports
import os
from typing import List

# Third-party imports
import pandas as pd
import numpy as np

# Local imports
from trading_system.models import Bar

# Constants
DEFAULT_WINDOW = 20

# Classes and functions
class MyClass:
    """Class docstring."""
    pass

def my_function():
    """Function docstring."""
    pass
```

### File Organization

- One class per file (when possible)
- Related functions grouped together
- Constants at module level
- Private helpers at end of file

## Testing Standards

### Test Naming

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Test Structure

```python
def test_calculate_ma_basic():
    """Test basic moving average calculation."""
    data = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
    result = calculate_ma(data, window=2)
    assert len(result) == 5
    assert pd.isna(result.iloc[0])
    assert result.iloc[2] == 2.5
```

## Configuration

### Pydantic Models

Use Pydantic for configuration validation:

```python
from pydantic import BaseModel, Field

class StrategyConfig(BaseModel):
    """Strategy configuration."""
    risk_pct: float = Field(0.75, ge=0.0, le=1.0, description="Risk per trade")
    window: int = Field(20, gt=0, description="Indicator window")
```

## Documentation

- All public functions/classes must have docstrings
- Complex logic should have inline comments
- Update documentation when changing APIs

## Tools

### Formatting

```bash
black trading_system/ tests/
```

### Linting

```bash
flake8 trading_system/ tests/
```

### Type Checking

```bash
mypy trading_system/
```

## Best Practices

1. **Keep functions focused**: One responsibility per function
2. **Avoid deep nesting**: Use early returns, extract functions
3. **Use meaningful names**: Code should be self-documenting
4. **Handle errors explicitly**: Don't ignore exceptions
5. **Write tests**: Test new functionality
6. **Update docs**: Keep documentation in sync with code

---

For more details, see [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
