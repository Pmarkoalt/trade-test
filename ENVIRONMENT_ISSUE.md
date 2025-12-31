# Environment Issue: NumPy Segmentation Fault on macOS

## Problem

The test suite may fail with a segmentation fault during numpy initialization on macOS:

```
Fatal Python error: Segmentation fault
File ".../numpy/__init__.py", line 386 in _mac_os_check
```

This occurs when numpy performs its macOS compatibility check during import. This is a known issue with certain numpy versions in conda environments on macOS, particularly with:
- Python 3.9 on macOS
- Older NumPy versions (< 1.24.0)
- Conda environments with mixed pip/conda installations

## Impact

- **All pytest runs fail** during test collection (before any tests execute)
- Tests cannot be run until this is resolved
- This is an environment/installation issue, not a code issue

## Universal Prevention Strategies

### ✅ Recommended: Use Docker (Best Solution)

**Docker completely avoids this issue** by providing a consistent Linux environment:

```bash
# Build and run tests in Docker
docker-compose build
docker-compose run --rm trading-system pytest tests/ -v
```

This is the **most reliable solution** and works identically across all machines.

### ✅ Recommended: Use Python 3.11+ with pip

Python 3.11+ has better compatibility with NumPy on macOS:

```bash
# Create environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### ✅ Use Automated Setup Script

We provide `scripts/setup_environment.sh` that automatically:
- Detects macOS
- Checks NumPy compatibility
- Fixes common issues
- Verifies the environment

```bash
./scripts/setup_environment.sh
```

### ✅ Pin NumPy Version

The project now recommends NumPy >= 1.24.0 which has better macOS compatibility. This is specified in `pyproject.toml`.

## Manual Solutions (If Prevention Didn't Work)

### Option 1: Reinstall NumPy (Recommended for Conda)

```bash
# Update conda
conda update conda

# Reinstall numpy with conda-forge (more reliable on macOS)
conda install -c conda-forge numpy>=1.24.0 pandas

# Or reinstall via pip
pip uninstall numpy
pip install "numpy>=1.24.0"
```

### Option 2: Use Python 3.11 with Fresh Environment

```bash
# Create new environment with Python 3.11
conda create -n trade_test_py311 python=3.11
conda activate trade_test_py311
pip install -e ".[dev]"
```

### Option 3: Rebuild Environment Completely

```bash
# Remove old environment
conda deactivate
conda env remove -n trade_test_clean  # if exists

# Create fresh environment
conda create -n trade_test_clean python=3.11
conda activate trade_test_clean
pip install --upgrade pip
pip install -e ".[dev]"
```

### Option 4: Use System Python (if available)

If you have a system Python installation that works:

```bash
/usr/bin/python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Verification

After applying a fix, verify numpy imports correctly:

```bash
python -c "import numpy; import pandas; print('NumPy version:', numpy.__version__); print('Imports OK')"
```

If this works, then pytest should work:

```bash
pytest tests/ -v --tb=short
```

## Quick Diagnosis

Run the environment check script:

```bash
./scripts/setup_environment.sh --check
```

Or manually check:

```bash
python -c "import numpy; print('NumPy OK')" 2>&1
```

If this fails with a segfault, you have the issue.

## Root Cause Analysis

**Is this a machine-by-machine problem?**

**Partially yes, but preventable:**

1. **Machine-specific factors:**
   - macOS version differences
   - Conda vs pip vs system Python
   - Python version (3.9 more prone than 3.11+)
   - Existing system libraries that conflict

2. **Universal factors we can control:**
   - ✅ NumPy version pinning (>= 1.24.0)
   - ✅ Python version recommendation (3.11+)
   - ✅ Docker support (avoids issue entirely)
   - ✅ Automated setup scripts
   - ✅ Pre-flight checks in test scripts

**Conclusion:** While the issue is more common on certain macOS setups, we can prevent it universally by:
- Using Docker (recommended)
- Using Python 3.11+ with pip
- Pinning NumPy to known-good versions
- Providing automated setup scripts

## Next Steps

Once the environment issue is resolved:

1. Run full test suite: `pytest tests/ -v`
2. Check for any failing tests
3. Debug and fix any code-level issues found
4. Generate coverage report: `pytest --cov=trading_system --cov-report=html`

## Additional Resources

- NumPy GitHub Issues: https://github.com/numpy/numpy/issues
- Conda-forge: https://conda-forge.org/
- Known macOS/NumPy issues: Search for "numpy _mac_os_check segfault"
- Docker documentation: See README.md Docker section

