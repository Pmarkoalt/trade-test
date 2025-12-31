#!/usr/bin/env python3
"""Verify environment variable handling for production readiness.

This script checks:
1. No hardcoded paths or secrets in production code
2. API keys are loaded from environment variables
3. Missing environment variables fail gracefully
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Colors for output
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
NC = "\033[0m"  # No Color


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓{NC} {msg}")


def print_error(msg: str) -> None:
    """Print error message."""
    print(f"{RED}✗{NC} {msg}")


def print_warning(msg: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}⚠{NC} {msg}")


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"{BLUE}ℹ{NC} {msg}")


def check_hardcoded_secrets() -> Tuple[bool, List[str]]:
    """Check for hardcoded secrets in production code.

    Returns:
        Tuple of (passed, issues)
    """
    print_section("Checking for hardcoded secrets...")
    issues = []
    passed = True

    # Patterns to check (excluding test files and comments)
    patterns = [
        (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
        (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
        (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token"),
    ]

    # Directories to check
    code_dirs = ["trading_system"]
    exclude_dirs = ["__pycache__", ".pytest_cache", "htmlcov"]
    exclude_files = ["verify_env_vars.py"]  # Exclude this script

    for code_dir in code_dirs:
        if not Path(code_dir).exists():
            continue

        for py_file in Path(code_dir).rglob("*.py"):
            # Skip excluded directories
            if any(excluded in str(py_file) for excluded in exclude_dirs):
                continue

            # Skip excluded files
            if py_file.name in exclude_files:
                continue

            try:
                content = py_file.read_text()
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    # Skip comments
                    if line.strip().startswith("#"):
                        continue

                    # Skip docstrings
                    if '"""' in line or "'''" in line:
                        continue

                    for pattern, issue_type in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Check if it's in a test context (test files are OK)
                            if "test" in str(py_file).lower():
                                continue

                            # Check if it's a default value in function signature (OK)
                            if "def " in line or "=" in line and "Optional" in line:
                                # Might be a default parameter, check context
                                if "Optional[" in line or "= None" in line:
                                    continue

                            issues.append(f"{py_file}:{line_num} - {issue_type}: {line.strip()}")
                            passed = False
            except Exception as e:
                print_warning(f"Could not read {py_file}: {e}")

    if passed:
        print_success("No hardcoded secrets found in production code")
    else:
        print_error(f"Found {len(issues)} potential hardcoded secrets:")
        for issue in issues:
            print(f"  {issue}")

    return passed, issues


def check_hardcoded_paths() -> Tuple[bool, List[str]]:
    """Check for hardcoded absolute paths.

    Returns:
        Tuple of (passed, issues)
    """
    print_section("Checking for hardcoded paths...")
    issues = []
    passed = True

    # Patterns for hardcoded paths
    patterns = [
        (r"/Users/[^/\s]+", "Hardcoded macOS user path"),
        (r"/home/[^/\s]+", "Hardcoded Linux user path"),
        (r"C:\\Users\\[^\\\s]+", "Hardcoded Windows user path"),
    ]

    code_dirs = ["trading_system"]
    exclude_dirs = ["__pycache__", ".pytest_cache"]

    for code_dir in code_dirs:
        if not Path(code_dir).exists():
            continue

        for py_file in Path(code_dir).rglob("*.py"):
            if any(excluded in str(py_file) for excluded in exclude_dirs):
                continue

            try:
                content = py_file.read_text()
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    # Skip comments
                    if line.strip().startswith("#"):
                        continue

                    for pattern, issue_type in patterns:
                        if re.search(pattern, line):
                            issues.append(f"{py_file}:{line_num} - {issue_type}: {line.strip()}")
                            passed = False
            except Exception as e:
                print_warning(f"Could not read {py_file}: {e}")

    if passed:
        print_success("No hardcoded paths found")
    else:
        print_error(f"Found {len(issues)} hardcoded paths:")
        for issue in issues:
            print(f"  {issue}")

    return passed, issues


def check_env_var_usage() -> Tuple[bool, Dict[str, List[str]]]:
    """Check that API keys are loaded from environment variables.

    Returns:
        Tuple of (passed, env_var_usage)
    """
    print_section("Checking environment variable usage...")
    env_vars = {
        "MASSIVE_API_KEY": [],
        "ALPHA_VANTAGE_API_KEY": [],
        "NEWSAPI_KEY": [],
        "SENDGRID_API_KEY": [],
        "SMTP_PASSWORD": [],
        "EMAIL_RECIPIENTS": [],
        "DASHBOARD_PASSWORD_HASH": [],
    }

    code_dirs = ["trading_system"]
    exclude_dirs = ["__pycache__", ".pytest_cache"]

    for code_dir in code_dirs:
        if not Path(code_dir).exists():
            continue

        for py_file in Path(code_dir).rglob("*.py"):
            if any(excluded in str(py_file) for excluded in exclude_dirs):
                continue

            try:
                content = py_file.read_text()
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    for env_var in env_vars.keys():
                        # Check for os.getenv or os.environ usage
                        if f'os.getenv("{env_var}"' in line or "os.getenv('{env_var}'" in line:
                            env_vars[env_var].append(f"{py_file}:{line_num}")
                        elif f'os.environ.get("{env_var}"' in line or f"os.environ.get('{env_var}'" in line:
                            env_vars[env_var].append(f"{py_file}:{line_num}")
                        elif f'os.environ["{env_var}"]' in line or f"os.environ['{env_var}']" in line:
                            env_vars[env_var].append(f"{py_file}:{line_num}")
            except Exception as e:
                print_warning(f"Could not read {py_file}: {e}")

    # Report findings
    passed = True
    for env_var, locations in env_vars.items():
        if locations:
            print_success(f"{env_var} is loaded from environment variables ({len(locations)} locations)")
            if len(locations) <= 3:  # Show locations if few
                for loc in locations:
                    print_info(f"  {loc}")
        else:
            print_warning(f"{env_var} not found in code (may not be used)")

    return passed, env_vars


def test_missing_env_vars() -> Tuple[bool, List[str]]:
    """Test that missing environment variables fail gracefully.

    Returns:
        Tuple of (passed, issues)
    """
    print_section("Testing missing environment variables...")
    issues = []
    passed = True

    # Save current environment
    original_env = {}
    test_vars = [
        "MASSIVE_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "NEWSAPI_KEY",
        "SENDGRID_API_KEY",
    ]

    for var in test_vars:
        if var in os.environ:
            original_env[var] = os.environ[var]
            del os.environ[var]

    try:
        # Test imports that might use env vars
        try:
            from trading_system.scheduler.jobs.daily_signals_job import load_config

            # Test loading config without env vars
            config = load_config()
            if config:
                # Check if None values are handled
                data_config = config.get("data_pipeline")
                if data_config:
                    if data_config.massive_api_key is None:
                        print_success("MASSIVE_API_KEY: None handled gracefully")
                    else:
                        issues.append("MASSIVE_API_KEY: Should be None when not set")
                        passed = False

                    if data_config.alpha_vantage_api_key is None:
                        print_success("ALPHA_VANTAGE_API_KEY: None handled gracefully")
                    else:
                        issues.append("ALPHA_VANTAGE_API_KEY: Should be None when not set")
                        passed = False

                research_config = config.get("research")
                if research_config:
                    if research_config.newsapi_key is None:
                        print_success("NEWSAPI_KEY: None handled gracefully")
                    else:
                        issues.append("NEWSAPI_KEY: Should be None when not set")
                        passed = False

        except Exception as e:
            issues.append(f"Error testing config loading: {e}")
            passed = False

        # Test classes that require API keys
        try:
            from trading_system.data_pipeline.sources.news.newsapi_client import NewsAPIClient

            try:
                client = NewsAPIClient(api_key=None)
                issues.append("NewsAPIClient should raise ValueError for None api_key")
                passed = False
            except ValueError:
                print_success("NewsAPIClient correctly raises ValueError for None api_key")
            except Exception as e:
                issues.append(f"NewsAPIClient raised unexpected error: {e}")
                passed = False
        except ImportError:
            print_warning("Could not import NewsAPIClient (may not be installed)")

        try:
            from trading_system.adapters.alpaca_adapter import AlpacaAdapter, AdapterConfig

            try:
                config = AdapterConfig(api_key=None, api_secret=None)
                adapter = AlpacaAdapter(config)
                issues.append("AlpacaAdapter should raise ValueError for None credentials")
                passed = False
            except ValueError:
                print_success("AlpacaAdapter correctly raises ValueError for None credentials")
            except Exception as e:
                issues.append(f"AlpacaAdapter raised unexpected error: {e}")
                passed = False
        except ImportError:
            print_warning("Could not import AlpacaAdapter (may not be installed)")

        try:
            from trading_system.data_pipeline.live_data_fetcher import LiveDataFetcher, DataPipelineConfig

            config = DataPipelineConfig(massive_api_key=None)
            fetcher = LiveDataFetcher(config)

            # Test that it fails gracefully when trying to use equity without API key
            try:
                fetcher._get_source("equity")
                issues.append("LiveDataFetcher should raise error for equity without API key")
                passed = False
            except Exception as e:
                if "Polygon API key required" in str(e):
                    print_success("LiveDataFetcher correctly raises error for missing API key")
                else:
                    issues.append(f"LiveDataFetcher raised unexpected error: {e}")
                    passed = False
        except ImportError:
            print_warning("Could not import LiveDataFetcher (may not be installed)")

    finally:
        # Restore original environment
        for var, value in original_env.items():
            os.environ[var] = value

    if passed and not issues:
        print_success("Missing environment variables are handled gracefully")
    else:
        print_error(f"Found {len(issues)} issues with missing environment variables:")
        for issue in issues:
            print(f"  {issue}")

    return passed, issues


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{BLUE}{'=' * 60}{NC}")
    print(f"{BLUE}{title}{NC}")
    print(f"{BLUE}{'=' * 60}{NC}")


def main() -> int:
    """Run all verification checks."""
    print_section("Environment Variable Verification")
    print("Verifying environment variable handling for production readiness...\n")

    all_passed = True
    results = {}

    # Check 1: Hardcoded secrets
    passed, issues = check_hardcoded_secrets()
    results["hardcoded_secrets"] = (passed, issues)
    if not passed:
        all_passed = False

    # Check 2: Hardcoded paths
    passed, issues = check_hardcoded_paths()
    results["hardcoded_paths"] = (passed, issues)
    if not passed:
        all_passed = False

    # Check 3: Environment variable usage
    passed, env_usage = check_env_var_usage()
    results["env_var_usage"] = (passed, env_usage)
    if not passed:
        all_passed = False

    # Check 4: Missing environment variables
    passed, issues = test_missing_env_vars()
    results["missing_env_vars"] = (passed, issues)
    if not passed:
        all_passed = False

    # Summary
    print_section("Summary")
    if all_passed:
        print_success("All environment variable checks passed!")
        return 0
    else:
        print_error("Some environment variable checks failed. Review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
