"""Tests for configuration validation of example configs."""

from pathlib import Path

import pytest

from trading_system.configs.validation import validate_config_file

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLE_CONFIGS_DIR = PROJECT_ROOT / "EXAMPLE_CONFIGS"


class TestExampleConfigsValidation:
    """Test that all example configuration files validate correctly."""

    @pytest.fixture
    def example_configs(self):
        """Get all example config files."""
        config_files = list(EXAMPLE_CONFIGS_DIR.glob("*.yaml"))
        # Filter out README if it exists as YAML
        return [f for f in config_files if f.name != "README.yaml"]

    def test_all_example_configs_exist(self, example_configs):
        """Verify that we found example config files."""
        assert len(example_configs) > 0, "No example config files found"

    @pytest.mark.parametrize(
        "config_path",
        [
            "equity_config.yaml",
            "crypto_config.yaml",
            "mean_reversion_config.yaml",
            "pairs_config.yaml",
            "multi_timeframe_config.yaml",
            "factor_config.yaml",
            "run_config.yaml",
        ],
    )
    def test_example_config_validates(self, config_path):
        """Test that each example config file validates correctly."""
        full_path = EXAMPLE_CONFIGS_DIR / config_path

        if not full_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        # Determine config type based on filename
        if "run" in config_path.lower():
            config_type = "run"
        else:
            config_type = "strategy"

        # Validate the config
        is_valid, error_message, config = validate_config_file(str(full_path), config_type=config_type)

        assert is_valid, f"Config file {config_path} failed validation:\n{error_message}"
        assert config is not None, f"Config file {config_path} returned None"

    def test_all_example_configs_validate(self, example_configs):
        """Test that all example configs in the directory validate."""
        failed_configs = []

        for config_file in example_configs:
            # Determine config type
            if "run" in config_file.name.lower():
                config_type = "run"
            else:
                config_type = "strategy"

            # Validate
            is_valid, error_message, config = validate_config_file(str(config_file), config_type=config_type)

            if not is_valid:
                failed_configs.append((config_file.name, error_message))

        if failed_configs:
            error_summary = "\n\n".join(f"{name}:\n{error}" for name, error in failed_configs)
            pytest.fail(f"The following config files failed validation:\n\n{error_summary}")

    def test_run_config_references_valid_strategy_configs(self):
        """Test that run_config.yaml references valid strategy configs if they exist."""
        run_config_path = EXAMPLE_CONFIGS_DIR / "run_config.yaml"

        if not run_config_path.exists():
            pytest.skip("run_config.yaml not found")

        # Validate run config
        is_valid, error_message, run_config = validate_config_file(str(run_config_path), config_type="run")

        assert is_valid, f"run_config.yaml failed validation: {error_message}"
        assert run_config is not None

        # Check referenced strategy configs if they exist
        from trading_system.configs.run_config import RunConfig

        if isinstance(run_config, RunConfig):
            # Check equity strategy config if enabled
            if run_config.strategies.equity and run_config.strategies.equity.enabled:
                equity_path = Path(run_config.strategies.equity.config_path)
                # Resolve relative paths
                if not equity_path.is_absolute():
                    equity_path = PROJECT_ROOT / equity_path

                if equity_path.exists():
                    equity_valid, equity_error, _ = validate_config_file(str(equity_path), config_type="strategy")
                    assert equity_valid, (
                        "Equity strategy config referenced by run_config.yaml " f"failed validation: {equity_error}"
                    )

            # Check crypto strategy config if enabled
            if run_config.strategies.crypto and run_config.strategies.crypto.enabled:
                crypto_path = Path(run_config.strategies.crypto.config_path)
                # Resolve relative paths
                if not crypto_path.is_absolute():
                    crypto_path = PROJECT_ROOT / crypto_path

                if crypto_path.exists():
                    crypto_valid, crypto_error, _ = validate_config_file(str(crypto_path), config_type="strategy")
                    assert crypto_valid, (
                        "Crypto strategy config referenced by run_config.yaml " f"failed validation: {crypto_error}"
                    )
