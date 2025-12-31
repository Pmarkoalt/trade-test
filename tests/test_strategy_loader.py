"""Unit tests for strategy_loader.py."""

import tempfile
from pathlib import Path

import pytest
import yaml

from trading_system.configs.strategy_config import StrategyConfig
from trading_system.strategies.strategy_loader import (
    load_strategies_from_configs,
    load_strategies_from_run_config,
    load_strategy_from_config,
)


class TestLoadStrategyFromConfig:
    """Tests for load_strategy_from_config function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_load_strategy_from_config_valid(self):
        """Test loading strategy from valid config file."""
        # Create a valid strategy config file
        config_path = self.temp_dir / "test_strategy.yaml"
        config_data = {
            "name": "test_momentum",
            "asset_class": "equity",
            "universe": "NASDAQ-100",
            "benchmark": "SPY",
            "parameters": {
                "stop_atr_mult": 2.5,
                "fast_clearance": 0.005,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        strategy = load_strategy_from_config(str(config_path))
        assert strategy is not None
        assert strategy.config.name == "test_momentum"
        assert strategy.config.asset_class == "equity"

    def test_load_strategy_from_config_file_not_found(self):
        """Test loading strategy from non-existent file."""
        config_path = self.temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_strategy_from_config(str(config_path))

    def test_load_strategy_from_config_invalid_config(self):
        """Test loading strategy from invalid config file."""
        config_path = self.temp_dir / "invalid.yaml"

        # Create invalid YAML
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        # Should raise an error when trying to parse
        with pytest.raises((ValueError, yaml.YAMLError)):
            load_strategy_from_config(str(config_path))

    def test_load_strategy_from_config_factor_strategy(self):
        """Test loading factor strategy from config."""
        config_path = self.temp_dir / "factor_strategy.yaml"
        config_data = {
            "name": "equity_factor",
            "asset_class": "equity",
            "universe": "NASDAQ-100",
            "benchmark": "SPY",
            "parameters": {
                "factors": {"momentum": 0.4, "value": 0.3, "quality": 0.3},
                "rebalance_frequency": "monthly",
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        strategy = load_strategy_from_config(str(config_path))
        assert strategy is not None
        assert "factor" in strategy.config.name.lower() or hasattr(strategy, "factors")

    def test_load_strategy_from_config_multi_timeframe_strategy(self):
        """Test loading multi-timeframe strategy from config."""
        config_path = self.temp_dir / "mtf_strategy.yaml"
        config_data = {
            "name": "equity_mtf",
            "asset_class": "equity",
            "universe": "NASDAQ-100",
            "benchmark": "SPY",
            "parameters": {
                "higher_tf_ma": 50,
                "weekly_lookback": 55,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        strategy = load_strategy_from_config(str(config_path))
        assert strategy is not None


class TestLoadStrategiesFromConfigs:
    """Tests for load_strategies_from_configs function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_load_strategies_from_configs_multiple(self):
        """Test loading multiple strategies from config files."""
        # Create multiple config files
        config_paths = []
        for i in range(3):
            config_path = self.temp_dir / f"strategy_{i}.yaml"
            config_data = {
                "name": f"test_strategy_{i}",
                "asset_class": "equity",
                "universe": "NASDAQ-100",
                "benchmark": "SPY",
            }
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)
            config_paths.append(str(config_path))

        strategies = load_strategies_from_configs(config_paths)
        assert len(strategies) == 3
        assert all(s is not None for s in strategies)

    def test_load_strategies_from_configs_one_fails(self):
        """Test loading strategies when one config fails."""
        # Create one valid and one invalid config
        valid_path = self.temp_dir / "valid.yaml"
        config_data = {
            "name": "valid_strategy",
            "asset_class": "equity",
            "universe": "NASDAQ-100",
            "benchmark": "SPY",
        }
        with open(valid_path, "w") as f:
            yaml.dump(config_data, f)

        invalid_path = self.temp_dir / "invalid.yaml"
        invalid_path.touch()  # Empty file

        config_paths = [str(valid_path), str(invalid_path)]

        # Should raise an error when one fails
        with pytest.raises(Exception):
            load_strategies_from_configs(config_paths)

    def test_load_strategies_from_configs_empty_list(self):
        """Test loading strategies from empty list."""
        strategies = load_strategies_from_configs([])
        assert len(strategies) == 0


class TestLoadStrategiesFromRunConfig:
    """Tests for load_strategies_from_run_config function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_load_strategies_from_run_config_equity_only(self):
        """Test loading strategies with equity config only."""
        equity_config_path = self.temp_dir / "equity.yaml"
        config_data = {
            "name": "equity_momentum",
            "asset_class": "equity",
            "universe": "NASDAQ-100",
            "benchmark": "SPY",
        }
        with open(equity_config_path, "w") as f:
            yaml.dump(config_data, f)

        strategies = load_strategies_from_run_config(equity_config_path=str(equity_config_path))
        assert len(strategies) == 1
        assert strategies[0].config.asset_class == "equity"

    def test_load_strategies_from_run_config_crypto_only(self):
        """Test loading strategies with crypto config only."""
        crypto_config_path = self.temp_dir / "crypto.yaml"
        config_data = {
            "name": "crypto_momentum",
            "asset_class": "crypto",
            "universe": "fixed",
            "benchmark": "BTC",
            "exit": {"mode": "staged", "tightened_stop_atr_mult": 1.5},  # Required for crypto
        }
        with open(crypto_config_path, "w") as f:
            yaml.dump(config_data, f)

        strategies = load_strategies_from_run_config(crypto_config_path=str(crypto_config_path))
        assert len(strategies) == 1
        assert strategies[0].config.asset_class == "crypto"

    def test_load_strategies_from_run_config_both(self):
        """Test loading strategies with both equity and crypto configs."""
        equity_config_path = self.temp_dir / "equity.yaml"
        config_data = {
            "name": "equity_momentum",
            "asset_class": "equity",
            "universe": "NASDAQ-100",
            "benchmark": "SPY",
        }
        with open(equity_config_path, "w") as f:
            yaml.dump(config_data, f)

        crypto_config_path = self.temp_dir / "crypto.yaml"
        config_data = {
            "name": "crypto_momentum",
            "asset_class": "crypto",
            "universe": "fixed",
            "benchmark": "BTC",
            "exit": {"mode": "staged", "tightened_stop_atr_mult": 1.5},  # Required for crypto
        }
        with open(crypto_config_path, "w") as f:
            yaml.dump(config_data, f)

        strategies = load_strategies_from_run_config(
            equity_config_path=str(equity_config_path), crypto_config_path=str(crypto_config_path)
        )
        assert len(strategies) == 2
        asset_classes = {s.config.asset_class for s in strategies}
        assert "equity" in asset_classes
        assert "crypto" in asset_classes

    def test_load_strategies_from_run_config_no_configs(self):
        """Test loading strategies with no configs provided."""
        with pytest.raises(ValueError, match="No strategy configs provided"):
            load_strategies_from_run_config()
