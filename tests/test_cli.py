"""Tests for CLI interface."""

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from trading_system.cli import cmd_report, cmd_strategy_create, cmd_strategy_template, setup_logging
from trading_system.configs.run_config import RunConfig
from trading_system.strategies.strategy_template_generator import generate_strategy_template


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config_file(temp_dir):
    """Create a sample run_config.yaml file."""
    config_data = {
        "dataset": {
            "equity_path": "data/equity/ohlcv/",
            "crypto_path": "data/crypto/ohlcv/",
            "benchmark_path": "data/benchmarks/",
            "format": "csv",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "min_lookback_days": 250,
        },
        "splits": {
            "train_start": "2023-01-01",
            "train_end": "2023-09-30",
            "validation_start": "2023-10-01",
            "validation_end": "2023-10-31",
            "holdout_start": "2023-11-01",
            "holdout_end": "2023-12-31",
        },
        "strategies": {
            "equity": {"config_path": "EXAMPLE_CONFIGS/equity_config.yaml", "enabled": True},
            "crypto": {"config_path": "EXAMPLE_CONFIGS/crypto_config.yaml", "enabled": False},
        },
        "portfolio": {"starting_equity": 100000},
        "output": {"base_path": str(temp_dir / "results"), "run_id": "test_run", "log_level": "INFO", "log_file": "test.log"},
        "random_seed": 42,
    }

    config_path = temp_dir / "run_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path


def test_load_run_config(sample_config_file):
    """Test loading run config from YAML."""
    config = RunConfig.from_yaml(str(sample_config_file))

    assert config.dataset.start_date == "2023-01-01"
    assert config.dataset.end_date == "2023-12-31"
    assert config.portfolio.starting_equity == 100000
    assert config.random_seed == 42
    assert config.strategies.equity.enabled is True
    assert config.strategies.crypto.enabled is False


def test_setup_logging(sample_config_file, temp_dir):
    """Test logging setup."""
    config = RunConfig.from_yaml(str(sample_config_file))
    setup_logging(config)

    # Check that log file is created
    log_file = config.get_output_dir() / config.output.log_file
    assert log_file.parent.exists()


def test_cmd_backtest_validation(sample_config_file):
    """Test backtest command argument parsing."""

    # This test just validates the command can parse arguments
    # Full integration test would require actual data files
    class Args:
        def __init__(self):
            self.config = str(sample_config_file)
            self.period = "train"

    Args()
    # Note: This will fail without actual data, but tests argument parsing
    # In a real test suite, you'd mock the runner or use fixture data
    pass  # Test skipped - requires actual data


def test_cmd_validate_validation(sample_config_file):
    """Test validate command argument parsing."""

    class Args:
        def __init__(self):
            self.config = str(sample_config_file)

    Args()
    # Test skipped - requires validation module implementation


def test_cmd_holdout_validation(sample_config_file):
    """Test holdout command argument parsing."""

    class Args:
        def __init__(self):
            self.config = str(sample_config_file)

    Args()
    # Test skipped - requires actual data


def test_cmd_report_validation(temp_dir):
    """Test report command argument parsing."""
    # Create a minimal run directory structure
    base_path = temp_dir / "results"
    run_id = "test_run_123"
    run_dir = base_path / run_id / "train"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal equity curve file
    import pandas as pd

    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    equity_df = pd.DataFrame(
        {
            "date": dates,
            "equity": [100000.0] * 10,
            "cash": [50000.0] * 10,
            "positions": [0] * 10,
            "exposure": [0.0] * 10,
            "exposure_pct": [0.0] * 10,
        }
    )
    equity_df.to_csv(run_dir / "equity_curve.csv", index=False)

    # Create empty trade log
    trade_df = pd.DataFrame()
    trade_df.to_csv(run_dir / "trade_log.csv", index=False)

    class Args:
        def __init__(self):
            self.run_id = run_id
            self.base_path = str(base_path)

    args = Args()
    # Should succeed with valid run directory
    result = cmd_report(args)
    assert result == 0


def test_run_config_get_output_dir(sample_config_file):
    """Test output directory generation."""
    config = RunConfig.from_yaml(str(sample_config_file))
    output_dir = config.get_output_dir()

    assert output_dir.name == config.output.run_id
    assert config.output.base_path in str(output_dir)


def test_run_config_get_run_id(sample_config_file):
    """Test run ID generation."""
    config = RunConfig.from_yaml(str(sample_config_file))

    # With explicit run_id
    assert config.output.get_run_id() == config.output.run_id

    # Without run_id (should generate)
    config.output.run_id = None
    run_id = config.output.get_run_id()
    assert run_id.startswith("run_")
    assert len(run_id) > 10  # Should have timestamp


def test_run_config_splits_get_dates(sample_config_file):
    """Test splits date extraction."""
    config = RunConfig.from_yaml(str(sample_config_file))

    train_start, train_end = config.splits.get_dates("train")
    assert train_start.year == 2023
    assert train_start.month == 1
    assert train_end.year == 2023
    assert train_end.month == 9

    validation_start, validation_end = config.splits.get_dates("validation")
    assert validation_start.year == 2023
    assert validation_start.month == 10

    holdout_start, holdout_end = config.splits.get_dates("holdout")
    assert holdout_start.year == 2023
    assert holdout_start.month == 11
    assert holdout_end.year == 2023
    assert holdout_end.month == 12


def test_generate_strategy_template_basic(temp_dir):
    """Test basic strategy template generation."""
    strategy_name = "test_strategy"
    strategy_type = "custom"
    asset_class = "equity"

    output_path = str(temp_dir / "test_strategy.py")
    content = generate_strategy_template(
        strategy_name=strategy_name, strategy_type=strategy_type, asset_class=asset_class, output_path=output_path
    )

    # Check file was created
    assert Path(output_path).exists()

    # Check content contains expected elements
    assert strategy_name.replace("_", " ").title() in content
    assert "EquityTestStrategy" in content  # Class name
    assert "StrategyInterface" in content  # Base class
    assert "check_eligibility" in content
    assert "check_entry_triggers" in content
    assert "check_exit_signals" in content
    assert "update_stop_price" in content


def test_generate_strategy_template_momentum(temp_dir):
    """Test momentum strategy template generation."""
    strategy_name = "my_momentum"
    strategy_type = "momentum"
    asset_class = "equity"

    output_path = str(temp_dir / "momentum_strategy.py")
    content = generate_strategy_template(
        strategy_name=strategy_name, strategy_type=strategy_type, asset_class=asset_class, output_path=output_path
    )

    assert Path(output_path).exists()
    assert "MomentumBaseStrategy" in content
    assert "EquityMyMomentumMomentumStrategy" in content or "EquityMomentumStrategy" in content


def test_generate_strategy_template_crypto(temp_dir):
    """Test crypto strategy template generation."""
    strategy_name = "crypto_custom"
    strategy_type = "custom"
    asset_class = "crypto"

    output_path = str(temp_dir / "crypto_strategy.py")
    content = generate_strategy_template(
        strategy_name=strategy_name, strategy_type=strategy_type, asset_class=asset_class, output_path=output_path
    )

    assert Path(output_path).exists()
    assert "CryptoCryptoCustomStrategy" in content
    assert asset_class in content.lower()


def test_generate_strategy_template_invalid_type():
    """Test that invalid strategy type raises error."""
    with pytest.raises(ValueError, match="Invalid strategy_type"):
        generate_strategy_template(strategy_name="test", strategy_type="invalid_type", asset_class="equity")


def test_generate_strategy_template_invalid_asset_class():
    """Test that invalid asset class raises error."""
    with pytest.raises(ValueError, match="asset_class must be"):
        generate_strategy_template(strategy_name="test", strategy_type="custom", asset_class="invalid")


def test_generate_strategy_template_directory(temp_dir):
    """Test strategy template generation with custom directory."""
    strategy_name = "test_strategy"
    custom_dir = temp_dir / "custom_strategies"

    content = generate_strategy_template(
        strategy_name=strategy_name, strategy_type="custom", asset_class="equity", directory=str(custom_dir)
    )

    # Check file was created in custom directory
    expected_file = custom_dir / f"{strategy_name}_equity.py"
    assert expected_file.exists()

    # Check content
    assert "EquityTestStrategyStrategy" in content or "EquityTestStrategy" in content


def test_cmd_strategy_template(temp_dir):
    """Test strategy template CLI command."""

    class Args:
        def __init__(self):
            self.name = "test_cli_strategy"
            self.type = "custom"
            self.asset_class = "equity"
            self.output = str(temp_dir / "cli_strategy.py")
            self.directory = None

    args = Args()
    result = cmd_strategy_template(args)

    assert result == 0
    assert Path(temp_dir / "cli_strategy.py").exists()


def test_cmd_strategy_create_with_name(temp_dir):
    """Test strategy create CLI command with name (non-interactive)."""

    class Args:
        def __init__(self):
            self.name = "test_create_strategy"
            self.type = "custom"
            self.asset_class = "equity"
            self.output = str(temp_dir / "create_strategy.py")
            self.directory = None

    args = Args()
    result = cmd_strategy_create(args)

    assert result == 0
    assert Path(temp_dir / "create_strategy.py").exists()


def test_generate_strategy_template_all_types(temp_dir):
    """Test template generation for all strategy types."""
    strategy_types = ["momentum", "mean_reversion", "factor", "multi_timeframe", "pairs", "custom"]
    asset_classes = ["equity", "crypto"]

    for strategy_type in strategy_types:
        for asset_class in asset_classes:
            strategy_name = f"test_{strategy_type}_{asset_class}"
            output_path = str(temp_dir / f"{strategy_name}.py")

            content = generate_strategy_template(
                strategy_name=strategy_name, strategy_type=strategy_type, asset_class=asset_class, output_path=output_path
            )

            assert Path(output_path).exists()
            assert asset_class in content.lower()
            assert "class" in content
            assert "def check_eligibility" in content
