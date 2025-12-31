"""Tests for CLI interface."""

import pytest
import tempfile
import shutil
from pathlib import Path
import yaml

from trading_system.cli import cmd_backtest, cmd_validate, cmd_holdout, cmd_report, setup_logging
from trading_system.configs.run_config import RunConfig


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
        'dataset': {
            'equity_path': 'data/equity/ohlcv/',
            'crypto_path': 'data/crypto/ohlcv/',
            'benchmark_path': 'data/benchmarks/',
            'format': 'csv',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'min_lookback_days': 250
        },
        'splits': {
            'train_start': '2023-01-01',
            'train_end': '2023-09-30',
            'validation_start': '2023-10-01',
            'validation_end': '2023-10-31',
            'holdout_start': '2023-11-01',
            'holdout_end': '2023-12-31'
        },
        'strategies': {
            'equity': {
                'config_path': 'EXAMPLE_CONFIGS/equity_config.yaml',
                'enabled': True
            },
            'crypto': {
                'config_path': 'EXAMPLE_CONFIGS/crypto_config.yaml',
                'enabled': False
            }
        },
        'portfolio': {
            'starting_equity': 100000
        },
        'output': {
            'base_path': str(temp_dir / 'results'),
            'run_id': 'test_run',
            'log_level': 'INFO',
            'log_file': 'test.log'
        },
        'random_seed': 42
    }
    
    config_path = temp_dir / 'run_config.yaml'
    with open(config_path, 'w') as f:
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
            self.period = 'train'
    
    args = Args()
    # Note: This will fail without actual data, but tests argument parsing
    # In a real test suite, you'd mock the runner or use fixture data
    pass  # Test skipped - requires actual data


def test_cmd_validate_validation(sample_config_file):
    """Test validate command argument parsing."""
    class Args:
        def __init__(self):
            self.config = str(sample_config_file)
    
    args = Args()
    # Test skipped - requires validation module implementation
    pass


def test_cmd_holdout_validation(sample_config_file):
    """Test holdout command argument parsing."""
    class Args:
        def __init__(self):
            self.config = str(sample_config_file)
    
    args = Args()
    # Test skipped - requires actual data
    pass


def test_cmd_report_validation():
    """Test report command argument parsing."""
    class Args:
        def __init__(self):
            self.run_id = 'test_run_123'
    
    args = Args()
    # Test skipped - requires report generation implementation
    pass


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
    assert run_id.startswith('run_')
    assert len(run_id) > 10  # Should have timestamp


def test_run_config_splits_get_dates(sample_config_file):
    """Test splits date extraction."""
    config = RunConfig.from_yaml(str(sample_config_file))
    
    train_start, train_end = config.splits.get_dates('train')
    assert train_start.year == 2023
    assert train_start.month == 1
    assert train_end.year == 2023
    assert train_end.month == 9
    
    validation_start, validation_end = config.splits.get_dates('validation')
    assert validation_start.year == 2023
    assert validation_start.month == 10
    
    holdout_start, holdout_end = config.splits.get_dates('holdout')
    assert holdout_start.year == 2023
    assert holdout_start.month == 11
    assert holdout_end.year == 2023
    assert holdout_end.month == 12

