"""Tests for scheduler jobs."""

import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest

from trading_system.scheduler.config import SchedulerConfig
from trading_system.scheduler.cron_runner import CronRunner
from trading_system.scheduler.jobs.daily_signals_job import (
    daily_signals_job,
    get_market_summary,
    load_config,
    send_error_alert,
)
from trading_system.scheduler.jobs.ml_retrain_job import MLRetrainJob


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("POLYGON_API_KEY", "test_polygon_key")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_av_key")
    monkeypatch.setenv("DATA_CACHE_PATH", "data/cache")
    monkeypatch.setenv("CACHE_TTL_HOURS", "24")
    monkeypatch.setenv("MAX_RECOMMENDATIONS", "5")
    monkeypatch.setenv("MIN_CONVICTION", "MEDIUM")
    monkeypatch.setenv("TECHNICAL_WEIGHT", "0.6")
    monkeypatch.setenv("NEWS_WEIGHT", "0.4")
    monkeypatch.setenv("NEWS_ENABLED", "true")
    monkeypatch.setenv("NEWS_LOOKBACK_HOURS", "48")
    monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "test_user")
    monkeypatch.setenv("SMTP_PASSWORD", "test_password")
    monkeypatch.setenv("FROM_EMAIL", "test@example.com")
    monkeypatch.setenv("FROM_NAME", "Test Sender")
    monkeypatch.setenv("EMAIL_RECIPIENTS", "recipient1@example.com,recipient2@example.com")


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
    return {
        "AAPL": pd.DataFrame(
            {
                "open": [150.0] * len(dates),
                "high": [155.0] * len(dates),
                "low": [148.0] * len(dates),
                "close": [152.0] * len(dates),
                "volume": [1000000] * len(dates),
            },
            index=dates,
        ),
        "MSFT": pd.DataFrame(
            {
                "open": [300.0] * len(dates),
                "high": [305.0] * len(dates),
                "low": [298.0] * len(dates),
                "close": [302.0] * len(dates),
                "volume": [2000000] * len(dates),
            },
            index=dates,
        ),
    }


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_with_env_vars(self, mock_env_vars):
        """Test loading config from environment variables."""
        config = load_config()

        assert "data_pipeline" in config
        assert "signals" in config
        assert "email" in config
        assert "research" in config
        assert "universe" in config

        # Check data pipeline config
        assert config["data_pipeline"].polygon_api_key == "test_polygon_key"
        assert config["data_pipeline"].alpha_vantage_api_key == "test_av_key"

        # Check signals config
        assert config["signals"].max_recommendations == 5
        assert config["signals"].min_conviction == "MEDIUM"

        # Check email config
        assert config["email"].smtp_host == "smtp.test.com"
        assert config["email"].from_email == "test@example.com"

    def test_load_config_with_config_path(self, mock_env_vars, tmp_path):
        """Test loading config with config file path."""
        # Create a dummy config file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("test: config")

        # Mock RunConfig.from_yaml to avoid actual file loading
        with patch("trading_system.scheduler.jobs.daily_signals_job.RunConfig") as mock_run_config:
            mock_config = MagicMock()
            mock_config.strategies.equity = None
            mock_config.strategies.crypto = None
            mock_run_config.from_yaml.return_value = mock_config

            config = load_config(str(config_file))
            assert config is not None

    def test_load_config_default_universes(self, mock_env_vars):
        """Test that default universes are set when config file doesn't exist."""
        config = load_config(config_path="nonexistent.yaml")

        assert "universe" in config
        assert "equity" in config["universe"]
        assert "crypto" in config["universe"]


class TestGetMarketSummary:
    """Tests for get_market_summary function."""

    def test_get_market_summary_empty(self):
        """Test market summary with empty data."""
        summary = get_market_summary({})

        assert summary["total_symbols"] == 0
        assert summary["symbols"] == []
        assert summary["date_range"] is None
        assert summary["avg_volume"] == 0.0

    def test_get_market_summary_with_data(self, sample_ohlcv_data):
        """Test market summary with actual data."""
        summary = get_market_summary(sample_ohlcv_data)

        assert summary["total_symbols"] == 2
        assert "AAPL" in summary["symbols"]
        assert "MSFT" in summary["symbols"]
        assert summary["date_range"] is not None
        assert summary["avg_volume"] > 0

    def test_get_market_summary_date_range(self, sample_ohlcv_data):
        """Test that date range is correctly extracted."""
        summary = get_market_summary(sample_ohlcv_data)

        assert summary["date_range"] is not None
        assert "start" in summary["date_range"]
        assert "end" in summary["date_range"]


class TestSendErrorAlert:
    """Tests for send_error_alert function."""

    @pytest.mark.asyncio
    async def test_send_error_alert_no_recipients(self, monkeypatch):
        """Test error alert when no recipients configured."""
        monkeypatch.setenv("EMAIL_RECIPIENTS", "")
        error = ValueError("Test error")

        # Should not raise, just log warning
        await send_error_alert(error)

    @pytest.mark.asyncio
    async def test_send_error_alert_with_recipients(self, mock_env_vars):
        """Test error alert with recipients configured."""
        error = ValueError("Test error")

        with patch("trading_system.scheduler.jobs.daily_signals_job.EmailService") as mock_email_service:
            mock_service = MagicMock()
            mock_service._send_email = AsyncMock()
            mock_email_service.return_value = mock_service

            await send_error_alert(error)

            # Verify email service was called
            mock_service._send_email.assert_called_once()


class TestDailySignalsJob:
    """Tests for daily_signals_job function."""

    @pytest.mark.asyncio
    async def test_daily_signals_job_no_strategies(self, mock_env_vars):
        """Test daily signals job when no strategies are loaded."""
        with patch("trading_system.scheduler.jobs.daily_signals_job.load_strategies_from_run_config") as mock_load:
            mock_load.return_value = []

            # Should return early without error
            await daily_signals_job("equity")

    @pytest.mark.asyncio
    async def test_daily_signals_job_no_symbols(self, mock_env_vars):
        """Test daily signals job when no symbols are found."""
        with (
            patch("trading_system.scheduler.jobs.daily_signals_job.load_strategies_from_run_config") as mock_load_strategies,
            patch("trading_system.scheduler.jobs.daily_signals_job.load_universe") as mock_load_universe,
        ):

            mock_strategy = MagicMock()
            mock_load_strategies.return_value = [mock_strategy]
            mock_load_universe.return_value = []

            # Should return early without error
            await daily_signals_job("equity")

    @pytest.mark.asyncio
    async def test_daily_signals_job_success(self, mock_env_vars, sample_ohlcv_data):
        """Test successful daily signals job execution."""
        with (
            patch("trading_system.scheduler.jobs.daily_signals_job.load_config") as mock_load_config,
            patch("trading_system.scheduler.jobs.daily_signals_job.LiveDataFetcher") as mock_fetcher_class,
            patch("trading_system.scheduler.jobs.daily_signals_job.load_strategies_from_run_config") as mock_load_strategies,
            patch("trading_system.scheduler.jobs.daily_signals_job.load_universe") as mock_load_universe,
            patch("trading_system.scheduler.jobs.daily_signals_job.LiveSignalGenerator") as mock_signal_gen_class,
            patch("trading_system.scheduler.jobs.daily_signals_job.EmailService") as mock_email_class,
        ):

            # Setup mocks
            mock_config = {
                "data_pipeline": MagicMock(),
                "signals": MagicMock(),
                "email": MagicMock(),
                "research": MagicMock(enabled=False),
                "universe": {"equity": "NASDAQ-100"},
            }
            mock_load_config.return_value = mock_config

            mock_fetcher = AsyncMock()
            mock_fetcher.fetch_daily_data = AsyncMock(return_value=sample_ohlcv_data)
            mock_fetcher_class.return_value = mock_fetcher

            mock_strategy = MagicMock()
            mock_load_strategies.return_value = [mock_strategy]
            mock_load_universe.return_value = ["AAPL", "MSFT"]

            mock_signal_gen = MagicMock()
            mock_signal_gen.generate_recommendations = AsyncMock(return_value=[])
            mock_signal_gen.tracker = None
            mock_signal_gen_class.return_value = mock_signal_gen

            mock_email = MagicMock()
            mock_email.send_daily_report = AsyncMock(return_value=True)
            mock_email_class.return_value = mock_email

            # Run job
            await daily_signals_job("equity")

            # Verify calls
            mock_fetcher.fetch_daily_data.assert_called_once()
            mock_signal_gen.generate_recommendations.assert_called_once()
            mock_email.send_daily_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_daily_signals_job_error_handling(self, mock_env_vars):
        """Test error handling in daily signals job."""
        with (
            patch("trading_system.scheduler.jobs.daily_signals_job.load_config") as mock_load_config,
            patch("trading_system.scheduler.jobs.daily_signals_job.send_error_alert") as mock_send_alert,
        ):

            mock_load_config.side_effect = ValueError("Config error")
            mock_send_alert.return_value = None

            # Should raise the error after sending alert
            with pytest.raises(ValueError):
                await daily_signals_job("equity")

            # Verify error alert was sent
            mock_send_alert.assert_called_once()


class TestCronRunner:
    """Tests for CronRunner class."""

    def test_cron_runner_initialization(self):
        """Test CronRunner initialization."""
        config = SchedulerConfig(enabled=True)
        runner = CronRunner(config)

        assert runner.config == config
        assert runner.scheduler is not None

    def test_cron_runner_default_config(self):
        """Test CronRunner with default config."""
        runner = CronRunner()

        assert runner.config is not None
        assert isinstance(runner.config, SchedulerConfig)

    def test_cron_runner_register_jobs_enabled(self):
        """Test job registration when scheduler is enabled."""
        config = SchedulerConfig(enabled=True)
        runner = CronRunner(config)

        runner.register_jobs()

        # Verify jobs are registered (check scheduler has jobs)
        jobs = runner.scheduler.get_jobs()
        assert len(jobs) == 2  # equity and crypto signals

    def test_cron_runner_register_jobs_disabled(self):
        """Test job registration when scheduler is disabled."""
        config = SchedulerConfig(enabled=False)
        runner = CronRunner(config)

        runner.register_jobs()

        # Jobs should not be registered
        jobs = runner.scheduler.get_jobs()
        assert len(jobs) == 0

    def test_cron_runner_start_stop(self):
        """Test starting and stopping the scheduler."""
        config = SchedulerConfig(enabled=True)
        runner = CronRunner(config)

        # Start scheduler
        runner.start()
        assert runner.is_running()

        # Stop scheduler
        runner.stop()
        assert not runner.is_running()

    def test_cron_runner_start_disabled(self):
        """Test starting scheduler when disabled."""
        config = SchedulerConfig(enabled=False)
        runner = CronRunner(config)

        runner.start()

        # Scheduler should not be running
        assert not runner.is_running()

    def test_cron_runner_stop_when_not_running(self):
        """Test stopping scheduler when not running."""
        config = SchedulerConfig(enabled=True)
        runner = CronRunner(config)

        # Stop without starting should not error
        runner.stop()
        assert not runner.is_running()


class TestMLRetrainJob:
    """Tests for MLRetrainJob class."""

    @pytest.fixture
    def mock_feature_db(self):
        """Create mock feature database."""
        db = MagicMock()
        db.count_samples = MagicMock(return_value=1000)
        db.get_active_model = MagicMock(return_value=None)
        db.activate_model = MagicMock(return_value=True)
        return db

    @pytest.fixture
    def mock_model_registry(self, mock_feature_db):
        """Create mock model registry."""
        registry = MagicMock()
        registry.get_active = MagicMock(return_value=None)
        registry.activate = MagicMock(return_value=True)
        return registry

    @pytest.fixture
    def mock_trainer(self):
        """Create mock trainer."""
        trainer = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.model_id = "test_model_123"
        mock_result.cv_metrics = {"auc": 0.85}
        mock_result.train_samples = 1000
        mock_result.error_message = None
        trainer.train = MagicMock(return_value=mock_result)
        return trainer

    @pytest.fixture
    def ml_config(self):
        """Create ML config for testing."""
        from trading_system.ml_refinement.config import MLConfig, ModelType, TrainingConfig

        return MLConfig(
            min_new_samples_for_retrain=100,
            training=TrainingConfig(
                train_window_days=180,
                validation_window_days=30,
                test_window_days=30,
            ),
        )

    def test_ml_retrain_job_initialization(self, ml_config, mock_feature_db, mock_model_registry):
        """Test MLRetrainJob initialization."""
        job = MLRetrainJob(
            config=ml_config,
            feature_db=mock_feature_db,
            model_registry=mock_model_registry,
        )

        assert job.config == ml_config
        assert job.feature_db == mock_feature_db
        assert job.model_registry == mock_model_registry

    def test_ml_retrain_job_default_registry(self, ml_config, mock_feature_db):
        """Test MLRetrainJob with default model registry."""
        with patch("trading_system.scheduler.jobs.ml_retrain_job.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry

            job = MLRetrainJob(
                config=ml_config,
                feature_db=mock_feature_db,
            )

            assert job.model_registry == mock_registry

    def test_ml_retrain_job_run_no_models(self, ml_config, mock_feature_db, mock_model_registry, mock_trainer):
        """Test running retrain job with no models to retrain."""
        with patch("trading_system.scheduler.jobs.ml_retrain_job.ModelTrainer") as mock_trainer_class:
            mock_trainer_class.return_value = mock_trainer

            job = MLRetrainJob(
                config=ml_config,
                feature_db=mock_feature_db,
                model_registry=mock_model_registry,
            )

            # Mock no active models
            mock_model_registry.get_active.return_value = None

            results = job.run(force=True)

            assert "models_retrained" in results
            assert "models_skipped" in results
            assert "errors" in results
            assert "started_at" in results
            assert "completed_at" in results

    def test_ml_retrain_job_check_retrain_needed_no_model(self, ml_config, mock_feature_db, mock_model_registry):
        """Test checking if retrain is needed when no active model exists."""
        job = MLRetrainJob(
            config=ml_config,
            feature_db=mock_feature_db,
            model_registry=mock_model_registry,
        )

        mock_model_registry.get_active.return_value = None
        mock_feature_db.count_samples.return_value = 500

        from trading_system.ml_refinement.config import ModelType

        status = job.check_retrain_needed(ModelType.SIGNAL_QUALITY)

        assert status["needed"] is True
        assert status["reason"] == "No active model"
        assert status["new_samples"] == 500

    def test_ml_retrain_job_check_retrain_needed_insufficient_samples(self, ml_config, mock_feature_db, mock_model_registry):
        """Test checking if retrain is needed with insufficient new samples."""
        job = MLRetrainJob(
            config=ml_config,
            feature_db=mock_feature_db,
            model_registry=mock_model_registry,
        )

        # Mock active model
        mock_model = MagicMock()
        mock_model.train_end_date = "2024-01-01"
        mock_model.model_id = "model_123"
        mock_model.validation_metrics = {"auc": 0.80}
        mock_model_registry.get_active.return_value = mock_model

        # Mock insufficient new samples
        mock_feature_db.count_samples.return_value = 50  # Less than threshold of 100

        from trading_system.ml_refinement.config import ModelType

        status = job.check_retrain_needed(ModelType.SIGNAL_QUALITY)

        assert status["needed"] is False
        assert "Insufficient new samples" in status["reason"]
        assert status["new_samples"] == 50

    def test_ml_retrain_job_check_retrain_needed_sufficient_samples(self, ml_config, mock_feature_db, mock_model_registry):
        """Test checking if retrain is needed with sufficient new samples."""
        job = MLRetrainJob(
            config=ml_config,
            feature_db=mock_feature_db,
            model_registry=mock_model_registry,
        )

        # Mock active model
        mock_model = MagicMock()
        mock_model.train_end_date = "2024-01-01"
        mock_model.model_id = "model_123"
        mock_model.validation_metrics = {"auc": 0.80}
        mock_model_registry.get_active.return_value = mock_model

        # Mock sufficient new samples
        mock_feature_db.count_samples.return_value = 150  # More than threshold of 100

        from trading_system.ml_refinement.config import ModelType

        status = job.check_retrain_needed(ModelType.SIGNAL_QUALITY)

        assert status["needed"] is True
        assert "new samples" in status["reason"]
        assert status["new_samples"] == 150
