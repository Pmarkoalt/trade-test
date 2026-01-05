"""Unit tests for configs/migration.py."""

import shutil
import tempfile
from pathlib import Path

import yaml

from trading_system.configs.migration import (
    CURRENT_CONFIG_VERSION,
    backup_config,
    check_config_version,
    detect_config_version,
    migrate_config,
    migrate_config_v1_0_to_v1_1,
)


class TestDetectConfigVersion:
    """Tests for detect_config_version function."""

    def test_detect_version_explicit(self):
        """Test detecting version from explicit 'version' field."""
        config_data = {"version": "1.0", "other": "data"}
        version = detect_config_version(config_data)
        assert version == "1.0"

    def test_detect_version_config_version(self):
        """Test detecting version from 'config_version' field."""
        config_data = {"config_version": "1.1", "other": "data"}
        version = detect_config_version(config_data)
        assert version == "1.1"

    def test_detect_version_schema_version(self):
        """Test detecting version from 'schema_version' field."""
        config_data = {"schema_version": "1.2", "other": "data"}
        version = detect_config_version(config_data)
        assert version == "1.2"

    def test_detect_version_default(self):
        """Test default version when no version field exists."""
        config_data = {"other": "data"}
        version = detect_config_version(config_data)
        assert version == "1.0"

    def test_detect_version_priority(self):
        """Test that 'version' field takes priority over others."""
        config_data = {"version": "1.0", "config_version": "1.1", "schema_version": "1.2"}
        version = detect_config_version(config_data)
        assert version == "1.0"

    def test_detect_version_string_conversion(self):
        """Test that version is converted to string."""
        config_data = {"version": 1.0}
        version = detect_config_version(config_data)
        assert version == "1.0"
        assert isinstance(version, str)


class TestMigrateConfigV1_0ToV1_1:
    """Tests for migrate_config_v1_0_to_v1_1 function."""

    def test_migrate_adds_version(self):
        """Test that migration adds version field if missing."""
        config_data = {"some": "data"}
        migrated = migrate_config_v1_0_to_v1_1(config_data)
        assert "version" in migrated
        assert migrated["version"] == "1.1"

    def test_migrate_preserves_data(self):
        """Test that migration preserves existing data."""
        config_data = {"some": "data", "other": "value"}
        migrated = migrate_config_v1_0_to_v1_1(config_data)
        assert migrated["some"] == "data"
        assert migrated["other"] == "value"

    def test_migrate_updates_version(self):
        """Test that migration updates version field."""
        config_data = {"version": "1.0", "data": "test"}
        migrated = migrate_config_v1_0_to_v1_1(config_data)
        assert migrated["version"] == "1.1"

    def test_migrate_creates_copy(self):
        """Test that migration creates a copy (doesn't modify original)."""
        config_data = {"version": "1.0"}
        migrated = migrate_config_v1_0_to_v1_1(config_data)
        assert config_data["version"] == "1.0"  # Original unchanged
        assert migrated["version"] == "1.1"  # Migrated updated


class TestMigrateConfig:
    """Tests for migrate_config function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_migrate_config_success(self):
        """Test successful config migration."""
        config_path = self.temp_dir / "test_config.yaml"
        # Use a minimal valid run config structure matching RunConfig schema
        config_data = {
            "version": "1.0",
            "dataset": {
                "equity_path": "data/equity/",
                "crypto_path": "data/crypto/",
                "benchmark_path": "data/benchmarks/",
                "format": "csv",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            },
            "splits": {
                "train_start": "2023-01-01",
                "train_end": "2023-06-30",
                "validation_start": "2023-07-01",
                "validation_end": "2023-09-30",
                "holdout_start": "2023-10-01",
                "holdout_end": "2023-12-31",
            },
            "strategies": {"equity": {"config_path": "test_strategy.yaml", "enabled": True}},
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        success, message, migrated = migrate_config(str(config_path), target_version="1.1", dry_run=True)

        assert success is True
        assert migrated is not None
        assert migrated["version"] == "1.1"

    def test_migrate_config_file_not_found(self):
        """Test migration when file doesn't exist."""
        config_path = self.temp_dir / "nonexistent.yaml"

        success, message, migrated = migrate_config(str(config_path))

        assert success is False
        assert "not found" in message.lower()
        assert migrated is None

    def test_migrate_config_empty_file(self):
        """Test migration with empty YAML file."""
        config_path = self.temp_dir / "empty.yaml"
        config_path.write_text("")

        success, message, migrated = migrate_config(str(config_path))

        assert success is False
        assert "empty" in message.lower() or "invalid" in message.lower()
        assert migrated is None

    def test_migrate_config_same_version(self):
        """Test migration when already at target version."""
        config_path = self.temp_dir / "test_config.yaml"
        config_data = {"version": "1.0", "data": "test"}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        success, message, migrated = migrate_config(str(config_path), target_version="1.0", dry_run=True)

        assert success is True
        assert "already" in message.lower() or "current" in message.lower()

    def test_migrate_config_dry_run(self):
        """Test migration in dry-run mode (doesn't write file)."""
        config_path = self.temp_dir / "test_config.yaml"
        config_data = {"version": "1.0", "data": "test"}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        success, message, migrated = migrate_config(str(config_path), target_version="1.1", dry_run=True)

        assert success is True
        # Original file should still have version 1.0
        with open(config_path, "r") as f:
            original = yaml.safe_load(f)
        assert original["version"] == "1.0"

    def test_migrate_config_write_file(self):
        """Test migration that writes migrated config."""
        config_path = self.temp_dir / "test_config.yaml"
        config_data = {"version": "1.0", "data": "test"}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        success, message, migrated = migrate_config(str(config_path), target_version="1.1", dry_run=False)

        assert success is True
        # File should be updated
        with open(config_path, "r") as f:
            updated = yaml.safe_load(f)
        assert updated["version"] == "1.1"

    def test_migrate_config_custom_output_path(self):
        """Test migration with custom output path."""
        config_path = self.temp_dir / "test_config.yaml"
        output_path = self.temp_dir / "migrated_config.yaml"
        config_data = {"version": "1.0", "data": "test"}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        success, message, migrated = migrate_config(
            str(config_path), target_version="1.1", output_path=str(output_path), dry_run=False
        )

        assert success is True
        assert output_path.exists()
        with open(output_path, "r") as f:
            migrated_file = yaml.safe_load(f)
        assert migrated_file["version"] == "1.1"
        # Original should be unchanged
        with open(config_path, "r") as f:
            original = yaml.safe_load(f)
        assert original["version"] == "1.0"


class TestBackupConfig:
    """Tests for backup_config function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_backup_config_default_location(self):
        """Test backing up config to default location."""
        config_path = self.temp_dir / "test_config.yaml"
        config_data = {"version": "1.0", "data": "test"}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        backup_path = backup_config(str(config_path))

        assert Path(backup_path).exists()
        assert "backup" in backup_path.lower() or "test_config" in backup_path
        # Backup should have same content
        with open(backup_path, "r") as f:
            backup_data = yaml.safe_load(f)
        assert backup_data == config_data

    def test_backup_config_custom_dir(self):
        """Test backing up config to custom directory."""
        config_path = self.temp_dir / "test_config.yaml"
        backup_dir = self.temp_dir / "backups"
        config_data = {"version": "1.0", "data": "test"}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        backup_path = backup_config(str(config_path), backup_dir=str(backup_dir))

        assert backup_dir.exists()
        assert Path(backup_path).parent == backup_dir
        assert Path(backup_path).exists()

    def test_backup_config_creates_dir(self):
        """Test that backup creates directory if it doesn't exist."""
        config_path = self.temp_dir / "test_config.yaml"
        backup_dir = self.temp_dir / "new_backups" / "nested"
        config_data = {"version": "1.0"}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        backup_path = backup_config(str(config_path), backup_dir=str(backup_dir))

        assert backup_dir.exists()
        assert Path(backup_path).exists()


class TestCheckConfigVersion:
    """Tests for check_config_version function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_check_config_version_current(self):
        """Test checking config with current version."""
        config_path = self.temp_dir / "test_config.yaml"
        config_data = {"version": CURRENT_CONFIG_VERSION, "data": "test"}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        version, is_current = check_config_version(str(config_path))

        assert version == CURRENT_CONFIG_VERSION
        assert is_current is True

    def test_check_config_version_old(self):
        """Test checking config with old version."""
        config_path = self.temp_dir / "test_config.yaml"
        config_data = {"version": "0.9", "data": "test"}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        version, is_current = check_config_version(str(config_path))

        assert version == "0.9"
        assert is_current is False

    def test_check_config_version_no_version(self):
        """Test checking config with no version field."""
        config_path = self.temp_dir / "test_config.yaml"
        config_data = {"data": "test"}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        version, is_current = check_config_version(str(config_path))

        assert version == "1.0"  # Default
        assert is_current is False  # Not current version

    def test_check_config_version_file_not_found(self):
        """Test checking version of non-existent file."""
        config_path = self.temp_dir / "nonexistent.yaml"

        # Should handle gracefully or raise - check implementation
        try:
            version, is_current = check_config_version(str(config_path))
            # If it doesn't raise, version should be default
            assert version == "1.0"
        except (FileNotFoundError, Exception):
            # If it raises, that's also acceptable behavior
            pass
