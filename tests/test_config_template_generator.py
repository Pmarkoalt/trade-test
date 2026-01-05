"""Unit tests for configs/template_generator.py."""

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from trading_system.configs.template_generator import generate_run_config_template, generate_strategy_config_template


class TestGenerateRunConfigTemplate:
    """Tests for generate_run_config_template function."""

    def test_generate_template_returns_string(self):
        """Test that template generation returns a string."""
        template = generate_run_config_template()
        assert isinstance(template, str)
        assert len(template) > 0

    def test_generate_template_valid_yaml(self):
        """Test that generated template is valid YAML."""
        template = generate_run_config_template()
        config_data = yaml.safe_load(template)
        assert config_data is not None
        assert isinstance(config_data, dict)

    def test_generate_template_has_required_fields(self):
        """Test that template has required fields."""
        template = generate_run_config_template()
        config_data = yaml.safe_load(template)

        assert "dataset" in config_data
        assert "splits" in config_data
        assert "strategies" in config_data

    def test_generate_template_with_comments(self):
        """Test template generation with comments."""
        template = generate_run_config_template(include_comments=True)
        assert "#" in template  # Should have comments
        assert "Dataset configuration" in template or "dataset:" in template

    def test_generate_template_without_comments(self):
        """Test template generation without comments."""
        template = generate_run_config_template(include_comments=False)
        # Should still be valid YAML, but fewer comment lines
        config_data = yaml.safe_load(template)
        assert config_data is not None

    def test_generate_template_save_to_file(self):
        """Test saving template to file."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            output_path = temp_dir / "run_config_template.yaml"
            template = generate_run_config_template(output_path=str(output_path))

            assert output_path.exists()
            assert isinstance(template, str)

            # Verify file content
            with open(output_path, "r") as f:
                file_content = f.read()
            assert file_content == template

            # Verify it's valid YAML
            config_data = yaml.safe_load(file_content)
            assert config_data is not None
        finally:
            shutil.rmtree(temp_dir)

    def test_generate_template_creates_parent_dir(self):
        """Test that template generation creates parent directory."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            output_path = temp_dir / "nested" / "deep" / "template.yaml"
            generate_run_config_template(output_path=str(output_path))

            assert output_path.parent.exists()
            assert output_path.exists()
        finally:
            shutil.rmtree(temp_dir)


class TestGenerateStrategyConfigTemplate:
    """Tests for generate_strategy_config_template function."""

    def test_generate_equity_template(self):
        """Test generating equity strategy template."""
        template = generate_strategy_config_template(asset_class="equity")
        assert isinstance(template, str)
        assert len(template) > 0

        config_data = yaml.safe_load(template)
        assert config_data is not None
        assert config_data.get("asset_class") == "equity"
        assert "equity" in config_data.get("name", "").lower()

    def test_generate_crypto_template(self):
        """Test generating crypto strategy template."""
        template = generate_strategy_config_template(asset_class="crypto")
        assert isinstance(template, str)
        assert len(template) > 0

        config_data = yaml.safe_load(template)
        assert config_data is not None
        assert config_data.get("asset_class") == "crypto"
        assert "crypto" in config_data.get("name", "").lower()

    def test_generate_template_invalid_asset_class(self):
        """Test that invalid asset class raises ValueError."""
        with pytest.raises(ValueError, match="asset_class must be"):
            generate_strategy_config_template(asset_class="invalid")

    def test_generate_template_with_comments(self):
        """Test template generation with comments."""
        template = generate_strategy_config_template(asset_class="equity", include_comments=True)
        assert "#" in template  # Should have comments

    def test_generate_template_without_comments(self):
        """Test template generation without comments."""
        template = generate_strategy_config_template(asset_class="equity", include_comments=False)
        config_data = yaml.safe_load(template)
        assert config_data is not None

    def test_generate_template_save_to_file(self):
        """Test saving template to file."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            output_path = temp_dir / "strategy_template.yaml"
            template = generate_strategy_config_template(asset_class="equity", output_path=str(output_path))

            assert output_path.exists()
            assert isinstance(template, str)

            # Verify file content
            with open(output_path, "r") as f:
                file_content = f.read()
            assert file_content == template

            # Verify it's valid YAML
            config_data = yaml.safe_load(file_content)
            assert config_data is not None
            assert config_data.get("asset_class") == "equity"
        finally:
            shutil.rmtree(temp_dir)

    def test_generate_template_creates_parent_dir(self):
        """Test that template generation creates parent directory."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            output_path = temp_dir / "nested" / "deep" / "template.yaml"
            generate_strategy_config_template(asset_class="crypto", output_path=str(output_path))

            assert output_path.parent.exists()
            assert output_path.exists()
        finally:
            shutil.rmtree(temp_dir)

    def test_equity_template_has_equity_specific_fields(self):
        """Test that equity template has equity-specific configuration."""
        template = generate_strategy_config_template(asset_class="equity")
        config_data = yaml.safe_load(template)

        assert config_data.get("asset_class") == "equity"
        assert "universe" in config_data or "benchmark" in config_data

    def test_crypto_template_has_crypto_specific_fields(self):
        """Test that crypto template has crypto-specific configuration."""
        template = generate_strategy_config_template(asset_class="crypto")
        config_data = yaml.safe_load(template)

        assert config_data.get("asset_class") == "crypto"
        # Crypto might have different exit mode
        if "exit" in config_data:
            exit_config = config_data["exit"]
            # Crypto might use "staged" exit mode
            assert isinstance(exit_config, dict)
