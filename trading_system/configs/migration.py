"""Configuration migration tool for upgrading configs between versions."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .validation import validate_config_file

# Current config schema version
CURRENT_CONFIG_VERSION = "1.0"


def detect_config_version(config_data: Dict[str, Any]) -> str:
    """Detect the version of a configuration file.

    Args:
        config_data: Parsed YAML/JSON config data

    Returns:
        Version string (e.g., "1.0") or "1.0" if not specified
    """
    # Check for explicit version field
    if "version" in config_data:
        return str(config_data["version"])

    # Check for config_version field
    if "config_version" in config_data:
        return str(config_data["config_version"])

    # Check for schema_version field
    if "schema_version" in config_data:
        return str(config_data["schema_version"])

    # Default to 1.0 if no version specified
    return "1.0"


def migrate_config_v1_0_to_v1_1(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate config from version 1.0 to 1.1.

    This is a placeholder for future migrations. Currently, 1.0 and 1.1 are the same.

    Args:
        config_data: Config data in version 1.0 format

    Returns:
        Config data in version 1.1 format
    """
    migrated = config_data.copy()

    # Add version field if not present
    if "version" not in migrated:
        migrated["version"] = "1.1"

    # Future migrations would go here
    # Example:
    # if "old_field" in migrated:
    #     migrated["new_field"] = migrated.pop("old_field")

    return migrated


def migrate_config(
    config_path: str, target_version: Optional[str] = None, output_path: Optional[str] = None, dry_run: bool = False
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Migrate a configuration file to a target version.

    Args:
        config_path: Path to the configuration file to migrate
        target_version: Target version (defaults to CURRENT_CONFIG_VERSION)
        output_path: Optional path to save migrated config (defaults to overwrite)
        dry_run: If True, don't write the migrated config, just return it

    Returns:
        Tuple of (success, message, migrated_config_dict)
    """
    if target_version is None:
        target_version = CURRENT_CONFIG_VERSION

    config_path_obj = Path(config_path)

    if not config_path_obj.exists():
        return False, f"Config file not found: {config_path}", None

    try:
        # Load config
        with open(config_path_obj, "r") as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            return False, "Config file is empty or invalid YAML", None

        # Detect current version
        current_version = detect_config_version(config_data)

        if current_version == target_version:
            return True, f"Config is already at version {target_version}", config_data

        # Perform migration
        migrated_data = config_data.copy()

        # Migration chain: 1.0 -> 1.1 -> ... -> target_version
        version_chain = _get_version_chain(current_version, target_version)

        for i in range(len(version_chain) - 1):
            from_ver = version_chain[i]
            to_ver = version_chain[i + 1]

            if from_ver == "1.0" and to_ver == "1.1":
                migrated_data = migrate_config_v1_0_to_v1_1(migrated_data)
            else:
                return False, f"Migration from {from_ver} to {to_ver} not implemented", None

        # Update version field
        migrated_data["version"] = target_version

        # Validate migrated config
        # Determine config type
        if "dataset" in migrated_data or "splits" in migrated_data:
            config_type = "run"
        else:
            config_type = "strategy"

        # Create a temporary file to validate
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(migrated_data, tmp, default_flow_style=False, sort_keys=False)
            tmp_path = tmp.name
            tmp.flush()  # Ensure data is written

        try:
            # Check if file exists before validation
            if not Path(tmp_path).exists():
                return False, "Failed to create temporary file for validation", None

            is_valid, error_message, _ = validate_config_file(tmp_path, config_type=config_type)
            if not is_valid:
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()
                return False, f"Migrated config failed validation: {error_message}", None
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

        # Write migrated config
        if not dry_run:
            output_path_obj = Path(output_path) if output_path else config_path_obj
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path_obj, "w") as f:
                yaml.dump(migrated_data, f, default_flow_style=False, sort_keys=False)

            message = f"Successfully migrated config from {current_version} to {target_version}"
            if output_path:
                message += f" (saved to {output_path})"
        else:
            message = f"Dry run: Would migrate config from {current_version} to {target_version}"

        return True, message, migrated_data

    except yaml.YAMLError as e:
        return False, f"Failed to parse YAML: {str(e)}", None
    except Exception as e:
        return False, f"Migration failed: {str(e)}", None


def _get_version_chain(from_version: str, to_version: str) -> List[str]:
    """Get the chain of versions to migrate through.

    Args:
        from_version: Starting version
        to_version: Target version

    Returns:
        List of versions in migration order
    """
    # Parse versions
    [int(x) for x in from_version.split(".")]
    [int(x) for x in to_version.split(".")]

    # Simple linear migration for now
    # In the future, this could handle more complex versioning
    chain = [from_version]

    # For now, just add target version
    # Future: could add intermediate versions if needed
    if from_version != to_version:
        chain.append(to_version)

    return chain


def backup_config(config_path: str, backup_dir: Optional[str] = None) -> str:
    """Create a backup of a configuration file.

    Args:
        config_path: Path to config file to backup
        backup_dir: Directory to store backup (defaults to config_backups/)

    Returns:
        Path to backup file
    """
    config_path_obj = Path(config_path)

    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if backup_dir is None:
        backup_dir = "config_backups"

    backup_dir_obj = Path(backup_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir_obj / timestamp / config_path_obj.name

    backup_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy file
    import shutil

    shutil.copy2(config_path_obj, backup_path)

    return str(backup_path)


def check_config_version(config_path: str) -> Tuple[str, bool]:
    """Check the version of a configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (version_string, is_current)
    """
    config_path_obj = Path(config_path)

    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path_obj, "r") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        raise ValueError("Config file is empty or invalid YAML")

    version = detect_config_version(config_data)
    is_current = version == CURRENT_CONFIG_VERSION

    return version, is_current
