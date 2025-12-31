#!/usr/bin/env python3
"""Dependency update automation script for trading-system.

This script helps manage and update dependencies:
- Check for outdated packages
- Update dependency versions in pyproject.toml
- Run tests after updates
- Generate dependency reports

Usage:
    python scripts/update_dependencies.py check          # Check for outdated packages
    python scripts/update_dependencies.py update        # Update to latest compatible versions
    python scripts/update_dependencies.py update --test  # Update and run tests
    python scripts/update_dependencies.py report         # Generate dependency report
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

try:
    import tomli  # For Python < 3.11

    tomllib = None
except ImportError:
    try:
        import tomllib  # For Python 3.11+

        tomli = None
    except ImportError:
        print("Error: tomli or tomllib required. Install with: pip install tomli")
        sys.exit(1)


def load_pyproject() -> dict:
    """Load and parse pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
        sys.exit(1)

    content = pyproject_path.read_text()

    # Use tomllib for Python 3.11+, tomli for older versions
    if tomli is None:
        return tomllib.loads(content)  # type: ignore[union-attr]
    else:
        return tomli.loads(content)


def parse_version_spec(spec: str) -> Tuple[str, str, str]:
    """Parse version specifier like '>=1.5.0,<3.0.0' into (min, max, current).

    Returns:
        (min_version, max_version, current_version)
    """
    # Extract version numbers
    min_match = re.search(r">=([\d.]+)", spec)
    max_match = re.search(r"<([\d.]+)", spec)

    min_version = min_match.group(1) if min_match else None
    max_version = max_match.group(1) if max_match else None

    # Try to get current installed version
    try:
        result = subprocess.run(  # noqa: S603 - subprocess needed for pip execution
            [sys.executable, "-m", "pip", "show", spec.split(">=")[0].split(",")[0].strip()],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    current_version = line.split(":", 1)[1].strip()
                    return (min_version, max_version, current_version)
    except Exception:  # noqa: S110 - exception handling needed for version parsing
        pass

    return (min_version, max_version, None)


def check_outdated() -> Dict[str, Dict]:
    """Check for outdated packages."""
    print("Checking for outdated packages...")

    try:
        result = subprocess.run(  # noqa: S603 - subprocess needed for pip execution
            [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"], capture_output=True, text=True, check=True
        )
        outdated = json.loads(result.stdout)

        pyproject = load_pyproject()
        dependencies = pyproject.get("project", {}).get("dependencies", [])
        optional_deps = pyproject.get("project", {}).get("optional-dependencies", {})

        # Build dependency map
        dep_map = {}
        for dep in dependencies:
            name = dep.split(">=")[0].split(",")[0].strip()
            dep_map[name] = {"spec": dep, "group": "core"}

        for group, deps in optional_deps.items():
            for dep in deps:
                name = dep.split(">=")[0].split(",")[0].strip()
                dep_map[name] = {"spec": dep, "group": group}

        # Check which outdated packages are in our dependencies
        relevant_outdated = {}
        for pkg in outdated:
            name = pkg["name"]
            if name in dep_map:
                relevant_outdated[name] = {
                    "current": pkg["version"],
                    "latest": pkg["latest_version"],
                    "spec": dep_map[name]["spec"],
                    "group": dep_map[name]["group"],
                }

        return relevant_outdated
    except subprocess.CalledProcessError as e:
        print(f"Error checking outdated packages: {e}")
        return {}
    except json.JSONDecodeError:
        print("No outdated packages found (or pip list failed)")
        return {}


def print_outdated_report(outdated: Dict[str, Dict]):
    """Print a report of outdated packages."""
    if not outdated:
        print("\n✓ All dependencies are up to date!")
        return

    print(f"\nFound {len(outdated)} outdated dependencies:\n")

    # Group by dependency group
    by_group = {}
    for name, info in outdated.items():
        group = info["group"]
        if group not in by_group:
            by_group[group] = []
        by_group[group].append((name, info))

    for group in sorted(by_group.keys()):
        print(f"\n{group.upper()} dependencies:")
        print("-" * 70)
        for name, info in sorted(by_group[group]):
            current = info["current"]
            latest = info["latest"]
            spec = info["spec"]
            print(f"  {name:25} {current:12} → {latest:12}  (spec: {spec})")


def update_pyproject(outdated: Dict[str, Dict], dry_run: bool = True) -> bool:
    """Update pyproject.toml with latest compatible versions."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    updates = []
    for name, info in outdated.items():
        current_spec = info["spec"]
        latest_version = info["latest"]

        # Parse current spec to maintain version constraints
        min_match = re.search(r">=([\d.]+)", current_spec)
        max_match = re.search(r"<([\d.]+)", current_spec)

        if min_match:
            # Update to latest but maintain upper bound if present
            if max_match:
                # Keep major version constraint
                major_version = latest_version.split(".")[0]
                new_spec = f">={latest_version},<{int(major_version) + 1}.0.0"
            else:
                new_spec = f">={latest_version}"

            if new_spec != current_spec:
                updates.append((name, current_spec, new_spec))
                if not dry_run:
                    # Replace in content
                    pattern = re.escape(current_spec)
                    content = re.sub(pattern, new_spec, content)

    if not updates:
        print("\nNo updates needed.")
        return False

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Would update {len(updates)} dependencies:")
    for name, old, new in updates:
        print(f"  {name}: {old} → {new}")

    if not dry_run:
        pyproject_path.write_text(content)
        print(f"\n✓ Updated {pyproject_path}")

    return len(updates) > 0


def generate_report():
    """Generate a comprehensive dependency report."""
    print("Generating dependency report...\n")

    pyproject = load_pyproject()
    project = pyproject.get("project", {})

    print("=" * 70)
    print(f"DEPENDENCY REPORT: {project.get('name', 'trading-system')}")
    print("=" * 70)

    # Core dependencies
    core_deps = project.get("dependencies", [])
    print(f"\nCore Dependencies ({len(core_deps)}):")
    print("-" * 70)
    for dep in sorted(core_deps):
        name = dep.split(">=")[0].split(",")[0].strip()
        print(f"  {name:30} {dep}")

    # Optional dependencies
    optional_deps = project.get("optional-dependencies", {})
    print(f"\nOptional Dependency Groups ({len(optional_deps)}):")
    print("-" * 70)
    for group, deps in sorted(optional_deps.items()):
        print(f"\n  [{group}] ({len(deps)} packages):")
        for dep in sorted(deps):
            name = dep.split(">=")[0].split(",")[0].strip()
            print(f"    {name:28} {dep}")

    # Python version requirement
    python_req = project.get("requires-python", "Not specified")
    print(f"\nPython Version Requirement: {python_req}")

    # Check installed packages
    print("\n" + "=" * 70)
    print("INSTALLED PACKAGES STATUS")
    print("=" * 70)

    outdated = check_outdated()
    if outdated:
        print_outdated_report(outdated)
    else:
        print("\n✓ All dependencies are up to date!")


def run_tests() -> bool:
    """Run the test suite."""
    print("\nRunning tests...")
    try:
        result = subprocess.run(  # noqa: S603 - subprocess needed for pytest execution
            [sys.executable, "-m", "pytest", "tests/", "-v"], check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Dependency management automation for trading-system")
    parser.add_argument("command", choices=["check", "update", "report"], help="Command to execute")
    parser.add_argument("--test", action="store_true", help="Run tests after update (only for 'update' command)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run mode (don't actually update files)")
    parser.add_argument("--force", action="store_true", help="Actually perform updates (overrides --dry-run)")

    args = parser.parse_args()

    if args.command == "check":
        outdated = check_outdated()
        print_outdated_report(outdated)

    elif args.command == "update":
        outdated = check_outdated()
        if not outdated:
            print("\n✓ All dependencies are up to date!")
            return

        print_outdated_report(outdated)

        dry_run = args.dry_run and not args.force
        if update_pyproject(outdated, dry_run=dry_run):
            if not dry_run:
                print("\n✓ Dependencies updated in pyproject.toml")
                print("  Next steps:")
                print("  1. Review the changes")
                print("  2. Run: pip install -e '.[dev]'")
                if args.test:
                    if run_tests():
                        print("  3. ✓ Tests passed!")
                    else:
                        print("  3. ✗ Tests failed - please review changes")
                else:
                    print("  3. Run: python scripts/update_dependencies.py update --test")
            else:
                print("\n[DRY RUN] Use --force to actually update files")

    elif args.command == "report":
        generate_report()


if __name__ == "__main__":
    main()
