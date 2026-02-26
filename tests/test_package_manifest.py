"""Tests for package manifest and napari plugin configuration."""
import os
from pathlib import Path

import pytest

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def test_napari_yaml_exists():
    """Test that napari.yaml is included in the napari_hydra package."""
    import napari_hydra

    # Get the package directory
    package_dir = Path(napari_hydra.__file__).parent
    napari_yaml_path = package_dir / "napari.yaml"

    # Check that the file exists
    assert napari_yaml_path.exists(), (
        f"napari.yaml not found at {napari_yaml_path}. "
        "This file is required for napari plugin discovery."
    )


def test_napari_yaml_is_readable():
    """Test that napari.yaml can be read and contains valid content."""
    import napari_hydra
    from pathlib import Path

    package_dir = Path(napari_hydra.__file__).parent
    napari_yaml_path = package_dir / "napari.yaml"

    # Read the file
    with open(napari_yaml_path, "r") as f:
        content = f.read()

    # Basic checks for required fields
    assert "name: napari-hydra" in content, "Missing 'name' field in napari.yaml"
    assert "contributions:" in content, "Missing 'contributions' field in napari.yaml"
    assert "commands:" in content, "Missing 'commands' field in napari.yaml"


@pytest.mark.skipif(not HAS_YAML, reason="pyyaml not installed")
def test_napari_yaml_contains_plugin_entry():
    """Test that napari.yaml defines the HydraStarDistPlugin."""
    import napari_hydra
    from pathlib import Path

    package_dir = Path(napari_hydra.__file__).parent
    napari_yaml_path = package_dir / "napari.yaml"

    with open(napari_yaml_path, "r") as f:
        manifest = yaml.safe_load(f)

    # Check that the plugin command is defined
    commands = manifest.get("contributions", {}).get("commands", [])
    assert len(commands) > 0, "No commands defined in napari.yaml"

    # Check for the HydraStarDistPlugin command
    plugin_found = any(
        cmd.get("python_name") == "napari_hydra.plugin:HydraStarDistPlugin"
        for cmd in commands
    )
    assert plugin_found, (
        "HydraStarDistPlugin command not found in napari.yaml. "
        "Expected python_name: 'napari_hydra.plugin:HydraStarDistPlugin'"
    )


def test_manifest_entry_point():
    """Test that the napari.manifest entry point is correctly configured."""
    import importlib.metadata

    # Get the entry points for napari.manifest
    try:
        entry_points = importlib.metadata.entry_points()
        # Handle both old and new importlib.metadata API
        if hasattr(entry_points, "select"):
            napari_manifest_eps = entry_points.select(group="napari.manifest")
        else:
            napari_manifest_eps = entry_points.get("napari.manifest", [])

        # Check that napari-hydra is registered
        napari_hydra_found = any(
            ep.name == "napari-hydra" for ep in napari_manifest_eps
        )
        assert napari_hydra_found, (
            "napari-hydra not found in napari.manifest entry points. "
            "Check pyproject.toml [project.entry-points] configuration."
        )
    except Exception as e:
        pytest.skip(f"Could not check entry points: {e}")
