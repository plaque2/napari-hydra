"""Tests for napari plugin discovery and registration (headless-safe).

These tests verify that napari can discover and register the napari-hydra
plugin purely via Python, without requiring a GUI or model downloads.
They use the npe2 PluginManager to introspect the plugin manifest.
"""
import pytest


def test_napari_discovers_plugin():
    """Verify that napari can discover napari-hydra via its manifest."""
    from npe2 import PluginManager

    pm = PluginManager.instance()
    pm.discover()

    manifest = pm.get_manifest("napari-hydra")
    assert manifest is not None
    assert manifest.display_name == "napari-hydra"


def test_manifest_declares_widget_contribution():
    """Verify the manifest declares the expected widget command."""
    from npe2 import PluginManager

    pm = PluginManager.instance()
    pm.discover()

    manifest = pm.get_manifest("napari-hydra")

    # Check the command is registered
    command_ids = [cmd.id for cmd in manifest.contributions.commands]
    assert "napari-hydra.hydra_widget" in command_ids

    # Check the widget contribution references the command
    widget_commands = [w.command for w in manifest.contributions.widgets]
    assert "napari-hydra.hydra_widget" in widget_commands


def test_widget_command_points_to_correct_class():
    """Verify the command's python_name resolves to the right class."""
    from npe2 import PluginManager

    pm = PluginManager.instance()
    pm.discover()

    manifest = pm.get_manifest("napari-hydra")
    cmd = next(
        c for c in manifest.contributions.commands
        if c.id == "napari-hydra.hydra_widget"
    )
    assert cmd.python_name == "napari_hydra.plugin:HydraStarDistPlugin"


def test_plugin_module_is_importable():
    """Verify the plugin module and class can be imported."""
    from napari_hydra.plugin import HydraStarDistPlugin

    assert HydraStarDistPlugin is not None
    # Verify it's a class (not an instance or something else)
    assert isinstance(HydraStarDistPlugin, type)
