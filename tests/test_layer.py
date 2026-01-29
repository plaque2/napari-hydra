from unittest.mock import MagicMock
from napari_hydra.utils.layer import create_hydra_colormaps, get_or_create_layer

def test_create_hydra_colormaps():
    white, blue = create_hydra_colormaps()
    # Check if they are Colormap objects (or similar)
    # Since we can't easily assert on napari types in minimal env without napari installed
    # we just check they returned something not None
    assert white is not None
    assert blue is not None
    # If napari is installed, we could check type(white).__name__ == 'DirectLabelColormap'

def test_get_or_create_layer_existing():
    viewer = MagicMock()
    # Mock dictionary behavior for viewer.layers
    viewer.layers = {"my_layer": "existing_layer_obj"}
    
    layer = get_or_create_layer(viewer, "my_layer", None, None, None)
    assert layer == "existing_layer_obj"
    viewer.add_labels.assert_not_called()

def test_get_or_create_layer_new():
    viewer = MagicMock()
    # Mock empty layers
    viewer.layers = {}
    
    mock_layer = MagicMock()
    viewer.add_labels.return_value = mock_layer
    
    data = [1, 2, 3]
    layer = get_or_create_layer(viewer, "new_layer", data, "cmap", "additive")
    
    assert layer == mock_layer
    viewer.add_labels.assert_called_once_with(
        data, name="new_layer", blending="additive", colormap="cmap"
    )
