from napari.utils.colormaps import DirectLabelColormap

def create_hydra_colormaps():
    """
    Create the standard white (wells) and blue (plaque) colormaps used by the plugin.

    Returns:
        tuple: A tuple containing two colormaps:
            - white_cmap (napari.utils.colormaps.DirectLabelColormap): White colormap for well labels.
            - blue_cmap (napari.utils.colormaps.DirectLabelColormap): Blue colormap for plaque labels.
    """
    color_dict = {0: (0, 0, 0, 0)}
    color_dict[None] = (1, 1, 1, 1)
    white_cmap = DirectLabelColormap(color_dict=color_dict)

    color_dict_blue = {0: (0, 0, 0, 0)}
    color_dict_blue[None] = (0, 194/255, 1, 1)
    blue_cmap = DirectLabelColormap(color_dict=color_dict_blue)
    
    return white_cmap, blue_cmap

def get_or_create_layer(viewer, layer_name, layer_data, colormap, blending):
    """
    Get an existing labels layer by name or create a new one with the specified parameters.

    Args:
        viewer (napari.Viewer): The napari viewer instance.
        layer_name (str): Name of the layer to retrieve or create.
        layer_data (numpy.ndarray): Data to initialize the layer with if creating new.
        colormap (napari.utils.colormaps.DirectLabelColormap): Colormap to use for the layer.
        blending (str): Blending mode for the layer (e.g., 'additive', 'multiplicative').

    Returns:
        napari.layers.Labels: The retrieved or newly created labels layer.
    """
    if layer_name in viewer.layers:
        layer = viewer.layers[layer_name]
    else:
        layer = viewer.add_labels(
            layer_data, name=layer_name,
            blending=blending, colormap=colormap
        )
    return layer
