from napari.utils.colormaps import DirectLabelColormap

def create_hydra_colormaps():
    """
    Create the standard white (wells) and blue (plaque) colormaps used by the plugin.
    Returns:
        white_cmap, blue_cmap
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
    """
    if layer_name in viewer.layers:
        layer = viewer.layers[layer_name]
    else:
        layer = viewer.add_labels(
            layer_data, name=layer_name,
            blending=blending, colormap=colormap
        )
    return layer
