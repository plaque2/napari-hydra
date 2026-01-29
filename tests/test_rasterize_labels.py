import numpy as np
from napari_hydra.utils.rasterize_labels import rasterize_labels

def test_rasterize_labels():
    # Define a simple square polygon
    # Coords: (2,2), (2,5), (5,5), (5,2)
    # Label dict format: 'coord': [[X_arr, Y_arr]]
    coord_x = np.array([2, 2, 5, 5])
    coord_y = np.array([2, 5, 5, 2])
    
    label_dict = {"coord": [[coord_x, coord_y]]}
    out_shape = (10, 10)
    
    labels, polys = rasterize_labels(label_dict, out_shape)
    
    assert labels.shape == (10, 10)
    assert np.any(labels == 1)
    # Center should be filled
    assert labels[3, 3] == 1
    assert labels[0, 0] == 0
