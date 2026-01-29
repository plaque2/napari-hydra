# Helper to get polygons and rasterize
import numpy as np
from skimage.draw import polygon

def rasterize_labels(label_dict, out_shape):
    """
    Convert polygon coordinates from StarDist prediction into a rasterized label array.

    Args:
        label_dict (dict): Dictionary containing prediction results, specifically 'coord'.
                           'coord' is a list of coordinates (arrays of [x, y]).
        out_shape (tuple): Shape of the output array (height, width).

    Returns:
        tuple: A tuple containing two elements:
            - labels_arr (numpy.ndarray): Rasterized integer label mask.
            - poly_coords (list): List of polygon coordinates (list of [x, y] pairs).
    """
    unmatched_coords = label_dict["coord"]
    poly_coords = []
    for coord in unmatched_coords:
        X = coord[0]
        Y = coord[1]
        single_polygon = [[X[i], Y[i]] for i in range(len(X))]
        poly_coords.append(single_polygon)
    labels_arr = np.zeros(out_shape, dtype=np.int32)
    def scale_poly(poly):
        poly = np.array(poly)
        return poly
    for label, poly in enumerate(poly_coords):
        poly_scaled = scale_poly(poly)
        rr, cc = polygon(poly_scaled[:, 0], poly_scaled[:, 1], shape=labels_arr.shape)
        labels_arr[rr, cc] = label + 1
    return labels_arr, poly_coords