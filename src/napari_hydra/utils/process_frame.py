import numpy as np
from skimage import img_as_float32
from csbdeep.utils import normalize
from stardist.utils import edt_prob
from stardist.geometry import star_dist
from .make_divisible import make_divisible

def process_frame(img2d, wells2d, plaque2d, target_width, target_height, config):
    """
    Process a single frame for training, resizing and generating StarDist targets.

    Args:
        img2d (numpy.ndarray): Input 2D image (or 2D + channels).
        wells2d (numpy.ndarray): 2D well labels.
        plaque2d (numpy.ndarray): 2D plaque labels.
        target_width (int): Target width for resizing.
        target_height (int): Target height for resizing.
        config (object): StarDist configuration object containing grid and n_rays.

    Returns:
        tuple: A tuple containing five elements:
            - X (numpy.ndarray): Normalized and resized image with channel dimension.
            - dist1 (numpy.ndarray): StarDist distances for wells.
            - prob1 (numpy.ndarray): StarDist probabilities for wells.
            - dist2 (numpy.ndarray): StarDist distances for plaques.
            - prob2 (numpy.ndarray): StarDist probabilities for plaques.
    """
    # Resize frame and labels
    h, w = img2d.shape[:2]
    scale = min(w / target_width, h / target_height)
    new_w, new_h = int(w / scale), int(h / scale)
    new_w, new_h = make_divisible(new_w), make_divisible(new_h)
    y_indices = np.linspace(0, h - 1, new_h).astype(int)
    x_indices = np.linspace(0, w - 1, new_w).astype(int)

    # Resize image
    if img2d.ndim == 3:
        resized_img = img2d[y_indices[:, None], x_indices[None, :], :]
    else:
        resized_img = img2d[y_indices[:, None], x_indices[None, :]]
    img = img_as_float32(resized_img)
    axis_norm = (0, 1)
    image_norm = normalize(img, 1, 99.8, axis=axis_norm)

    # Resize labels
    wells_resized = wells2d[y_indices[:, None], x_indices[None, :]]
    plaque_resized = plaque2d[y_indices[:, None], x_indices[None, :]]

    # Ensure input has channel axis
    X = image_norm.astype(np.float32)
    if X.ndim == 2:
        X = np.expand_dims(X, -1)
    Y1 = wells_resized.astype(np.int32)
    Y2 = plaque_resized.astype(np.int32)

    # Prepare targets
    prob1_full = edt_prob(Y1)
    prob2_full = edt_prob(Y2)

    y_indices_prob = np.arange(0, prob1_full.shape[0], config.grid[0])
    x_indices_prob = np.arange(0, prob1_full.shape[1], config.grid[1])
    prob1 = np.expand_dims(prob1_full[y_indices_prob[:, None], x_indices_prob[None, :]], -1)
    prob2 = np.expand_dims(prob2_full[y_indices_prob[:, None], x_indices_prob[None, :]], -1)

    dist1 = star_dist(Y1, config.n_rays, mode="cpp", grid=config.grid)
    dist2 = star_dist(Y2, config.n_rays, mode="cpp", grid=config.grid)

    return X, dist1, prob1, dist2, prob2
