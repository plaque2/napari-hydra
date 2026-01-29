import numpy as np
from .process_frame import process_frame

def prepare_training_batches(image, wells, plaque, is_stack, target_width, target_height, config):
    """
    Prepare batches for fine-tuning the model from image and label data.

    Processes single images or stacks, resizing and normalizing them, and generating
    targets (distances and probabilities) for the StarDist model.

    Args:
        image (numpy.ndarray): Training image data (single frame or stack).
        wells (numpy.ndarray): Ground truth well labels.
        plaque (numpy.ndarray): Ground truth plaque labels.
        is_stack (bool): Flag indicating if the input data represents a stack of frames.
        target_width (int): Target width for resizing images.
        target_height (int): Target height for resizing images.
        config (object): StarDist configuration object containing grid and n_rays.

    Returns:
        tuple: A tuple containing two elements:
            - X_train (numpy.ndarray): Training images array suitable for Keras model input.
            - Y_train (dict): Dictionary of training targets with keys 'dist1', 'prob1', 'dist2', 'prob2'.
    """
    if is_stack:
        n_frames = image.shape[0]
        # If image has more than 3 dims, select first channel if needed
        Xs, dist1s, prob1s, dist2s, prob2s = [], [], [], [], []
        for z in range(n_frames):
            # Handle 4D (t,z,y,x) or (z,y,x,c) etc
            img2d = image[z]
            wells2d = wells[z]
            plaque2d = plaque[z]
            X, dist1, prob1, dist2, prob2 = process_frame(
                img2d, wells2d, plaque2d, 
                target_width, target_height, config
            )
            Xs.append(X)
            dist1s.append(dist1)
            prob1s.append(prob1)
            dist2s.append(dist2)
            prob2s.append(prob2)
        X_train = np.stack(Xs, axis=0)
        Y_train = {
            'dist1': np.stack(dist1s, axis=0),
            'prob1': np.stack(prob1s, axis=0),
            'dist2': np.stack(dist2s, axis=0),
            'prob2': np.stack(prob2s, axis=0),
        }
    else:
        if image.ndim == 3 and image.shape[-1] in (1, 3):
            img2d = image
            wells2d = wells
            plaque2d = plaque
        elif image.ndim == 2:
            img2d = image
            wells2d = wells
            plaque2d = plaque
        else:
            # If image is 3D with shape (1, h, w) or similar, squeeze
            img2d = np.squeeze(image)
            wells2d = np.squeeze(wells)
            plaque2d = np.squeeze(plaque)
        X, dist1, prob1, dist2, prob2 = process_frame(img2d, wells2d, plaque2d)
        X_train = np.expand_dims(X, 0)
        Y_train = {
            'dist1': np.expand_dims(dist1, 0),
            'prob1': np.expand_dims(prob1, 0),
            'dist2': np.expand_dims(dist2, 0),
            'prob2': np.expand_dims(prob2, 0),
        }
    return X_train, Y_train
