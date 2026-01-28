import numpy as np


def split_into_patches(img, target_height, target_width):
    num_blocks = int(np.ceil(img.shape[0]/target_height)), int(np.ceil(img.shape[1]/target_width))

    y_breakpoints = np.linspace(0, img.shape[0]-target_height, num=num_blocks[0], endpoint=True, dtype=int)
    x_breakpoints = np.linspace(0, img.shape[1]-target_width, num=num_blocks[1], endpoint=True, dtype=int)
    breakpoints = np.dstack(np.meshgrid(y_breakpoints, x_breakpoints)).reshape(-1, 2)

    patches = []
    for y, x in breakpoints:
        patches.append(img[y: y+target_height, x: x+target_width])

    return np.array(patches), breakpoints

def combine_patches(patches, breakpoints):
    # patches: shape = (num_patches, patch_height, patch_width, num_channels)
    patch_shape = patches.shape[1], patches.shape[2]
    result_height, result_width = np.max(breakpoints, axis=0) + patch_shape
    result = np.zeros((result_height, result_width, 1)) - np.inf
    for patch, start_coords in zip(patches, breakpoints):
        y_start, x_start = start_coords
        y_end, x_end = start_coords + patch_shape
        result[y_start: y_end, x_start: x_end] = np.maximum(result[y_start: y_end, x_start: x_end], patch)

    return result.astype(np.float32)
