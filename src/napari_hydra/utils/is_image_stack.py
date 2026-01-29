def is_image_stack(image):
    """
    Determine if an image array is a stack of frames or a single image.

    This function uses heuristics based on spatial dimensions (height, width) generally being large (>= 50 px)
    and channel/frame dimensions being small (<= 50 px) to distinguish single images from stacks.

    Args:
        image (numpy.ndarray): Input image array of varying dimensionality (2D, 3D, or 4D).

    Returns:
        bool: True if the image is detected as a stack of frames, False if it is a single image.
    """
    if image.ndim == 2:
        # (height, width) - single 2D image
        return False
    elif image.ndim == 3:
        # Could be (h, w, c), (n_frames, h, w), or other
        dims = image.shape
        
        # Check if dims[0] and dims[1] are consecutive spatial dims (both >= 50)
        if dims[0] >= 50 and dims[1] >= 50:
            # dims[2] is likely channels (1-50) or frames (1-N)
            if dims[2] <= 50:
                # Likely (h, w, c) - single image with channels
                return False
            else:
                # dims[2] > 50, likely (h, w, frames) - stack without channels
                return True
        # Check if dims[1] and dims[2] are consecutive spatial dims (both >= 50)
        elif dims[1] >= 50 and dims[2] >= 50:
            # dims[0] is likely frames or channels
            if dims[0] <= 50:
                # Likely (frames/channels, h, w)
                if dims[0] == 1:
                    return False  # Single frame
                else:
                    return True   # Multiple frames
            else:
                # All three dims >= 50, ambiguous - assume (h, w, c)
                return False
        # Check if dims[0] and dims[2] are >= 50 (less likely but possible)
        elif dims[0] >= 50 and dims[2] >= 50:
            # dims[1] is small, likely channels
            if dims[1] <= 50:
                # Likely (h, c, w) or permutation - assume single image
                return False
            else:
                return False
        else:
            # No clear spatial dims >= 50, assume single image
            return False
    elif image.ndim == 4:
        # Could be (n_frames, h, w, c) or (h, w, c, frames) or other
        dims = image.shape
        # Look for two consecutive dims >= 50 (spatial)
        for i in range(len(dims) - 1):
            if dims[i] >= 50 and dims[i+1] >= 50:
                # Found spatial dims at positions i and i+1
                # Check remaining dims: one should be frames, one should be channels
                other_dims = [dims[j] for j in range(4) if j != i and j != i+1]
                # If both other dims are small (<=50), likely frames and channels
                if all(d <= 50 for d in other_dims):
                    # At least one should be > 1 to indicate multiple frames
                    if max(other_dims) > 1:
                        return True
                    else:
                        return False
                return False
        # No consecutive spatial dims found
        return False
    else:
        # image.ndim > 4: assume stack (high-dimensional data)
        return True