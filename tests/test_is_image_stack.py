import numpy as np
from napari_hydra.utils.is_image_stack import is_image_stack

def test_is_image_stack_2d():
    # 2D image (h, w) -> False
    img = np.zeros((100, 100))
    assert not is_image_stack(img)

def test_is_image_stack_3d_stack():
    # (frames, h, w) where frames > 50 is unlikely for channels
    # But heuristic says: consecutive spatial dims >= 50
    # (100, 100, 10) -> (h, w, c) -> False (if last dim <= 50)
    
    # Let's test the specific heuristics:
    # 1. (h, w, c) with small c: dims[0]>=50, dims[1]>=50, dims[2]<=50 -> False
    img = np.zeros((100, 100, 3))
    assert not is_image_stack(img)

    # 2. (frames, h, w) with frames > 50: dims[1]>=50, dims[2]>=50, dims[0]>50
    # ambiguous case (60, 60, 60). Code heuristic prioritizes dims[0], dims[1] being large spatial.
    # If dims[2] is also large, it assumes stack (frames) rather than channels.
    img = np.zeros((60, 60, 60))
    assert is_image_stack(img)

    # 3. (frames, h, w) with typical stack: (10, 100, 100)
    # dims[1]>=50, dims[2]>=50, dims[0]<=50 and > 1 -> True
    img = np.zeros((10, 100, 100))
    assert is_image_stack(img)

def test_is_image_stack_4d():
    # (t, z, y, x) -> True
    img = np.zeros((5, 1, 100, 100))
    assert is_image_stack(img)
