import sys
import importlib
import numpy as np
from unittest.mock import MagicMock, patch

# Force-import the actual module (not the function re-exported by __init__.py)
importlib.import_module('napari_hydra.utils.process_frame')
_mod = sys.modules['napari_hydra.utils.process_frame']
MODULE = _mod.__name__

@patch(f'{MODULE}.star_dist')
@patch(f'{MODULE}.edt_prob')
@patch(f'{MODULE}.normalize')
@patch(f'{MODULE}.img_as_float32')
def test_process_frame(mock_as_float, mock_norm, mock_edt, mock_stardist):
    from napari_hydra.utils.process_frame import process_frame
    # Inputs
    img2d = np.zeros((100, 100))
    wells2d = np.zeros((100, 100), dtype=int)
    plaque2d = np.zeros((100, 100), dtype=int)
    target_width = 50
    target_height = 50
    config = MagicMock()
    config.grid = (2, 2)
    config.n_rays = 32
    
    # Mock returns
    mock_as_float.return_value = np.zeros((32, 32))
    mock_norm.return_value = np.zeros((32, 32))
    mock_edt.return_value = np.zeros((32, 32)) 
    mock_stardist.return_value = np.zeros((16, 16, 32))
    
    X, dist1, prob1, dist2, prob2 = process_frame(
        img2d, wells2d, plaque2d, target_width, target_height, config
    )
    
    # Assert output shapes
    assert X.shape == (32, 32, 1) or X.shape == (32, 32)
    
    # Verify mocked functions were called correctly
    mock_norm.assert_called_once()
    assert mock_edt.call_count == 2  # one for wells, one for plaques
    assert mock_stardist.call_count == 2
