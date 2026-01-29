import numpy as np
from unittest.mock import MagicMock, patch
from napari_hydra.utils.process_frame import process_frame

@patch('napari_hydra.utils.process_frame.star_dist')
@patch('napari_hydra.utils.process_frame.edt_prob')
@patch('napari_hydra.utils.process_frame.normalize')
@patch('napari_hydra.utils.process_frame.img_as_float32')
def test_process_frame(mock_as_float, mock_norm, mock_edt, mock_stardist):
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
    # normalize returns same shape
    mock_as_float.return_value = np.zeros((32, 32)) # transformed size
    mock_norm.return_value = np.zeros((32, 32))
    
    # edt_prob returns probability map
    mock_edt.return_value = np.zeros((32, 32)) 
    
    # star_dist returns dist map
    mock_stardist.return_value = np.zeros((16, 16, 32)) # subsampled by grid
    
    X, dist1, prob1, dist2, prob2 = process_frame(
        img2d, wells2d, plaque2d, target_width, target_height, config
    )
    
    # Assert output shapes
    # X should be normalized image with channel dim if 2D
    assert X.shape == (32, 32, 1) or X.shape == (32, 32)
    
    # Since we mocked inputs and internal funcs, mainly check calls
    mock_norm.assert_called_once()
    assert mock_edt.call_count == 2 # one for wells, one for plaques
    assert mock_stardist.call_count == 2
