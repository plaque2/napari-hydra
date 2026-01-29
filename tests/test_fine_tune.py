import numpy as np
from unittest.mock import MagicMock, patch
from napari_hydra.utils.fine_tune import prepare_training_batches

@patch('napari_hydra.utils.fine_tune.process_frame')
def test_prepare_training_batches_single_image(mock_process):
    # Setup
    img = np.zeros((100, 100))
    wells = np.zeros((100, 100))
    plaque = np.zeros((100, 100))
    is_stack = False
    
    # Mock process_frame return
    # X, dist1, prob1, dist2, prob2
    mock_process.return_value = (
        np.zeros((64, 64, 1)), # X
        np.zeros((32, 32, 32)), # dist1
        np.zeros((32, 32, 1)), # prob1
        np.zeros((32, 32, 32)), # dist2
        np.zeros((32, 32, 1))  # prob2
    )
    
    X_train, Y_train = prepare_training_batches(
        img, wells, plaque, is_stack, 100, 100, MagicMock()
    )
    
    # Expect expanded dims for single image
    assert X_train.shape == (1, 64, 64, 1)
    assert Y_train['dist1'].shape == (1, 32, 32, 32)
    mock_process.assert_called_once()

@patch('napari_hydra.utils.fine_tune.process_frame')
def test_prepare_training_batches_stack(mock_process):
    # Setup stack of 2 frames
    img = np.zeros((2, 100, 100))
    wells = np.zeros((2, 100, 100))
    plaque = np.zeros((2, 100, 100))
    is_stack = True
    
    mock_process.return_value = (
        np.zeros((64, 64, 1)), # X
        np.zeros((32, 32, 32)), # dist1
        np.zeros((32, 32, 1)), # prob1
        np.zeros((32, 32, 32)), # dist2
        np.zeros((32, 32, 1))  # prob2
    )
    
    X_train, Y_train = prepare_training_batches(
        img, wells, plaque, is_stack, 100, 100, MagicMock()
    )
    
    # Expect stack dimension
    assert X_train.shape == (2, 64, 64, 1)
    assert Y_train['dist1'].shape == (2, 32, 32, 32)
    assert mock_process.call_count == 2
