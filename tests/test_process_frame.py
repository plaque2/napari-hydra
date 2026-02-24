import importlib
import numpy as np
from unittest.mock import MagicMock, patch

# Get the actual module object to use with patch.object,
# bypassing the name shadowing in __init__.py entirely.
_mod = importlib.import_module('napari_hydra.utils.process_frame')


def test_process_frame():
    with patch.object(_mod, 'img_as_float32') as mock_as_float, \
         patch.object(_mod, 'normalize') as mock_norm, \
         patch.object(_mod, 'edt_prob') as mock_edt, \
         patch.object(_mod, 'star_dist') as mock_stardist:

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

        X, dist1, prob1, dist2, prob2 = _mod.process_frame(
            img2d, wells2d, plaque2d, target_width, target_height, config
        )

        # Assert output shapes
        assert X.shape == (32, 32, 1) or X.shape == (32, 32)

        # Verify mocked functions were called correctly
        mock_norm.assert_called_once()
        assert mock_edt.call_count == 2  # one for wells, one for plaques
        assert mock_stardist.call_count == 2
