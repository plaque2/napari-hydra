import importlib
from unittest.mock import MagicMock, patch

# Get the actual module object to use with patch.object,
# bypassing the name shadowing in __init__.py entirely.
_mod = importlib.import_module('napari_hydra.utils.ensure_default_model')


def test_ensure_default_model_download():
    with patch.object(_mod.os.path, 'isdir') as mock_isdir, \
         patch.object(_mod.os, 'listdir') as mock_listdir, \
         patch.object(_mod.zipfile, 'ZipFile') as mock_zipfile, \
         patch.object(_mod.urllib.request, 'urlretrieve') as mock_urlretrieve:

        # Setup
        dest_dir = "/fake/models"
        zip_url = "http://fake.com/model.zip"

        # Case 1: Models already exist
        mock_listdir.return_value = ["existing_model"]
        mock_isdir.return_value = True

        dirs = _mod.ensure_default_model(dest_dir, zip_url)
        assert "existing_model" in dirs
        mock_urlretrieve.assert_not_called()

        # Case 2: No models, download required
        mock_listdir.side_effect = [[], ["downloaded_model"]]

        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

        dirs = _mod.ensure_default_model(dest_dir, zip_url)

        mock_urlretrieve.assert_called_once()
        mock_zip_instance.extractall.assert_called_once_with(dest_dir)
        assert "downloaded_model" in dirs


def test_ensure_default_model_fallback():
    import zipfile

    with patch.object(_mod.os, 'listdir') as mock_listdir, \
         patch.object(_mod.zipfile, 'ZipFile') as mock_zipfile, \
         patch.object(_mod.shutil, 'unpack_archive') as mock_unpack, \
         patch.object(_mod.urllib.request, 'urlretrieve') as mock_urlretrieve:

        dest_dir = "/fake/models"
        zip_url = "http://fake.com/model.zip"

        mock_listdir.side_effect = [[], ["unpacked_model"]]
        mock_zipfile.side_effect = zipfile.BadZipFile

        _mod.ensure_default_model(dest_dir, zip_url)

        mock_unpack.assert_called_once()
