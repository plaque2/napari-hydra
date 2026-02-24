import os
import sys
import importlib
from unittest.mock import MagicMock, patch

# Force-import the actual module (not the function re-exported by __init__.py)
importlib.import_module('napari_hydra.utils.ensure_default_model')
_mod = sys.modules['napari_hydra.utils.ensure_default_model']
MODULE = _mod.__name__

@patch(f'{MODULE}.urllib.request.urlretrieve')
@patch(f'{MODULE}.zipfile.ZipFile')
@patch(f'{MODULE}.os.listdir')
@patch(f'{MODULE}.os.path.isdir')
def test_ensure_default_model_download(mock_isdir, mock_listdir, mock_zipfile, mock_urlretrieve):
    from napari_hydra.utils.ensure_default_model import ensure_default_model
    # Setup
    dest_dir = "/fake/models"
    zip_url = "http://fake.com/model.zip"
    
    # Case 1: Models already exist
    mock_listdir.return_value = ["existing_model"]
    mock_isdir.return_value = True
    
    dirs = ensure_default_model(dest_dir, zip_url)
    assert "existing_model" in dirs
    mock_urlretrieve.assert_not_called()
    
    # Case 2: No models, download required
    mock_listdir.side_effect = [[], ["downloaded_model"]]
    
    mock_zip_instance = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    
    dirs = ensure_default_model(dest_dir, zip_url)
    
    mock_urlretrieve.assert_called_once()
    mock_zip_instance.extractall.assert_called_once_with(dest_dir)
    assert "downloaded_model" in dirs

@patch(f'{MODULE}.urllib.request.urlretrieve')
@patch(f'{MODULE}.shutil.unpack_archive')
@patch(f'{MODULE}.zipfile.ZipFile')
@patch(f'{MODULE}.os.listdir')
def test_ensure_default_model_fallback(mock_listdir, mock_zipfile, mock_unpack, mock_urlretrieve):
    from napari_hydra.utils.ensure_default_model import ensure_default_model
    import zipfile
    dest_dir = "/fake/models"
    zip_url = "http://fake.com/model.zip"
    
    mock_listdir.side_effect = [[], ["unpacked_model"]]
    mock_zipfile.side_effect = zipfile.BadZipFile
    
    ensure_default_model(dest_dir, zip_url)
    
    mock_unpack.assert_called_once()
