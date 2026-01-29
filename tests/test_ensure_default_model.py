import os
from unittest.mock import MagicMock, patch
from napari_hydra.utils.ensure_default_model import ensure_default_model

@patch('napari_hydra.utils.ensure_default_model.urllib.request.urlretrieve')
@patch('napari_hydra.utils.ensure_default_model.zipfile.ZipFile')
@patch('napari_hydra.utils.ensure_default_model.os.listdir')
@patch('napari_hydra.utils.ensure_default_model.os.path.isdir')
def test_ensure_default_model_download(mock_isdir, mock_listdir, mock_zipfile, mock_urlretrieve):
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
    # First call to listdir returns empty (check for existing)
    # Second call (at end) returns downloaded model
    mock_listdir.side_effect = [[], ["downloaded_model"]]
    
    # Mock zipfile extraction
    mock_zip_instance = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    
    dirs = ensure_default_model(dest_dir, zip_url)
    
    mock_urlretrieve.assert_called_once()
    mock_zip_instance.extractall.assert_called_once_with(dest_dir)
    assert "downloaded_model" in dirs

@patch('napari_hydra.utils.ensure_default_model.urllib.request.urlretrieve')
@patch('napari_hydra.utils.ensure_default_model.shutil.unpack_archive')
@patch('napari_hydra.utils.ensure_default_model.zipfile.ZipFile')
@patch('napari_hydra.utils.ensure_default_model.os.listdir')
def test_ensure_default_model_fallback(mock_listdir, mock_zipfile, mock_unpack, mock_urlretrieve):
    # Test fallback to shutil if BadZipFile
    import zipfile
    dest_dir = "/fake/models"
    zip_url = "http://fake.com/model.zip"
    
    mock_listdir.side_effect = [[], ["unpacked_model"]]
    mock_zipfile.side_effect = zipfile.BadZipFile
    
    ensure_default_model(dest_dir, zip_url)
    
    mock_unpack.assert_called_once()
