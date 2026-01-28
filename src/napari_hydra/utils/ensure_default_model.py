import os
import urllib
import tempfile
import zipfile
import shutil
from napari.utils.notifications import show_info

def ensure_default_model(dest_dir, zip_url):
    # Check for any directory in models folder
    dirs = [name for name in os.listdir(dest_dir) if os.path.isdir(os.path.join(dest_dir, name))]
    if dirs:
        return dirs
    # Download into a temp file then extract
    try:
        show_info("No models found locally — attempting to download default model (this may take a while)...")
    except Exception:
        pass
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        # download
        urllib.request.urlretrieve(zip_url, tmp_path)
        # extract
        try:
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                zf.extractall(dest_dir)
        except zipfile.BadZipFile:
            # fallback to shutil.unpack_archive which may handle other formats
            shutil.unpack_archive(tmp_path, dest_dir)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception as e:
        try:
            show_info(f"Failed to download default model: {e}")
        except Exception:
            pass
    # return any directories found after attempted download
    return [name for name in os.listdir(dest_dir) if os.path.isdir(os.path.join(dest_dir, name))]
