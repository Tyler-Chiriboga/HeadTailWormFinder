"""
Model weight downloader with progress feedback.
Downloads SAM weights if not present.
"""
import os
import requests
from pathlib import Path
from typing import Callable, Optional


def download_file_with_progress(
    url: str,
    destination: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> bool:
    """
    Download a file from URL with progress tracking.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        progress_callback: Optional callback function(downloaded_bytes, total_bytes)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directories if needed
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Start download with streaming
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress
        downloaded = 0
        chunk_size = 8192
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)
        
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        # Clean up partial download
        if destination.exists():
            destination.unlink()
        return False


def check_sam_weights(weights_path: Path) -> bool:
    """Check if SAM weights exist and are valid."""
    if not weights_path.exists():
        return False
    
    # Check file size (vit_b should be ~375MB)
    size_mb = weights_path.stat().st_size / (1024 * 1024)
    if size_mb < 300:  # Too small, likely corrupted
        return False
    
    return True


def download_sam_weights(
    url: str,
    destination: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> bool:
    """
    Download SAM weights if not present.
    
    Args:
        url: URL to download SAM weights from
        destination: Path to save the weights
        progress_callback: Optional callback for progress updates
        
    Returns:
        True if weights are available (downloaded or already present)
    """
    print(f"Downloading SAM weights from {url}...")
    print(f"Saving to: {destination}")
    
    return download_file_with_progress(url, destination, progress_callback)


class ModelDownloader:
    """Class to manage model downloads with Qt signals support."""
    
    def __init__(self):
        self.is_downloading = False
        self.current_progress = 0
        self.total_size = 0
    
    def download_sam(
        self,
        url: str,
        destination: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """Download SAM model weights."""
        if self.is_downloading:
            return False
        
        self.is_downloading = True
        try:
            result = download_sam_weights(url, destination, progress_callback)
            return result
        finally:
            self.is_downloading = False
    
    def get_download_progress(self) -> tuple:
        """Get current download progress."""
        return (self.current_progress, self.total_size)


if __name__ == "__main__":
    # Test download
    from config.settings import SAM_DOWNLOAD_URL, SAM_CHECKPOINT_PATH
    
    def print_progress(downloaded, total):
        if total > 0:
            percent = (downloaded / total) * 100
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
    
    success = download_sam_weights(SAM_DOWNLOAD_URL, SAM_CHECKPOINT_PATH, print_progress)
    print(f"\nDownload successful: {success}")
