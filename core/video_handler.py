"""
Video handling module for loading AVI files and extracting first frames.
Supports recursive subfolder scanning and folder navigation.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Information about a video file."""
    path: Path
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    folder: str  # Subfolder name for display
    
    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass
class FolderInfo:
    """Information about a folder containing videos."""
    path: Path
    name: str
    video_count: int
    relative_path: str  # Path relative to root folder


class VideoHandler:
    """
    Handles loading videos from a folder and extracting first frames.
    Supports navigation between multiple AVI files across subfolders.
    """
    
    def __init__(self):
        self.video_list: List[Path] = []
        self.current_index: int = 0
        self.root_folder: Optional[Path] = None
        self.current_folder: Optional[Path] = None
        self._frame_cache: dict = {}  # Cache first frames
        
        # Folder navigation
        self.folder_list: List[FolderInfo] = []
        self.current_folder_index: int = 0
        self.videos_by_folder: Dict[str, List[Path]] = {}  # folder_path -> videos
    
    def load_folder(self, folder_path: str, recursive: bool = True) -> int:
        """
        Load all AVI files from a folder and optionally all subfolders.
        
        Args:
            folder_path: Path to root folder containing AVI files
            recursive: If True, scan all subfolders recursively
            
        Returns:
            Number of AVI files found
        """
        self.root_folder = Path(folder_path)
        
        if not self.root_folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Clear previous data
        self.video_list = []
        self.folder_list = []
        self.videos_by_folder = {}
        self._frame_cache.clear()
        
        if recursive:
            # Find all subfolders containing AVI files
            self._scan_recursive(self.root_folder)
        else:
            # Only scan the root folder
            videos = self._get_videos_in_folder(self.root_folder)
            if videos:
                self.videos_by_folder[str(self.root_folder)] = videos
                self.folder_list.append(FolderInfo(
                    path=self.root_folder,
                    name=self.root_folder.name,
                    video_count=len(videos),
                    relative_path=""
                ))
                self.video_list.extend(videos)
        
        # Sort folder list by path
        self.folder_list.sort(key=lambda f: f.relative_path)
        
        # Set current folder to first one with videos
        self.current_folder_index = 0
        self.current_index = 0
        
        if self.folder_list:
            self.current_folder = self.folder_list[0].path
            # Update video_list to only show current folder's videos
            self._update_video_list_for_current_folder()
        
        return self.get_total_video_count()
    
    def _scan_recursive(self, folder: Path, relative_path: str = ""):
        """Recursively scan folder for AVI files."""
        videos = self._get_videos_in_folder(folder)
        
        if videos:
            folder_key = str(folder)
            self.videos_by_folder[folder_key] = videos
            self.folder_list.append(FolderInfo(
                path=folder,
                name=folder.name,
                video_count=len(videos),
                relative_path=relative_path
            ))
        
        # Scan subfolders
        try:
            for subfolder in sorted(folder.iterdir()):
                if subfolder.is_dir() and not subfolder.name.startswith('.'):
                    sub_relative = f"{relative_path}/{subfolder.name}" if relative_path else subfolder.name
                    self._scan_recursive(subfolder, sub_relative)
        except PermissionError:
            pass  # Skip folders we can't access
    
    def _get_videos_in_folder(self, folder: Path) -> List[Path]:
        """Get all AVI files in a single folder (not recursive)."""
        try:
            return sorted([
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() == '.avi'
            ])
        except PermissionError:
            return []
    
    def _update_video_list_for_current_folder(self):
        """Update video_list to show only current folder's videos."""
        if self.current_folder:
            folder_key = str(self.current_folder)
            self.video_list = self.videos_by_folder.get(folder_key, [])
            self.current_index = 0
    
    def get_total_video_count(self) -> int:
        """Get total number of videos across all folders."""
        return sum(len(v) for v in self.videos_by_folder.values())
    
    def get_video_count(self) -> int:
        """Get number of videos in current folder."""
        return len(self.video_list)
    
    def get_folder_count(self) -> int:
        """Get total number of folders with videos."""
        return len(self.folder_list)
    
    def get_current_folder_index(self) -> int:
        """Get current folder index."""
        return self.current_folder_index
    
    def get_current_folder_info(self) -> Optional[FolderInfo]:
        """Get info about current folder."""
        if 0 <= self.current_folder_index < len(self.folder_list):
            return self.folder_list[self.current_folder_index]
        return None
    
    def navigate_to_folder(self, index: int) -> bool:
        """
        Navigate to a specific folder by index.
        
        Args:
            index: Folder index (0-based)
            
        Returns:
            True if navigation successful
        """
        if 0 <= index < len(self.folder_list):
            self.current_folder_index = index
            self.current_folder = self.folder_list[index].path
            self._update_video_list_for_current_folder()
            return True
        return False
    
    def next_folder(self) -> bool:
        """Navigate to next folder. Returns True if successful."""
        return self.navigate_to_folder(self.current_folder_index + 1)
    
    def previous_folder(self) -> bool:
        """Navigate to previous folder. Returns True if successful."""
        return self.navigate_to_folder(self.current_folder_index - 1)
    
    def has_next_folder(self) -> bool:
        """Check if there's a next folder."""
        return self.current_folder_index < len(self.folder_list) - 1
    
    def has_previous_folder(self) -> bool:
        """Check if there's a previous folder."""
        return self.current_folder_index > 0
    
    def get_folder_list(self) -> List[Tuple[int, str, int]]:
        """Get list of all folders with their indices and video counts."""
        return [(i, f.relative_path or f.name, f.video_count) 
                for i, f in enumerate(self.folder_list)]
    
    def get_folder_position_string(self) -> str:
        """Get current folder position as string (e.g., 'Folder 3/47')."""
        total = len(self.folder_list)
        if total == 0:
            return "Folder 0/0"
        return f"Folder {self.current_folder_index + 1}/{total}"
    
    def get_video_count(self) -> int:
        """Get total number of videos in current folder."""
        return len(self.video_list)
    
    def get_current_index(self) -> int:
        """Get current video index (0-based)."""
        return self.current_index
    
    def get_current_video_path(self) -> Optional[Path]:
        """Get path to current video."""
        if not self.video_list:
            return None
        return self.video_list[self.current_index]
    
    def get_video_info(self, video_path: Optional[Path] = None) -> Optional[VideoInfo]:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to video, or None for current video
            
        Returns:
            VideoInfo object or None if video cannot be read
        """
        if video_path is None:
            video_path = self.get_current_video_path()
        
        if video_path is None:
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        try:
            # Get folder name relative to root
            folder_name = ""
            if self.root_folder and video_path.parent != self.root_folder:
                try:
                    folder_name = str(video_path.parent.relative_to(self.root_folder))
                except ValueError:
                    folder_name = video_path.parent.name
            
            info = VideoInfo(
                path=video_path,
                filename=video_path.name,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=cap.get(cv2.CAP_PROP_FPS),
                frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                folder=folder_name
            )
            return info
        finally:
            cap.release()
    
    def get_first_frame(self, video_path: Optional[Path] = None, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Extract the first frame from a video.
        
        Args:
            video_path: Path to video, or None for current video
            use_cache: Whether to use cached frames
            
        Returns:
            First frame as RGB numpy array, or None if failed
        """
        if video_path is None:
            video_path = self.get_current_video_path()
        
        if video_path is None:
            return None
        
        # Check cache
        cache_key = str(video_path)
        if use_cache and cache_key in self._frame_cache:
            return self._frame_cache[cache_key].copy()
        
        # Read first frame
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Cache the frame
            if use_cache:
                self._frame_cache[cache_key] = frame_rgb.copy()
            
            return frame_rgb
            
        finally:
            cap.release()
    
    def navigate_to(self, index: int) -> bool:
        """
        Navigate to a specific video by index.
        
        Args:
            index: Video index (0-based)
            
        Returns:
            True if navigation successful
        """
        if 0 <= index < len(self.video_list):
            self.current_index = index
            return True
        return False
    
    def next_video(self) -> bool:
        """Navigate to next video. Returns True if successful."""
        return self.navigate_to(self.current_index + 1)
    
    def previous_video(self) -> bool:
        """Navigate to previous video. Returns True if successful."""
        return self.navigate_to(self.current_index - 1)
    
    def has_next(self) -> bool:
        """Check if there's a next video."""
        return self.current_index < len(self.video_list) - 1
    
    def has_previous(self) -> bool:
        """Check if there's a previous video."""
        return self.current_index > 0
    
    def get_video_list(self) -> List[Tuple[int, str]]:
        """Get list of all videos with their indices."""
        return [(i, p.name) for i, p in enumerate(self.video_list)]
    
    def clear_cache(self):
        """Clear the frame cache."""
        self._frame_cache.clear()
    
    def get_position_string(self) -> str:
        """Get current position as string (e.g., '3/47')."""
        total = len(self.video_list)
        if total == 0:
            return "0/0"
        return f"{self.current_index + 1}/{total}"


if __name__ == "__main__":
    # Test video handler
    handler = VideoHandler()
    
    test_folder = "/media/hedtpc/BulkStorage/LeakyGut/CPLT-001 LHK-1B QCd/Movies/34860"
    
    try:
        count = handler.load_folder(test_folder)
        print(f"Found {count} AVI files")
        
        if count > 0:
            info = handler.get_video_info()
            print(f"Current video: {info.filename}")
            print(f"Resolution: {info.resolution}")
            print(f"FPS: {info.fps}")
            
            frame = handler.get_first_frame()
            if frame is not None:
                print(f"First frame shape: {frame.shape}")
    except FileNotFoundError as e:
        print(f"Test folder not found: {e}")
