"""
Annotation manager for storing and persisting worm annotations.
Supports JSON persistence for resuming work and model training data export.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import cv2


@dataclass
class WormAnnotation:
    """Annotation for a single worm in a video frame."""
    worm_id: int
    detection_box: Optional[Tuple[float, float, float, float]] = None  # YOLO detection
    head_box: Optional[Tuple[float, float, float, float]] = None       # User head annotation
    tail_box: Optional[Tuple[float, float, float, float]] = None       # User tail annotation
    head_line: Optional[Tuple[float, float, float, float]] = None      # Head line (x1,y1,x2,y2)
    tail_line: Optional[Tuple[float, float, float, float]] = None      # Tail line (x1,y1,x2,y2)
    segmentation_mask_path: Optional[str] = None                       # Path to worm mask
    head_mask_path: Optional[str] = None                               # Path to head mask
    tail_mask_path: Optional[str] = None                               # Path to tail mask
    confidence: float = 0.0
    notes: str = ""
    censored: bool = False                                             # Exclude from analysis
    health_score: Optional[float] = None                               # Health CNN score (0-1)
    health_classification: Optional[str] = None                        # "Healthy" or "Leaky"
    health_class: Optional[str] = None                                 # A, B, C, D, E (A=0, B=0.25, C=0.5, D=0.75, E=1)
    modified_by: Optional[str] = None                                  # Username who last modified
    modified_at: Optional[str] = None                                  # ISO timestamp of last modification
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'worm_id': self.worm_id,
            'detection_box': list(self.detection_box) if self.detection_box else None,
            'head_box': list(self.head_box) if self.head_box else None,
            'tail_box': list(self.tail_box) if self.tail_box else None,
            'head_line': list(self.head_line) if self.head_line else None,
            'tail_line': list(self.tail_line) if self.tail_line else None,
            'segmentation_mask_path': self.segmentation_mask_path,
            'head_mask_path': self.head_mask_path,
            'tail_mask_path': self.tail_mask_path,
            'confidence': self.confidence,
            'notes': self.notes,
            'censored': self.censored,
            'health_score': self.health_score,
            'health_classification': self.health_classification,
            'health_class': self.health_class,
            'modified_by': self.modified_by,
            'modified_at': self.modified_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'WormAnnotation':
        """Create from dictionary."""
        return cls(
            worm_id=data['worm_id'],
            detection_box=tuple(data['detection_box']) if data.get('detection_box') else None,
            head_box=tuple(data['head_box']) if data.get('head_box') else None,
            tail_box=tuple(data['tail_box']) if data.get('tail_box') else None,
            head_line=tuple(data['head_line']) if data.get('head_line') else None,
            tail_line=tuple(data['tail_line']) if data.get('tail_line') else None,
            segmentation_mask_path=data.get('segmentation_mask_path'),
            head_mask_path=data.get('head_mask_path'),
            tail_mask_path=data.get('tail_mask_path'),
            confidence=data.get('confidence', 0.0),
            notes=data.get('notes', ''),
            censored=data.get('censored', False),
            health_score=data.get('health_score'),
            health_classification=data.get('health_classification'),
            health_class=data.get('health_class'),
            modified_by=data.get('modified_by'),
            modified_at=data.get('modified_at')
        )
    
    def is_complete(self) -> bool:
        """Check if annotation has both head and tail (box or line)."""
        has_head = self.head_box is not None or self.head_line is not None
        has_tail = self.tail_box is not None or self.tail_line is not None
        return has_head and has_tail


@dataclass
class VideoAnnotations:
    """All annotations for a single video."""
    video_path: str
    video_filename: str
    frame_width: int = 0
    frame_height: int = 0
    annotations: Dict[int, WormAnnotation] = None  # worm_id -> annotation
    created_at: str = ""
    modified_at: str = ""
    qc_complete: bool = False  # Whether QC has been completed for this video
    qc_timestamp: str = ""  # When QC was completed
    
    def __post_init__(self):
        if self.annotations is None:
            self.annotations = {}
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.modified_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'video_path': self.video_path,
            'video_filename': self.video_filename,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'annotations': {
                str(k): v.to_dict() for k, v in self.annotations.items()
            },
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'qc_complete': self.qc_complete,
            'qc_timestamp': self.qc_timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VideoAnnotations':
        """Create from dictionary."""
        annotations = {}
        # Renumber worms sequentially starting from 0
        for new_id, (old_id, v) in enumerate(sorted(data.get('annotations', {}).items(), key=lambda x: int(x[0]))):
            worm = WormAnnotation.from_dict(v)
            worm.worm_id = new_id  # Update the worm_id to the new sequential ID
            annotations[new_id] = worm
        
        return cls(
            video_path=data['video_path'],
            video_filename=data['video_filename'],
            frame_width=data.get('frame_width', 0),
            frame_height=data.get('frame_height', 0),
            annotations=annotations,
            created_at=data.get('created_at', ''),
            modified_at=data.get('modified_at', ''),
            qc_complete=data.get('qc_complete', False),
            qc_timestamp=data.get('qc_timestamp', '')
        )


class AnnotationManager:
    """
    Manages annotations across all videos in a folder.
    Handles persistence to JSON and mask images for training data.
    """
    
    def __init__(self):
        self.folder_path: Optional[Path] = None
        self.annotations: Dict[str, VideoAnnotations] = {}  # video_path -> VideoAnnotations
        self._next_worm_id: Dict[str, int] = {}  # Track next worm ID per video
        self._unsaved_changes: bool = False
    
    def set_folder(self, folder_path: str):
        """
        Set the working folder and load existing annotations.
        
        Args:
            folder_path: Path to the folder containing videos
        """
        self.folder_path = Path(folder_path)
        self.annotations.clear()
        self._next_worm_id.clear()
        
        # Load existing annotations if present
        self.load_annotations()
    
    def _get_annotations_file_path(self) -> Path:
        """Get path to the annotations JSON file."""
        return self.folder_path / "worm_annotations.json"
    
    def _get_masks_folder(self) -> Path:
        """Get path to the folder for segmentation masks."""
        return self.folder_path / "segmentation_masks"
    
    def _get_crops_folder(self) -> Path:
        """Get path to the folder for cropped images."""
        return self.folder_path / "cropped_worms"
    
    def load_annotations(self) -> bool:
        """
        Load annotations from JSON file.
        
        Returns:
            True if loaded successfully, False if no file exists
        """
        json_path = self._get_annotations_file_path()
        
        if not json_path.exists():
            return False
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            for video_path, video_data in data.get('videos', {}).items():
                self.annotations[video_path] = VideoAnnotations.from_dict(video_data)
                
                # Note: from_dict renumbers worms starting from 0, so _next_worm_id 
                # is no longer needed (we find lowest available ID dynamically)
            
            self._unsaved_changes = False
            print(f"Loaded annotations for {len(self.annotations)} videos")
            return True
            
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return False
    
    def save_annotations(self) -> bool:
        """
        Save annotations to JSON file.
        
        Returns:
            True if saved successfully
        """
        if self.folder_path is None:
            return False
        
        json_path = self._get_annotations_file_path()
        
        try:
            data = {
                'folder_path': str(self.folder_path),
                'saved_at': datetime.now().isoformat(),
                'videos': {
                    k: v.to_dict() for k, v in self.annotations.items()
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self._unsaved_changes = False
            print(f"Saved annotations to {json_path}")
            return True
            
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False
    
    def get_or_create_video_annotations(
        self, 
        video_path: str,
        frame_width: int = 0,
        frame_height: int = 0
    ) -> VideoAnnotations:
        """
        Get annotations for a video, creating if needed.
        
        Args:
            video_path: Path to the video file
            frame_width: Width of video frame
            frame_height: Height of video frame
            
        Returns:
            VideoAnnotations object
        """
        if video_path not in self.annotations:
            self.annotations[video_path] = VideoAnnotations(
                video_path=video_path,
                video_filename=Path(video_path).name,
                frame_width=frame_width,
                frame_height=frame_height
            )
            self._next_worm_id[video_path] = 1
        
        return self.annotations[video_path]
    
    def add_worm_annotation(
        self,
        video_path: str,
        detection_box: Optional[Tuple[float, float, float, float]] = None,
        confidence: float = 0.0
    ) -> WormAnnotation:
        """
        Add a new worm annotation.
        
        Args:
            video_path: Path to the video file
            detection_box: YOLO detection bounding box
            confidence: Detection confidence
            
        Returns:
            New WormAnnotation object
        """
        video_annot = self.get_or_create_video_annotations(video_path)
        
        # Find the lowest available worm ID (starting from 0)
        existing_ids = set(video_annot.annotations.keys())
        worm_id = 0
        while worm_id in existing_ids:
            worm_id += 1
        
        annotation = WormAnnotation(
            worm_id=worm_id,
            detection_box=detection_box,
            confidence=confidence
        )
        
        video_annot.annotations[worm_id] = annotation
        video_annot.modified_at = datetime.now().isoformat()
        self._unsaved_changes = True
        
        return annotation
    
    def set_head_box(
        self,
        video_path: str,
        worm_id: int,
        head_box: Tuple[float, float, float, float],
        modified_by: Optional[str] = None
    ) -> bool:
        """Set head bounding box for a worm."""
        if video_path not in self.annotations:
            return False
        
        video_annot = self.annotations[video_path]
        if worm_id not in video_annot.annotations:
            return False
        
        annot = video_annot.annotations[worm_id]
        annot.head_box = head_box
        if modified_by:
            annot.modified_by = modified_by
            annot.modified_at = datetime.now().isoformat()
        video_annot.modified_at = datetime.now().isoformat()
        self._unsaved_changes = True
        return True
    
    def set_tail_box(
        self,
        video_path: str,
        worm_id: int,
        tail_box: Tuple[float, float, float, float],
        modified_by: Optional[str] = None
    ) -> bool:
        """Set tail bounding box for a worm."""
        if video_path not in self.annotations:
            return False
        
        video_annot = self.annotations[video_path]
        if worm_id not in video_annot.annotations:
            return False
        
        annot = video_annot.annotations[worm_id]
        annot.tail_box = tail_box
        if modified_by:
            annot.modified_by = modified_by
            annot.modified_at = datetime.now().isoformat()
        video_annot.modified_at = datetime.now().isoformat()
        self._unsaved_changes = True
        return True
    
    def set_detection_box(
        self,
        video_path: str,
        worm_id: int,
        detection_box: Tuple[float, float, float, float],
        modified_by: Optional[str] = None
    ) -> bool:
        """Set detection bounding box for a worm."""
        if video_path not in self.annotations:
            return False
        
        video_annot = self.annotations[video_path]
        if worm_id not in video_annot.annotations:
            return False
        
        annot = video_annot.annotations[worm_id]
        annot.detection_box = detection_box
        if modified_by:
            annot.modified_by = modified_by
            annot.modified_at = datetime.now().isoformat()
        video_annot.modified_at = datetime.now().isoformat()
        self._unsaved_changes = True
        return True
    
    def set_head_line(
        self,
        video_path: str,
        worm_id: int,
        head_line: Tuple[float, float, float, float],
        modified_by: Optional[str] = None
    ) -> bool:
        """Set head line annotation for a worm."""
        if video_path not in self.annotations:
            return False
        
        video_annot = self.annotations[video_path]
        if worm_id not in video_annot.annotations:
            return False
        
        annot = video_annot.annotations[worm_id]
        annot.head_line = head_line
        if modified_by:
            annot.modified_by = modified_by
            annot.modified_at = datetime.now().isoformat()
        video_annot.modified_at = datetime.now().isoformat()
        self._unsaved_changes = True
        return True
    
    def set_tail_line(
        self,
        video_path: str,
        worm_id: int,
        tail_line: Tuple[float, float, float, float],
        modified_by: Optional[str] = None
    ) -> bool:
        """Set tail line annotation for a worm."""
        if video_path not in self.annotations:
            return False
        
        video_annot = self.annotations[video_path]
        if worm_id not in video_annot.annotations:
            return False
        
        annot = video_annot.annotations[worm_id]
        annot.tail_line = tail_line
        if modified_by:
            annot.modified_by = modified_by
            annot.modified_at = datetime.now().isoformat()
        video_annot.modified_at = datetime.now().isoformat()
        self._unsaved_changes = True
        return True
    
    def set_worm_censored(
        self,
        video_path: str,
        worm_id: int,
        censored: bool,
        modified_by: Optional[str] = None
    ) -> bool:
        """Set censored status for a worm (exclude from analysis)."""
        if video_path not in self.annotations:
            return False
        
        video_annot = self.annotations[video_path]
        if worm_id not in video_annot.annotations:
            return False
        
        annot = video_annot.annotations[worm_id]
        annot.censored = censored
        if modified_by:
            annot.modified_by = modified_by
            annot.modified_at = datetime.now().isoformat()
        video_annot.modified_at = datetime.now().isoformat()
        self._unsaved_changes = True
        return True
    
    def set_video_qc_complete(
        self,
        video_path: str,
        qc_complete: bool
    ) -> bool:
        """Set QC complete status for a video."""
        if video_path not in self.annotations:
            return False
        
        video_annot = self.annotations[video_path]
        video_annot.qc_complete = qc_complete
        video_annot.qc_timestamp = datetime.now().isoformat() if qc_complete else ""
        video_annot.modified_at = datetime.now().isoformat()
        self._unsaved_changes = True
        return True
    
    def get_video_qc_status(self, video_path: str) -> Tuple[bool, str]:
        """Get QC complete status for a video. Returns (qc_complete, qc_timestamp)."""
        if video_path not in self.annotations:
            return False, ""
        video_annot = self.annotations[video_path]
        return video_annot.qc_complete, video_annot.qc_timestamp
    
    def delete_worm_annotation(self, video_path: str, worm_id: int) -> bool:
        """Delete a worm annotation."""
        if video_path not in self.annotations:
            return False
        
        video_annot = self.annotations[video_path]
        if worm_id not in video_annot.annotations:
            return False
        
        del video_annot.annotations[worm_id]
        video_annot.modified_at = datetime.now().isoformat()
        self._unsaved_changes = True
        return True
    
    def get_worm_annotation(
        self, 
        video_path: str, 
        worm_id: int
    ) -> Optional[WormAnnotation]:
        """Get a specific worm annotation."""
        if video_path not in self.annotations:
            return None
        return self.annotations[video_path].annotations.get(worm_id)
    
    def get_all_worm_annotations(self, video_path: str) -> List[WormAnnotation]:
        """Get all worm annotations for a video."""
        if video_path not in self.annotations:
            return []
        return list(self.annotations[video_path].annotations.values())
    
    def save_segmentation_mask(
        self,
        video_path: str,
        worm_id: int,
        mask: np.ndarray,
        mask_type: str = "worm"
    ) -> Optional[str]:
        """
        Save a segmentation mask for training data.
        
        Args:
            video_path: Path to the video
            worm_id: Worm ID
            mask: Binary mask array
            mask_type: Type of mask - "worm", "head", or "tail"
            
        Returns:
            Path to saved mask file, or None if failed
        """
        masks_folder = self._get_masks_folder()
        masks_folder.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on mask type
        video_name = Path(video_path).stem
        if mask_type == "worm":
            mask_filename = f"{video_name}_worm{worm_id}_mask.png"
        else:
            mask_filename = f"{video_name}_worm{worm_id}_{mask_type}_mask.png"
        mask_path = masks_folder / mask_filename
        
        try:
            # Save mask as grayscale image
            mask_uint8 = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_uint8)
            
            # Update annotation
            if video_path in self.annotations:
                annot = self.annotations[video_path].annotations.get(worm_id)
                if annot:
                    if mask_type == "worm":
                        annot.segmentation_mask_path = str(mask_path)
                    elif mask_type == "head":
                        annot.head_mask_path = str(mask_path)
                    elif mask_type == "tail":
                        annot.tail_mask_path = str(mask_path)
                    self._unsaved_changes = True
            
            return str(mask_path)
            
        except Exception as e:
            print(f"Error saving mask: {e}")
            return None
    
    def save_cropped_regions(
        self,
        video_path: str,
        worm_id: int,
        frame: np.ndarray,
        annotation: WormAnnotation
    ) -> Dict[str, str]:
        """
        Save cropped regions (head, tail, full worm) for training.
        
        Args:
            video_path: Path to the video
            worm_id: Worm ID
            frame: Full frame image (RGB)
            annotation: WormAnnotation with boxes
            
        Returns:
            Dictionary of region_type -> saved path
        """
        crops_folder = self._get_crops_folder()
        crops_folder.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        saved_paths = {}
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        def save_crop(box: Tuple[float, float, float, float], suffix: str) -> Optional[str]:
            if box is None:
                return None
            
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = frame.shape[:2]
            
            # Clip to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            crop = frame_bgr[y1:y2, x1:x2]
            filename = f"{video_name}_worm{worm_id}_{suffix}.png"
            path = crops_folder / filename
            
            try:
                cv2.imwrite(str(path), crop)
                return str(path)
            except:
                return None
        
        # Save head crop
        if annotation.head_box:
            path = save_crop(annotation.head_box, "head")
            if path:
                saved_paths['head'] = path
        
        # Save tail crop
        if annotation.tail_box:
            path = save_crop(annotation.tail_box, "tail")
            if path:
                saved_paths['tail'] = path
        
        # Save full detection crop
        if annotation.detection_box:
            path = save_crop(annotation.detection_box, "full")
            if path:
                saved_paths['full'] = path
        
        return saved_paths
    
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self._unsaved_changes
    
    def get_statistics(self) -> dict:
        """Get annotation statistics."""
        total_videos = len(self.annotations)
        total_worms = sum(
            len(v.annotations) for v in self.annotations.values()
        )
        complete_annotations = sum(
            sum(1 for a in v.annotations.values() if a.is_complete())
            for v in self.annotations.values()
        )
        
        return {
            'total_videos': total_videos,
            'total_worms': total_worms,
            'complete_annotations': complete_annotations,
            'incomplete_annotations': total_worms - complete_annotations
        }


if __name__ == "__main__":
    # Test annotation manager
    manager = AnnotationManager()
    
    # Test with dummy data
    test_video = "/test/video.avi"
    
    annot = manager.add_worm_annotation(
        test_video,
        detection_box=(100, 100, 200, 200),
        confidence=0.95
    )
    
    manager.set_head_box(test_video, annot.worm_id, (100, 100, 130, 130))
    manager.set_tail_box(test_video, annot.worm_id, (170, 170, 200, 200))
    
    print(f"Annotation: {annot}")
    print(f"Is complete: {manager.get_worm_annotation(test_video, annot.worm_id).is_complete()}")
    print(f"Statistics: {manager.get_statistics()}")
