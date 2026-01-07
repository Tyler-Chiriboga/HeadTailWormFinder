"""
SAM (Segment Anything Model) wrapper for worm segmentation.
Supports automatic model download and GPU selection.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

from app_utils.model_downloader import download_sam_weights, check_sam_weights


@dataclass
class SegmentationResult:
    """Result of a segmentation operation."""
    mask: np.ndarray  # Binary mask (H, W)
    score: float      # Confidence score
    bbox: Tuple[float, float, float, float]  # Input bounding box
    
    def to_dict(self) -> dict:
        """Convert to dictionary (mask saved separately)."""
        return {
            'score': self.score,
            'bbox': self.bbox,
            'mask_shape': self.mask.shape
        }


class SAMSegmenter:
    """
    SAM-based segmenter for worms.
    Uses bounding box prompts from YOLO or user selection.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_b",
        device: Optional[str] = None,
        download_url: Optional[str] = None
    ):
        """
        Initialize SAM segmenter.
        
        Args:
            checkpoint_path: Path to SAM checkpoint. Will download if not present.
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            device: Device to use. Auto-detect if None.
            download_url: URL to download checkpoint from
        """
        from config.settings import (
            SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE, 
            SAM_DOWNLOAD_URL, WEIGHTS_DIR, SAM_MODELS
        )
        
        self.model_type = model_type or SAM_MODEL_TYPE
        
        # Get checkpoint path for this model type
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        elif self.model_type in SAM_MODELS:
            self.checkpoint_path = WEIGHTS_DIR / SAM_MODELS[self.model_type]['checkpoint']
        else:
            self.checkpoint_path = SAM_CHECKPOINT_PATH
        
        # Get download URL for this model type
        if download_url:
            self.download_url = download_url
        elif self.model_type in SAM_MODELS:
            self.download_url = SAM_MODELS[self.model_type]['url']
        else:
            self.download_url = SAM_DOWNLOAD_URL
        
        # Get expected size for validation
        self.expected_size_mb = SAM_MODELS.get(self.model_type, {}).get('size_mb', 300)
        
        # Auto-detect device
        if device is None:
            device = self._get_device()
        self.device = device
        
        # Model components
        self.sam = None
        self.predictor = None
        self._current_image = None
        self._image_set = False
    
    def _get_device(self) -> str:
        """Get the best available device for SAM."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                # Use second GPU if available (YOLO uses first)
                return "cuda:1"
            elif num_gpus == 1:
                return "cuda:0"
        return "cpu"
    
    def ensure_model_downloaded(self, progress_callback=None) -> bool:
        """
        Ensure SAM model weights are downloaded.
        
        Args:
            progress_callback: Optional callback(downloaded, total) for progress
            
        Returns:
            True if weights are available
        """
        # Check if file exists and has reasonable size for this model
        if self.checkpoint_path.exists():
            size_mb = self.checkpoint_path.stat().st_size / (1024 * 1024)
            # Check if size is at least 90% of expected (to account for compression differences)
            min_size = self.expected_size_mb * 0.9
            if size_mb >= min_size:
                print(f"SAM weights found at {self.checkpoint_path} ({size_mb:.0f}MB)")
                return True
            else:
                print(f"SAM weights file too small ({size_mb:.0f}MB < {min_size:.0f}MB expected), re-downloading...")
                self.checkpoint_path.unlink()  # Delete corrupted file
        
        print(f"SAM weights not found at {self.checkpoint_path}")
        print(f"Downloading SAM {self.model_type} weights (~{self.expected_size_mb}MB)...")
        
        return download_sam_weights(
            self.download_url,
            self.checkpoint_path,
            progress_callback
        )
    
    def load_model(self, progress_callback=None) -> bool:
        """
        Load the SAM model.
        
        Args:
            progress_callback: Optional callback for download progress
            
        Returns:
            True if model loaded successfully
        """
        # Ensure weights are downloaded
        if not self.ensure_model_downloaded(progress_callback):
            raise RuntimeError("Failed to download SAM weights")
        
        print(f"Loading SAM model ({self.model_type}) from {self.checkpoint_path}")
        print(f"Using device: {self.device}")
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            # Load model
            self.sam = sam_model_registry[self.model_type](
                checkpoint=str(self.checkpoint_path)
            )
            self.sam.to(device=self.device)
            
            # Create predictor
            self.predictor = SamPredictor(self.sam)
            
            print("SAM model loaded successfully")
            return True
            
        except ImportError:
            print("segment_anything not installed. Install with:")
            print("pip install git+https://github.com/facebookresearch/segment-anything.git")
            raise
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.predictor is not None
    
    def set_image(self, image: np.ndarray):
        """
        Set the image for segmentation.
        Must be called before segment().
        
        Args:
            image: RGB image as numpy array (H, W, C)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.predictor.set_image(image)
        self._current_image = image
        self._image_set = True
    
    def _keep_largest_contour(self, mask: np.ndarray) -> np.ndarray:
        """
        Filter a binary mask to keep only the largest connected component.
        
        Args:
            mask: Binary mask array (H, W)
            
        Returns:
            Filtered mask with only the largest contour
        """
        import cv2
        
        # Ensure mask is uint8
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        # Find all contours
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return mask
        
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a new mask with only the largest contour
        filtered_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(filtered_mask, [largest_contour], -1, 1, cv2.FILLED)
        
        return filtered_mask
    
    def segment(
        self,
        bbox: Tuple[float, float, float, float],
        multimask_output: bool = False,
        keep_largest_only: bool = True
    ) -> Optional[SegmentationResult]:
        """
        Segment a region using a bounding box prompt.
        
        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            multimask_output: Whether to output multiple masks
            keep_largest_only: If True, keep only the largest contour in the mask
            
        Returns:
            SegmentationResult or None if failed
        """
        if not self._image_set:
            raise RuntimeError("Image not set. Call set_image() first.")
        
        # Convert bbox to numpy array format expected by SAM
        box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
        
        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],  # Add batch dimension
                multimask_output=multimask_output
            )
            
            # Get best mask
            if len(masks) > 0:
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
                best_score = float(scores[best_idx])
                
                # Filter to keep only the largest contour
                if keep_largest_only:
                    best_mask = self._keep_largest_contour(best_mask)
                
                return SegmentationResult(
                    mask=best_mask,
                    score=best_score,
                    bbox=bbox
                )
            
            return None
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            return None
    
    def segment_multiple(
        self,
        bboxes: List[Tuple[float, float, float, float]]
    ) -> List[Optional[SegmentationResult]]:
        """
        Segment multiple regions.
        
        Args:
            bboxes: List of bounding boxes
            
        Returns:
            List of SegmentationResults (None for failed segmentations)
        """
        results = []
        for bbox in bboxes:
            result = self.segment(bbox)
            results.append(result)
        return results
    
    def get_device(self) -> str:
        """Get current device."""
        return self.device
    
    def clear_image(self):
        """Clear the current image to free memory."""
        self._current_image = None
        self._image_set = False
        if self.predictor is not None:
            self.predictor.reset_image()
    
    @staticmethod
    def get_available_model_types() -> List[str]:
        """Get list of available SAM model types."""
        return ["vit_h", "vit_l", "vit_b"]
    
    @staticmethod
    def get_model_info(model_type: str) -> dict:
        """Get information about a model type."""
        info = {
            "vit_h": {
                "name": "ViT-H (Huge)",
                "size_mb": 2564,
                "description": "Most accurate, slowest"
            },
            "vit_l": {
                "name": "ViT-L (Large)", 
                "size_mb": 1249,
                "description": "Good balance"
            },
            "vit_b": {
                "name": "ViT-B (Base)",
                "size_mb": 375,
                "description": "Fastest, good accuracy"
            }
        }
        return info.get(model_type, {})


def mask_to_polygon(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Convert binary mask to polygon contours.
    
    Args:
        mask: Binary mask (H, W)
        
    Returns:
        List of contours, each as list of (x, y) points
    """
    import cv2
    
    # Ensure mask is uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(
        mask_uint8, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Convert to list of point tuples
    polygons = []
    for contour in contours:
        points = [(int(p[0][0]), int(p[0][1])) for p in contour]
        if len(points) >= 3:
            polygons.append(points)
    
    return polygons


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Get bounding box from binary mask.
    
    Args:
        mask: Binary mask (H, W)
        
    Returns:
        Bounding box as (x1, y1, x2, y2) or None if mask is empty
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    return (int(x1), int(y1), int(x2), int(y2))


if __name__ == "__main__":
    # Test SAM segmenter
    segmenter = SAMSegmenter()
    
    print(f"Available model types: {segmenter.get_available_model_types()}")
    print(f"Model info: {segmenter.get_model_info('vit_b')}")
    print(f"Device: {segmenter._get_device()}")
