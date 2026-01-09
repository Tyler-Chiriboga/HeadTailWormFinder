"""
FastAPI web application for Worm Head/Tail Annotation.
Provides REST API for video navigation, YOLO detection, SAM segmentation, and annotation management.
"""
import os
import sys
import json
import base64
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.video_handler import VideoHandler
from core.annotation_manager import AnnotationManager
from ml_models.yolo_detector import YOLODetector
from ml_models.sam_segmenter import SAMSegmenter
from ml_models.health_classifier import HealthClassifier
from web.auth import auth_manager, get_current_user, require_auth, require_admin


# Global state
class BatchProcessingState:
    """State for background batch processing."""
    def __init__(self):
        self.is_running = False
        self.should_cancel = False
        self.current_video = 0
        self.total_videos = 0
        self.current_folder = 0
        self.total_folders = 0
        self.current_video_name = ""
        self.current_folder_name = ""
        self.total_detections = 0
        self.total_segmentations = 0
        self.skipped_qc = 0
        self.skipped_detections = 0
        self.processed = 0
        self.error = None
        self.complete = False


class ExportProcessingState:
    """State for background export processing."""
    def __init__(self):
        self.is_running = False
        self.should_cancel = False
        self.current_worm = 0
        self.total_worms = 0
        self.current_video = ""
        self.masks_processed = 0
        self.error = None
        self.complete = False
        self.result_data = None  # Hold the final export bytes
    
    def reset(self):
        self.is_running = False
        self.should_cancel = False
        self.current_worm = 0
        self.total_worms = 0
        self.current_video = ""
        self.masks_processed = 0
        self.error = None
        self.complete = False
        self.result_data = None


class AppState:
    def __init__(self):
        self.video_handler: Optional[VideoHandler] = None
        self.annotation_manager: Optional[AnnotationManager] = None
        self.yolo_detector: Optional[YOLODetector] = None
        self.sam_segmenter: Optional[SAMSegmenter] = None
        self.health_classifier: Optional[HealthClassifier] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_video_path: Optional[str] = None
        self.models_loaded = False
        self.confidence_threshold = 0.25  # YOLO detection confidence threshold
        # Cache for straightened worms: {worm_id: {"image": np.array, "path": list, "video": str}}
        self.straightened_cache = {}
        # Batch processing state
        self.batch_state = BatchProcessingState()
        # Export processing state
        self.export_state = ExportProcessingState()
        
state = AppState()


# Pydantic models for API
class FolderPath(BaseModel):
    path: str

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    worm_id: Optional[int] = None
    box_type: str = "detection"  # detection, head, tail

class SegmentRequest(BaseModel):
    bbox: BoundingBox
    worm_id: int

class AnnotationUpdate(BaseModel):
    worm_id: int
    head_box: Optional[BoundingBox] = None
    tail_box: Optional[BoundingBox] = None


# Auth models
class LoginRequest(BaseModel):
    username: str
    password: str
    remember: bool = True


class ConfidenceSettings(BaseModel):
    confidence: float


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "annotator"

class ChangePasswordRequest(BaseModel):
    username: str
    new_password: str


class BrushStroke(BaseModel):
    x: float  # Can be float from canvas, will be converted to int
    y: float
    add: bool  # True = add to mask, False = remove from mask


class BrushMaskRequest(BaseModel):
    worm_id: int
    mask_type: str  # worm, head, tail
    strokes: List[BrushStroke]
    brush_size: int = 15


# Default dataset path
DEFAULT_DATASET_PATH = "/media/hedtpc/BulkStorage/LeakyGut/CPLT-001 LHK-1B QCd/Movies/"


def _load_current_video():
    """Internal: Load the current video and first frame."""
    if state.video_handler:
        video_info = state.video_handler.get_video_info()
        if video_info:
            state.current_video_path = str(video_info.path)
            # Clear straightened cache when video changes
            state.straightened_cache = {}
            frame = state.video_handler.get_first_frame()
            if frame is not None:
                # get_first_frame already returns RGB, no conversion needed
                state.current_frame = frame


async def _run_auto_detection() -> bool:
    """
    Run automatic detection and segmentation on the current frame.
    Returns True if successful, False otherwise.
    """
    if state.yolo_detector is None or state.current_frame is None:
        return False
    
    try:
        # Run detection
        detections = state.yolo_detector.detect(
            state.current_frame,
            conf_threshold=state.confidence_threshold
        )
        
        print(f"[auto_detect] Found {len(detections)} detections")
        
        if len(detections) == 0:
            return True  # Success, but no worms found
        
        if state.annotation_manager and state.current_video_path:
            # Clear any existing annotations
            if state.current_video_path in state.annotation_manager.annotations:
                state.annotation_manager.annotations[state.current_video_path].annotations.clear()
            
            # Pre-set the SAM image once for all segmentations
            if state.sam_segmenter is not None and state.sam_segmenter.is_loaded():
                print(f"[auto_detect] Setting SAM image for {len(detections)} worms")
                state.sam_segmenter.set_image(state.current_frame)
            
            segmented_count = 0
            for i, det in enumerate(detections):
                annot = state.annotation_manager.add_worm_annotation(
                    state.current_video_path,
                    detection_box=det.bbox,
                    confidence=det.confidence
                )
                
                # Auto-segment this worm
                if auto_segment_worm_no_set_image(annot.worm_id, det.bbox):
                    segmented_count += 1
            
            state.annotation_manager.save_annotations()
            print(f"[auto_detect] Complete: {len(detections)} detections, {segmented_count} segmented")
            return True
            
    except Exception as e:
        print(f"[auto_detect] Error: {e}")
        return False
    
    return False


async def _run_folder_auto_detection() -> dict:
    """
    Run automatic detection and segmentation on ALL videos in the current folder.
    Only processes videos that don't already have annotations.
    Returns dict with detection stats.
    """
    if state.yolo_detector is None or state.video_handler is None:
        return {"processed": 0, "total_detections": 0, "total_segmented": 0}
    
    video_count = state.video_handler.get_video_count()
    if video_count == 0:
        return {"processed": 0, "total_detections": 0, "total_segmented": 0}
    
    # Store current video index to restore later
    current_idx = state.video_handler.get_current_index()
    
    total_detections = 0
    total_segmented = 0
    processed = 0
    
    print(f"[folder_auto_detect] Processing {video_count} videos in folder...")
    
    for idx in range(video_count):
        state.video_handler.navigate_to(idx)
        video_info = state.video_handler.get_video_info()
        if not video_info:
            continue
        
        video_path = str(video_info.path)
        
        # Skip if already has annotations
        existing = state.annotation_manager.get_all_worm_annotations(video_path)
        if existing:
            print(f"[folder_auto_detect] Skipping {video_info.path.name} - already has {len(existing)} annotations")
            total_detections += len(existing)
            for annot in existing:
                if annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists():
                    total_segmented += 1
            continue
        
        # Get first frame
        frame = state.video_handler.get_first_frame()
        if frame is None:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        detections = state.yolo_detector.detect(
            frame_rgb,
            conf_threshold=state.confidence_threshold
        )
        
        print(f"[folder_auto_detect] {video_info.path.name}: {len(detections)} detections")
        processed += 1
        
        if len(detections) == 0:
            continue
        
        # Set SAM image once for this video
        if state.sam_segmenter is not None and state.sam_segmenter.is_loaded():
            state.sam_segmenter.set_image(frame_rgb)
        
        # Add annotations and segment
        for det in detections:
            annot = state.annotation_manager.add_worm_annotation(
                video_path,
                detection_box=det.bbox,
                confidence=det.confidence
            )
            total_detections += 1
            
            # Auto-segment and adjust detection box to match mask
            if state.sam_segmenter is not None and state.sam_segmenter.is_loaded():
                try:
                    result = state.sam_segmenter.segment(det.bbox, keep_largest_only=True)
                    if result is not None:
                        state.annotation_manager.save_segmentation_mask(
                            video_path,
                            annot.worm_id,
                            result.mask,
                            mask_type="worm"
                        )
                        total_segmented += 1
                        
                        # Adjust detection box to match mask bounding box
                        mask_binary = (result.mask > 0.5).astype(np.uint8) if result.mask.max() <= 1 else (result.mask > 128).astype(np.uint8)
                        coords = np.where(mask_binary > 0)
                        if len(coords[0]) > 0:
                            y_min, y_max = int(coords[0].min()), int(coords[0].max())
                            x_min, x_max = int(coords[1].min()), int(coords[1].max())
                            padding = 5
                            x_min = max(0, x_min - padding)
                            y_min = max(0, y_min - padding)
                            x_max = min(frame_rgb.shape[1] - 1, x_max + padding)
                            y_max = min(frame_rgb.shape[0] - 1, y_max + padding)
                            mask_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                            state.annotation_manager.set_detection_box(video_path, annot.worm_id, mask_bbox)
                except Exception as e:
                    print(f"[folder_auto_detect] Segment failed for {video_info.path.name} worm {annot.worm_id}: {e}")
    
    # Save all annotations
    state.annotation_manager.save_annotations()
    
    # Restore to original video
    state.video_handler.navigate_to(current_idx)
    _load_current_video()
    
    print(f"[folder_auto_detect] Complete: {total_detections} detections, {total_segmented} segmented")
    
    return {
        "processed": processed,
        "total_detections": total_detections,
        "total_segmented": total_segmented
    }


def _load_default_dataset():
    """Load the default dataset folder on startup."""
    default_path = Path(DEFAULT_DATASET_PATH)
    if default_path.exists():
        print(f"Loading default dataset from: {DEFAULT_DATASET_PATH}")
        state.video_handler = VideoHandler()
        state.video_handler.load_folder(DEFAULT_DATASET_PATH, recursive=True)
        
        state.annotation_manager = AnnotationManager()
        state.annotation_manager.set_folder(DEFAULT_DATASET_PATH)
        
        # Load first video
        if state.video_handler.get_video_count() > 0:
            state.video_handler.navigate_to(0)
            _load_current_video()
            print(f"Loaded {state.video_handler.get_video_count()} videos from default dataset")
        else:
            print("No videos found in default dataset")
    else:
        print(f"Default dataset path not found: {DEFAULT_DATASET_PATH}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and default dataset on startup."""
    print("Loading ML models...")
    
    # Load YOLO
    yolo_weights = "/home/hedtpc/Downloads/YOLO314Full/yolov7-custom_lambda/yolov7-custom/best_Leaky_v3.pt"
    if Path(yolo_weights).exists():
        try:
            state.yolo_detector = YOLODetector(weights_path=yolo_weights)
            print("YOLO detector loaded")
        except Exception as e:
            print(f"Failed to load YOLO: {e}")
    
    # Load SAM
    try:
        state.sam_segmenter = SAMSegmenter(model_type="vit_h")
        state.sam_segmenter.load_model()
        print("SAM segmenter loaded")
    except Exception as e:
        print(f"Failed to load SAM: {e}")
    
    # Load Health Classifier
    health_weights = Path(__file__).parent.parent / "weights" / "worm_leaky_model.pth"
    if health_weights.exists():
        try:
            state.health_classifier = HealthClassifier(str(health_weights))
            state.health_classifier.load_model()
            print("Health classifier loaded")
        except Exception as e:
            print(f"Failed to load health classifier: {e}")
    else:
        print(f"Health classifier weights not found at {health_weights}")
    
    state.models_loaded = True
    print("Models loaded!")
    
    # Load default dataset
    _load_default_dataset()
    
    yield
    
    # Cleanup
    print("Shutting down...")


app = FastAPI(title="Worm Annotation Tool", lifespan=lifespan)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_image_to_base64(image: np.ndarray, quality: int = 85) -> str:
    """Encode numpy image to base64 JPEG string.
    
    Args:
        image: RGB numpy array
        quality: JPEG compression quality
    """
    # Convert RGB to BGR for cv2.imencode
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Encode binary mask to base64 PNG string."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', mask_uint8)
    return base64.b64encode(buffer).decode('utf-8')


def compute_mask_statistics(mask: np.ndarray) -> Dict[str, Any]:
    """Compute statistics for a binary mask including area and skeleton length."""
    # Ensure binary mask
    binary = (mask > 128).astype(np.uint8) if mask.max() > 1 else (mask > 0.5).astype(np.uint8)
    
    # Compute area (number of pixels)
    area = int(np.sum(binary))
    
    if area == 0:
        return {
            "area": 0,
            "skeleton_length": 0,
            "skeleton": None
        }
    
    # Compute skeleton
    skeleton = skeletonize(binary > 0)
    skeleton_pixels = int(np.sum(skeleton))
    
    # Encode skeleton as base64 image
    skeleton_img = (skeleton * 255).astype(np.uint8)
    _, skeleton_encoded = cv2.imencode('.png', skeleton_img)
    skeleton_b64 = base64.b64encode(skeleton_encoded.tobytes()).decode('utf-8')
    
    # Compute skeleton length by finding endpoints and tracing path
    # Find endpoints (pixels with only 1 neighbor in skeleton)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
    endpoints = skeleton & (neighbor_count == 1)
    endpoint_coords = np.where(endpoints)
    
    skeleton_length = skeleton_pixels  # Default to pixel count
    
    # If we have exactly 2 endpoints, compute geodesic distance
    if len(endpoint_coords[0]) >= 2:
        # Use the two most distant endpoints
        start = (endpoint_coords[0][0], endpoint_coords[1][0])
        end = (endpoint_coords[0][-1], endpoint_coords[1][-1])
        
        # Create cost array (low cost on skeleton, high elsewhere)
        cost = np.where(skeleton, 1, 1000)
        
        try:
            # Find path through skeleton
            path, cost_val = route_through_array(cost, start, end, fully_connected=True)
            skeleton_length = len(path)
        except:
            pass  # Keep pixel count as length
    
    return {
        "area": area,
        "skeleton_length": skeleton_length,
        "skeleton": skeleton_b64
    }


def straighten_worm(image: np.ndarray, mask: np.ndarray, width: int = None) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
    """
    Straighten a worm image along its skeleton.
    
    Args:
        image: RGB image containing the worm
        mask: Binary mask of the worm
        width: Width of straightened image (auto-detect if None)
    
    Returns:
        Tuple of (straightened_image, skeleton_path, global_min_d)
        - global_min_d is the offset used to compute Y positions (d=global_min_d -> row 0)
    """
    # Ensure binary mask
    binary = (mask > 128).astype(np.uint8) if mask.max() > 1 else (mask > 0.5).astype(np.uint8)
    
    if np.sum(binary) == 0:
        return None, []
    
    # Compute skeleton
    skeleton = skeletonize(binary > 0)
    
    # Find endpoints
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
    endpoints = skeleton & (neighbor_count == 1)
    endpoint_coords = np.where(endpoints)
    
    if len(endpoint_coords[0]) < 2:
        return None, []
    
    # Find the two most distant endpoints
    from scipy.spatial.distance import cdist
    points = np.column_stack([endpoint_coords[0], endpoint_coords[1]])
    if len(points) > 2:
        dists = cdist(points, points)
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        start = (points[i][0], points[i][1])
        end = (points[j][0], points[j][1])
    else:
        start = (endpoint_coords[0][0], endpoint_coords[1][0])
        end = (endpoint_coords[0][-1], endpoint_coords[1][-1])
    
    # Trace path through skeleton
    cost = np.where(skeleton, 1, 1000)
    try:
        path, _ = route_through_array(cost, start, end, fully_connected=True)
    except:
        return None, []
    
    if len(path) < 3:
        return None, []
    
    # Store original skeleton path length for verification
    original_path_length = len(path)
    
    # Store original endpoints BEFORE smoothing (these are the true worm tips)
    path = list(path)
    original_start = path[0]
    original_end = path[-1]
    
    # Smooth the path to reduce spiky edges
    # Apply Gaussian smoothing to path coordinates
    path_y = np.array([p[0] for p in path], dtype=float)
    path_x = np.array([p[1] for p in path], dtype=float)
    
    # Use a smoothing window proportional to path length
    smooth_sigma = max(3, len(path) // 30)
    path_y_smooth = ndimage.gaussian_filter1d(path_y, sigma=smooth_sigma, mode='nearest')
    path_x_smooth = ndimage.gaussian_filter1d(path_x, sigma=smooth_sigma, mode='nearest')
    
    # CRITICAL: Preserve original endpoints to prevent cropping at worm tips
    # Gaussian smoothing pulls endpoints inward - we need to blend back to original
    # Use a fade zone at each end to smoothly transition from original to smoothed
    fade_length = min(smooth_sigma * 2, len(path) // 4)
    for i in range(fade_length):
        # Blend factor: 0 at endpoint (use original), 1 at fade_length (use smoothed)
        blend = i / fade_length
        # Start region
        path_y_smooth[i] = (1 - blend) * path_y[i] + blend * path_y_smooth[i]
        path_x_smooth[i] = (1 - blend) * path_x[i] + blend * path_x_smooth[i]
        # End region
        end_i = len(path) - 1 - i
        path_y_smooth[end_i] = (1 - blend) * path_y[end_i] + blend * path_y_smooth[end_i]
        path_x_smooth[end_i] = (1 - blend) * path_x[end_i] + blend * path_x_smooth[end_i]
    
    # Rebuild smoothed path (same length as original)
    path = [(int(round(y)), int(round(x))) for y, x in zip(path_y_smooth, path_x_smooth)]
    
    # Verify path length is preserved
    assert len(path) == original_path_length, f"Path length changed: {original_path_length} -> {len(path)}"
    
    # Pre-compute smoothed tangent directions to avoid spiky sampling
    # Use a larger window for tangent computation
    tangent_window = max(5, len(path) // 20)
    tangents = []
    for i in range(len(path)):
        # Use wider window for smoother tangent estimation
        prev_idx = max(0, i - tangent_window)
        next_idx = min(len(path) - 1, i + tangent_window)
        dy = path[next_idx][0] - path[prev_idx][0]
        dx = path[next_idx][1] - path[prev_idx][1]
        length = np.sqrt(dx*dx + dy*dy)
        if length > 0:
            tangents.append((dx/length, dy/length))
        elif tangents:
            tangents.append(tangents[-1])  # Use previous tangent
        else:
            tangents.append((1, 0))  # Default
    
    # Recompute width using the SAME tangent directions we'll use for sampling
    # This ensures consistency
    # IMPORTANT: Only count the contiguous mask segment that contains the skeleton point (d=0)
    # This prevents counting other parts of the worm that the perpendicular might cross
    all_min_d = []
    all_max_d = []
    for i in range(len(path)):
        y, x = path[i]
        tx, ty = tangents[i]
        # Perpendicular direction (same as sampling below)
        nx, ny = -ty, tx
        
        # Find the contiguous segment containing d=0
        # Scan outward from center in both directions until we hit non-mask
        min_d = 0
        for d in range(0, -300, -1):
            py, px = int(y + d * ny), int(x + d * nx)
            if 0 <= py < binary.shape[0] and 0 <= px < binary.shape[1]:
                if binary[py, px]:
                    min_d = d
                else:
                    break  # Hit edge of this segment
            else:
                break  # Out of bounds
        
        max_d = 0
        for d in range(0, 300):
            py, px = int(y + d * ny), int(x + d * nx)
            if 0 <= py < binary.shape[0] and 0 <= px < binary.shape[1]:
                if binary[py, px]:
                    max_d = d
                else:
                    break  # Hit edge of this segment
            else:
                break  # Out of bounds
        
        all_min_d.append(min_d)
        all_max_d.append(max_d)
    
    # Use the extremes across all slices to ensure nothing is cropped
    global_min_d = min(all_min_d) if all_min_d else -30
    global_max_d = max(all_max_d) if all_max_d else 30
    
    # Add generous padding
    padding = 20
    global_min_d -= padding
    global_max_d += padding
    
    width = global_max_d - global_min_d + 1
    width = max(50, width)
    
    # Sample perpendicular slices along path
    # Start with white background (255, 255, 255)
    straightened = np.full((width, len(path), 3), 255, dtype=np.uint8)
    straightened_mask = np.zeros((width, len(path)), dtype=np.uint8)
    
    for i, (y, x) in enumerate(path):
        tx, ty = tangents[i]
        
        # Perpendicular direction (rotate tangent 90 degrees)
        nx, ny = -ty, tx
        
        # Find the local contiguous segment containing d=0 for this slice
        local_min_d = 0
        for d in range(0, global_min_d - 1, -1):
            py_int = int(round(y + d * ny))
            px_int = int(round(x + d * nx))
            if 0 <= py_int < binary.shape[0] and 0 <= px_int < binary.shape[1]:
                if binary[py_int, px_int]:
                    local_min_d = d
                else:
                    break
            else:
                break
        
        local_max_d = 0
        for d in range(0, global_max_d + 1):
            py_int = int(round(y + d * ny))
            px_int = int(round(x + d * nx))
            if 0 <= py_int < binary.shape[0] and 0 <= px_int < binary.shape[1]:
                if binary[py_int, px_int]:
                    local_max_d = d
                else:
                    break
            else:
                break
        
        # Sample only the local contiguous segment
        for d in range(local_min_d, local_max_d + 1):
            py = y + d * ny
            px = x + d * nx
            
            # Output row index (d=global_min_d -> row 0)
            out_row = d - global_min_d
            
            if 0 <= out_row < width:
                py_int, px_int = int(round(py)), int(round(px))
                straightened_mask[out_row, i] = 255
                
                # Bilinear interpolation for sub-pixel sampling
                if 0 <= py < image.shape[0] - 1 and 0 <= px < image.shape[1] - 1:
                    y0, x0 = int(py), int(px)
                    y1, x1 = y0 + 1, x0 + 1
                    fy, fx = py - y0, px - x0
                    
                    # Interpolate
                    val = ((1-fy) * (1-fx) * image[y0, x0].astype(float) +
                           (1-fy) * fx * image[y0, x1].astype(float) +
                           fy * (1-fx) * image[y1, x0].astype(float) +
                           fy * fx * image[y1, x1].astype(float))
                    straightened[out_row, i] = val.astype(np.uint8)
                elif 0 <= py_int < image.shape[0] and 0 <= px_int < image.shape[1]:
                    straightened[out_row, i] = image[py_int, px_int]
    
    # Verify skeleton length matches output width
    output_width = straightened.shape[1]
    print(f"[straighten_worm] Skeleton path length: {original_path_length}, Output width: {output_width}, Match: {original_path_length == output_width}")
    print(f"[straighten_worm] Original endpoints: start={original_start}, end={original_end}")
    print(f"[straighten_worm] Smoothed endpoints: start={path[0]}, end={path[-1]}")
    print(f"[straighten_worm] global_min_d: {global_min_d}, global_max_d: {global_max_d}, width: {width}")
    
    return straightened, path, global_min_d


def auto_segment_worm(worm_id: int, bbox: tuple, return_mask_bounds: bool = False):
    """
    Automatically segment a worm using SAM when a detection box is created.
    Also adjusts the detection box to fit the segmentation mask.
    
    Args:
        worm_id: The worm ID
        bbox: Bounding box (x1, y1, x2, y2)
        return_mask_bounds: If True, returns (success, mask_bbox) tuple
        
    Returns:
        If return_mask_bounds=False: True/False for success
        If return_mask_bounds=True: (success, mask_bbox) where mask_bbox is (x1,y1,x2,y2) or None
    """
    if state.sam_segmenter is None or not state.sam_segmenter.is_loaded():
        return (False, None) if return_mask_bounds else False
    if state.current_frame is None:
        return (False, None) if return_mask_bounds else False
    if state.annotation_manager is None or state.current_video_path is None:
        return (False, None) if return_mask_bounds else False
    
    try:
        # Set image for SAM
        state.sam_segmenter.set_image(state.current_frame)
        
        # Run segmentation
        result = state.sam_segmenter.segment(bbox, keep_largest_only=True)
        
        if result is None:
            return (False, None) if return_mask_bounds else False
        
        # Save the mask
        state.annotation_manager.save_segmentation_mask(
            state.current_video_path,
            worm_id,
            result.mask,
            mask_type="worm"
        )
        
        # Adjust detection box to fit the mask
        _adjust_detection_box_to_mask(worm_id, result.mask)
        
        # Calculate mask bounding box if requested
        mask_bbox = None
        if return_mask_bounds:
            # Get the updated annotation with adjusted box
            annot = state.annotation_manager.get_worm_annotation(state.current_video_path, worm_id)
            if annot and annot.detection_box:
                mask_bbox = annot.detection_box
        
        return (True, mask_bbox) if return_mask_bounds else True
    except Exception as e:
        print(f"Auto-segment failed for worm {worm_id}: {e}")
        return (False, None) if return_mask_bounds else False


def auto_segment_worm_no_set_image(worm_id: int, bbox: tuple) -> bool:
    """
    Segment a worm using SAM - assumes set_image was already called.
    Also adjusts the detection box to fit the segmentation mask.
    Returns True if segmentation was successful, False otherwise.
    """
    if state.sam_segmenter is None or not state.sam_segmenter.is_loaded():
        print(f"[auto_segment_worm_no_set_image] SAM not loaded")
        return False
    if state.current_frame is None:
        print(f"[auto_segment_worm_no_set_image] No current frame")
        return False
    if state.annotation_manager is None or state.current_video_path is None:
        print(f"[auto_segment_worm_no_set_image] No annotation manager or video path")
        return False
    
    try:
        # Run segmentation (image already set)
        result = state.sam_segmenter.segment(bbox, keep_largest_only=True)
        
        if result is None:
            print(f"[auto_segment_worm_no_set_image] Segment returned None for worm {worm_id}")
            return False
        
        # Save the mask
        state.annotation_manager.save_segmentation_mask(
            state.current_video_path,
            worm_id,
            result.mask,
            mask_type="worm"
        )
        
        # Adjust detection box to fit the mask
        _adjust_detection_box_to_mask(worm_id, result.mask)
        
        return True
    except Exception as e:
        print(f"Auto-segment failed for worm {worm_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def _adjust_detection_box_to_mask(worm_id: int, mask: np.ndarray, padding: int = 5):
    """
    Adjust the detection box to tightly fit the segmentation mask.
    
    Args:
        worm_id: The worm ID
        mask: Binary mask array
        padding: Pixels of padding around the mask bounding box
    """
    if state.annotation_manager is None or state.current_video_path is None:
        return
    
    annot = state.annotation_manager.get_worm_annotation(state.current_video_path, worm_id)
    if annot is None:
        return
    
    # Find contours to get tight bounding box
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return
    
    # Get bounding rect of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding and clamp to image bounds
    if state.current_frame is not None:
        img_h, img_w = state.current_frame.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)
    else:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = x + w + padding
        y2 = y + h + padding
    
    # Update the annotation's detection box
    annot.detection_box = (x1, y1, x2, y2)
    state.annotation_manager._unsaved_changes = True
    print(f"[_adjust_detection_box_to_mask] Adjusted worm {worm_id} box to ({x1}, {y1}, {x2}, {y2})")


def classify_worm_health(worm_id: int, video_path: str = None) -> Tuple[Optional[float], Optional[str]]:
    """
    Classify a worm's health status using the health CNN.
    
    Args:
        worm_id: The worm ID to classify
        video_path: Optional video path (uses current if not provided)
        
    Returns:
        Tuple of (health_score, health_classification) or (None, None) if failed
    """
    if state.health_classifier is None or not state.health_classifier.is_loaded():
        print(f"[classify_worm_health] Health classifier not loaded")
        return None, None
    
    if state.current_frame is None:
        print(f"[classify_worm_health] No current frame")
        return None, None
        
    if state.annotation_manager is None:
        print(f"[classify_worm_health] No annotation manager")
        return None, None
    
    vpath = video_path or state.current_video_path
    if not vpath:
        print(f"[classify_worm_health] No video path")
        return None, None
    
    try:
        # Get the annotation
        annot = state.annotation_manager.get_worm_annotation(vpath, worm_id)
        if annot is None:
            print(f"[classify_worm_health] No annotation for worm {worm_id}")
            return None, None
            
        # Get the detection box
        if annot.detection_box is None:
            print(f"[classify_worm_health] No detection box for worm {worm_id}")
            return None, None
        
        # Classify using the bounding box
        score, classification = state.health_classifier.classify_from_frame(
            state.current_frame, 
            annot.detection_box
        )
        
        # Update the annotation with health info
        annot.health_score = score
        annot.health_classification = classification
        annot.health_class = score_to_health_class(score)
        state.annotation_manager._unsaved_changes = True
        
        print(f"[classify_worm_health] Worm {worm_id}: {classification} (score={score:.4f}, class={annot.health_class})")
        return score, classification
        
    except Exception as e:
        print(f"[classify_worm_health] Failed for worm {worm_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def score_to_health_class(score: Optional[float]) -> Optional[str]:
    """
    Convert a health score (0-1) to a health class (A-E).
    
    Class mapping:
        A = 0     (most healthy)
        B = 0.25
        C = 0.5
        D = 0.75
        E = 1     (most leaky)
    
    Args:
        score: Health score 0-1 (higher = more leaky)
        
    Returns:
        Health class A-E or None if score is None
    """
    if score is None:
        return None
    
    # Round to nearest class threshold
    if score < 0.125:
        return "A"
    elif score < 0.375:
        return "B"
    elif score < 0.625:
        return "C"
    elif score < 0.875:
        return "D"
    else:
        return "E"


def health_class_to_score(health_class: str) -> Optional[float]:
    """
    Convert a health class (A-E) to a health score (0-1).
    
    Args:
        health_class: Class letter A-E
        
    Returns:
        Health score or None if invalid class
    """
    class_map = {
        "A": 0.0,
        "B": 0.25,
        "C": 0.5,
        "D": 0.75,
        "E": 1.0
    }
    return class_map.get(health_class.upper())


# ============== API Routes ==============

# ------ Authentication Routes ------

@app.get("/")
async def root(user: Optional[dict] = Depends(get_current_user)):
    """Redirect to login or app based on auth status."""
    if user:
        return RedirectResponse(url="/app")
    return RedirectResponse(url="/login")


@app.get("/login")
async def login_page():
    """Serve the login page."""
    html_path = Path(__file__).parent / "static" / "login.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Login page not found</h1>")


@app.get("/app")
async def app_page(request: Request, token: Optional[str] = None):
    """Serve the main app (requires auth)."""
    # Check for auth via multiple methods
    user = await get_current_user(request, None)
    
    # Also check query param token
    if not user and token:
        session = auth_manager.validate_token(token)
        if session:
            user = session
    
    if not user:
        return RedirectResponse(url="/login")
    
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>App not found</h1>")


@app.post("/api/auth/login")
async def login(request: LoginRequest, response: Response):
    """Login and get session token."""
    token = auth_manager.login(request.username, request.password)
    
    if not token:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Set cookie
    max_age = 86400 * 7 if request.remember else 86400  # 7 days or 1 day
    response.set_cookie(
        key="auth_token",
        value=token,
        max_age=max_age,
        httponly=True,
        samesite="lax"
    )
    
    user = auth_manager.get_user(request.username)
    return {
        "success": True,
        "token": token,
        "username": request.username,
        "role": user.role if user else "annotator"
    }


@app.post("/api/auth/logout")
async def logout(response: Response, user: Optional[dict] = Depends(get_current_user)):
    """Logout and invalidate session."""
    # Clear cookie
    response.delete_cookie("auth_token")
    
    # Invalidate token from request
    if user:
        # Find and remove the session
        pass
    
    return {"success": True}


@app.get("/api/auth/me")
async def get_me(user: dict = Depends(require_auth)):
    """Get current user info."""
    return {
        "username": user["username"],
        "role": user["role"]
    }


# ------ Admin Routes ------

@app.get("/api/admin/users")
async def list_users(user: dict = Depends(require_admin)):
    """List all users (admin only)."""
    return {"users": auth_manager.list_users()}


@app.post("/api/admin/users")
async def create_user(request: CreateUserRequest, user: dict = Depends(require_admin)):
    """Create a new user (admin only)."""
    if auth_manager.create_user(request.username, request.password, request.role):
        return {"success": True, "username": request.username}
    raise HTTPException(status_code=400, detail="User already exists")


@app.delete("/api/admin/users/{username}")
async def delete_user(username: str, user: dict = Depends(require_admin)):
    """Delete a user (admin only)."""
    if auth_manager.delete_user(username):
        return {"success": True}
    raise HTTPException(status_code=400, detail="Cannot delete user")


@app.post("/api/admin/users/{username}/password")
async def change_user_password(username: str, request: ChangePasswordRequest, user: dict = Depends(require_admin)):
    """Change a user's password (admin only)."""
    if auth_manager.change_password(username, request.new_password):
        return {"success": True}
    raise HTTPException(status_code=404, detail="User not found")


# ------ File Browser Routes ------

class BrowseRequest(BaseModel):
    path: str = "/"

@app.post("/api/browse")
async def browse_directory(request: BrowseRequest, user: dict = Depends(require_auth)):
    """Browse server filesystem directories."""
    path = Path(request.path)
    
    # Handle special paths
    if request.path == "/" or request.path == "":
        # Return common starting points
        entries = []
        common_paths = [
            Path.home(),
            Path("/media"),
            Path("/mnt"),
            Path("/home"),
            Path("/"),
        ]
        for p in common_paths:
            if p.exists():
                entries.append({
                    "name": str(p),
                    "path": str(p),
                    "is_dir": True,
                    "is_shortcut": True
                })
        return {
            "current_path": "/",
            "parent_path": None,
            "entries": entries
        }
    
    if not path.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    
    if not path.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")
    
    try:
        entries = []
        for item in sorted(path.iterdir()):
            # Skip hidden files
            if item.name.startswith('.'):
                continue
            try:
                is_dir = item.is_dir()
                # For directories, check if they might contain videos
                has_videos = False
                if is_dir:
                    try:
                        # Quick check for video files
                        for ext in ['.avi', '.mp4', '.mov', '.mkv']:
                            if list(item.glob(f'*{ext}'))[:1]:
                                has_videos = True
                                break
                    except:
                        pass
                
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "is_dir": is_dir,
                    "has_videos": has_videos
                })
            except PermissionError:
                continue
        
        # Sort: directories first, then files
        entries.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        
        return {
            "current_path": str(path),
            "parent_path": str(path.parent) if path.parent != path else None,
            "entries": entries
        }
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")


# ------ App Status Routes ------

@app.get("/api/status")
async def get_status(user: dict = Depends(require_auth)):
    """Get current application status."""
    return {
        "models_loaded": state.models_loaded,
        "yolo_loaded": state.yolo_detector is not None,
        "sam_loaded": state.sam_segmenter is not None and state.sam_segmenter.is_loaded(),
        "folder_loaded": state.video_handler is not None,
        "current_video": state.current_video_path,
    }


@app.post("/api/open_folder")
async def open_folder(folder: FolderPath, user: dict = Depends(require_auth)):
    """Open a folder containing videos and auto-detect/segment all worms."""
    if not Path(folder.path).exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    state.video_handler = VideoHandler()
    state.video_handler.load_folder(folder.path, recursive=True)
    
    state.annotation_manager = AnnotationManager()
    state.annotation_manager.set_folder(folder.path)
    
    video_count = state.video_handler.get_video_count()
    
    # Auto-detect and segment all videos if YOLO and SAM are loaded
    total_detections = 0
    total_segmented = 0
    
    if state.yolo_detector is not None and video_count > 0:
        print(f"[open_folder] Auto-detecting worms in {video_count} videos...")
        
        for idx in range(video_count):
            state.video_handler.navigate_to(idx)
            video_info = state.video_handler.get_video_info()
            if not video_info:
                continue
            
            video_path = str(video_info.path)
            
            # Skip if already has annotations
            existing = state.annotation_manager.get_all_worm_annotations(video_path)
            if existing:
                print(f"[open_folder] Skipping {video_info.path.name} - already has {len(existing)} annotations")
                total_detections += len(existing)
                # Count segmented
                for annot in existing:
                    if annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists():
                        total_segmented += 1
                continue
            
            # Get first frame
            frame = state.video_handler.get_first_frame()
            if frame is None:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection
            detections = state.yolo_detector.detect(
                frame_rgb,
                conf_threshold=state.confidence_threshold
            )
            
            print(f"[open_folder] {video_info.path.name}: {len(detections)} detections")
            
            if len(detections) == 0:
                continue
            
            # Set SAM image once for this video
            if state.sam_segmenter is not None and state.sam_segmenter.is_loaded():
                state.sam_segmenter.set_image(frame_rgb)
            
            # Add annotations and segment
            for det in detections:
                annot = state.annotation_manager.add_worm_annotation(
                    video_path,
                    detection_box=det.bbox,
                    confidence=det.confidence
                )
                total_detections += 1
                
                # Auto-segment and adjust detection box to match mask
                if state.sam_segmenter is not None and state.sam_segmenter.is_loaded():
                    try:
                        result = state.sam_segmenter.segment(det.bbox, keep_largest_only=True)
                        if result is not None:
                            state.annotation_manager.save_segmentation_mask(
                                video_path,
                                annot.worm_id,
                                result.mask,
                                mask_type="worm"
                            )
                            total_segmented += 1
                            
                            # Adjust detection box to match mask bounding box
                            mask_binary = (result.mask > 0.5).astype(np.uint8) if result.mask.max() <= 1 else (result.mask > 128).astype(np.uint8)
                            coords = np.where(mask_binary > 0)
                            if len(coords[0]) > 0:
                                y_min, y_max = int(coords[0].min()), int(coords[0].max())
                                x_min, x_max = int(coords[1].min()), int(coords[1].max())
                                # Add small padding
                                padding = 5
                                x_min = max(0, x_min - padding)
                                y_min = max(0, y_min - padding)
                                x_max = min(frame_rgb.shape[1] - 1, x_max + padding)
                                y_max = min(frame_rgb.shape[0] - 1, y_max + padding)
                                mask_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                                state.annotation_manager.set_detection_box(video_path, annot.worm_id, mask_bbox)
                    except Exception as e:
                        print(f"[open_folder] Segment failed for {video_info.path.name} worm {annot.worm_id}: {e}")
        
        # Save all annotations
        state.annotation_manager.save_annotations()
        print(f"[open_folder] Complete: {total_detections} detections, {total_segmented} segmented")
    
    # Load first video
    if video_count > 0:
        state.video_handler.navigate_to(0)
        _load_current_video()
    
    return {
        "success": True,
        "video_count": video_count,
        "folder_count": state.video_handler.get_folder_count(),
        "videos": [name for _, name in state.video_handler.get_video_list()],
        "total_detections": total_detections,
        "total_segmented": total_segmented
    }


@app.get("/api/videos")
async def get_videos(user: dict = Depends(require_auth)):
    """Get list of videos in current folder."""
    if state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    current_index = state.video_handler.get_current_index()
    videos = []
    for idx, name in state.video_handler.get_video_list():
        # Build full path using current_folder
        video_path = str(state.video_handler.current_folder / name) if state.video_handler.current_folder else name
        qc_complete = False
        if state.annotation_manager:
            qc_complete, _ = state.annotation_manager.get_video_qc_status(video_path)
        videos.append({
            "index": idx, 
            "name": name, 
            "is_current": idx == current_index,
            "qc_complete": qc_complete
        })
    
    return {
        "videos": videos,
        "current_index": current_index,
        "current_video": state.current_video_path,
        "folder_info": _get_folder_info()
    }


@app.get("/api/video/{index}")
async def select_video(index: int, user: dict = Depends(require_auth)):
    """Select a video by index."""
    if state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    # Save current annotations first
    if state.annotation_manager:
        state.annotation_manager.save_annotations()
    
    if state.video_handler.navigate_to(index):
        _load_current_video()
        return await get_current_frame()
    
    raise HTTPException(status_code=404, detail="Video not found")


@app.get("/api/frame")
async def get_current_frame(user: dict = Depends(require_auth)):
    """Get the current frame with annotations."""
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame loaded")
    
    # Get annotations for current video
    annotations = []
    if state.annotation_manager and state.current_video_path:
        annots = state.annotation_manager.get_all_worm_annotations(state.current_video_path)
        for annot in annots:
            annot_data = {
                "worm_id": annot.worm_id,
                "detection_box": list(annot.detection_box) if annot.detection_box else None,
                "head_box": list(annot.head_box) if annot.head_box else None,
                "tail_box": list(annot.tail_box) if annot.tail_box else None,
                "head_line": list(annot.head_line) if annot.head_line else None,
                "tail_line": list(annot.tail_line) if annot.tail_line else None,
                "confidence": annot.confidence,
                "censored": annot.censored,
                "has_worm_mask": annot.segmentation_mask_path is not None,
                "has_head_mask": annot.head_mask_path is not None,
                "has_tail_mask": annot.tail_mask_path is not None,
                "health_score": annot.health_score,
                "health_classification": annot.health_classification,
                "health_class": annot.health_class or score_to_health_class(annot.health_score),
            }
            
            # Include mask data if available
            if annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists():
                mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    annot_data["worm_mask"] = encode_mask_to_base64(mask / 255.0)
                    # Compute mask statistics
                    stats = compute_mask_statistics(mask)
                    
                    # Auto-classify health if segmentation exists but health is missing
                    if annot.health_score is None and state.health_classifier is not None:
                        score, classification = classify_worm_health(annot.worm_id)
                        if score is not None:
                            annot.health_score = score
                            annot.health_classification = classification
                            annot.health_class = score_to_health_class(score)
                            annot_data["health_score"] = score
                            annot_data["health_classification"] = classification
                            annot_data["health_class"] = annot.health_class
                    
                    # Add health classification to stats
                    stats["health_score"] = annot.health_score
                    stats["health_classification"] = annot.health_classification
                    stats["health_class"] = annot.health_class or score_to_health_class(annot.health_score)
                    annot_data["worm_mask_stats"] = stats
            
            if annot.head_mask_path and Path(annot.head_mask_path).exists():
                mask = cv2.imread(annot.head_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    annot_data["head_mask"] = encode_mask_to_base64(mask / 255.0)
                    stats = compute_mask_statistics(mask)
                    annot_data["head_mask_stats"] = stats
            
            if annot.tail_mask_path and Path(annot.tail_mask_path).exists():
                mask = cv2.imread(annot.tail_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    annot_data["tail_mask"] = encode_mask_to_base64(mask / 255.0)
                    stats = compute_mask_statistics(mask)
                    annot_data["tail_mask_stats"] = stats
                    
            annotations.append(annot_data)
    
    return {
        "image": encode_image_to_base64(state.current_frame),
        "width": state.current_frame.shape[1],
        "height": state.current_frame.shape[0],
        "video_path": state.current_video_path,
        "annotations": annotations,
        "folder_info": _get_folder_info()
    }


@app.post("/api/detect")
async def run_detection(user: dict = Depends(require_auth)):
    """Run YOLO detection on current frame and auto-segment each worm."""
    if state.yolo_detector is None:
        raise HTTPException(status_code=400, detail="YOLO model not loaded")
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame loaded")
    
    # Run detection with confidence threshold
    detections = state.yolo_detector.detect(
        state.current_frame, 
        conf_threshold=state.confidence_threshold
    )
    
    print(f"[run_detection] Found {len(detections)} detections")
    
    # Clear existing annotations and add new ones
    if state.annotation_manager and state.current_video_path:
        # Remove old annotations for this video
        if state.current_video_path in state.annotation_manager.annotations:
            state.annotation_manager.annotations[state.current_video_path].annotations.clear()
        
        # Clear straightened cache for this video
        keys_to_remove = [k for k in state.straightened_cache.keys() if k.startswith(state.current_video_path)]
        for k in keys_to_remove:
            del state.straightened_cache[k]
        
        results = []
        segmented_count = 0
        
        # Pre-set the SAM image once for all segmentations (much faster)
        if state.sam_segmenter is not None and state.sam_segmenter.is_loaded() and len(detections) > 0:
            print(f"[run_detection] Setting SAM image once for {len(detections)} worms")
            state.sam_segmenter.set_image(state.current_frame)
        
        for i, det in enumerate(detections):
            annot = state.annotation_manager.add_worm_annotation(
                state.current_video_path,
                detection_box=det.bbox,
                confidence=det.confidence
            )
            
            # Auto-segment this worm (image already set)
            print(f"[run_detection] Segmenting worm {i+1}/{len(detections)} (id={annot.worm_id})")
            if auto_segment_worm_no_set_image(annot.worm_id, det.bbox):
                segmented_count += 1
                print(f"[run_detection] Worm {annot.worm_id} segmented successfully")
            else:
                print(f"[run_detection] Worm {annot.worm_id} segmentation FAILED")
            
            results.append({
                "worm_id": annot.worm_id,
                "bbox": list(det.bbox),
                "confidence": det.confidence
            })
        
        state.annotation_manager.save_annotations()
        
        print(f"[run_detection] Complete: {len(results)} detections, {segmented_count} segmented")
        
        return {
            "success": True,
            "count": len(results),
            "segmented": segmented_count,
            "detections": results
        }
    
    raise HTTPException(status_code=500, detail="Annotation manager not initialized")


@app.post("/api/detect/folder")
async def detect_and_segment_folder(user: dict = Depends(require_auth)):
    """Run YOLO detection AND SAM segmentation on all videos in the folder."""
    if state.yolo_detector is None:
        raise HTTPException(status_code=400, detail="YOLO model not loaded")
    if state.sam_segmenter is None or not state.sam_segmenter.is_loaded():
        raise HTTPException(status_code=400, detail="SAM model not loaded")
    if state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    video_count = state.video_handler.get_video_count()
    if video_count == 0:
        raise HTTPException(status_code=400, detail="No videos in folder")
    
    # Store current video index to restore later
    current_idx = state.video_handler.get_current_index()
    
    results = {
        "processed": 0,
        "total_detections": 0,
        "total_segmentations": 0,
        "videos": []
    }
    
    # Process each video
    for idx in range(video_count):
        state.video_handler.navigate_to(idx)
        video_info = state.video_handler.get_video_info()
        if not video_info:
            continue
            
        video_path = str(video_info.path)
        
        # Get first frame
        frame = state.video_handler.get_first_frame()
        if frame is None:
            continue
        
        # Store the frame for segmentation
        state.current_frame = frame
        state.current_video_path = video_path
        
        # Run detection
        detections = state.yolo_detector.detect(
            frame,
            conf_threshold=state.confidence_threshold
        )
        
        # Clear existing annotations
        if video_path in state.annotation_manager.annotations:
            state.annotation_manager.annotations[video_path].annotations.clear()
        
        video_results = {
            "video": video_info.path.name,
            "detections": len(detections),
            "segmentations": 0
        }
        
        # Add annotations and run segmentation for each detection
        if detections:
            # Set image for SAM once per frame
            state.sam_segmenter.set_image(frame)
            
            for det in detections:
                annot = state.annotation_manager.add_worm_annotation(
                    video_path,
                    detection_box=det.bbox,
                    confidence=det.confidence
                )
                
                # Run segmentation
                if auto_segment_worm_no_set_image(annot.worm_id, det.bbox):
                    video_results["segmentations"] += 1
        
        results["processed"] += 1
        results["total_detections"] += len(detections)
        results["total_segmentations"] += video_results["segmentations"]
        results["videos"].append(video_results)
    
    # Save all annotations
    state.annotation_manager.save_annotations()
    
    # Restore to original video
    state.video_handler.navigate_to(current_idx)
    _load_current_video()
    
    return {
        "success": True,
        **results
    }


@app.get("/api/detect/preview")
async def get_detection_preview(user: dict = Depends(require_auth)):
    """Get preview stats for batch detection."""
    if state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    # Count videos in current folder
    current_folder_videos = state.video_handler.get_video_count()
    
    # Count all videos across all folders
    total_videos = 0
    total_folders = len(state.video_handler.folder_list) if hasattr(state.video_handler, 'folder_list') else 1
    
    # Count videos with detections and QC complete
    with_detections = 0
    qc_complete = 0
    
    if state.annotation_manager:
        for video_path, video_annot in state.annotation_manager.annotations.items():
            total_videos += 1
            if len(video_annot.annotations) > 0:
                with_detections += 1
            if video_annot.qc_complete:
                qc_complete += 1
    
    # If no annotations exist, count videos from video handler
    if total_videos == 0:
        # Get count from all folders
        current_folder_idx = state.video_handler.current_folder_index
        for folder_idx in range(total_folders):
            state.video_handler.navigate_to_folder(folder_idx)
            total_videos += state.video_handler.get_video_count()
        state.video_handler.navigate_to_folder(current_folder_idx)
    
    return {
        "current_folder_videos": current_folder_videos,
        "total_videos": total_videos,
        "total_folders": total_folders,
        "with_detections": with_detections,
        "qc_complete": qc_complete
    }


class BatchDetectRequest(BaseModel):
    scope: str = "folder"  # "folder" or "project"
    skip_qc: bool = True
    skip_with_detections: bool = True


@app.post("/api/detect/batch")
async def batch_detect_and_segment(request: BatchDetectRequest, user: dict = Depends(require_auth)):
    """Run YOLO detection AND SAM segmentation with options to skip certain videos."""
    if state.yolo_detector is None:
        raise HTTPException(status_code=400, detail="YOLO model not loaded")
    if state.sam_segmenter is None or not state.sam_segmenter.is_loaded():
        raise HTTPException(status_code=400, detail="SAM model not loaded")
    if state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    # Store current position to restore later
    current_folder_idx = state.video_handler.current_folder_index
    current_video_idx = state.video_handler.get_current_index()
    
    results = {
        "processed": 0,
        "total_detections": 0,
        "total_segmentations": 0,
        "skipped_qc": 0,
        "skipped_detections": 0,
        "videos": []
    }
    
    # Determine which folders to process
    if request.scope == "project":
        folder_range = range(len(state.video_handler.folder_list))
    else:
        folder_range = [current_folder_idx]
    
    print(f"[batch_detect] Starting batch detection: scope={request.scope}, skip_qc={request.skip_qc}, skip_detections={request.skip_with_detections}")
    
    for folder_idx in folder_range:
        state.video_handler.navigate_to_folder(folder_idx)
        video_count = state.video_handler.get_video_count()
        
        folder_name = state.video_handler.folder_list[folder_idx].name if folder_idx < len(state.video_handler.folder_list) else "unknown"
        print(f"[batch_detect] Processing folder {folder_idx + 1}: {folder_name} ({video_count} videos)")
        
        for idx in range(video_count):
            state.video_handler.navigate_to(idx)
            video_info = state.video_handler.get_video_info()
            if not video_info:
                continue
            
            video_path = str(video_info.path)
            
            # Check if should skip this video
            existing_annot = state.annotation_manager.annotations.get(video_path)
            
            # Skip QC'd videos if requested
            if request.skip_qc and existing_annot and existing_annot.qc_complete:
                results["skipped_qc"] += 1
                print(f"[batch_detect] Skipping {video_info.path.name} - QC complete")
                continue
            
            # Skip videos with detections if requested
            if request.skip_with_detections and existing_annot and len(existing_annot.annotations) > 0:
                results["skipped_detections"] += 1
                print(f"[batch_detect] Skipping {video_info.path.name} - has {len(existing_annot.annotations)} detections")
                continue
            
            # Get first frame
            frame = state.video_handler.get_first_frame()
            if frame is None:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection
            detections = state.yolo_detector.detect(
                frame_rgb,
                conf_threshold=state.confidence_threshold
            )
            
            print(f"[batch_detect] {video_info.path.name}: {len(detections)} detections")
            
            video_results = {
                "video": video_info.path.name,
                "detections": len(detections),
                "segmentations": 0
            }
            
            # Clear existing annotations for this video (only if we're processing it)
            if video_path in state.annotation_manager.annotations:
                state.annotation_manager.annotations[video_path].annotations.clear()
            
            # Add annotations and run segmentation for each detection
            if detections:
                # Set image for SAM once per frame
                state.sam_segmenter.set_image(frame_rgb)
                
                for det in detections:
                    annot = state.annotation_manager.add_worm_annotation(
                        video_path,
                        detection_box=det.bbox,
                        confidence=det.confidence
                    )
                    
                    # Run segmentation and adjust box to match mask
                    try:
                        result = state.sam_segmenter.segment(det.bbox, keep_largest_only=True)
                        if result is not None:
                            state.annotation_manager.save_segmentation_mask(
                                video_path,
                                annot.worm_id,
                                result.mask,
                                mask_type="worm"
                            )
                            video_results["segmentations"] += 1
                            
                            # Adjust detection box to match mask bounding box
                            mask_binary = (result.mask > 0.5).astype(np.uint8) if result.mask.max() <= 1 else (result.mask > 128).astype(np.uint8)
                            coords = np.where(mask_binary > 0)
                            if len(coords[0]) > 0:
                                y_min, y_max = int(coords[0].min()), int(coords[0].max())
                                x_min, x_max = int(coords[1].min()), int(coords[1].max())
                                padding = 5
                                x_min = max(0, x_min - padding)
                                y_min = max(0, y_min - padding)
                                x_max = min(frame_rgb.shape[1] - 1, x_max + padding)
                                y_max = min(frame_rgb.shape[0] - 1, y_max + padding)
                                mask_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                                state.annotation_manager.set_detection_box(video_path, annot.worm_id, mask_bbox)
                    except Exception as e:
                        print(f"[batch_detect] Segment failed for {video_info.path.name} worm {annot.worm_id}: {e}")
            
            results["processed"] += 1
            results["total_detections"] += len(detections)
            results["total_segmentations"] += video_results["segmentations"]
            results["videos"].append(video_results)
    
    # Save all annotations
    state.annotation_manager.save_annotations()
    
    # Restore to original position
    state.video_handler.navigate_to_folder(current_folder_idx)
    state.video_handler.navigate_to(current_video_idx)
    _load_current_video()
    
    print(f"[batch_detect] Complete: {results['processed']} processed, {results['total_detections']} detections, {results['total_segmentations']} segmentations")
    print(f"[batch_detect] Skipped: {results['skipped_qc']} QC'd, {results['skipped_detections']} with detections")
    
    return {
        "success": True,
        **results
    }


@app.get("/api/detect/batch/status")
async def get_batch_status(user: dict = Depends(require_auth)):
    """Get current batch processing status."""
    bs = state.batch_state
    return {
        "is_running": bs.is_running,
        "current_video": bs.current_video,
        "total_videos": bs.total_videos,
        "current_folder": bs.current_folder,
        "total_folders": bs.total_folders,
        "current_video_name": bs.current_video_name,
        "current_folder_name": bs.current_folder_name,
        "total_detections": bs.total_detections,
        "total_segmentations": bs.total_segmentations,
        "skipped_qc": bs.skipped_qc,
        "skipped_detections": bs.skipped_detections,
        "processed": bs.processed,
        "error": bs.error,
        "complete": bs.complete
    }


@app.post("/api/detect/batch/cancel")
async def cancel_batch(user: dict = Depends(require_auth)):
    """Cancel ongoing batch processing."""
    if state.batch_state.is_running:
        state.batch_state.should_cancel = True
        return {"success": True, "message": "Cancellation requested"}
    return {"success": False, "message": "No batch process running"}


@app.get("/api/detect/batch/stream")
async def batch_detect_stream(
    scope: str = "folder",
    skip_qc: bool = True,
    skip_with_detections: bool = True,
    user: dict = Depends(require_auth)
):
    """
    Stream batch detection progress via Server-Sent Events.
    This allows the UI to show real-time progress.
    """
    if state.yolo_detector is None:
        raise HTTPException(status_code=400, detail="YOLO model not loaded")
    if state.sam_segmenter is None or not state.sam_segmenter.is_loaded():
        raise HTTPException(status_code=400, detail="SAM model not loaded")
    if state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    if state.batch_state.is_running:
        raise HTTPException(status_code=400, detail="Batch process already running")
    
    async def generate():
        bs = state.batch_state
        bs.is_running = True
        bs.should_cancel = False
        bs.complete = False
        bs.error = None
        bs.processed = 0
        bs.total_detections = 0
        bs.total_segmentations = 0
        bs.skipped_qc = 0
        bs.skipped_detections = 0
        
        # Store current position to restore later
        current_folder_idx = state.video_handler.current_folder_index
        current_video_idx = state.video_handler.get_current_index()
        
        try:
            # Determine which folders to process
            if scope == "project":
                folder_range = list(range(len(state.video_handler.folder_list)))
            else:
                folder_range = [current_folder_idx]
            
            bs.total_folders = len(folder_range)
            
            # Count total videos first
            total_videos = 0
            for folder_idx in folder_range:
                state.video_handler.navigate_to_folder(folder_idx)
                total_videos += state.video_handler.get_video_count()
            bs.total_videos = total_videos
            
            # Reset to start
            video_counter = 0
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'start', 'total_videos': total_videos, 'total_folders': len(folder_range)})}\n\n"
            
            for folder_idx_pos, folder_idx in enumerate(folder_range):
                if bs.should_cancel:
                    yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                    break
                
                state.video_handler.navigate_to_folder(folder_idx)
                video_count = state.video_handler.get_video_count()
                
                folder_name = state.video_handler.folder_list[folder_idx].name if folder_idx < len(state.video_handler.folder_list) else "unknown"
                bs.current_folder = folder_idx_pos + 1
                bs.current_folder_name = folder_name
                
                yield f"data: {json.dumps({'type': 'folder', 'folder': folder_idx_pos + 1, 'total_folders': len(folder_range), 'folder_name': folder_name, 'videos_in_folder': video_count})}\n\n"
                
                for idx in range(video_count):
                    if bs.should_cancel:
                        break
                    
                    video_counter += 1
                    bs.current_video = video_counter
                    
                    state.video_handler.navigate_to(idx)
                    video_info = state.video_handler.get_video_info()
                    if not video_info:
                        continue
                    
                    video_path = str(video_info.path)
                    bs.current_video_name = video_info.path.name
                    
                    # Check if should skip this video
                    existing_annot = state.annotation_manager.annotations.get(video_path)
                    
                    # Skip QC'd videos if requested
                    if skip_qc and existing_annot and existing_annot.qc_complete:
                        bs.skipped_qc += 1
                        yield f"data: {json.dumps({'type': 'skip', 'video': video_counter, 'total': total_videos, 'name': video_info.path.name, 'reason': 'qc_complete'})}\n\n"
                        continue
                    
                    # Skip videos with detections if requested
                    if skip_with_detections and existing_annot and len(existing_annot.annotations) > 0:
                        bs.skipped_detections += 1
                        yield f"data: {json.dumps({'type': 'skip', 'video': video_counter, 'total': total_videos, 'name': video_info.path.name, 'reason': 'has_detections'})}\n\n"
                        continue
                    
                    # Send progress update
                    yield f"data: {json.dumps({'type': 'processing', 'video': video_counter, 'total': total_videos, 'name': video_info.path.name, 'folder': folder_name})}\n\n"
                    
                    # Get first frame
                    frame = state.video_handler.get_first_frame()
                    if frame is None:
                        continue
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run detection
                    detections = state.yolo_detector.detect(
                        frame_rgb,
                        conf_threshold=state.confidence_threshold
                    )
                    
                    segmentations = 0
                    
                    # Clear existing annotations for this video
                    if video_path in state.annotation_manager.annotations:
                        state.annotation_manager.annotations[video_path].annotations.clear()
                    
                    # Add annotations and run segmentation
                    if detections:
                        state.sam_segmenter.set_image(frame_rgb)
                        
                        for det in detections:
                            annot = state.annotation_manager.add_worm_annotation(
                                video_path,
                                detection_box=det.bbox,
                                confidence=det.confidence
                            )
                            
                            try:
                                result = state.sam_segmenter.segment(det.bbox, keep_largest_only=True)
                                if result is not None:
                                    state.annotation_manager.save_segmentation_mask(
                                        video_path,
                                        annot.worm_id,
                                        result.mask,
                                        mask_type="worm"
                                    )
                                    segmentations += 1
                                    
                                    # Adjust detection box to match mask
                                    mask_binary = (result.mask > 0.5).astype(np.uint8) if result.mask.max() <= 1 else (result.mask > 128).astype(np.uint8)
                                    coords = np.where(mask_binary > 0)
                                    if len(coords[0]) > 0:
                                        y_min, y_max = int(coords[0].min()), int(coords[0].max())
                                        x_min, x_max = int(coords[1].min()), int(coords[1].max())
                                        padding = 5
                                        x_min = max(0, x_min - padding)
                                        y_min = max(0, y_min - padding)
                                        x_max = min(frame_rgb.shape[1] - 1, x_max + padding)
                                        y_max = min(frame_rgb.shape[0] - 1, y_max + padding)
                                        mask_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                                        state.annotation_manager.set_detection_box(video_path, annot.worm_id, mask_bbox)
                            except Exception as e:
                                print(f"[batch_stream] Segment failed: {e}")
                    
                    bs.processed += 1
                    bs.total_detections += len(detections)
                    bs.total_segmentations += segmentations
                    
                    # Send result for this video
                    yield f"data: {json.dumps({'type': 'result', 'video': video_counter, 'total': total_videos, 'name': video_info.path.name, 'detections': len(detections), 'segmentations': segmentations, 'processed': bs.processed, 'total_detections': bs.total_detections, 'total_segmentations': bs.total_segmentations})}\n\n"
                    
                    # Allow other tasks to run
                    await asyncio.sleep(0)
            
            # Save all annotations
            state.annotation_manager.save_annotations()
            
            # Restore position
            state.video_handler.navigate_to_folder(current_folder_idx)
            state.video_handler.navigate_to(current_video_idx)
            _load_current_video()
            
            bs.complete = True
            
            # Send completion
            yield f"data: {json.dumps({'type': 'complete', 'processed': bs.processed, 'total_detections': bs.total_detections, 'total_segmentations': bs.total_segmentations, 'skipped_qc': bs.skipped_qc, 'skipped_detections': bs.skipped_detections})}\n\n"
            
        except Exception as e:
            bs.error = str(e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            
            # Try to restore position
            try:
                state.video_handler.navigate_to_folder(current_folder_idx)
                state.video_handler.navigate_to(current_video_idx)
                _load_current_video()
            except:
                pass
        finally:
            bs.is_running = False
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/segment")
async def run_segmentation(request: SegmentRequest, user: dict = Depends(require_auth)):
    """Run SAM segmentation on a bounding box."""
    if state.sam_segmenter is None or not state.sam_segmenter.is_loaded():
        raise HTTPException(status_code=400, detail="SAM model not loaded")
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame loaded")
    
    # Set image for SAM
    state.sam_segmenter.set_image(state.current_frame)
    
    # Run segmentation
    bbox = (request.bbox.x1, request.bbox.y1, request.bbox.x2, request.bbox.y2)
    result = state.sam_segmenter.segment(bbox, keep_largest_only=True)
    
    if result is None:
        raise HTTPException(status_code=500, detail="Segmentation failed")
    
    # Save the mask
    mask_type = request.bbox.box_type if request.bbox.box_type in ["head", "tail"] else "worm"
    
    if state.annotation_manager and state.current_video_path:
        saved_path = state.annotation_manager.save_segmentation_mask(
            state.current_video_path,
            request.worm_id,
            result.mask,
            mask_type=mask_type
        )
        state.annotation_manager.save_annotations()
    
    return {
        "success": True,
        "mask": encode_mask_to_base64(result.mask),
        "score": result.score,
        "worm_id": request.worm_id,
        "mask_type": mask_type
    }


@app.post("/api/segment/{worm_id}/{mask_type}")
async def resegment_mask(worm_id: int, mask_type: str, user: dict = Depends(require_auth)):
    """Re-run segmentation for a specific worm's box (worm, head, or tail)."""
    if state.sam_segmenter is None or not state.sam_segmenter.is_loaded():
        raise HTTPException(status_code=400, detail="SAM model not loaded")
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame loaded")
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    if mask_type not in ["worm", "head", "tail"]:
        raise HTTPException(status_code=400, detail="Invalid mask_type. Must be 'worm', 'head', or 'tail'")
    
    # Get the worm annotation
    video_annots = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annots:
        raise HTTPException(status_code=404, detail="No annotations for current video")
    
    annot = video_annots.annotations.get(worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {worm_id} not found")
    
    # Get the appropriate box
    if mask_type == "worm":
        bbox = annot.detection_box
    elif mask_type == "head":
        bbox = annot.head_box
    else:  # tail
        bbox = annot.tail_box
    
    if not bbox:
        raise HTTPException(status_code=400, detail=f"Worm {worm_id} has no {mask_type} box to segment")
    
    # Set image and run segmentation
    state.sam_segmenter.set_image(state.current_frame)
    result = state.sam_segmenter.segment(bbox, keep_largest_only=True)
    
    if result is None:
        raise HTTPException(status_code=500, detail="Segmentation failed")
    
    # Save the mask
    saved_path = state.annotation_manager.save_segmentation_mask(
        state.current_video_path,
        worm_id,
        result.mask,
        mask_type=mask_type
    )
    state.annotation_manager.save_annotations()
    
    return {
        "success": True,
        "mask": encode_mask_to_base64(result.mask),
        "score": result.score,
        "worm_id": worm_id,
        "mask_type": mask_type
    }


@app.post("/api/segment/all")
async def segment_all_worms(user: dict = Depends(require_auth)):
    """Run SAM segmentation on all worms in the current video."""
    if state.sam_segmenter is None or not state.sam_segmenter.is_loaded():
        raise HTTPException(status_code=400, detail="SAM model not loaded")
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame loaded")
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    # Get annotations for current video
    video_annotations = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annotations or not video_annotations.annotations:
        raise HTTPException(status_code=400, detail="No annotations for current video")
    
    # Set image for SAM once
    state.sam_segmenter.set_image(state.current_frame)
    
    results = {
        "processed": 0,
        "success": 0,
        "failed": 0,
        "worms": []
    }
    
    for annot in video_annotations.annotations.values():
        worm_result = {"worm_id": annot.worm_id, "masks": []}
        
        # Segment worm box (detection box)
        if annot.detection_box:
            try:
                result = state.sam_segmenter.segment(annot.detection_box, keep_largest_only=True)
                if result:
                    state.annotation_manager.save_segmentation_mask(
                        state.current_video_path,
                        annot.worm_id,
                        result.mask,
                        mask_type="worm"
                    )
                    worm_result["masks"].append({"type": "worm", "score": result.score})
                    results["success"] += 1
                else:
                    results["failed"] += 1
            except Exception as e:
                results["failed"] += 1
            results["processed"] += 1
        
        # Segment head box
        if annot.head_box:
            try:
                result = state.sam_segmenter.segment(annot.head_box, keep_largest_only=True)
                if result:
                    state.annotation_manager.save_segmentation_mask(
                        state.current_video_path,
                        annot.worm_id,
                        result.mask,
                        mask_type="head"
                    )
                    worm_result["masks"].append({"type": "head", "score": result.score})
                    results["success"] += 1
                else:
                    results["failed"] += 1
            except Exception as e:
                results["failed"] += 1
            results["processed"] += 1
        
        # Segment tail box
        if annot.tail_box:
            try:
                result = state.sam_segmenter.segment(annot.tail_box, keep_largest_only=True)
                if result:
                    state.annotation_manager.save_segmentation_mask(
                        state.current_video_path,
                        annot.worm_id,
                        result.mask,
                        mask_type="tail"
                    )
                    worm_result["masks"].append({"type": "tail", "score": result.score})
                    results["success"] += 1
                else:
                    results["failed"] += 1
            except Exception as e:
                results["failed"] += 1
            results["processed"] += 1
        
        results["worms"].append(worm_result)
    
    # Save annotations
    state.annotation_manager.save_annotations()
    
    return {
        "success": True,
        **results
    }


@app.post("/api/brush-mask")
async def apply_brush_to_mask(request: BrushMaskRequest, user: dict = Depends(require_auth)):
    """Apply brush strokes to refine a segmentation mask."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame loaded")
    
    # Get annotation
    video_annotations = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annotations:
        raise HTTPException(status_code=404, detail="No annotations for current video")
    
    annot = video_annotations.annotations.get(request.worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {request.worm_id} not found")
    
    # Load existing mask or create blank one
    mask_path = None
    if request.mask_type == "worm":
        mask_path = annot.segmentation_mask_path
    elif request.mask_type == "head":
        mask_path = annot.head_mask_path
    elif request.mask_type == "tail":
        mask_path = annot.tail_mask_path
    
    # Get frame dimensions
    h, w = state.current_frame.shape[:2]
    
    # Load or create mask
    if mask_path and Path(mask_path).exists():
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            # Resize if needed
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
    
    # Apply brush strokes
    for stroke in request.strokes:
        x, y = int(stroke.x), int(stroke.y)
        # Clamp to image bounds
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        
        if stroke.add:
            # Add to mask (white)
            cv2.circle(mask, (x, y), request.brush_size, 255, -1)
        else:
            # Remove from mask (black)
            cv2.circle(mask, (x, y), request.brush_size, 0, -1)
    
    # Convert to boolean mask for saving
    mask_bool = mask > 128
    
    # Save the updated mask
    state.annotation_manager.save_segmentation_mask(
        state.current_video_path,
        request.worm_id,
        mask_bool,
        mask_type=request.mask_type
    )
    
    return {
        "success": True,
        "strokes_applied": len(request.strokes),
        "worm_id": request.worm_id,
        "mask_type": request.mask_type
    }


@app.post("/api/annotation/worm")
async def add_manual_worm(box: BoundingBox, user: dict = Depends(require_auth)):
    """Add a manual worm detection (when YOLO misses a worm) and auto-segment."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    coords = (box.x1, box.y1, box.x2, box.y2)
    
    # Add new worm annotation with manual detection box
    annot = state.annotation_manager.add_worm_annotation(
        state.current_video_path,
        detection_box=coords,
        confidence=1.0  # Manual detections have 100% confidence
    )
    
    # Auto-segment this worm and get the mask bounding box
    segmented, mask_bbox = auto_segment_worm(annot.worm_id, coords, return_mask_bounds=True)
    
    # If segmentation succeeded and we got a mask bbox, update the detection box to match
    final_coords = coords
    if segmented and mask_bbox:
        final_coords = mask_bbox
        # Update the annotation with the adjusted box
        state.annotation_manager.set_detection_box(state.current_video_path, annot.worm_id, final_coords)
        print(f"[add_manual_worm] Adjusted detection box from {coords} to {final_coords} based on SAM mask")
    
    state.annotation_manager.save_annotations()
    
    return {
        "success": True, 
        "worm_id": annot.worm_id, 
        "detection_box": final_coords,
        "original_box": coords if final_coords != coords else None,
        "segmented": segmented,
        "box_adjusted": final_coords != coords,
        "message": f"Added manual worm detection (Worm {annot.worm_id})" + (" with mask" if segmented else "")
    }


@app.post("/api/annotation/head")
async def set_head_box(box: BoundingBox, user: dict = Depends(require_auth)):
    """Set head box for a worm."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    if box.worm_id is None:
        raise HTTPException(status_code=400, detail="worm_id required")
    
    coords = (box.x1, box.y1, box.x2, box.y2)
    state.annotation_manager.set_head_box(state.current_video_path, box.worm_id, coords)
    state.annotation_manager.save_annotations()
    
    return {"success": True, "worm_id": box.worm_id, "head_box": coords}


@app.post("/api/annotation/tail")
async def set_tail_box(box: BoundingBox, user: dict = Depends(require_auth)):
    """Set tail box for a worm."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    if box.worm_id is None:
        raise HTTPException(status_code=400, detail="worm_id required")
    
    coords = (box.x1, box.y1, box.x2, box.y2)
    state.annotation_manager.set_tail_box(state.current_video_path, box.worm_id, coords)
    state.annotation_manager.save_annotations()
    
    return {"success": True, "worm_id": box.worm_id, "tail_box": coords}


@app.post("/api/annotation/detection")
async def set_detection_box_endpoint(box: BoundingBox, user: dict = Depends(require_auth)):
    """Set detection box for a worm."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    if box.worm_id is None:
        raise HTTPException(status_code=400, detail="worm_id required")
    
    coords = (box.x1, box.y1, box.x2, box.y2)
    state.annotation_manager.set_detection_box(state.current_video_path, box.worm_id, coords)
    state.annotation_manager.save_annotations()
    
    return {"success": True, "worm_id": box.worm_id, "detection_box": coords}

@app.post("/api/annotation/head-line")
async def set_head_line(box: BoundingBox, user: dict = Depends(require_auth)):
    """Set head line for a worm (x1,y1 to x2,y2)."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    if box.worm_id is None:
        raise HTTPException(status_code=400, detail="worm_id required")
    
    coords = (box.x1, box.y1, box.x2, box.y2)
    state.annotation_manager.set_head_line(state.current_video_path, box.worm_id, coords)
    state.annotation_manager.save_annotations()
    
    return {"success": True, "worm_id": box.worm_id, "head_line": coords}


@app.post("/api/annotation/tail-line")
async def set_tail_line(box: BoundingBox, user: dict = Depends(require_auth)):
    """Set tail line for a worm (x1,y1 to x2,y2)."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    if box.worm_id is None:
        raise HTTPException(status_code=400, detail="worm_id required")
    
    coords = (box.x1, box.y1, box.x2, box.y2)
    state.annotation_manager.set_tail_line(state.current_video_path, box.worm_id, coords)
    state.annotation_manager.save_annotations()
    
    return {"success": True, "worm_id": box.worm_id, "tail_line": coords}


@app.delete("/api/annotation/{worm_id}")
async def delete_annotation(worm_id: int, user: dict = Depends(require_auth)):
    """Delete a worm annotation."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    state.annotation_manager.delete_worm_annotation(state.current_video_path, worm_id)
    state.annotation_manager.save_annotations()
    
    return {"success": True, "deleted_worm_id": worm_id}


@app.post("/api/annotation/{worm_id}/censor")
async def toggle_worm_censored(worm_id: int, request: Request, user: dict = Depends(require_auth)):
    """Set censored status for a worm (exclude from analysis)."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    video_annots = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annots:
        raise HTTPException(status_code=404, detail="No annotations for current video")
    
    annot = video_annots.annotations.get(worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {worm_id} not found")
    
    # Get censored value from request body, or toggle if not specified
    try:
        body = await request.json()
        new_status = body.get('censored', not annot.censored)
    except:
        new_status = not annot.censored
    
    state.annotation_manager.set_worm_censored(state.current_video_path, worm_id, new_status)
    state.annotation_manager.save_annotations()
    
    return {"success": True, "worm_id": worm_id, "censored": new_status}


@app.post("/api/annotation/{worm_id}/health_class")
async def set_worm_health_class(worm_id: int, request: Request, user: dict = Depends(require_auth)):
    """Set health class for a worm (A, B, C, D, E)."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    video_annots = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annots:
        raise HTTPException(status_code=404, detail="No annotations for current video")
    
    annot = video_annots.annotations.get(worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {worm_id} not found")
    
    try:
        body = await request.json()
        health_class = body.get('health_class', '').upper()
    except:
        raise HTTPException(status_code=400, detail="Invalid request body")
    
    if health_class not in ['A', 'B', 'C', 'D', 'E']:
        raise HTTPException(status_code=400, detail="Invalid health class. Must be A, B, C, D, or E")
    
    # Update the annotation
    annot.health_class = health_class
    annot.health_score = health_class_to_score(health_class)
    annot.health_classification = "Healthy" if annot.health_score < 0.5 else "Leaky"
    
    state.annotation_manager._unsaved_changes = True
    state.annotation_manager.save_annotations()
    
    return {
        "success": True, 
        "worm_id": worm_id, 
        "health_class": health_class,
        "health_score": annot.health_score,
        "health_classification": annot.health_classification
    }


@app.post("/api/video/qc")
async def toggle_video_qc(request: Request, user: dict = Depends(require_auth)):
    """Toggle QC complete status for the current video."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    try:
        body = await request.json()
        qc_complete = body.get('qc_complete', None)
    except:
        # Toggle if no explicit value provided
        qc_complete = None
    
    # Get current QC status
    current_qc, _ = state.annotation_manager.get_video_qc_status(state.current_video_path)
    
    # If no explicit value, toggle
    if qc_complete is None:
        qc_complete = not current_qc
    
    # Set the new QC status
    state.annotation_manager.set_video_qc_complete(state.current_video_path, qc_complete)
    state.annotation_manager.save_annotations()
    
    # Get updated status
    new_qc, qc_timestamp = state.annotation_manager.get_video_qc_status(state.current_video_path)
    
    return {
        "success": True,
        "qc_complete": new_qc,
        "qc_timestamp": qc_timestamp
    }


@app.get("/api/video/qc")
async def get_video_qc(user: dict = Depends(require_auth)):
    """Get QC status for the current video."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    qc_complete, qc_timestamp = state.annotation_manager.get_video_qc_status(state.current_video_path)
    
    return {
        "qc_complete": qc_complete,
        "qc_timestamp": qc_timestamp
    }


@app.post("/api/precompute_straightened")
async def precompute_straightened(user: dict = Depends(require_auth)):
    """Precompute straightened views for all worms with masks in the current frame."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame loaded")
    
    video_annots = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annots:
        return {"success": True, "precomputed": 0, "message": "No annotations"}
    
    precomputed = 0
    errors = 0
    
    for worm_id, annot in video_annots.annotations.items():
        if not annot.segmentation_mask_path or not Path(annot.segmentation_mask_path).exists():
            continue
        
        cache_key = f"{state.current_video_path}_{worm_id}"
        
        # Skip if already cached
        if cache_key in state.straightened_cache:
            precomputed += 1
            continue
        
        try:
            mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                errors += 1
                continue
            
            straightened, path, global_min_d = straighten_worm(state.current_frame, mask)
            
            if straightened is not None and path:
                state.straightened_cache[cache_key] = {
                    "image": straightened,
                    "path": path,
                    "global_min_d": global_min_d,
                    "video": state.current_video_path
                }
                precomputed += 1
            else:
                errors += 1
        except Exception as e:
            print(f"Error precomputing straightened view for worm {worm_id}: {e}")
            errors += 1
    
    return {
        "success": True,
        "precomputed": precomputed,
        "errors": errors,
        "message": f"Precomputed {precomputed} straightened views"
    }


@app.get("/api/worm/{worm_id}/crop")
async def get_worm_crop(worm_id: int, user: dict = Depends(require_auth)):
    """Get cropped image of a worm from its detection box."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame loaded")
    
    video_annots = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annots:
        raise HTTPException(status_code=404, detail="No annotations for current video")
    
    annot = video_annots.annotations.get(worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {worm_id} not found")
    
    if not annot.detection_box:
        raise HTTPException(status_code=400, detail=f"Worm {worm_id} has no detection box")
    
    # Crop from detection box
    x1, y1, x2, y2 = [int(c) for c in annot.detection_box]
    h, w = state.current_frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    crop = state.current_frame[y1:y2, x1:x2].copy()
    
    # Encode as base64
    crop_b64 = encode_image_to_base64(crop)
    
    return {
        "success": True,
        "worm_id": worm_id,
        "image": crop_b64,
        "width": x2 - x1,
        "height": y2 - y1
    }


@app.get("/api/worm/{worm_id}/masked")
async def get_worm_masked(worm_id: int, user: dict = Depends(require_auth)):
    """Get masked worm image (worm pixels on white background)."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame loaded")
    
    video_annots = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annots:
        raise HTTPException(status_code=404, detail="No annotations for current video")
    
    annot = video_annots.annotations.get(worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {worm_id} not found")
    
    if not annot.segmentation_mask_path or not Path(annot.segmentation_mask_path).exists():
        raise HTTPException(status_code=400, detail=f"Worm {worm_id} has no segmentation mask")
    
    # Load mask
    mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=500, detail="Failed to load mask")
    
    # Find bounding box of mask
    coords = np.where(mask > 127)
    if len(coords[0]) == 0:
        raise HTTPException(status_code=400, detail="Mask is empty")
    
    y1, y2 = coords[0].min(), coords[0].max() + 1
    x1, x2 = coords[1].min(), coords[1].max() + 1
    
    # Crop mask and image
    mask_crop = mask[y1:y2, x1:x2]
    img_crop = state.current_frame[y1:y2, x1:x2].copy()
    
    # Create white background and apply mask
    result = np.full_like(img_crop, 255)
    result[mask_crop > 127] = img_crop[mask_crop > 127]
    
    # Encode as base64
    result_b64 = encode_image_to_base64(result)
    
    return {
        "success": True,
        "worm_id": worm_id,
        "image": result_b64,
        "width": x2 - x1,
        "height": y2 - y1
    }


@app.get("/api/worm/{worm_id}/straightened")
async def get_worm_straightened(worm_id: int, user: dict = Depends(require_auth)):
    """Get a straightened view of a worm (simplified version for tile view)."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame loaded")
    
    video_annots = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annots:
        raise HTTPException(status_code=404, detail="No annotations for current video")
    
    annot = video_annots.annotations.get(worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {worm_id} not found")
    
    if not annot.segmentation_mask_path or not Path(annot.segmentation_mask_path).exists():
        raise HTTPException(status_code=400, detail=f"Worm {worm_id} has no segmentation mask")
    
    # Check cache first
    cache_key = f"{state.current_video_path}_{worm_id}"
    cached = state.straightened_cache.get(cache_key)
    
    if cached and cached.get("video") == state.current_video_path:
        straightened = cached["image"]
    else:
        # Load mask
        mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise HTTPException(status_code=500, detail="Failed to load mask")
        
        # Straighten the worm
        straightened, path, global_min_d = straighten_worm(state.current_frame, mask)
        
        if straightened is None:
            raise HTTPException(status_code=500, detail="Failed to straighten worm")
        
        # Cache the result
        state.straightened_cache[cache_key] = {
            "image": straightened,
            "path": path,
            "global_min_d": global_min_d,
            "video": state.current_video_path
        }
    
    # Encode as base64
    straightened_b64 = encode_image_to_base64(straightened)
    
    return {
        "success": True,
        "worm_id": worm_id,
        "image": straightened_b64,
        "width": straightened.shape[1],
        "height": straightened.shape[0]
    }


@app.get("/api/worm/{worm_id}/annotated")
async def get_worm_annotated(worm_id: int, user: dict = Depends(require_auth)):
    """Get cropped worm image with detection box and annotations overlaid."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame loaded")
    
    video_annots = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annots:
        raise HTTPException(status_code=404, detail="No annotations for current video")
    
    annot = video_annots.annotations.get(worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {worm_id} not found")
    
    if not annot.detection_box:
        raise HTTPException(status_code=400, detail=f"Worm {worm_id} has no detection box")
    
    # Get detection box with some padding
    x1, y1, x2, y2 = [int(c) for c in annot.detection_box]
    h, w = state.current_frame.shape[:2]
    
    # Add 10% padding
    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    
    # Create a copy and draw annotations
    crop = state.current_frame[y1:y2, x1:x2].copy()
    
    # Offset for drawing within crop
    offset_x, offset_y = x1, y1
    
    # Draw worm mask contour if available
    if annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists():
        mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(crop, contours, -1, (0, 255, 0), 1)
    
    # Draw head box if available
    if annot.head_box:
        hx1, hy1, hx2, hy2 = [int(c) for c in annot.head_box]
        cv2.rectangle(crop, 
                     (hx1 - offset_x, hy1 - offset_y), 
                     (hx2 - offset_x, hy2 - offset_y), 
                     (0, 255, 0), 1)
    
    # Draw tail box if available  
    if annot.tail_box:
        tx1, ty1, tx2, ty2 = [int(c) for c in annot.tail_box]
        cv2.rectangle(crop,
                     (tx1 - offset_x, ty1 - offset_y),
                     (tx2 - offset_x, ty2 - offset_y),
                     (0, 0, 255), 1)
    
    # Add worm ID and health class label
    label_parts = [f"#{worm_id}"]
    if annot.health_class:
        label_parts.append(annot.health_class)
    label = " ".join(label_parts)
    cv2.putText(crop, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Encode as base64
    crop_b64 = encode_image_to_base64(crop)
    
    return {
        "success": True,
        "worm_id": worm_id,
        "image": crop_b64,
        "width": x2 - x1,
        "height": y2 - y1
    }


@app.get("/api/tile/all_worms")
async def get_all_worms_for_tile(
    source: str = "current_video",
    user: dict = Depends(require_auth)
):
    """
    Get list of all worms for tile view.
    
    Args:
        source: "current_video", "current_folder", or "all_folders"
        
    Returns:
        List of worm metadata including video path and worm_id for each
    """
    if state.annotation_manager is None or state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    worms = []
    
    if source == "current_video":
        # Just current video
        if state.current_video_path:
            video_annots = state.annotation_manager.annotations.get(state.current_video_path)
            if video_annots:
                for worm_id, annot in video_annots.annotations.items():
                        worms.append({
                            "video_path": state.current_video_path,
                            "video_name": Path(state.current_video_path).name,
                            "folder_index": state.video_handler.current_folder_index,
                            "worm_id": worm_id,
                            "has_mask": bool(annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists()),
                            "has_head": bool(annot.head_box),
                            "has_tail": bool(annot.tail_box),
                            "health_class": annot.health_class,
                            "censored": annot.censored,
                            "detection_box": annot.detection_box
                        })
    elif source == "current_folder":
        # All videos in current folder
        current_folder_path = str(state.video_handler.current_folder) if state.video_handler.current_folder else None
        if current_folder_path:
            folder_videos = state.video_handler.videos_by_folder.get(current_folder_path, [])
            for video_path in folder_videos:
                video_path_str = str(video_path)
                video_annots = state.annotation_manager.annotations.get(video_path_str)
                if video_annots:
                    for worm_id, annot in video_annots.annotations.items():
                        worms.append({
                            "video_path": video_path_str,
                            "video_name": Path(video_path_str).name,
                            "folder_index": state.video_handler.current_folder_index,
                            "worm_id": worm_id,
                            "has_mask": bool(annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists()),
                            "has_head": bool(annot.head_box),
                            "has_tail": bool(annot.tail_box),
                            "health_class": annot.health_class,
                            "censored": annot.censored,
                            "detection_box": annot.detection_box
                        })
    
    elif source == "all_folders":
        # All videos across all folders
        for folder_idx, folder_info in enumerate(state.video_handler.folder_list):
            folder_videos = state.video_handler.videos_by_folder.get(str(folder_info.path), [])
            for video_path in folder_videos:
                video_path_str = str(video_path)
                video_annots = state.annotation_manager.annotations.get(video_path_str)
                if video_annots:
                    for worm_id, annot in video_annots.annotations.items():
                        worms.append({
                            "video_path": video_path_str,
                            "video_name": Path(video_path_str).name,
                            "folder_name": folder_info.name,
                            "folder_index": folder_idx,
                            "worm_id": worm_id,
                            "has_mask": bool(annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists()),
                            "has_head": bool(annot.head_box),
                            "has_tail": bool(annot.tail_box),
                            "health_class": annot.health_class,
                            "censored": annot.censored,
                            "detection_box": annot.detection_box
                        })
    
    return {
        "success": True,
        "source": source,
        "worm_count": len(worms),
        "worms": worms
    }


@app.get("/api/tile/worm_image")
async def get_tile_worm_image(
    video_path: str,
    worm_id: int,
    show_type: str = "crop",
    show_mask: bool = True,
    user: dict = Depends(require_auth)
):
    """
    Get a worm image for tile view from a specific video.
    
    This endpoint allows getting images from videos other than the current one,
    which is needed for multi-video tile view.
    
    Args:
        video_path: Full path to the video
        worm_id: Worm ID
        show_type: "crop", "mask", "straightened", or "annotated"
        show_mask: Whether to draw mask contour on annotated view
    """
    if state.annotation_manager is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    video_annots = state.annotation_manager.annotations.get(video_path)
    if not video_annots:
        raise HTTPException(status_code=404, detail="No annotations for video")
    
    annot = video_annots.annotations.get(worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {worm_id} not found")
    
    # Load the frame for this video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open video")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        raise HTTPException(status_code=500, detail="Cannot read frame from video")
    
    h, w = frame.shape[:2]
    
    if show_type == "crop":
        if not annot.detection_box:
            return {"success": False, "error": "No detection box"}
        
        x1, y1, x2, y2 = [int(c) for c in annot.detection_box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = frame[y1:y2, x1:x2].copy()
        # Convert BGR to RGB for encode_image_to_base64
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_b64 = encode_image_to_base64(crop_rgb)
        
    elif show_type == "mask":
        if not annot.segmentation_mask_path or not Path(annot.segmentation_mask_path).exists():
            return {"success": False, "error": "No segmentation mask"}
        
        mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return {"success": False, "error": "Cannot load mask"}
        
        # Find bounding box of mask
        coords = np.where(mask > 127)
        if len(coords[0]) == 0:
            return {"success": False, "error": "Mask is empty"}
        
        y1, y2 = coords[0].min(), coords[0].max() + 1
        x1, x2 = coords[1].min(), coords[1].max() + 1
        
        mask_crop = mask[y1:y2, x1:x2]
        img_crop = frame[y1:y2, x1:x2].copy()
        
        result = np.full_like(img_crop, 255)
        result[mask_crop > 127] = img_crop[mask_crop > 127]
        # Convert BGR to RGB for encode_image_to_base64
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        img_b64 = encode_image_to_base64(result_rgb)
        
    elif show_type == "straightened":
        if not annot.segmentation_mask_path or not Path(annot.segmentation_mask_path).exists():
            return {"success": False, "error": "No segmentation mask"}
        
        # Check cache first
        cache_key = f"{video_path}_{worm_id}_straightened"
        cached = state.straightened_cache.get(cache_key)
        
        if cached and cached.get("video") == video_path:
            # Use cached straightened image (already RGB)
            straightened_rgb = cached["image"]
        else:
            # Compute straightened image on-demand
            try:
                mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    return {"success": False, "error": "Cannot load mask"}
                
                # Convert frame to RGB for straightening
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = straighten_worm(frame_rgb, mask)
                
                if result is None or result[0] is None:
                    return {"success": False, "error": "Failed to straighten worm"}
                
                straightened_rgb = result[0]
                
                # Cache the result (store as RGB)
                state.straightened_cache[cache_key] = {
                    "image": straightened_rgb,
                    "video": video_path
                }
            except Exception as e:
                print(f"Error straightening worm {worm_id}: {e}")
                return {"success": False, "error": f"Straightening failed: {str(e)}"}
        
        # encode_image_to_base64 expects RGB input
        img_b64 = encode_image_to_base64(straightened_rgb)
        
    elif show_type == "annotated":
        if not annot.detection_box:
            return {"success": False, "error": "No detection box"}
        
        x1, y1, x2, y2 = [int(c) for c in annot.detection_box]
        
        # Add 10% padding
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        crop = frame[y1:y2, x1:x2].copy()
        offset_x, offset_y = x1, y1
        
        # Draw worm mask contour if available and show_mask is True
        if show_mask and annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists():
            mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask_crop = mask[y1:y2, x1:x2]
                contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(crop, contours, -1, (0, 255, 0), 1)
        
        # Draw head box if available
        if annot.head_box:
            hx1, hy1, hx2, hy2 = [int(c) for c in annot.head_box]
            cv2.rectangle(crop,
                         (hx1 - offset_x, hy1 - offset_y),
                         (hx2 - offset_x, hy2 - offset_y),
                         (0, 255, 0), 1)
        
        # Draw tail box if available
        if annot.tail_box:
            tx1, ty1, tx2, ty2 = [int(c) for c in annot.tail_box]
            cv2.rectangle(crop,
                         (tx1 - offset_x, ty1 - offset_y),
                         (tx2 - offset_x, ty2 - offset_y),
                         (0, 0, 255), 1)
        
        # Add label
        label_parts = [f"#{worm_id}"]
        if annot.health_class:
            label_parts.append(annot.health_class)
        label = " ".join(label_parts)
        cv2.putText(crop, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Convert BGR to RGB for encode_image_to_base64
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_b64 = encode_image_to_base64(crop_rgb)
    
    else:
        raise HTTPException(status_code=400, detail=f"Invalid show_type: {show_type}")
    
    return {
        "success": True,
        "video_path": video_path,
        "worm_id": worm_id,
        "image": img_b64
    }


@app.get("/api/tile/all_frames")
async def get_all_frames_for_tile(
    source: str = "current_folder",
    user: dict = Depends(require_auth)
):
    """
    Get list of all video frames for tile view.
    
    Args:
        source: "current_folder" or "all_folders"
        
    Returns:
        List of frame metadata including video path and worm count
    """
    if state.annotation_manager is None or state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    frames = []
    
    if source == "current_folder":
        # All videos in current folder
        current_folder_path = str(state.video_handler.current_folder) if state.video_handler.current_folder else None
        if current_folder_path:
            folder_videos = state.video_handler.videos_by_folder.get(current_folder_path, [])
            for video_path in folder_videos:
                video_path_str = str(video_path)
                worm_count = 0
                video_annots = state.annotation_manager.annotations.get(video_path_str)
                if video_annots:
                    worm_count = len(video_annots.annotations)
                
                # Get QC status
                qc_complete, _ = state.annotation_manager.get_video_qc_status(video_path_str)
                
                frames.append({
                    "video_path": video_path_str,
                    "video_name": Path(video_path_str).name,
                    "folder_name": state.video_handler.current_folder.name if state.video_handler.current_folder else "",
                    "folder_index": state.video_handler.current_folder_index,
                    "worm_count": worm_count,
                    "qc_complete": qc_complete
                })
    
    elif source == "all_folders":
        # All videos across all folders
        for folder_idx, folder_info in enumerate(state.video_handler.folder_list):
            folder_videos = state.video_handler.videos_by_folder.get(str(folder_info.path), [])
            for video_path in folder_videos:
                video_path_str = str(video_path)
                worm_count = 0
                video_annots = state.annotation_manager.annotations.get(video_path_str)
                if video_annots:
                    worm_count = len(video_annots.annotations)
                
                # Get QC status
                qc_complete, _ = state.annotation_manager.get_video_qc_status(video_path_str)
                
                frames.append({
                    "video_path": video_path_str,
                    "video_name": Path(video_path_str).name,
                    "folder_name": folder_info.name,
                    "folder_index": folder_idx,
                    "worm_count": worm_count,
                    "qc_complete": qc_complete
                })
    
    return {
        "success": True,
        "source": source,
        "frame_count": len(frames),
        "frames": frames
    }


@app.get("/api/tile/frame_image")
async def get_tile_frame_image(
    video_path: str,
    show_annotations: bool = True,
    show_mask: bool = True,
    user: dict = Depends(require_auth)
):
    """
    Get a full frame image for tile view.
    
    Args:
        video_path: Full path to the video
        show_annotations: Whether to draw annotations on the frame
        show_mask: Whether to show filled mask overlay (semi-transparent)
    """
    if state.annotation_manager is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    # Load the frame for this video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open video")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        raise HTTPException(status_code=500, detail="Cannot read frame from video")
    
    h, w = frame.shape[:2]
    
    if show_annotations:
        video_annots = state.annotation_manager.annotations.get(video_path)
        if video_annots:
            # First pass: draw filled masks if show_mask is True
            if show_mask:
                mask_overlay = frame.copy()
                for worm_id, annot in video_annots.annotations.items():
                    if annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists():
                        mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Color based on health class
                            color = (0, 255, 0)  # Default green BGR
                            if annot.health_class == "Healthy":
                                color = (0, 255, 0)  # Green
                            elif annot.health_class == "Slightly Unhealthy":
                                color = (0, 255, 255)  # Yellow
                            elif annot.health_class == "Unhealthy":
                                color = (0, 165, 255)  # Orange
                            elif annot.health_class == "Very Unhealthy":
                                color = (0, 0, 255)  # Red
                            
                            if annot.censored:
                                color = (128, 128, 128)
                            
                            # Fill mask area with color
                            mask_overlay[mask > 127] = color
                
                # Blend with original frame (40% mask, 60% original)
                frame = cv2.addWeighted(frame, 0.6, mask_overlay, 0.4, 0)
            
            # Second pass: draw boxes, labels, head/tail markers
            for worm_id, annot in video_annots.annotations.items():
                # Draw detection box
                if annot.detection_box:
                    x1, y1, x2, y2 = [int(c) for c in annot.detection_box]
                    
                    # Color based on health class
                    color = (0, 255, 0)  # Default green
                    if annot.health_class == "Healthy":
                        color = (0, 255, 0)  # Green
                    elif annot.health_class == "Slightly Unhealthy":
                        color = (0, 255, 255)  # Yellow
                    elif annot.health_class == "Unhealthy":
                        color = (0, 165, 255)  # Orange
                    elif annot.health_class == "Very Unhealthy":
                        color = (0, 0, 255)  # Red
                    
                    # Make censored worms gray
                    if annot.censored:
                        color = (128, 128, 128)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label_parts = [f"#{worm_id}"]
                    if annot.health_class:
                        label_parts.append(annot.health_class[:1])  # First letter
                    label = " ".join(label_parts)
                    
                    # Background for label
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # Draw head marker
                    if annot.head_box:
                        hx1, hy1, hx2, hy2 = [int(c) for c in annot.head_box]
                        hcx, hcy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
                        cv2.circle(frame, (hcx, hcy), 5, (0, 255, 0), -1)
                        cv2.circle(frame, (hcx, hcy), 6, (255, 255, 255), 1)
                    
                    # Draw tail marker
                    if annot.tail_box:
                        tx1, ty1, tx2, ty2 = [int(c) for c in annot.tail_box]
                        tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
                        cv2.circle(frame, (tcx, tcy), 5, (0, 0, 255), -1)
                        cv2.circle(frame, (tcx, tcy), 6, (255, 255, 255), 1)
                
                # Draw mask contour (always draw contours for visibility)
                if annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists():
                    mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cv2.drawContours(frame, contours, -1, (255, 255, 255), 1)
    
    # Convert BGR to RGB for encode_image_to_base64
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_b64 = encode_image_to_base64(frame_rgb)
    
    return {
        "success": True,
        "video_path": video_path,
        "image": img_b64,
        "width": w,
        "height": h
    }


@app.post("/api/tile/qc_video")
async def mark_video_qc(
    video_path: str,
    qc_complete: bool = True,
    user: dict = Depends(require_auth)
):
    """Mark a single video as QC complete/incomplete."""
    if state.annotation_manager is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    state.annotation_manager.set_video_qc_complete(video_path, qc_complete)
    state.annotation_manager.save_annotations()
    
    return {
        "success": True,
        "video_path": video_path,
        "qc_complete": qc_complete
    }


class QCAllRequest(BaseModel):
    video_paths: list[str]
    qc_complete: bool = True


@app.post("/api/tile/qc_all_visible")
async def mark_all_visible_qc(
    request: QCAllRequest,
    user: dict = Depends(require_auth)
):
    """Mark multiple videos as QC complete/incomplete."""
    if state.annotation_manager is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    updated = []
    for video_path in request.video_paths:
        state.annotation_manager.set_video_qc_complete(video_path, request.qc_complete)
        updated.append(video_path)
    
    state.annotation_manager.save_annotations()
    
    return {
        "success": True,
        "updated_count": len(updated),
        "qc_complete": request.qc_complete
    }


@app.get("/api/annotation/{worm_id}/straightened")
async def get_straightened_worm(worm_id: int, user: dict = Depends(require_auth)):
    """Get a straightened view of a worm along its skeleton."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame loaded")
    
    video_annots = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annots:
        raise HTTPException(status_code=404, detail="No annotations for current video")
    
    annot = video_annots.annotations.get(worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {worm_id} not found")
    
    if not annot.segmentation_mask_path or not Path(annot.segmentation_mask_path).exists():
        raise HTTPException(status_code=400, detail=f"Worm {worm_id} has no segmentation mask")
    
    # Check cache first
    cache_key = f"{state.current_video_path}_{worm_id}"
    cached = state.straightened_cache.get(cache_key)
    
    if cached and cached.get("video") == state.current_video_path:
        straightened = cached["image"]
        path = cached["path"]
        global_min_d = cached.get("global_min_d", 0)
    else:
        # Load mask
        mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise HTTPException(status_code=500, detail="Failed to load mask")
        
        # Straighten the worm
        straightened, path, global_min_d = straighten_worm(state.current_frame, mask)
        
        if straightened is None:
            raise HTTPException(status_code=500, detail="Failed to straighten worm - skeleton extraction failed")
        
        # Cache the result
        state.straightened_cache[cache_key] = {
            "image": straightened,
            "path": path,
            "global_min_d": global_min_d,
            "video": state.current_video_path
        }
    
    # Map existing head/tail annotations to straightened coordinates
    existing_head_x = None
    existing_tail_x = None
    
    if path:
        # Helper to find closest path index to a point
        def find_closest_path_index(orig_x, orig_y):
            """Find the path index closest to the given original image coordinates."""
            min_dist = float('inf')
            best_idx = 0
            for idx, (py, px) in enumerate(path):
                dist = (px - orig_x) ** 2 + (py - orig_y) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            return best_idx
        
        # Check for existing head annotation
        if annot.head_box:
            # Use center of head box
            hx = (annot.head_box[0] + annot.head_box[2]) / 2
            hy = (annot.head_box[1] + annot.head_box[3]) / 2
            existing_head_x = find_closest_path_index(hx, hy)
        elif hasattr(annot, 'head_line') and annot.head_line:
            # Use midpoint of head line
            hx = (annot.head_line[0] + annot.head_line[2]) / 2
            hy = (annot.head_line[1] + annot.head_line[3]) / 2
            existing_head_x = find_closest_path_index(hx, hy)
        
        # Check for existing tail annotation
        if annot.tail_box:
            # Use center of tail box
            tx = (annot.tail_box[0] + annot.tail_box[2]) / 2
            ty = (annot.tail_box[1] + annot.tail_box[3]) / 2
            existing_tail_x = find_closest_path_index(tx, ty)
        elif hasattr(annot, 'tail_line') and annot.tail_line:
            # Use midpoint of tail line
            tx = (annot.tail_line[0] + annot.tail_line[2]) / 2
            ty = (annot.tail_line[1] + annot.tail_line[3]) / 2
            existing_tail_x = find_closest_path_index(tx, ty)
    
    # Encode as base64
    straightened_b64 = encode_image_to_base64(straightened)
    
    return {
        "success": True,
        "worm_id": worm_id,
        "straightened_image": straightened_b64,
        "width": straightened.shape[1],
        "height": straightened.shape[0],
        "path_length": len(path),
        "existing_head_x": existing_head_x,
        "existing_tail_x": existing_tail_x
    }


class StraightenedLine(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class ApplyStraightenedAnnotation(BaseModel):
    worm_id: int
    head_line: StraightenedLine
    tail_line: StraightenedLine


@app.post("/api/annotation/straightened/apply")
async def apply_straightened_annotation(data: ApplyStraightenedAnnotation, user: dict = Depends(require_auth)):
    """
    Apply head/tail annotations from straightened view back to original coordinates.
    Maps the line positions from the straightened image back to the original image
    using the skeleton path.
    """
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    video_annots = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annots:
        raise HTTPException(status_code=404, detail="No annotations for current video")
    
    annot = video_annots.annotations.get(data.worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {data.worm_id} not found")
    
    if not annot.segmentation_mask_path or not Path(annot.segmentation_mask_path).exists():
        raise HTTPException(status_code=400, detail=f"Worm {data.worm_id} has no segmentation mask")
    
    # Load mask and regenerate skeleton path
    mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=500, detail="Failed to load mask")
    
    # Get the skeleton path (same as straighten_worm does)
    _, path, _ = straighten_worm(state.current_frame, mask)
    
    if not path:
        raise HTTPException(status_code=500, detail="Failed to extract skeleton path")
    
    # Map straightened X positions to skeleton path indices
    # The X coordinate in the straightened image corresponds to the path index
    def line_midpoint_x(line):
        return (line.x1 + line.x2) / 2
    
    head_x = line_midpoint_x(data.head_line)
    tail_x = line_midpoint_x(data.tail_line)
    
    print(f"[apply_straightened] Head line midpoint X: {head_x}, Tail line midpoint X: {tail_x}")
    print(f"[apply_straightened] Path length: {len(path)}")
    
    # Clamp to valid path indices
    head_idx = max(0, min(len(path) - 1, int(round(head_x))))
    tail_idx = max(0, min(len(path) - 1, int(round(tail_x))))
    
    print(f"[apply_straightened] Head path idx: {head_idx}, Tail path idx: {tail_idx}")
    
    # Get original image coordinates from path
    # Path is list of (row, col) tuples = (y, x)
    head_point = path[head_idx]  # (y, x) in original image
    tail_point = path[tail_idx]  # (y, x) in original image
    
    print(f"[apply_straightened] Head point (y,x): {head_point}, Tail point (y,x): {tail_point}")
    
    # Create small boxes around the points (lines would be better, but boxes work with current system)
    # Use a small box size centered on the point
    box_half_size = 15
    
    # Box format is (x1, y1, x2, y2)
    head_box = (
        head_point[1] - box_half_size,  # x1 (col - half)
        head_point[0] - box_half_size,  # y1 (row - half)
        head_point[1] + box_half_size,  # x2 (col + half)
        head_point[0] + box_half_size   # y2 (row + half)
    )
    
    tail_box = (
        tail_point[1] - box_half_size,  # x1
        tail_point[0] - box_half_size,  # y1
        tail_point[1] + box_half_size,  # x2
        tail_point[0] + box_half_size   # y2
    )
    
    print(f"[apply_straightened] Head box (x1,y1,x2,y2): {head_box}")
    print(f"[apply_straightened] Tail box (x1,y1,x2,y2): {tail_box}")
    
    # To properly map a point from straightened view to original image,
    # we need to account for the perpendicular offset (Y in straightened = offset from skeleton)
    # We need the tangent at each path point to compute perpendicular direction
    
    # Recompute tangents (same logic as straighten_worm)
    tangent_window = max(5, len(path) // 20)
    tangents = []
    for i in range(len(path)):
        prev_idx = max(0, i - tangent_window)
        next_idx = min(len(path) - 1, i + tangent_window)
        dy = path[next_idx][0] - path[prev_idx][0]
        dx = path[next_idx][1] - path[prev_idx][1]
        length = np.sqrt(dx*dx + dy*dy)
        if length > 0:
            tangents.append((dx/length, dy/length))
        elif tangents:
            tangents.append(tangents[-1])
        else:
            tangents.append((1, 0))
    
    # Get global_min_d from cache - this tells us how Y coordinates map to perpendicular offsets
    # Y=0 in straightened corresponds to d=global_min_d
    # Y=row in straightened corresponds to d=global_min_d + row
    # The skeleton (d=0) is at row = -global_min_d (since global_min_d is negative)
    cache_key = f"{state.current_video_path}_{data.worm_id}"
    cached = state.straightened_cache.get(cache_key, {})
    global_min_d = cached.get("global_min_d", 0)
    straightened_height = cached.get("image").shape[0] if cached.get("image") is not None else 150
    
    print(f"[apply_straightened] global_min_d: {global_min_d}, straightened_height: {straightened_height}")
    print(f"[apply_straightened] Skeleton row in straightened: {-global_min_d}")
    
    def map_straightened_to_original(sx, sy):
        """
        Map a point from straightened image coordinates to original image coordinates.
        
        sx: X coordinate in straightened image (= path index along skeleton)
        sy: Y coordinate in straightened image (= row index)
        
        The perpendicular offset d = sy + global_min_d
        (since row 0 corresponds to d=global_min_d)
        """
        path_idx = max(0, min(len(path) - 1, int(round(sx))))
        skel_y, skel_x = path[path_idx]  # Skeleton point in original image
        
        # Get perpendicular direction at this path point
        tx, ty = tangents[path_idx]
        nx, ny = -ty, tx  # Perpendicular: rotate tangent 90 degrees
        
        # Convert Y row to perpendicular offset
        # Row 0 -> d = global_min_d
        # Row sy -> d = global_min_d + sy
        perp_offset = global_min_d + sy
        
        # Map to original coordinates
        # The skeleton point is at d=0, we offset by perp_offset along perpendicular
        orig_y = skel_y + perp_offset * ny
        orig_x = skel_x + perp_offset * nx
        
        return (orig_x, orig_y)  # Return in (x, y) format
    
    # Map line endpoints to original image coordinates
    head_p1_orig = map_straightened_to_original(data.head_line.x1, data.head_line.y1)
    head_p2_orig = map_straightened_to_original(data.head_line.x2, data.head_line.y2)
    tail_p1_orig = map_straightened_to_original(data.tail_line.x1, data.tail_line.y1)
    tail_p2_orig = map_straightened_to_original(data.tail_line.x2, data.tail_line.y2)
    
    print(f"[apply_straightened] Head line orig: ({head_p1_orig[0]:.1f},{head_p1_orig[1]:.1f}) -> ({head_p2_orig[0]:.1f},{head_p2_orig[1]:.1f})")
    print(f"[apply_straightened] Tail line orig: ({tail_p1_orig[0]:.1f},{tail_p1_orig[1]:.1f}) -> ({tail_p2_orig[0]:.1f},{tail_p2_orig[1]:.1f})")
    
    # Store lines in (x1, y1, x2, y2) format
    head_line_orig = (head_p1_orig[0], head_p1_orig[1], head_p2_orig[0], head_p2_orig[1])
    tail_line_orig = (tail_p1_orig[0], tail_p1_orig[1], tail_p2_orig[0], tail_p2_orig[1])
    
    # Set the annotations - use lines if available, otherwise boxes
    state.annotation_manager.set_head_box(state.current_video_path, data.worm_id, head_box)
    state.annotation_manager.set_tail_box(state.current_video_path, data.worm_id, tail_box)
    
    # Also set lines if supported
    if hasattr(annot, 'head_line'):
        annot.head_line = head_line_orig
    if hasattr(annot, 'tail_line'):
        annot.tail_line = tail_line_orig
    
    state.annotation_manager.save_annotations()
    
    return {
        "success": True,
        "worm_id": data.worm_id,
        "head_box": head_box,
        "tail_box": tail_box,
        "head_point": (head_point[1], head_point[0]),  # (x, y) format
        "tail_point": (tail_point[1], tail_point[0]),  # (x, y) format
        "head_path_idx": head_idx,
        "tail_path_idx": tail_idx,
        "path_length": len(path)
    }


@app.delete("/api/annotation/{worm_id}/box/{box_type}")
async def delete_box(worm_id: int, box_type: str, user: dict = Depends(require_auth)):
    """Delete a specific box for a worm (head or tail)."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    video_annots = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annots:
        raise HTTPException(status_code=404, detail="No annotations for current video")
    
    annot = video_annots.annotations.get(worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {worm_id} not found")
    
    if box_type == "head":
        annot.head_box = None
        annot.head_line = None
    elif box_type == "tail":
        annot.tail_box = None
        annot.tail_line = None
    else:
        raise HTTPException(status_code=400, detail=f"Invalid box type: {box_type}. Use 'head' or 'tail'")
    
    state.annotation_manager.save_annotations()
    return {"success": True, "worm_id": worm_id, "deleted_box": box_type}


@app.delete("/api/annotation/{worm_id}/mask/{mask_type}")
async def delete_mask(worm_id: int, mask_type: str, user: dict = Depends(require_auth)):
    """Delete a specific mask for a worm (worm, head, or tail)."""
    if state.annotation_manager is None or state.current_video_path is None:
        raise HTTPException(status_code=400, detail="No video loaded")
    
    video_annots = state.annotation_manager.annotations.get(state.current_video_path)
    if not video_annots:
        raise HTTPException(status_code=404, detail="No annotations for current video")
    
    annot = video_annots.annotations.get(worm_id)
    if not annot:
        raise HTTPException(status_code=404, detail=f"Worm {worm_id} not found")
    
    # Delete the mask file and clear the path
    mask_path = None
    if mask_type == "worm" and annot.segmentation_mask_path:
        mask_path = Path(annot.segmentation_mask_path)
        annot.segmentation_mask_path = None
    elif mask_type == "head" and annot.head_mask_path:
        mask_path = Path(annot.head_mask_path)
        annot.head_mask_path = None
    elif mask_type == "tail" and annot.tail_mask_path:
        mask_path = Path(annot.tail_mask_path)
        annot.tail_mask_path = None
    else:
        raise HTTPException(status_code=400, detail=f"Invalid mask type or mask not found: {mask_type}")
    
    # Delete the actual file
    if mask_path and mask_path.exists():
        mask_path.unlink()
    
    state.annotation_manager.save_annotations()
    return {"success": True, "worm_id": worm_id, "deleted_mask": mask_type}


@app.post("/api/save")
async def save_annotations(user: dict = Depends(require_auth)):
    """Manually save annotations."""
    if state.annotation_manager is None:
        raise HTTPException(status_code=400, detail="No annotations to save")
    
    success = state.annotation_manager.save_annotations()
    return {"success": success}


@app.post("/api/settings/confidence")
async def set_confidence_threshold(settings: ConfidenceSettings, user: dict = Depends(require_auth)):
    """Set YOLO detection confidence threshold."""
    if settings.confidence < 0.1 or settings.confidence > 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.1 and 1.0")
    
    state.confidence_threshold = settings.confidence
    return {"success": True, "confidence": state.confidence_threshold}


@app.get("/api/settings")
async def get_settings(user: dict = Depends(require_auth)):
    """Get current settings."""
    return {
        "confidence_threshold": state.confidence_threshold
    }


@app.get("/api/export/summary")
async def get_export_summary(user: dict = Depends(require_auth)):
    """Get summary statistics for export modal display."""
    if state.annotation_manager is None:
        raise HTTPException(status_code=400, detail="No annotations available")
    
    # Collect stats by chip
    chip_stats = {}  # chip_name -> {total_worms, healthy_count, leaky_count, scores}
    total_worms = 0
    total_healthy = 0
    total_leaky = 0
    all_scores = []
    videos_set = set()
    with_masks = 0
    qc_complete_count = 0
    
    for video_path, video_annot in state.annotation_manager.annotations.items():
        videos_set.add(video_path)
        
        # Check QC status
        if video_annot.qc_complete:
            qc_complete_count += 1
        
        # Extract chip name from path (e.g., "CHIP1" from path)
        video_path_obj = Path(video_path)
        # Try to find chip in path components
        chip_name = "Unknown"
        for part in video_path_obj.parts:
            if "chip" in part.lower():
                chip_name = part
                break
        # If not found, use parent folder name
        if chip_name == "Unknown":
            chip_name = video_path_obj.parent.name
        
        if chip_name not in chip_stats:
            chip_stats[chip_name] = {
                "total_worms": 0,
                "healthy_count": 0,
                "leaky_count": 0,
                "scores": [],
                "videos": set()
            }
        
        chip_stats[chip_name]["videos"].add(video_path)
        
        for worm_id, annot in video_annot.annotations.items():
            total_worms += 1
            chip_stats[chip_name]["total_worms"] += 1
            
            # Count masks
            if annot.segmentation_mask_path:
                with_masks += 1
            
            # Health classification
            if annot.health_score is not None:
                score = annot.health_score
                all_scores.append(score)
                chip_stats[chip_name]["scores"].append(score)
                
                # Score >= 0.5 = Leaky, < 0.5 = Healthy
                if score >= 0.5:
                    total_leaky += 1
                    chip_stats[chip_name]["leaky_count"] += 1
                else:
                    total_healthy += 1
                    chip_stats[chip_name]["healthy_count"] += 1
    
    # Calculate project summary
    avg_score = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0
    healthy_pct = round(total_healthy / total_worms * 100, 1) if total_worms > 0 else 0
    leaky_pct = round(total_leaky / total_worms * 100, 1) if total_worms > 0 else 0
    
    project_summary = {
        "total_worms": total_worms,
        "total_videos": len(videos_set),
        "total_chips": len(chip_stats),
        "healthy_count": total_healthy,
        "leaky_count": total_leaky,
        "healthy_pct": healthy_pct,
        "leaky_pct": leaky_pct,
        "avg_score": avg_score,
        "with_masks": with_masks,
        "qc_complete_count": qc_complete_count
    }
    
    # Calculate chip summaries
    chip_summary = []
    for chip_name, stats in sorted(chip_stats.items()):
        chip_avg = round(sum(stats["scores"]) / len(stats["scores"]), 3) if stats["scores"] else 0
        chip_healthy_pct = round(stats["healthy_count"] / stats["total_worms"] * 100, 1) if stats["total_worms"] > 0 else 0
        chip_summary.append({
            "chip": chip_name,
            "total_worms": stats["total_worms"],
            "healthy_count": stats["healthy_count"],
            "leaky_count": stats["leaky_count"],
            "healthy_pct": chip_healthy_pct,
            "avg_score": chip_avg,
            "video_count": len(stats["videos"])
        })
    
    return {
        "project_summary": project_summary,
        "chip_summary": chip_summary
    }


@app.get("/api/export/training/preview")
async def get_training_preview(
    qc_only: bool = True,
    skip_censored: bool = True,
    require_mask: bool = True,
    user: dict = Depends(require_auth)
):
    """Get preview statistics for training dataset export."""
    if state.annotation_manager is None:
        raise HTTPException(status_code=400, detail="No annotations available")
    
    total_videos = 0
    total_worms = 0
    with_masks = 0
    qc_videos = 0
    non_censored = 0
    eligible_videos = 0
    eligible_worms = 0
    
    for video_path, video_annot in state.annotation_manager.annotations.items():
        total_videos += 1
        
        if video_annot.qc_complete:
            qc_videos += 1
        
        video_eligible = not qc_only or video_annot.qc_complete
        worms_in_video = 0
        
        for worm_id, annot in video_annot.annotations.items():
            total_worms += 1
            
            if annot.segmentation_mask_path:
                with_masks += 1
            
            if not annot.censored:
                non_censored += 1
            
            # Check if this worm would be included
            if not annot.detection_box:
                continue
            if skip_censored and annot.censored:
                continue
            if require_mask and not annot.segmentation_mask_path:
                continue
            
            if video_eligible:
                eligible_worms += 1
                worms_in_video += 1
        
        if video_eligible and worms_in_video > 0:
            eligible_videos += 1
    
    return {
        "total_videos": total_videos,
        "total_worms": total_worms,
        "qc_videos": qc_videos,
        "with_masks": with_masks,
        "non_censored": non_censored,
        "eligible_videos": eligible_videos,
        "eligible_worms": eligible_worms
    }


@app.get("/api/export/excel")
async def export_to_excel(mode: str = "project", exclude_censored: bool = True, user: dict = Depends(require_auth)):
    """Export annotations to Excel file.
    
    Args:
        mode: Export mode - 'project' (single file), 'chip' (per-chip files), 'both'
        exclude_censored: If True, exclude censored worms from export
    """
    if state.annotation_manager is None:
        raise HTTPException(status_code=400, detail="No annotations to export")
    
    try:
        import pandas as pd
        from io import BytesIO
        import zipfile
        from datetime import datetime
        
        # Collect all annotation data into rows and organize by chip
        rows = []
        chip_rows = {}  # chip_name -> list of rows
        
        for video_path, video_annot in state.annotation_manager.annotations.items():
            video_name = Path(video_path).name
            video_path_obj = Path(video_path)
            
            # Extract chip name from path
            chip_name = "Unknown"
            for part in video_path_obj.parts:
                if "chip" in part.lower():
                    chip_name = part
                    break
            if chip_name == "Unknown":
                chip_name = video_path_obj.parent.name
            
            if chip_name not in chip_rows:
                chip_rows[chip_name] = []
            
            for worm_id, annot in video_annot.annotations.items():
                # Skip censored worms if requested
                if exclude_censored and annot.censored:
                    continue
                    
                # Determine health classification
                health_class = "Unknown"
                if annot.health_score is not None:
                    health_class = "Leaky" if annot.health_score >= 0.5 else "Healthy"
                
                # Calculate worm area and length from mask if available
                worm_area = None
                worm_length = None
                if annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists():
                    try:
                        mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Calculate area (number of pixels in mask)
                            binary = (mask > 128).astype(np.uint8)
                            worm_area = int(np.sum(binary))
                            
                            # Calculate skeleton length
                            if worm_area > 0:
                                skeleton = skeletonize(binary > 0)
                                skeleton_pixels = int(np.sum(skeleton))
                                
                                # Find endpoints and trace for more accurate length
                                kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
                                neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
                                endpoints = skeleton & (neighbor_count == 1)
                                endpoint_coords = np.where(endpoints)
                                
                                worm_length = skeleton_pixels  # Default to pixel count
                                
                                # If we have 2+ endpoints, compute geodesic distance
                                if len(endpoint_coords[0]) >= 2:
                                    start = (endpoint_coords[0][0], endpoint_coords[1][0])
                                    end = (endpoint_coords[0][-1], endpoint_coords[1][-1])
                                    cost = np.where(skeleton, 1, 1000)
                                    try:
                                        path, _ = route_through_array(cost, start, end, fully_connected=True)
                                        worm_length = len(path)
                                    except:
                                        pass
                    except Exception as e:
                        print(f"[export] Error computing mask stats for {video_name} worm {worm_id}: {e}")
                
                row = {
                    'video_path': video_path,
                    'video_name': video_name,
                    'chip': chip_name,
                    'worm_id': worm_id,
                    'confidence': annot.confidence,
                    'health_score': annot.health_score,
                    'health_classification': health_class,
                    'worm_area': worm_area,
                    'worm_length': worm_length,
                    'detection_box_x1': annot.detection_box[0] if annot.detection_box else None,
                    'detection_box_y1': annot.detection_box[1] if annot.detection_box else None,
                    'detection_box_x2': annot.detection_box[2] if annot.detection_box else None,
                    'detection_box_y2': annot.detection_box[3] if annot.detection_box else None,
                    'head_box_x1': annot.head_box[0] if annot.head_box else None,
                    'head_box_y1': annot.head_box[1] if annot.head_box else None,
                    'head_box_x2': annot.head_box[2] if annot.head_box else None,
                    'head_box_y2': annot.head_box[3] if annot.head_box else None,
                    'tail_box_x1': annot.tail_box[0] if annot.tail_box else None,
                    'tail_box_y1': annot.tail_box[1] if annot.tail_box else None,
                    'tail_box_x2': annot.tail_box[2] if annot.tail_box else None,
                    'tail_box_y2': annot.tail_box[3] if annot.tail_box else None,
                    'head_line_x1': annot.head_line[0] if annot.head_line else None,
                    'head_line_y1': annot.head_line[1] if annot.head_line else None,
                    'head_line_x2': annot.head_line[2] if annot.head_line else None,
                    'head_line_y2': annot.head_line[3] if annot.head_line else None,
                    'tail_line_x1': annot.tail_line[0] if annot.tail_line else None,
                    'tail_line_y1': annot.tail_line[1] if annot.tail_line else None,
                    'tail_line_x2': annot.tail_line[2] if annot.tail_line else None,
                    'tail_line_y2': annot.tail_line[3] if annot.tail_line else None,
                    'has_worm_mask': annot.segmentation_mask_path is not None,
                    'has_head_mask': annot.head_mask_path is not None,
                    'has_tail_mask': annot.tail_mask_path is not None,
                    'censored': annot.censored,
                    'worm_mask_path': annot.segmentation_mask_path,
                    'head_mask_path': annot.head_mask_path,
                    'tail_mask_path': annot.tail_mask_path,
                }
                
                # Calculate line lengths
                if annot.head_line:
                    x1, y1, x2, y2 = annot.head_line
                    row['head_line_length'] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                else:
                    row['head_line_length'] = None
                    
                if annot.tail_line:
                    x1, y1, x2, y2 = annot.tail_line
                    row['tail_line_length'] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                else:
                    row['tail_line_length'] = None
                
                rows.append(row)
                chip_rows[chip_name].append(row)
        
        if not rows:
            raise HTTPException(status_code=400, detail="No annotation data to export")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        def create_project_excel(rows_data):
            """Create a single Excel file with all data and summaries."""
            df = pd.DataFrame(rows_data)
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main annotations sheet
                df.to_excel(writer, sheet_name='Annotations', index=False)
                
                # Project summary sheet
                total_worms = len(rows_data)
                healthy_count = sum(1 for r in rows_data if r.get('health_classification') == 'Healthy')
                leaky_count = sum(1 for r in rows_data if r.get('health_classification') == 'Leaky')
                scores = [r['health_score'] for r in rows_data if r.get('health_score') is not None]
                avg_score = sum(scores) / len(scores) if scores else 0
                with_masks = sum(1 for r in rows_data if r.get('has_worm_mask'))
                
                # Area and length statistics
                areas = [r['worm_area'] for r in rows_data if r.get('worm_area') is not None]
                lengths = [r['worm_length'] for r in rows_data if r.get('worm_length') is not None]
                avg_area = sum(areas) / len(areas) if areas else 0
                avg_length = sum(lengths) / len(lengths) if lengths else 0
                
                summary_data = {
                    'Metric': ['Total Worms', 'Healthy Count', 'Leaky Count', 'Healthy %', 'Leaky %', 'Avg Health Score', 'With Masks', 'Avg Worm Area (px)', 'Avg Worm Length (px)'],
                    'Value': [
                        total_worms,
                        healthy_count,
                        leaky_count,
                        round(healthy_count / total_worms * 100, 1) if total_worms > 0 else 0,
                        round(leaky_count / total_worms * 100, 1) if total_worms > 0 else 0,
                        round(avg_score, 3),
                        with_masks,
                        round(avg_area, 1),
                        round(avg_length, 1)
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Project Summary', index=False)
                
                # Chip summary sheet
                chip_summary = []
                for chip_name in sorted(chip_rows.keys()):
                    chip_data = chip_rows[chip_name]
                    chip_total = len(chip_data)
                    chip_healthy = sum(1 for r in chip_data if r.get('health_classification') == 'Healthy')
                    chip_leaky = sum(1 for r in chip_data if r.get('health_classification') == 'Leaky')
                    chip_scores = [r['health_score'] for r in chip_data if r.get('health_score') is not None]
                    chip_avg = sum(chip_scores) / len(chip_scores) if chip_scores else 0
                    chip_areas = [r['worm_area'] for r in chip_data if r.get('worm_area') is not None]
                    chip_lengths = [r['worm_length'] for r in chip_data if r.get('worm_length') is not None]
                    chip_avg_area = sum(chip_areas) / len(chip_areas) if chip_areas else 0
                    chip_avg_length = sum(chip_lengths) / len(chip_lengths) if chip_lengths else 0
                    
                    chip_summary.append({
                        'Chip': chip_name,
                        'Total Worms': chip_total,
                        'Healthy': chip_healthy,
                        'Leaky': chip_leaky,
                        'Healthy %': round(chip_healthy / chip_total * 100, 1) if chip_total > 0 else 0,
                        'Avg Score': round(chip_avg, 3),
                        'Avg Area (px)': round(chip_avg_area, 1),
                        'Avg Length (px)': round(chip_avg_length, 1)
                    })
                
                pd.DataFrame(chip_summary).to_excel(writer, sheet_name='Chip Summary', index=False)
            
            output.seek(0)
            return output
        
        def create_chip_excel(chip_name, chip_data):
            """Create Excel file for a single chip."""
            df = pd.DataFrame(chip_data)
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Annotations', index=False)
                
                # Chip summary
                total_worms = len(chip_data)
                healthy_count = sum(1 for r in chip_data if r.get('health_classification') == 'Healthy')
                leaky_count = sum(1 for r in chip_data if r.get('health_classification') == 'Leaky')
                scores = [r['health_score'] for r in chip_data if r.get('health_score') is not None]
                avg_score = sum(scores) / len(scores) if scores else 0
                
                summary_data = {
                    'Metric': ['Chip', 'Total Worms', 'Healthy', 'Leaky', 'Healthy %', 'Avg Score'],
                    'Value': [
                        chip_name,
                        total_worms,
                        healthy_count,
                        leaky_count,
                        round(healthy_count / total_worms * 100, 1) if total_worms > 0 else 0,
                        round(avg_score, 3)
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            output.seek(0)
            return output
        
        # Handle different export modes
        if mode == "project":
            output = create_project_excel(rows)
            filename = f"worm_annotations_{timestamp}.xlsx"
            return StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        
        elif mode == "chip":
            # Create a ZIP file with per-chip Excel files
            zip_output = BytesIO()
            with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zf:
                for chip_name, chip_data in sorted(chip_rows.items()):
                    if chip_data:
                        excel_output = create_chip_excel(chip_name, chip_data)
                        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in chip_name)
                        zf.writestr(f"{safe_name}_{timestamp}.xlsx", excel_output.read())
            
            zip_output.seek(0)
            filename = f"chip_exports_{timestamp}.zip"
            return StreamingResponse(
                zip_output,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        
        elif mode == "both":
            # Create a ZIP file with project file and per-chip files
            zip_output = BytesIO()
            with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add project-level file
                project_output = create_project_excel(rows)
                zf.writestr(f"project_annotations_{timestamp}.xlsx", project_output.read())
                
                # Add per-chip files in a subfolder
                for chip_name, chip_data in sorted(chip_rows.items()):
                    if chip_data:
                        excel_output = create_chip_excel(chip_name, chip_data)
                        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in chip_name)
                        zf.writestr(f"per_chip/{safe_name}_{timestamp}.xlsx", excel_output.read())
            
            zip_output.seek(0)
            filename = f"worm_export_all_{timestamp}.zip"
            return StreamingResponse(
                zip_output,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid export mode: {mode}")
        
    except ImportError as e:
        # Fallback to JSON if pandas/openpyxl not available
        all_annotations = {}
        for video_path, video_annot in state.annotation_manager.annotations.items():
            all_annotations[video_path] = video_annot.to_dict()
        
        return JSONResponse(
            content=all_annotations,
            headers={"Content-Disposition": "attachment; filename=worm_annotations.json"}
        )


@app.get("/api/export/excel/stream")
async def export_to_excel_stream(mode: str = "project", exclude_censored: bool = True, user: dict = Depends(require_auth)):
    """Export annotations to Excel file with streaming progress updates via SSE.
    
    Args:
        mode: Export mode - 'project' (single file), 'chip' (per-chip files), 'both'
        exclude_censored: If True, exclude censored worms from export
    """
    if state.annotation_manager is None:
        raise HTTPException(status_code=400, detail="No annotations to export")
    
    if state.export_state.is_running:
        raise HTTPException(status_code=400, detail="Export already in progress")
    
    async def generate_export_progress():
        """Generator for SSE progress events."""
        import pandas as pd
        from io import BytesIO
        import zipfile
        from datetime import datetime
        import base64
        
        state.export_state.reset()
        state.export_state.is_running = True
        
        try:
            # First pass: count total worms to export
            total_worms = 0
            for video_path, video_annot in state.annotation_manager.annotations.items():
                for worm_id, annot in video_annot.annotations.items():
                    if exclude_censored and annot.censored:
                        continue
                    total_worms += 1
            
            state.export_state.total_worms = total_worms
            
            yield f"data: {json.dumps({'type': 'start', 'total_worms': total_worms})}\n\n"
            
            # Collect all annotation data into rows and organize by chip
            rows = []
            chip_rows = {}  # chip_name -> list of rows
            processed = 0
            
            for video_path, video_annot in state.annotation_manager.annotations.items():
                if state.export_state.should_cancel:
                    yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                    return
                
                video_name = Path(video_path).name
                video_path_obj = Path(video_path)
                state.export_state.current_video = video_name
                
                # Extract chip name from path
                chip_name = "Unknown"
                for part in video_path_obj.parts:
                    if "chip" in part.lower():
                        chip_name = part
                        break
                if chip_name == "Unknown":
                    chip_name = video_path_obj.parent.name
                
                if chip_name not in chip_rows:
                    chip_rows[chip_name] = []
                
                for worm_id, annot in video_annot.annotations.items():
                    # Skip censored worms if requested
                    if exclude_censored and annot.censored:
                        continue
                    
                    processed += 1
                    state.export_state.current_worm = processed
                    
                    # Send progress update every 10 worms or on last worm
                    if processed % 10 == 0 or processed == total_worms:
                        yield f"data: {json.dumps({'type': 'progress', 'current': processed, 'total': total_worms, 'video': video_name})}\n\n"
                    
                    # Determine health classification
                    health_class = "Unknown"
                    if annot.health_score is not None:
                        health_class = "Leaky" if annot.health_score >= 0.5 else "Healthy"
                    
                    # Calculate worm area and length from mask if available
                    worm_area = None
                    worm_length = None
                    if annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists():
                        try:
                            mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
                            if mask is not None:
                                state.export_state.masks_processed += 1
                                # Calculate area (number of pixels in mask)
                                binary = (mask > 128).astype(np.uint8)
                                worm_area = int(np.sum(binary))
                                
                                # Calculate skeleton length
                                if worm_area > 0:
                                    skeleton = skeletonize(binary > 0)
                                    skeleton_pixels = int(np.sum(skeleton))
                                    
                                    # Find endpoints and trace for more accurate length
                                    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
                                    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
                                    endpoints = skeleton & (neighbor_count == 1)
                                    endpoint_coords = np.where(endpoints)
                                    
                                    worm_length = skeleton_pixels  # Default to pixel count
                                    
                                    # If we have 2+ endpoints, compute geodesic distance
                                    if len(endpoint_coords[0]) >= 2:
                                        start = (endpoint_coords[0][0], endpoint_coords[1][0])
                                        end = (endpoint_coords[0][-1], endpoint_coords[1][-1])
                                        cost = np.where(skeleton, 1, 1000)
                                        try:
                                            path, _ = route_through_array(cost, start, end, fully_connected=True)
                                            worm_length = len(path)
                                        except:
                                            pass
                        except Exception as e:
                            print(f"[export] Error computing mask stats for {video_name} worm {worm_id}: {e}")
                    
                    row = {
                        'video_path': video_path,
                        'video_name': video_name,
                        'chip': chip_name,
                        'worm_id': worm_id,
                        'confidence': annot.confidence,
                        'health_score': annot.health_score,
                        'health_classification': health_class,
                        'worm_area': worm_area,
                        'worm_length': worm_length,
                        'detection_box_x1': annot.detection_box[0] if annot.detection_box else None,
                        'detection_box_y1': annot.detection_box[1] if annot.detection_box else None,
                        'detection_box_x2': annot.detection_box[2] if annot.detection_box else None,
                        'detection_box_y2': annot.detection_box[3] if annot.detection_box else None,
                        'head_box_x1': annot.head_box[0] if annot.head_box else None,
                        'head_box_y1': annot.head_box[1] if annot.head_box else None,
                        'head_box_x2': annot.head_box[2] if annot.head_box else None,
                        'head_box_y2': annot.head_box[3] if annot.head_box else None,
                        'tail_box_x1': annot.tail_box[0] if annot.tail_box else None,
                        'tail_box_y1': annot.tail_box[1] if annot.tail_box else None,
                        'tail_box_x2': annot.tail_box[2] if annot.tail_box else None,
                        'tail_box_y2': annot.tail_box[3] if annot.tail_box else None,
                        'head_line_x1': annot.head_line[0] if annot.head_line else None,
                        'head_line_y1': annot.head_line[1] if annot.head_line else None,
                        'head_line_x2': annot.head_line[2] if annot.head_line else None,
                        'head_line_y2': annot.head_line[3] if annot.head_line else None,
                        'tail_line_x1': annot.tail_line[0] if annot.tail_line else None,
                        'tail_line_y1': annot.tail_line[1] if annot.tail_line else None,
                        'tail_line_x2': annot.tail_line[2] if annot.tail_line else None,
                        'tail_line_y2': annot.tail_line[3] if annot.tail_line else None,
                        'has_worm_mask': annot.segmentation_mask_path is not None,
                        'has_head_mask': annot.head_mask_path is not None,
                        'has_tail_mask': annot.tail_mask_path is not None,
                        'censored': annot.censored,
                        'worm_mask_path': annot.segmentation_mask_path,
                        'head_mask_path': annot.head_mask_path,
                        'tail_mask_path': annot.tail_mask_path,
                    }
                    
                    # Calculate line lengths
                    if annot.head_line:
                        x1, y1, x2, y2 = annot.head_line
                        row['head_line_length'] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                    else:
                        row['head_line_length'] = None
                        
                    if annot.tail_line:
                        x1, y1, x2, y2 = annot.tail_line
                        row['tail_line_length'] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                    else:
                        row['tail_line_length'] = None
                    
                    rows.append(row)
                    chip_rows[chip_name].append(row)
            
            if not rows:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No annotation data to export'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'building', 'message': 'Building Excel file...'})}\n\n"
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            def create_project_excel(rows_data):
                """Create a single Excel file with all data and summaries."""
                df = pd.DataFrame(rows_data)
                output = BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Main annotations sheet
                    df.to_excel(writer, sheet_name='Annotations', index=False)
                    
                    # Project summary sheet
                    total_worms = len(rows_data)
                    healthy_count = sum(1 for r in rows_data if r.get('health_classification') == 'Healthy')
                    leaky_count = sum(1 for r in rows_data if r.get('health_classification') == 'Leaky')
                    scores = [r['health_score'] for r in rows_data if r.get('health_score') is not None]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    with_masks = sum(1 for r in rows_data if r.get('has_worm_mask'))
                    
                    # Area and length statistics
                    areas = [r['worm_area'] for r in rows_data if r.get('worm_area') is not None]
                    lengths = [r['worm_length'] for r in rows_data if r.get('worm_length') is not None]
                    avg_area = sum(areas) / len(areas) if areas else 0
                    avg_length = sum(lengths) / len(lengths) if lengths else 0
                    
                    summary_data = {
                        'Metric': ['Total Worms', 'Healthy Count', 'Leaky Count', 'Healthy %', 'Leaky %', 'Avg Health Score', 'With Masks', 'Avg Worm Area (px)', 'Avg Worm Length (px)'],
                        'Value': [
                            total_worms,
                            healthy_count,
                            leaky_count,
                            round(healthy_count / total_worms * 100, 1) if total_worms > 0 else 0,
                            round(leaky_count / total_worms * 100, 1) if total_worms > 0 else 0,
                            round(avg_score, 3),
                            with_masks,
                            round(avg_area, 1),
                            round(avg_length, 1)
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Project Summary', index=False)
                    
                    # Chip summary sheet
                    chip_summary = []
                    for cn in sorted(chip_rows.keys()):
                        cd = chip_rows[cn]
                        chip_total = len(cd)
                        chip_healthy = sum(1 for r in cd if r.get('health_classification') == 'Healthy')
                        chip_leaky = sum(1 for r in cd if r.get('health_classification') == 'Leaky')
                        chip_scores = [r['health_score'] for r in cd if r.get('health_score') is not None]
                        chip_avg = sum(chip_scores) / len(chip_scores) if chip_scores else 0
                        chip_areas = [r['worm_area'] for r in cd if r.get('worm_area') is not None]
                        chip_lengths = [r['worm_length'] for r in cd if r.get('worm_length') is not None]
                        chip_avg_area = sum(chip_areas) / len(chip_areas) if chip_areas else 0
                        chip_avg_length = sum(chip_lengths) / len(chip_lengths) if chip_lengths else 0
                        
                        chip_summary.append({
                            'Chip': cn,
                            'Total Worms': chip_total,
                            'Healthy': chip_healthy,
                            'Leaky': chip_leaky,
                            'Healthy %': round(chip_healthy / chip_total * 100, 1) if chip_total > 0 else 0,
                            'Avg Score': round(chip_avg, 3),
                            'Avg Area (px)': round(chip_avg_area, 1),
                            'Avg Length (px)': round(chip_avg_length, 1)
                        })
                    
                    pd.DataFrame(chip_summary).to_excel(writer, sheet_name='Chip Summary', index=False)
                
                output.seek(0)
                return output
            
            def create_chip_excel(cn, cd):
                """Create Excel file for a single chip."""
                df = pd.DataFrame(cd)
                output = BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Annotations', index=False)
                    
                    # Chip summary
                    total_worms = len(cd)
                    healthy_count = sum(1 for r in cd if r.get('health_classification') == 'Healthy')
                    leaky_count = sum(1 for r in cd if r.get('health_classification') == 'Leaky')
                    scores = [r['health_score'] for r in cd if r.get('health_score') is not None]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    
                    summary_data = {
                        'Metric': ['Chip', 'Total Worms', 'Healthy', 'Leaky', 'Healthy %', 'Avg Score'],
                        'Value': [
                            cn,
                            total_worms,
                            healthy_count,
                            leaky_count,
                            round(healthy_count / total_worms * 100, 1) if total_worms > 0 else 0,
                            round(avg_score, 3)
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                output.seek(0)
                return output
            
            # Handle different export modes
            if mode == "project":
                output = create_project_excel(rows)
                filename = f"worm_annotations_{timestamp}.xlsx"
                file_data = base64.b64encode(output.read()).decode('utf-8')
                media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                
            elif mode == "chip":
                # Create a ZIP file with per-chip Excel files
                zip_output = BytesIO()
                with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for cn, cd in sorted(chip_rows.items()):
                        if cd:
                            excel_output = create_chip_excel(cn, cd)
                            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in cn)
                            zf.writestr(f"{safe_name}_{timestamp}.xlsx", excel_output.read())
                
                zip_output.seek(0)
                filename = f"chip_exports_{timestamp}.zip"
                file_data = base64.b64encode(zip_output.read()).decode('utf-8')
                media_type = "application/zip"
                
            elif mode == "both":
                # Create a ZIP file with project file and per-chip files
                zip_output = BytesIO()
                with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # Add project-level file
                    project_output = create_project_excel(rows)
                    zf.writestr(f"project_annotations_{timestamp}.xlsx", project_output.read())
                    
                    # Add per-chip files in a subfolder
                    for cn, cd in sorted(chip_rows.items()):
                        if cd:
                            excel_output = create_chip_excel(cn, cd)
                            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in cn)
                            zf.writestr(f"per_chip/{safe_name}_{timestamp}.xlsx", excel_output.read())
                
                zip_output.seek(0)
                filename = f"worm_export_all_{timestamp}.zip"
                file_data = base64.b64encode(zip_output.read()).decode('utf-8')
                media_type = "application/zip"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Invalid export mode: {mode}'})}\n\n"
                return
            
            state.export_state.complete = True
            yield f"data: {json.dumps({'type': 'complete', 'filename': filename, 'media_type': media_type, 'data': file_data, 'masks_processed': state.export_state.masks_processed})}\n\n"
            
        except Exception as e:
            state.export_state.error = str(e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            state.export_state.is_running = False
    
    return StreamingResponse(
        generate_export_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/export/cancel")
async def cancel_export(user: dict = Depends(require_auth)):
    """Cancel the ongoing export."""
    state.export_state.should_cancel = True
    return {"status": "cancellation requested"}


@app.get("/api/export/training")
async def export_training_dataset(
    format: str = "yolo_detect",
    train_split: float = 0.8,
    qc_only: bool = True,
    skip_censored: bool = True,
    require_mask: bool = True,
    user: dict = Depends(require_auth)
):
    """Export annotations as YOLO training dataset.
    
    Args:
        format: 'yolo_detect' (bounding boxes), 'yolo_segment' (polygons), or 'both'
        train_split: Fraction of data for training (0.5-0.95), rest goes to validation
        qc_only: Only include QC'd videos
        skip_censored: Exclude censored worms
        require_mask: Only include worms with segmentation masks (required for segment format)
    """
    if state.annotation_manager is None:
        raise HTTPException(status_code=400, detail="No annotations to export")
    
    if state.video_handler is None:
        raise HTTPException(status_code=400, detail="No video folder loaded")
    
    try:
        import cv2
        import numpy as np
        from io import BytesIO
        import zipfile
        from datetime import datetime
        import random
        
        # Collect all valid videos and their annotations
        video_data = []  # List of (video_path, frame, worm_annotations)
        
        for video_path, video_annot in state.annotation_manager.annotations.items():
            # Check if video has QC status
            if qc_only and not video_annot.qc_complete:
                continue
            
            # Get valid annotations for this video
            valid_annotations = []
            for worm_id, annot in video_annot.annotations.items():
                if skip_censored and annot.censored:
                    continue
                if require_mask and not annot.segmentation_mask_path:
                    continue
                if not annot.detection_box:
                    continue
                valid_annotations.append((worm_id, annot))
            
            if valid_annotations:
                video_data.append((video_path, valid_annotations))
        
        if not video_data:
            raise HTTPException(status_code=400, detail="No valid annotations found with the selected criteria")
        
        # Shuffle and split by video (to avoid train/val leakage)
        random.shuffle(video_data)
        split_idx = int(len(video_data) * train_split)
        train_videos = video_data[:split_idx]
        val_videos = video_data[split_idx:]
        
        # Ensure at least one video in each split
        if len(train_videos) == 0 and len(val_videos) > 0:
            train_videos = [val_videos.pop(0)]
        elif len(val_videos) == 0 and len(train_videos) > 1:
            val_videos = [train_videos.pop()]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create ZIP file with YOLO dataset structure
        zip_output = BytesIO()
        
        stats = {
            'train_images': 0,
            'train_labels': 0,
            'val_images': 0,
            'val_labels': 0,
            'total_worms': 0
        }
        
        with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zf:
            
            def process_videos(videos_list, split_name, format_type):
                """Process videos and create images/labels for a split."""
                nonlocal stats
                
                for video_path, annotations in videos_list:
                    # Get the video frame
                    try:
                        video_handler_temp = state.video_handler.__class__(Path(video_path).parent)
                        video_handler_temp._init_video(video_path)
                        frame = video_handler_temp.get_first_frame()
                        if frame is None:
                            continue
                        h, w = frame.shape[:2]
                    except Exception as e:
                        print(f"Error loading video {video_path}: {e}")
                        continue
                    
                    # Create unique image name
                    video_name = Path(video_path).stem
                    img_name = f"{video_name}.jpg"
                    
                    # Save image
                    img_encoded = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1]
                    
                    if format_type in ['yolo_detect', 'both']:
                        zf.writestr(f"detect/images/{split_name}/{img_name}", img_encoded.tobytes())
                        stats[f'{split_name}_images'] += 1
                    if format_type in ['yolo_segment', 'both']:
                        zf.writestr(f"segment/images/{split_name}/{img_name}", img_encoded.tobytes())
                        if format_type == 'yolo_segment':
                            stats[f'{split_name}_images'] += 1
                    
                    # Create labels
                    detect_labels = []
                    segment_labels = []
                    
                    for worm_id, annot in annotations:
                        stats['total_worms'] += 1
                        
                        # YOLO Detection format: class_id x_center y_center width height (normalized)
                        if annot.detection_box:
                            x1, y1, x2, y2 = annot.detection_box
                            x_center = ((x1 + x2) / 2) / w
                            y_center = ((y1 + y2) / 2) / h
                            box_w = (x2 - x1) / w
                            box_h = (y2 - y1) / h
                            
                            # Clamp values to [0, 1]
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            box_w = max(0, min(1, box_w))
                            box_h = max(0, min(1, box_h))
                            
                            detect_labels.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
                        
                        # YOLO Segmentation format: class_id x1 y1 x2 y2 ... xn yn (polygon, normalized)
                        if annot.segmentation_mask_path and format_type in ['yolo_segment', 'both']:
                            mask_path = Path(annot.segmentation_mask_path)
                            if mask_path.exists():
                                try:
                                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                                    if mask is not None:
                                        # Find contours
                                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        if contours:
                                            # Get the largest contour
                                            largest_contour = max(contours, key=cv2.contourArea)
                                            
                                            # Simplify contour to reduce points
                                            epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                                            simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
                                            
                                            # Convert to normalized coordinates
                                            points = []
                                            for pt in simplified:
                                                px = pt[0][0] / w
                                                py = pt[0][1] / h
                                                px = max(0, min(1, px))
                                                py = max(0, min(1, py))
                                                points.extend([f"{px:.6f}", f"{py:.6f}"])
                                            
                                            if len(points) >= 6:  # Need at least 3 points
                                                segment_labels.append(f"0 {' '.join(points)}")
                                except Exception as e:
                                    print(f"Error processing mask {mask_path}: {e}")
                    
                    # Write label files
                    label_name = f"{video_name}.txt"
                    
                    if format_type in ['yolo_detect', 'both'] and detect_labels:
                        zf.writestr(f"detect/labels/{split_name}/{label_name}", '\n'.join(detect_labels))
                        stats[f'{split_name}_labels'] += 1
                    
                    if format_type in ['yolo_segment', 'both'] and segment_labels:
                        zf.writestr(f"segment/labels/{split_name}/{label_name}", '\n'.join(segment_labels))
                        if format_type == 'yolo_segment':
                            stats[f'{split_name}_labels'] += 1
            
            # Process train and val splits
            process_videos(train_videos, 'train', format)
            process_videos(val_videos, 'val', format)
            
            # Create data.yaml file(s)
            if format in ['yolo_detect', 'both']:
                detect_yaml = f"""# YOLO Detection Dataset
# Generated: {timestamp}
# Train/Val Split: {train_split*100:.0f}% / {(1-train_split)*100:.0f}%

path: .
train: images/train
val: images/val

nc: 1
names:
  0: worm
"""
                zf.writestr("detect/data.yaml", detect_yaml)
            
            if format in ['yolo_segment', 'both']:
                segment_yaml = f"""# YOLO Segmentation Dataset
# Generated: {timestamp}
# Train/Val Split: {train_split*100:.0f}% / {(1-train_split)*100:.0f}%

path: .
train: images/train
val: images/val

nc: 1
names:
  0: worm
"""
                zf.writestr("segment/data.yaml", segment_yaml)
            
            # Create README
            readme = f"""# Worm Training Dataset
Generated: {timestamp}

## Statistics
- Train Images: {stats['train_images']}
- Train Labels: {stats['train_labels']}
- Val Images: {stats['val_images']}
- Val Labels: {stats['val_labels']}
- Total Worms: {stats['total_worms']}
- Train/Val Split: {train_split*100:.0f}% / {(1-train_split)*100:.0f}%

## Format
- Format: {format}
- QC Only: {qc_only}
- Skip Censored: {skip_censored}
- Require Mask: {require_mask}

## Usage

### Detection (YOLOv5/v7/v8)
```bash
cd detect
yolo detect train data=data.yaml model=yolov8n.pt epochs=100
```

### Segmentation (YOLOv8-seg)
```bash
cd segment
yolo segment train data=data.yaml model=yolov8n-seg.pt epochs=100
```

## Class
- 0: worm
"""
            zf.writestr("README.md", readme)
        
        zip_output.seek(0)
        filename = f"worm_training_{format}_{timestamp}.zip"
        
        return StreamingResponse(
            zip_output,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training export failed: {str(e)}")


@app.post("/api/navigate/{direction}")
async def navigate(direction: str, user: dict = Depends(require_auth)):
    """Navigate to next/previous video, crossing folder boundaries."""
    if state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    # Save current first
    if state.annotation_manager:
        state.annotation_manager.save_annotations()
    
    success = False
    folder_changed = False
    
    if direction == "next":
        success = state.video_handler.next_video()
        # If at end of folder, try next folder
        if not success and state.video_handler.has_next_folder():
            if state.video_handler.next_folder():
                success = True
                folder_changed = True
    elif direction == "prev":
        success = state.video_handler.previous_video()
        # If at start of folder, try previous folder and go to last video
        if not success and state.video_handler.has_previous_folder():
            if state.video_handler.previous_folder():
                # Go to last video in the folder
                video_count = state.video_handler.get_video_count()
                if video_count > 0:
                    state.video_handler.navigate_to(video_count - 1)
                success = True
                folder_changed = True
    elif direction == "next_folder":
        if state.video_handler.next_folder():
            success = True
            folder_changed = True
    elif direction == "prev_folder":
        if state.video_handler.previous_folder():
            success = True
            folder_changed = True
    
    if success:
        _load_current_video()
        
        # Auto-detect based on whether folder changed
        auto_detected = False
        folder_detection_stats = None
        
        if folder_changed and state.yolo_detector is not None:
            # New folder: auto-detect ALL videos in the folder
            print(f"[navigate] Folder changed, running auto-detection on entire folder...")
            folder_detection_stats = await _run_folder_auto_detection()
            auto_detected = folder_detection_stats["total_detections"] > 0
        elif state.current_video_path and state.annotation_manager:
            # Same folder: only auto-detect if this video has no annotations
            existing_annots = state.annotation_manager.get_all_worm_annotations(state.current_video_path)
            if len(existing_annots) == 0 and state.yolo_detector is not None and state.current_frame is not None:
                print(f"[navigate] No annotations for {state.current_video_path}, running auto-detection...")
                auto_detected = await _run_auto_detection()
        
        result = await get_current_frame(user)
        result["folder_changed"] = folder_changed
        result["folder_info"] = _get_folder_info()
        result["auto_detected"] = auto_detected
        if folder_detection_stats:
            result["folder_detection_stats"] = folder_detection_stats
        return result
    
    return {"success": False, "message": f"Cannot navigate {direction}"}


def _get_folder_info() -> dict:
    """Get current folder information."""
    if state.video_handler is None:
        return {}
    
    folder_info = state.video_handler.get_current_folder_info()
    return {
        "current_folder_index": state.video_handler.get_current_folder_index(),
        "folder_count": state.video_handler.get_folder_count(),
        "folder_name": folder_info.name if folder_info else "",
        "folder_path": folder_info.relative_path if folder_info else "",
        "video_count": state.video_handler.get_video_count(),
        "total_videos": state.video_handler.get_total_video_count(),
        "position_string": state.video_handler.get_folder_position_string(),
        "has_next_folder": state.video_handler.has_next_folder(),
        "has_prev_folder": state.video_handler.has_previous_folder()
    }


@app.get("/api/folders")
async def get_folders(user: dict = Depends(require_auth)):
    """Get list of all folders with their video counts."""
    if state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    folder_list = state.video_handler.get_folder_list()
    return {
        "folders": [
            {
                "index": idx,
                "name": name,
                "video_count": count,
                "is_current": idx == state.video_handler.get_current_folder_index()
            }
            for idx, name, count in folder_list
        ],
        "current_folder_index": state.video_handler.get_current_folder_index(),
        "folder_count": state.video_handler.get_folder_count(),
        **_get_folder_info()
    }


@app.get("/api/folder/{index}")
async def select_folder(index: int, user: dict = Depends(require_auth)):
    """Select a folder by index."""
    if state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    # Save current annotations first
    if state.annotation_manager:
        state.annotation_manager.save_annotations()
    
    if state.video_handler.navigate_to_folder(index):
        _load_current_video()
        result = await get_current_frame(user)
        result["folder_info"] = _get_folder_info()
        return result
    
    raise HTTPException(status_code=404, detail="Folder not found")


@app.post("/api/preload_adjacent_folders")
async def preload_adjacent_folders(user: dict = Depends(require_auth)):
    """
    Preload frames for adjacent folders in the background.
    This helps reduce lag when navigating between folders.
    """
    if state.video_handler is None:
        return {"success": False, "message": "No folder loaded"}
    
    prev_idx, next_idx = state.video_handler.get_adjacent_folder_indices()
    preloaded = []
    
    if prev_idx is not None:
        if state.video_handler.preload_folder_frames(prev_idx):
            preloaded.append(f"folder_{prev_idx}")
    
    if next_idx is not None:
        if state.video_handler.preload_folder_frames(next_idx):
            preloaded.append(f"folder_{next_idx}")
    
    return {"success": True, "preloaded": preloaded}


# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
