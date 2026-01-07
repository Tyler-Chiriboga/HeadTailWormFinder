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
from web.auth import auth_manager, get_current_user, require_auth, require_admin


# Global state
class AppState:
    def __init__(self):
        self.video_handler: Optional[VideoHandler] = None
        self.annotation_manager: Optional[AnnotationManager] = None
        self.yolo_detector: Optional[YOLODetector] = None
        self.sam_segmenter: Optional[SAMSegmenter] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_video_path: Optional[str] = None
        self.models_loaded = False
        self.confidence_threshold = 0.25  # YOLO detection confidence threshold
        
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
    x: int
    y: int
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
            frame = state.video_handler.get_first_frame()
            if frame is not None:
                state.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


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
    """Encode numpy image to base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
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


def straighten_worm(image: np.ndarray, mask: np.ndarray, width: int = None) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Straighten a worm image along its skeleton.
    
    Args:
        image: RGB image containing the worm
        mask: Binary mask of the worm
        width: Width of straightened image (auto-detect if None)
    
    Returns:
        Tuple of (straightened_image, skeleton_path)
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
    
    # Smooth the path to reduce spiky edges
    # Apply Gaussian smoothing to path coordinates
    path = list(path)
    path_y = np.array([p[0] for p in path], dtype=float)
    path_x = np.array([p[1] for p in path], dtype=float)
    
    # Use a smoothing window proportional to path length
    smooth_sigma = max(3, len(path) // 30)
    path_y_smooth = ndimage.gaussian_filter1d(path_y, sigma=smooth_sigma, mode='nearest')
    path_x_smooth = ndimage.gaussian_filter1d(path_x, sigma=smooth_sigma, mode='nearest')
    
    # Rebuild smoothed path
    path = [(int(round(y)), int(round(x))) for y, x in zip(path_y_smooth, path_x_smooth)]
    
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
    # Track both total width AND the offset from skeleton to mask center
    all_min_d = []
    all_max_d = []
    for i in range(len(path)):
        y, x = path[i]
        tx, ty = tangents[i]
        # Perpendicular direction (same as sampling below)
        nx, ny = -ty, tx
        # Measure width by scanning perpendicular
        min_d, max_d = None, None
        for d in range(-250, 251):
            py, px = int(y + d * ny), int(x + d * nx)
            if 0 <= py < binary.shape[0] and 0 <= px < binary.shape[1]:
                if binary[py, px]:
                    if min_d is None:
                        min_d = d
                    max_d = d
        if min_d is not None and max_d is not None:
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
        
        # Sample perpendicular line - use global_min_d to global_max_d range
        for d in range(global_min_d, global_max_d + 1):
            py = y + d * ny
            px = x + d * nx
            
            # Output row index (d=global_min_d -> row 0)
            out_row = d - global_min_d
            
            # Check if this point is inside the mask
            py_int, px_int = int(round(py)), int(round(px))
            is_in_mask = (0 <= py_int < binary.shape[0] and 
                          0 <= px_int < binary.shape[1] and 
                          binary[py_int, px_int] > 0)
            
            if is_in_mask:
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
            # else: keep white background (already set)
    
    return straightened, path


def auto_segment_worm(worm_id: int, bbox: tuple) -> bool:
    """
    Automatically segment a worm using SAM when a detection box is created.
    Returns True if segmentation was successful, False otherwise.
    """
    if state.sam_segmenter is None or not state.sam_segmenter.is_loaded():
        return False
    if state.current_frame is None:
        return False
    if state.annotation_manager is None or state.current_video_path is None:
        return False
    
    try:
        # Set image for SAM
        state.sam_segmenter.set_image(state.current_frame)
        
        # Run segmentation
        result = state.sam_segmenter.segment(bbox, keep_largest_only=True)
        
        if result is None:
            return False
        
        # Save the mask
        state.annotation_manager.save_segmentation_mask(
            state.current_video_path,
            worm_id,
            result.mask,
            mask_type="worm"
        )
        return True
    except Exception as e:
        print(f"Auto-segment failed for worm {worm_id}: {e}")
        return False


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
    """Open a folder containing videos."""
    if not Path(folder.path).exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    state.video_handler = VideoHandler()
    state.video_handler.load_folder(folder.path, recursive=True)
    
    state.annotation_manager = AnnotationManager()
    state.annotation_manager.set_folder(folder.path)
    
    # Load first video
    if state.video_handler.get_video_count() > 0:
        state.video_handler.navigate_to(0)
        _load_current_video()
    
    return {
        "success": True,
        "video_count": state.video_handler.get_video_count(),
        "folder_count": state.video_handler.get_folder_count(),
        "videos": [name for _, name in state.video_handler.get_video_list()]
    }


@app.get("/api/videos")
async def get_videos(user: dict = Depends(require_auth)):
    """Get list of videos in current folder."""
    if state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    current_index = state.video_handler.get_current_index()
    return {
        "videos": [
            {"index": idx, "name": name, "is_current": idx == current_index}
            for idx, name in state.video_handler.get_video_list()
        ],
        "current_index": current_index,
        "current_video": state.current_video_path
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
            }
            
            # Include mask data if available
            if annot.segmentation_mask_path and Path(annot.segmentation_mask_path).exists():
                mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    annot_data["worm_mask"] = encode_mask_to_base64(mask / 255.0)
                    # Compute mask statistics
                    stats = compute_mask_statistics(mask)
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
        "annotations": annotations
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
    
    # Clear existing annotations and add new ones
    if state.annotation_manager and state.current_video_path:
        # Remove old annotations for this video
        if state.current_video_path in state.annotation_manager.annotations:
            state.annotation_manager.annotations[state.current_video_path].annotations.clear()
        
        results = []
        segmented_count = 0
        
        for det in detections:
            annot = state.annotation_manager.add_worm_annotation(
                state.current_video_path,
                detection_box=det.bbox,
                confidence=det.confidence
            )
            
            # Auto-segment this worm
            if auto_segment_worm(annot.worm_id, det.bbox):
                segmented_count += 1
            
            results.append({
                "worm_id": annot.worm_id,
                "bbox": list(det.bbox),
                "confidence": det.confidence
            })
        
        state.annotation_manager.save_annotations()
        
        return {
            "success": True,
            "count": len(results),
            "segmented": segmented_count,
            "detections": results
        }
    
    raise HTTPException(status_code=500, detail="Annotation manager not initialized")


@app.post("/api/detect/batch")
async def run_batch_detection(user: dict = Depends(require_auth)):
    """Run YOLO detection on all videos in the loaded folder."""
    if state.yolo_detector is None:
        raise HTTPException(status_code=400, detail="YOLO model not loaded")
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
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        detections = state.yolo_detector.detect(
            frame_rgb,
            conf_threshold=state.confidence_threshold
        )
        
        # Clear existing and add new annotations
        if video_path in state.annotation_manager.annotations:
            state.annotation_manager.annotations[video_path].annotations.clear()
        
        video_results = {
            "video": video_info.path.name,
            "detections": len(detections)
        }
        
        for det in detections:
            state.annotation_manager.add_worm_annotation(
                video_path,
                detection_box=det.bbox,
                confidence=det.confidence
            )
        
        results["processed"] += 1
        results["total_detections"] += len(detections)
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
    
    # Auto-segment this worm
    segmented = auto_segment_worm(annot.worm_id, coords)
    
    state.annotation_manager.save_annotations()
    
    return {
        "success": True, 
        "worm_id": annot.worm_id, 
        "detection_box": coords,
        "segmented": segmented,
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
    
    # Load mask
    mask = cv2.imread(annot.segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=500, detail="Failed to load mask")
    
    # Straighten the worm
    straightened, path = straighten_worm(state.current_frame, mask)
    
    if straightened is None:
        raise HTTPException(status_code=500, detail="Failed to straighten worm - skeleton extraction failed")
    
    # Encode as base64
    straightened_b64 = encode_image_to_base64(straightened)
    
    return {
        "success": True,
        "worm_id": worm_id,
        "straightened_image": straightened_b64,
        "width": straightened.shape[1],
        "height": straightened.shape[0],
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


@app.get("/api/export/excel")
async def export_to_excel(user: dict = Depends(require_auth)):
    """Export annotations to Excel file."""
    if state.annotation_manager is None:
        raise HTTPException(status_code=400, detail="No annotations to export")
    
    try:
        import pandas as pd
        from io import BytesIO
        
        # Collect all annotation data into rows
        rows = []
        for video_path, video_annot in state.annotation_manager.annotations.items():
            video_name = Path(video_path).name
            for worm_id, annot in video_annot.annotations.items():
                row = {
                    'video_path': video_path,
                    'video_name': video_name,
                    'worm_id': worm_id,
                    'confidence': annot.confidence,
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
        
        if not rows:
            raise HTTPException(status_code=400, detail="No annotation data to export")
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Annotations', index=False)
        
        output.seek(0)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"worm_annotations_{timestamp}.xlsx"
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except ImportError as e:
        # Fallback to JSON if pandas/openpyxl not available
        all_annotations = {}
        for video_path, video_annot in state.annotation_manager.annotations.items():
            all_annotations[video_path] = video_annot.to_dict()
        
        return JSONResponse(
            content=all_annotations,
            headers={"Content-Disposition": "attachment; filename=worm_annotations.json"}
        )


@app.post("/api/navigate/{direction}")
async def navigate(direction: str, user: dict = Depends(require_auth)):
    """Navigate to next/previous video."""
    if state.video_handler is None:
        raise HTTPException(status_code=400, detail="No folder loaded")
    
    # Save current first
    if state.annotation_manager:
        state.annotation_manager.save_annotations()
    
    success = False
    if direction == "next":
        success = state.video_handler.next_video()
    elif direction == "prev":
        success = state.video_handler.previous_video()
    
    if success:
        _load_current_video()
        return await get_current_frame(user)
    
    return {"success": False, "message": f"Cannot navigate {direction}"}


# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
