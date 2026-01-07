"""
Configuration settings for HeadTailWormFinder application.
"""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESOURCES_DIR = PROJECT_ROOT / "resources"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Cache file for remembering last folder
CACHE_FILE = CACHE_DIR / "app_cache.json"

# YOLOv7 Model Configuration
YOLO_MODEL_PATH = "/home/hedtpc/Downloads/YOLO314Full/yolov7-custom_lambda/yolov7-custom/best_Leaky_v3.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_IMG_SIZE = 640

# SAM Model Configuration
SAM_MODEL_TYPE = "vit_b"  # Options: vit_h, vit_l, vit_b
SAM_CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_PATH = WEIGHTS_DIR / SAM_CHECKPOINT_NAME
SAM_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# SAM Model Options (for UI selection)
SAM_MODELS = {
    "vit_b": {
        "name": "SAM ViT-B (375MB) - Fast",
        "checkpoint": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "size_mb": 375,
        "vram_required_gb": 4  # Approximate VRAM needed
    },
    "vit_l": {
        "name": "SAM ViT-L (1.2GB) - Balanced",
        "checkpoint": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "size_mb": 1200,
        "vram_required_gb": 8  # Approximate VRAM needed
    },
    "vit_h": {
        "name": "SAM ViT-H (2.4GB) - Best Quality",
        "checkpoint": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "size_mb": 2400,
        "vram_required_gb": 12  # Approximate VRAM needed
    }
}


def get_best_sam_model() -> str:
    """
    Determine the best SAM model based on available GPU memory.
    
    Returns:
        Model type string: 'vit_h', 'vit_l', or 'vit_b'
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 'vit_b'  # Smallest for CPU
        
        # Check GPU memory (use the second GPU if available, else first)
        num_gpus = torch.cuda.device_count()
        gpu_idx = 1 if num_gpus > 1 else 0  # SAM uses second GPU
        
        gpu_memory_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
        print(f"GPU {gpu_idx} memory: {gpu_memory_gb:.1f} GB")
        
        # Select best model that fits in memory (with some headroom)
        if gpu_memory_gb >= 16:  # 16GB+ can run vit_h comfortably
            return 'vit_h'
        elif gpu_memory_gb >= 10:  # 10GB+ can run vit_l
            return 'vit_l'
        else:
            return 'vit_b'
            
    except Exception as e:
        print(f"Error detecting GPU memory: {e}")
        return 'vit_b'

# Default data paths
DEFAULT_VIDEO_FOLDER = "/media/hedtpc/BulkStorage/LeakyGut/CPLT-001 LHK-1B QCd/Movies"

# Annotation colors (RGB)
COLORS = {
    "yolo_detection": (0, 120, 255),      # Blue for YOLO detections
    "head_box": (0, 255, 0),               # Green for head
    "tail_box": (255, 0, 0),               # Red for tail
    "segmentation_mask": (255, 255, 0),    # Yellow for SAM mask
    "selected": (255, 165, 0),             # Orange for selected items
}

# UI Settings
UI_SETTINGS = {
    "window_title": "HeadTailWormFinder - Worm Annotation Tool",
    "default_width": 1400,
    "default_height": 900,
    "thumbnail_size": (120, 90),
    "box_line_width": 2,
    "mask_opacity": 0.4,
}

# Auto-run settings
AUTO_SETTINGS = {
    "auto_load_last_folder": True,
    "auto_run_detection": True,
}

# Export settings
EXPORT_SETTINGS = {
    "excel_filename": "worm_annotations.xlsx",
    "annotation_filename": "annotations.json",
    "cropped_images_folder": "cropped_worms",
}

# GPU Settings
GPU_SETTINGS = {
    "yolo_device": "cuda:0",  # Will be auto-detected
    "sam_device": "cuda:0",   # Can use cuda:1 if available
    "auto_detect": True,
}


def get_available_gpus():
    """Get list of available CUDA devices."""
    import torch
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def ensure_directories():
    """Ensure all required directories exist."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
