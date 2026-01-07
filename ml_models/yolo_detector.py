"""
YOLOv7 detector wrapper for worm detection.
Supports automatic GPU detection and selection.
"""
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Add YOLOv7 repo to path for model loading
YOLOV7_REPO_PATH = Path(__file__).parent.parent / "yolov7_repo"
if YOLOV7_REPO_PATH.exists() and str(YOLOV7_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(YOLOV7_REPO_PATH))

# Import YOLOv7 utilities
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox


@dataclass
class Detection:
    """Represents a single detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = "worm"
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Get bounding box as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of detection."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Detection':
        """Create from dictionary."""
        return cls(**data)


class YOLODetector:
    """
    YOLOv7-based worm detector with automatic GPU selection.
    """
    
    def __init__(
        self,
        weights_path: str,
        device: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        img_size: int = 640
    ):
        """
        Initialize YOLO detector.
        
        Args:
            weights_path: Path to YOLOv7 weights file (.pt)
            device: Device to use ('cuda:0', 'cuda:1', 'cpu'). Auto-detect if None.
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            img_size: Input image size for model
        """
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        # Auto-detect device if not specified
        if device is None:
            device = self._get_best_device()
        self.device = device
        
        # Load model
        self.model = None
        self._load_model()
    
    def _get_best_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            # Get GPU with most free memory
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                return "cpu"
            
            # For simplicity, use first GPU
            # Could be extended to check memory and select best
            print(f"Found {num_gpus} CUDA device(s)")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            
            return "cuda:0"
        return "cpu"
    
    def _load_model(self):
        """Load the YOLOv7 model."""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")
        
        print(f"Loading YOLOv7 model from {self.weights_path}")
        print(f"Using device: {self.device}")
        
        # Load model using torch
        try:
            # Load the checkpoint (weights_only=False needed for YOLOv7 custom models)
            ckpt = torch.load(str(self.weights_path), map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model' in ckpt:
                self.model = ckpt['model'].float()
            elif 'ema' in ckpt and ckpt['ema'] is not None:
                self.model = ckpt['ema'].float()
            else:
                self.model = ckpt.float()
            
            # Set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Fuse layers for faster inference
            if hasattr(self.model, 'fuse'):
                self.model.fuse()
            
            print("YOLOv7 model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple]:
        """
        Preprocess image for YOLO inference using YOLOv7's letterbox.
        
        Args:
            image: RGB image as numpy array (H, W, C)
            
        Returns:
            Tuple of (preprocessed tensor, original shape)
        """
        # Store original shape
        self._orig_shape = image.shape[:2]  # H, W
        
        # Convert RGB to BGR for letterbox (it expects BGR)
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Use YOLOv7's letterbox for proper preprocessing
        img_letterbox = letterbox(img_bgr, self.img_size, stride=32, auto=False)[0]
        
        # Convert BGR to RGB, HWC to CHW
        img_letterbox = img_letterbox[:, :, ::-1].transpose(2, 0, 1)
        img_letterbox = np.ascontiguousarray(img_letterbox)
        
        # Convert to tensor
        tensor = torch.from_numpy(img_letterbox).float().to(self.device)
        tensor /= 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def _postprocess(self, predictions: torch.Tensor, img_shape: Tuple[int, int]) -> List[Detection]:
        """
        Postprocess YOLO predictions using YOLOv7's NMS.
        
        Args:
            predictions: Raw model output
            img_shape: Shape of preprocessed image (H, W)
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # Apply NMS using YOLOv7's function
        pred = non_max_suppression(
            predictions, 
            self.conf_threshold, 
            self.iou_threshold,
            classes=None,
            agnostic=False
        )
        
        # Process detections
        for det in pred:  # per image
            if len(det) == 0:
                continue
            
            # Rescale boxes from img_size to original image size
            det[:, :4] = scale_coords(
                (self.img_size, self.img_size), 
                det[:, :4], 
                self._orig_shape
            ).round()
            
            # Convert to Detection objects
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [float(x.cpu().numpy()) for x in xyxy]
                
                detections.append(Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(conf.cpu().numpy()),
                    class_id=int(cls.cpu().numpy()),
                    class_name="worm"
                ))
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect worms in an image.
        
        Args:
            image: RGB image as numpy array (H, W, C)
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess
        tensor = self._preprocess(image)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(tensor)[0]
        
        # Postprocess
        detections = self._postprocess(predictions, tensor.shape[2:])
        
        return detections
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold."""
        self.conf_threshold = max(0.0, min(1.0, threshold))
    
    def set_iou_threshold(self, threshold: float):
        """Set IOU threshold for NMS."""
        self.iou_threshold = max(0.0, min(1.0, threshold))
    
    def get_device(self) -> str:
        """Get current device."""
        return self.device
    
    @staticmethod
    def get_available_devices() -> List[str]:
        """Get list of available devices."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        return devices


if __name__ == "__main__":
    # Test detector
    from config.settings import YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD
    
    detector = YOLODetector(
        weights_path=YOLO_MODEL_PATH,
        conf_threshold=YOLO_CONFIDENCE_THRESHOLD
    )
    
    print(f"Available devices: {detector.get_available_devices()}")
    print(f"Using device: {detector.get_device()}")
