"""
Health classification model for worms.
Uses a ResNet18 CNN to classify worms as Healthy or Leaky.
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import cv2


class HealthClassifier:
    """ResNet18-based worm health classifier."""
    
    def __init__(self, weights_path: Optional[str] = None):
        """
        Initialize the health classifier.
        
        Args:
            weights_path: Path to the model weights file. 
                         If None, uses default path in weights folder.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.transform: Optional[T.Compose] = None
        self.weights_path = weights_path
        self._loaded = False
        
    def load_model(self, weights_path: Optional[str] = None) -> bool:
        """
        Load the health classification model.
        
        Args:
            weights_path: Optional path to weights file (overrides init path)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if weights_path:
            self.weights_path = weights_path
            
        if not self.weights_path:
            # Default path in weights folder
            default_path = Path(__file__).parent.parent / "weights" / "worm_leaky_model.pth"
            self.weights_path = str(default_path)
            
        weights_file = Path(self.weights_path)
        if not weights_file.exists():
            print(f"[HealthClassifier] Weights file not found: {self.weights_path}")
            return False
            
        try:
            # Create ResNet18 model with sigmoid output
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 1),
                nn.Sigmoid()
            )
            
            # Load weights
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transform pipeline
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
            ])
            
            self._loaded = True
            print(f"[HealthClassifier] Model loaded from {self.weights_path}")
            return True
            
        except Exception as e:
            print(f"[HealthClassifier] Failed to load model: {e}")
            self._loaded = False
            return False
            
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._loaded and self.model is not None
        
    def classify(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, str]:
        """
        Classify a worm image as Healthy or Leaky.
        
        Args:
            image: RGB or BGR image of the worm (cropped region)
            mask: Optional binary mask to apply (not currently used, 
                  model was trained on full crops)
                  
        Returns:
            Tuple of (score, classification) where score is 0-1 
            and classification is "Healthy" or "Leaky"
        """
        if not self.is_loaded():
            raise RuntimeError("Health classifier model not loaded")
            
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if BGR by looking at the color pattern
            # Assume input could be BGR from OpenCV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Transform and predict
        inp_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            score = self.model(inp_tensor).item()
            
        # Classification based on threshold
        classification = "Healthy" if score >= 0.5 else "Leaky"
        
        return score, classification
        
    def classify_from_frame(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[float, float, float, float],
        mask: Optional[np.ndarray] = None
    ) -> Tuple[float, str]:
        """
        Classify a worm from a full frame given its bounding box.
        
        Args:
            frame: Full frame image (RGB)
            bbox: Bounding box (x1, y1, x2, y2)
            mask: Optional full-frame mask for the worm
            
        Returns:
            Tuple of (score, classification)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bounds are within frame
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Crop the worm region
        worm_crop = frame[y1:y2, x1:x2]
        
        if worm_crop.size == 0:
            return 0.0, "Unknown"
            
        # Optionally crop mask too (not used in current model)
        mask_crop = None
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            
        return self.classify(worm_crop, mask_crop)
