"""
OCCUR-CAM Core Utilities
Utility functions for image processing, file handling, and common operations.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import json
import time
from datetime import datetime
import hashlib
import base64
from PIL import Image, ImageEnhance

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    try:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(logs_dir / "occur_cam.log")
            ]
        )
        
        # Set specific loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        
        logging.info("Logging configured successfully")
        
    except Exception as e:
        print(f"Error setting up logging: {e}")

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image as numpy array
        target_size: Target size (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    try:
        if maintain_aspect_ratio:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create canvas with target size
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Center the resized image
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
            
    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        return image

def enhance_image(image: np.ndarray, brightness: float = 1.0, 
                 contrast: float = 1.0, sharpness: float = 1.0) -> np.ndarray:
    """
    Enhance image quality.
    
    Args:
        image: Input image as numpy array
        brightness: Brightness factor (1.0 = no change)
        contrast: Contrast factor (1.0 = no change)
        sharpness: Sharpness factor (1.0 = no change)
        
    Returns:
        Enhanced image
    """
    try:
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(sharpness)
        
        # Convert back to OpenCV format
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced
        
    except Exception as e:
        logging.error(f"Error enhancing image: {e}")
        return image

def normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize image pixel values.
    
    Args:
        image: Input image as numpy array
        method: Normalization method ('minmax', 'zscore', 'unit')
        
    Returns:
        Normalized image
    """
    try:
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                normalized = (image - img_min) / (img_max - img_min)
            else:
                normalized = image
        elif method == 'zscore':
            # Z-score normalization
            mean = image.mean()
            std = image.std()
            if std > 0:
                normalized = (image - mean) / std
            else:
                normalized = image
        elif method == 'unit':
            # Unit vector normalization
            norm = np.linalg.norm(image)
            if norm > 0:
                normalized = image / norm
            else:
                normalized = image
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.astype(np.float32)
        
    except Exception as e:
        logging.error(f"Error normalizing image: {e}")
        return image.astype(np.float32)

def crop_face_region(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                    padding: float = 0.2) -> np.ndarray:
    """
    Crop face region from image with padding.
    
    Args:
        image: Input image as numpy array
        bbox: Bounding box (x, y, width, height)
        padding: Padding factor around face
        
    Returns:
        Cropped face region
    """
    try:
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        
        # Calculate padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate crop coordinates
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)
        
        # Crop face region
        face_region = image[y1:y2, x1:x2]
        
        return face_region
        
    except Exception as e:
        logging.error(f"Error cropping face region: {e}")
        return np.array([])

def draw_face_annotations(image: np.ndarray, detections: List[Dict[str, Any]], 
                         recognitions: List[Dict[str, Any]] = None) -> np.ndarray:
    """
    Draw face detection and recognition annotations on image.
    
    Args:
        image: Input image as numpy array
        detections: List of face detection results
        recognitions: List of face recognition results
        
    Returns:
        Annotated image
    """
    try:
        annotated = image.copy()
        
        # Create recognition lookup
        recognition_lookup = {}
        if recognitions:
            for rec in recognitions:
                bbox = rec.get('bbox', (0, 0, 0, 0))
                recognition_lookup[bbox] = rec
        
        # Draw detections and recognitions
        for detection in detections:
            bbox = detection.get('bbox', (0, 0, 0, 0))
            x, y, w, h = bbox
            
            # Check if this detection has a recognition
            recognition = recognition_lookup.get(bbox)
            
            if recognition:
                # Draw recognition (green box)
                color = (0, 255, 0)  # Green
                thickness = 2
                
                # Draw bounding box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
                
                # Draw label
                label = f"{recognition.get('name', 'Unknown')} ({recognition.get('confidence', 0):.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(annotated, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), color, -1)
                
                # Draw label text
                cv2.putText(annotated, label, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # Draw detection only (red box)
                color = (0, 0, 255)  # Red
                thickness = 1
                
                # Draw bounding box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
                
                # Draw label
                label = f"Unknown ({detection.get('confidence', 0):.2f})"
                cv2.putText(annotated, label, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated
        
    except Exception as e:
        logging.error(f"Error drawing face annotations: {e}")
        return image

def save_image(image: np.ndarray, file_path: Union[str, Path], 
               quality: int = 95) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image as numpy array
        file_path: Output file path
        quality: JPEG quality (1-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine file format
        if file_path.suffix.lower() in ['.jpg', '.jpeg']:
            # Save as JPEG
            cv2.imwrite(str(file_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            # Save as other format
            cv2.imwrite(str(file_path), image)
        
        return True
        
    except Exception as e:
        logging.error(f"Error saving image: {e}")
        return False

def load_image(file_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load image from file.
    
    Args:
        file_path: Image file path
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logging.error(f"Image file not found: {file_path}")
            return None
        
        # Load image
        image = cv2.imread(str(file_path))
        
        if image is None:
            logging.error(f"Failed to load image: {file_path}")
            return None
        
        return image
        
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        return None

def calculate_image_hash(image: np.ndarray) -> str:
    """
    Calculate hash for image.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Image hash string
    """
    try:
        # Convert image to bytes
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        
        # Calculate MD5 hash
        hash_md5 = hashlib.md5()
        hash_md5.update(image_bytes)
        
        return hash_md5.hexdigest()
        
    except Exception as e:
        logging.error(f"Error calculating image hash: {e}")
        return ""

def encode_image_base64(image: np.ndarray, format: str = 'jpg') -> str:
    """
    Encode image as base64 string.
    
    Args:
        image: Image as numpy array
        format: Image format ('jpg', 'png')
        
    Returns:
        Base64 encoded image string
    """
    try:
        # Encode image
        if format.lower() == 'jpg':
            success, buffer = cv2.imencode('.jpg', image)
        elif format.lower() == 'png':
            success, buffer = cv2.imencode('.png', image)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if not success:
            raise ValueError("Failed to encode image")
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
        
    except Exception as e:
        logging.error(f"Error encoding image to base64: {e}")
        return ""

def decode_image_base64(image_base64: str) -> Optional[np.ndarray]:
    """
    Decode base64 string to image.
    
    Args:
        image_base64: Base64 encoded image string
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
        
    except Exception as e:
        logging.error(f"Error decoding base64 image: {e}")
        return None

def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """
    Create thumbnail of image.
    
    Args:
        image: Input image as numpy array
        size: Thumbnail size (width, height)
        
    Returns:
        Thumbnail image
    """
    try:
        return resize_image(image, size, maintain_aspect_ratio=True)
        
    except Exception as e:
        logging.error(f"Error creating thumbnail: {e}")
        return image

def validate_image(image: np.ndarray) -> bool:
    """
    Validate image for processing.
    
    Args:
        image: Image as numpy array
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if image is None:
            return False
        
        if len(image.shape) != 3:
            return False
        
        if image.shape[2] != 3:
            return False
        
        if image.size == 0:
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating image: {e}")
        return False

def get_image_info(image: np.ndarray) -> Dict[str, Any]:
    """
    Get image information.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Image information dictionary
    """
    try:
        if not validate_image(image):
            return {}
        
        return {
            "shape": image.shape,
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": image.shape[2],
            "dtype": str(image.dtype),
            "size_bytes": image.nbytes,
            "hash": calculate_image_hash(image)
        }
        
    except Exception as e:
        logging.error(f"Error getting image info: {e}")
        return {}

def create_timestamped_filename(prefix: str = "image", 
                              extension: str = "jpg") -> str:
    """
    Create timestamped filename.
    
    Args:
        prefix: Filename prefix
        extension: File extension
        
    Returns:
        Timestamped filename
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        return f"{prefix}_{timestamp}.{extension}"
        
    except Exception as e:
        logging.error(f"Error creating timestamped filename: {e}")
        return f"{prefix}_{int(time.time())}.{extension}"