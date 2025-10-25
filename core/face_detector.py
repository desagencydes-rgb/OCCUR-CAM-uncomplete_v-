"""
OCCUR-CAM Face Detection Engine
Face detection using InsightFace for high-accuracy detection.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
import time

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.data import get_image
except ImportError:
    logging.error("InsightFace not installed. Please install with: pip install insightface")
    raise

from models.face_models import FaceDetection, FaceQualityMetrics, FaceRecognitionConfig
from config.settings import config

class FaceDetector:
    """High-accuracy face detection using InsightFace."""
    
    def __init__(self, config: FaceRecognitionConfig = None):
        """Initialize face detector with configuration."""
        self.config = config or FaceRecognitionConfig()
        self.app = None
        self.is_initialized = False
        
        # Initialize the detector
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize InsightFace detector."""
        try:
            logging.info("Initializing InsightFace detector...")
            
            # Create FaceAnalysis app - CPU optimized
            self.app = FaceAnalysis(
                name=config.face_recognition.MODEL_NAME,
                providers=['CPUExecutionProvider'],  # Force CPU only
                allowed_modules=['detection', 'recognition']  # Only load needed modules
            )
            
            # Prepare the model - CPU optimized
            self.app.prepare(
                ctx_id=0,  # CPU context
                det_size=(640, 640)  # Larger detection size for better face detection
            )
            
            self.is_initialized = True
            logging.info("InsightFace detector initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize InsightFace detector: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        if not self.is_initialized:
            raise RuntimeError("Face detector not initialized")
        
        try:
            start_time = time.time()
            
            # Convert BGR to RGB for InsightFace
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Run face detection
            faces = self.app.get(image_rgb)
            
            detections = []
            for i, face in enumerate(faces):
                # Extract bounding box
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                width = x2 - x
                height = y2 - y
                
                # Use InsightFace detection score directly
                confidence = getattr(face, 'det_score', 0.8)
                
                # Extract landmarks if available
                landmarks = None
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = [(int(kp[0]), int(kp[1])) for kp in face.kps]
                
                # Create detection object
                detection = FaceDetection(
                    bbox=(x, y, width, height),
                    confidence=confidence,
                    landmarks=landmarks,
                    face_id=f"face_{i}_{int(time.time())}"
                )
                
                detections.append(detection)
            
            # Filter by confidence threshold
            detections = [d for d in detections if d.confidence >= self.config.detection_threshold]
            
            # Limit number of faces
            if len(detections) > self.config.max_faces_per_frame:
                detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
                detections = detections[:self.config.max_faces_per_frame]
            
            processing_time = time.time() - start_time
            logging.debug(f"Detected {len(detections)} faces in {processing_time:.3f}s")
            
            return detections
            
        except Exception as e:
            logging.error(f"Error in face detection: {e}")
            return []
    
    def _calculate_detection_confidence(self, face, image: np.ndarray) -> float:
        """Calculate detection confidence based on face quality and size."""
        try:
            # Base confidence from InsightFace (if available)
            base_confidence = getattr(face, 'det_score', 0.8)
            
            # Get face region
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox
            face_region = image[y:y2, x:x2]
            
            if face_region.size == 0:
                return 0.0
            
            # Calculate quality metrics
            quality_metrics = self._assess_face_quality(face_region)
            
            # Adjust confidence based on quality
            quality_factor = min(quality_metrics.overall_score * 1.2, 1.0)
            
            # Adjust for face size (larger faces are generally more reliable)
            face_area = (x2 - x) * (y2 - y)
            image_area = image.shape[0] * image.shape[1]
            size_factor = min(face_area / (image_area * 0.01), 1.0)  # Normalize to 1% of image
            
            # Combine factors
            final_confidence = base_confidence * quality_factor * (0.5 + 0.5 * size_factor)
            
            return min(final_confidence, 1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating detection confidence: {e}")
            return 0.5  # Default confidence
    
    def _assess_face_quality(self, face_region: np.ndarray) -> FaceQualityMetrics:
        """Assess face quality metrics."""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 1000.0, 1.0)  # Normalize
            
            # Calculate brightness
            brightness = np.mean(gray) / 255.0
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray) / 255.0
            
            # Calculate symmetry (simplified)
            h, w = gray.shape
            left_half = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)
            min_width = min(left_half.shape[1], right_half.shape[1])
            if min_width > 0:
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
                symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
            else:
                symmetry = 0.5
            
            # Calculate pose angle (simplified - would need landmarks for accurate calculation)
            pose_angle = 0.0  # Placeholder - would calculate from landmarks
            
            # Calculate overall score
            overall_score = (sharpness * 0.3 + brightness * 0.2 + contrast * 0.2 + 
                           symmetry * 0.2 + (1.0 - abs(pose_angle)) * 0.1)
            
            return FaceQualityMetrics(
                sharpness=sharpness,
                brightness=brightness,
                contrast=contrast,
                symmetry=symmetry,
                pose_angle=pose_angle,
                overall_score=overall_score
            )
            
        except Exception as e:
            logging.warning(f"Error assessing face quality: {e}")
            return FaceQualityMetrics(
                sharpness=0.5,
                brightness=0.5,
                contrast=0.5,
                symmetry=0.5,
                pose_angle=0.0,
                overall_score=0.5
            )
    
    def extract_face_region(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """Extract face region from image based on detection."""
        try:
            x, y, w, h = detection.bbox
            
            # Add padding around face
            padding = 0.2
            pad_w = int(w * padding)
            pad_h = int(h * padding)
            
            # Calculate padded coordinates
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(image.shape[1], x + w + pad_w)
            y2 = min(image.shape[0], y + h + pad_h)
            
            # Extract face region
            face_region = image[y1:y2, x1:x2]
            
            return face_region
            
        except Exception as e:
            logging.error(f"Error extracting face region: {e}")
            return np.array([])
    
    def resize_face(self, face_region: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Resize face region to target size."""
        if target_size is None:
            target_size = self.config.face_size
        
        try:
            resized = cv2.resize(face_region, target_size, interpolation=cv2.INTER_LANCZOS4)
            return resized
        except Exception as e:
            logging.error(f"Error resizing face: {e}")
            return face_region
    
    def preprocess_face(self, face_region: np.ndarray) -> np.ndarray:
        """Preprocess face region for recognition."""
        try:
            # Resize to target size
            processed = self.resize_face(face_region)
            
            # Convert to RGB if needed
            if len(processed.shape) == 3 and processed.shape[2] == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            processed = processed.astype(np.float32) / 255.0
            
            return processed
            
        except Exception as e:
            logging.error(f"Error preprocessing face: {e}")
            return face_region
    
    def detect_faces_batch(self, images: List[np.ndarray]) -> List[List[FaceDetection]]:
        """Detect faces in a batch of images."""
        results = []
        
        for image in images:
            detections = self.detect_faces(image)
            results.append(detections)
        
        return results
    
    def get_detector_info(self) -> dict:
        """Get detector information and status."""
        return {
            "is_initialized": self.is_initialized,
            "model_name": config.face_recognition.MODEL_NAME,
            "detection_threshold": self.config.detection_threshold,
            "max_faces_per_frame": self.config.max_faces_per_frame,
            "face_size": self.config.face_size,
            "providers": self.app.get_providers() if self.app else []
        }
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.app:
                del self.app
            self.app = None
            self.is_initialized = False
            logging.info("Face detector cleaned up")
        except Exception as e:
            logging.warning(f"Error during detector cleanup: {e}")

# Global detector instance
_detector_instance = None

def get_face_detector(config: FaceRecognitionConfig = None) -> FaceDetector:
    """Get global face detector instance."""
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = FaceDetector(config)
    
    return _detector_instance

def cleanup_face_detector():
    """Cleanup global face detector instance."""
    global _detector_instance
    
    if _detector_instance:
        _detector_instance.cleanup()
        _detector_instance = None
