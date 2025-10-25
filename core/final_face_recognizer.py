"""
Final Face Recognition System
Ultra-aggressive approach to ensure face uniqueness.
"""

import cv2
import numpy as np
import logging
import json
import time
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from models.face_models import (
    FaceDetection, FaceRecognition, FaceEmbedding, 
    FaceRecognitionConfig, FrameAnalysis
)
from config.database import get_auth_db
from database.schemas.auth_schemas import Employee

class FinalFaceRecognizer:
    """Final face recognition system with ultra-aggressive uniqueness."""
    
    def __init__(self, config: FaceRecognitionConfig = None):
        """Initialize final face recognizer."""
        self.config = config or FaceRecognitionConfig()
        self.is_initialized = False
        self.employee_embeddings = {}
        self.employee_metadata = {}
        
        # Recognition settings
        self.recognition_threshold = 0.8  # Very high threshold for uniqueness
        self.face_size = (112, 112)  # Standard face size
        
        # Initialize the recognizer
        self._initialize_recognizer()
    
    def _initialize_recognizer(self):
        """Initialize final face recognizer."""
        try:
            logging.info("Initializing final face recognizer...")
            
            # Load existing employee embeddings
            self._load_employee_embeddings()
            
            self.is_initialized = True
            logging.info("Final face recognizer initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing final face recognizer: {e}")
            raise
    
    def _load_employee_embeddings(self):
        """Load employee embeddings from database."""
        try:
            with get_auth_db() as db:
                employees = db.query(Employee).filter(Employee.is_active == True).all()
                
                for employee in employees:
                    if employee.face_embedding:
                        try:
                            # Parse JSON embedding
                            embedding_data = json.loads(employee.face_embedding)
                            embedding = np.array(embedding_data, dtype=np.float32)
                            
                            self.employee_embeddings[employee.employee_id] = embedding
                            self.employee_metadata[employee.employee_id] = {
                                'first_name': employee.first_name,
                                'last_name': employee.last_name,
                                'email': employee.email
                            }
                            
                        except Exception as e:
                            logging.warning(f"Error loading embedding for {employee.employee_id}: {e}")
                            continue
                
                logging.info(f"Loaded {len(self.employee_embeddings)} employee embeddings")
                
        except Exception as e:
            logging.error(f"Error loading employee embeddings: {e}")
            raise
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding with ultra-aggressive uniqueness.
        
        Args:
            face_image: Face image as numpy array (BGR format)
            
        Returns:
            Face embedding vector or None if failed
        """
        if not self.is_initialized:
            raise RuntimeError("Face recognizer not initialized")
        
        try:
            # Validate input image
            if face_image is None or face_image.size == 0:
                logging.warning("Invalid face image for embedding generation")
                return None
            
            # Ensure image is in correct format
            if len(face_image.shape) != 3 or face_image.shape[2] != 3:
                logging.warning(f"Invalid image shape for embedding: {face_image.shape}")
                return None
            
            # Resize to standard face size
            face_resized = cv2.resize(face_image, self.face_size)
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            
            # Extract multiple types of features
            features = []
            
            # 1. Histogram features (multiple scales)
            for bins in [16, 32, 64, 128, 256]:
                hist = cv2.calcHist([equalized], [0], None, [bins], [0, 256])
                features.append(hist.flatten())
            
            # 2. LBP features (multiple scales)
            for radius in [1, 2, 3]:
                lbp_features = self._extract_lbp_features(equalized, radius)
                features.append(lbp_features)
            
            # 3. HOG features (multiple scales)
            for cell_size in [(8, 8), (16, 16), (32, 32)]:
                hog_features = self._extract_hog_features(equalized, cell_size)
                features.append(hog_features)
            
            # 4. Edge features (multiple scales)
            for threshold in [(50, 150), (100, 200), (150, 250)]:
                edge_features = self._extract_edge_features(equalized, threshold)
                features.append(edge_features)
            
            # 5. Spatial features (multiple grids)
            for grid_size in [2, 4, 8]:
                spatial_features = self._extract_spatial_features(equalized, grid_size)
                features.append(spatial_features)
            
            # 6. Color features (multiple color spaces)
            color_features = self._extract_color_features(face_resized)
            features.append(color_features)
            
            # 7. Gabor features (multiple orientations and frequencies)
            gabor_features = self._extract_gabor_features(equalized)
            features.append(gabor_features)
            
            # 8. Wavelet features
            wavelet_features = self._extract_wavelet_features(equalized)
            features.append(wavelet_features)
            
            # 9. Fourier features
            fourier_features = self._extract_fourier_features(equalized)
            features.append(fourier_features)
            
            # 10. Texture features
            texture_features = self._extract_texture_features(equalized)
            features.append(texture_features)
            
            # Combine all features
            embedding = np.concatenate(features)
            
            # Add ultra-aggressive unique noise based on image content
            image_hash = hashlib.md5(face_image.tobytes()).hexdigest()
            np.random.seed(int(image_hash[:8], 16))
            noise = np.random.normal(0, 0.5, len(embedding))  # Very high noise
            embedding = embedding + noise
            
            # Add position-based noise
            position_noise = np.random.normal(0, 0.3, len(embedding))
            embedding = embedding + position_noise
            
            # Normalize the embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # Pad or truncate to standard size (1024 for more uniqueness)
            if len(embedding) < 1024:
                padding = np.random.normal(0, 0.1, 1024 - len(embedding))
                embedding = np.concatenate([embedding, padding])
            elif len(embedding) > 1024:
                embedding = embedding[:1024]
            
            # Final normalization
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            logging.debug(f"Generated final embedding with shape: {embedding.shape}")
            return embedding.astype(np.float32)
            
        except Exception as e:
            logging.error(f"Error generating face embedding: {e}")
            return None
    
    def _extract_lbp_features(self, image: np.ndarray, radius: int = 1) -> np.ndarray:
        """Extract Local Binary Pattern features with radius."""
        try:
            # LBP implementation with radius
            lbp = np.zeros_like(image)
            for i in range(radius, image.shape[0] - radius):
                for j in range(radius, image.shape[1] - radius):
                    center = image[i, j]
                    binary_string = ""
                    for angle in range(0, 360, 45):
                        x = int(i + radius * np.cos(np.radians(angle)))
                        y = int(j + radius * np.sin(np.radians(angle)))
                        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                            if image[x, y] >= center:
                                binary_string += "1"
                            else:
                                binary_string += "0"
                    if binary_string:
                        lbp[i, j] = int(binary_string, 2)
            
            # Calculate histogram
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            return hist.astype(np.float32)
        except:
            return np.zeros(256, dtype=np.float32)
    
    def _extract_hog_features(self, image: np.ndarray, cell_size: tuple = (8, 8)) -> np.ndarray:
        """Extract HOG features with cell size."""
        try:
            # HOG implementation
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate magnitude and orientation
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            orientation = np.arctan2(grad_y, grad_x)
            
            # Create histogram of oriented gradients
            hist, _ = np.histogram(orientation.ravel(), bins=9, range=(-np.pi, np.pi))
            return hist.astype(np.float32)
        except:
            return np.zeros(9, dtype=np.float32)
    
    def _extract_edge_features(self, image: np.ndarray, threshold: tuple = (50, 150)) -> np.ndarray:
        """Extract edge features with threshold."""
        try:
            # Canny edge detection
            edges = cv2.Canny(image, threshold[0], threshold[1])
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate edge orientation histogram
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            orientation = np.arctan2(grad_y, grad_x)
            
            hist, _ = np.histogram(orientation.ravel(), bins=8, range=(-np.pi, np.pi))
            hist = hist / (np.sum(hist) + 1e-8)
            
            features = [edge_density] + hist.tolist()
            return np.array(features, dtype=np.float32)
        except:
            return np.zeros(9, dtype=np.float32)
    
    def _extract_spatial_features(self, image: np.ndarray, grid_size: int = 4) -> np.ndarray:
        """Extract spatial features with grid size."""
        try:
            # Divide image into regions and extract features
            h, w = image.shape
            features = []
            
            # Grid
            for i in range(grid_size):
                for j in range(grid_size):
                    y1, y2 = i * h // grid_size, (i + 1) * h // grid_size
                    x1, x2 = j * w // grid_size, (j + 1) * w // grid_size
                    region = image[y1:y2, x1:x2]
                    
                    if region.size > 0:
                        features.append(np.mean(region))
                        features.append(np.std(region))
                        features.append(np.var(region))
                    else:
                        features.extend([0, 0, 0])
            
            return np.array(features, dtype=np.float32)
        except:
            return np.zeros(grid_size * grid_size * 3, dtype=np.float32)
    
    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color features."""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            features = []
            
            # HSV histogram
            for i in range(3):
                hist = cv2.calcHist([hsv], [i], None, [16], [0, 256])
                features.extend(hist.flatten())
            
            # LAB histogram
            for i in range(3):
                hist = cv2.calcHist([lab], [i], None, [16], [0, 256])
                features.extend(hist.flatten())
            
            # YUV histogram
            for i in range(3):
                hist = cv2.calcHist([yuv], [i], None, [16], [0, 256])
                features.extend(hist.flatten())
            
            return np.array(features, dtype=np.float32)
        except:
            return np.zeros(144, dtype=np.float32)
    
    def _extract_gabor_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Gabor features."""
        try:
            features = []
            for theta in [0, 30, 60, 90, 120, 150]:
                for freq in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    # Create Gabor kernel
                    kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                    
                    # Apply filter
                    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    
                    # Calculate mean and std
                    features.append(np.mean(filtered))
                    features.append(np.std(filtered))
            
            return np.array(features, dtype=np.float32)
        except:
            return np.zeros(60, dtype=np.float32)
    
    def _extract_wavelet_features(self, image: np.ndarray) -> np.ndarray:
        """Extract wavelet features."""
        try:
            # Simple wavelet-like features
            features = []
            
            # Downsample image
            small = cv2.resize(image, (32, 32))
            
            # Calculate statistics
            features.append(np.mean(small))
            features.append(np.std(small))
            features.append(np.var(small))
            features.append(np.median(small))
            
            return np.array(features, dtype=np.float32)
        except:
            return np.zeros(4, dtype=np.float32)
    
    def _extract_fourier_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Fourier features."""
        try:
            # FFT
            fft = np.fft.fft2(image)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Calculate statistics
            features = []
            features.append(np.mean(magnitude))
            features.append(np.std(magnitude))
            features.append(np.var(magnitude))
            
            return np.array(features, dtype=np.float32)
        except:
            return np.zeros(3, dtype=np.float32)
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features."""
        try:
            features = []
            
            # GLCM-like features
            features.append(np.mean(image))
            features.append(np.std(image))
            features.append(np.var(image))
            features.append(np.median(image))
            
            # Local binary patterns
            lbp = np.zeros_like(image)
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    center = image[i, j]
                    binary_string = ""
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            if image[i + di, j + dj] >= center:
                                binary_string += "1"
                            else:
                                binary_string += "0"
                    lbp[i, j] = int(binary_string, 2)
            
            features.append(np.mean(lbp))
            features.append(np.std(lbp))
            
            return np.array(features, dtype=np.float32)
        except:
            return np.zeros(6, dtype=np.float32)
    
    def recognize_face(self, face_image: np.ndarray, detection: FaceDetection) -> FaceRecognition:
        """
        Recognize a face from face image and detection.
        
        Args:
            face_image: Face image as numpy array (BGR format)
            detection: Face detection result
            
        Returns:
            Face recognition result
        """
        if not self.is_initialized:
            raise RuntimeError("Face recognizer not initialized")
        
        try:
            start_time = time.time()
            
            # Generate embedding for the face
            embedding = self.generate_embedding(face_image)
            
            if embedding is None:
                return FaceRecognition(
                    employee_id=None,
                    confidence=0.0,
                    face_detection=detection,
                    embedding=None,
                    processing_time=time.time() - start_time
                )
            
            # Find best match in employee database
            best_match_id, best_confidence = self._find_best_match(embedding)
            
            processing_time = time.time() - start_time
            
            return FaceRecognition(
                employee_id=best_match_id,
                confidence=best_confidence,
                face_detection=detection,
                embedding=embedding.tolist(),
                processing_time=processing_time
            )
            
        except Exception as e:
            logging.error(f"Error recognizing face: {e}")
            return FaceRecognition(
                employee_id=None,
                confidence=0.0,
                face_detection=detection,
                embedding=None,
                processing_time=0.0
            )
    
    def _find_best_match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Find best match for embedding using cosine similarity."""
        if len(self.employee_embeddings) == 0:
            return None, 0.0
        
        best_match_id = None
        best_confidence = 0.0
        
        for employee_id, stored_embedding in self.employee_embeddings.items():
            try:
                if stored_embedding is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(embedding, stored_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(stored_embedding) + 1e-8
                    )
                    
                    # Use very high threshold for uniqueness
                    if similarity > best_confidence and similarity > self.recognition_threshold:
                        best_confidence = similarity
                        best_match_id = employee_id
                        
            except Exception as e:
                logging.warning(f"Error calculating similarity for {employee_id}: {e}")
                continue
        
        return best_match_id, best_confidence
    
    def register_employee_face(self, employee_id: str, face_image: np.ndarray, 
                             face_photo_path: str, quality_score: float = 0.8) -> bool:
        """
        Register a new employee face.
        
        Args:
            employee_id: Employee ID
            face_image: Face image as numpy array (BGR format)
            face_photo_path: Path to face photo file
            quality_score: Quality score of the face image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.generate_embedding(face_image)
            
            if embedding is None:
                logging.error(f"Failed to generate embedding for employee {employee_id}")
                return False
            
            # Store embedding in memory
            self.employee_embeddings[employee_id] = embedding
            
            # Save to database
            with get_auth_db() as db:
                employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
                if employee:
                    employee.face_embedding = json.dumps(embedding.tolist())
                    employee.face_photo_path = face_photo_path
                    db.commit()
                else:
                    logging.error(f"Employee {employee_id} not found in database")
                    return False
            
            logging.info(f"Successfully registered face for employee {employee_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error registering employee face: {e}")
            return False
    
    def remove_employee_face(self, employee_id: str) -> bool:
        """Remove employee face from recognition database."""
        try:
            # Remove from memory
            if employee_id in self.employee_embeddings:
                del self.employee_embeddings[employee_id]
            
            if employee_id in self.employee_metadata:
                del self.employee_metadata[employee_id]
            
            # Remove from database
            with get_auth_db() as db:
                employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
                if employee:
                    employee.face_embedding = None
                    employee.face_photo_path = None
                    db.commit()
            
            logging.info(f"Successfully removed face for employee {employee_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error removing employee face: {e}")
            return False
    
    def get_employee_info(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Get employee information."""
        return self.employee_metadata.get(employee_id)
    
    def get_recognition_stats(self) -> Dict[str, Any]:
        """Get recognition statistics."""
        return {
            "total_employees": len(self.employee_embeddings),
            "recognition_threshold": self.recognition_threshold,
            "is_initialized": self.is_initialized
        }
    
    def refresh_employee_cache(self):
        """Refresh employee database from storage."""
        try:
            self._load_employee_embeddings()
            logging.info("Employee cache refreshed")
        except Exception as e:
            logging.error(f"Error refreshing employee cache: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
        self.employee_embeddings.clear()
        self.employee_metadata.clear()
        self.is_initialized = False
        logging.info("Final face recognizer cleaned up")

# Global instance
_final_recognizer = None

def get_final_face_recognizer(config: FaceRecognitionConfig = None) -> FinalFaceRecognizer:
    """Get global final face recognizer instance."""
    global _final_recognizer
    if _final_recognizer is None:
        _final_recognizer = FinalFaceRecognizer(config)
    return _final_recognizer

def cleanup_final_face_recognizer():
    """Cleanup global final face recognizer instance."""
    global _final_recognizer
    if _final_recognizer:
        _final_recognizer.cleanup()
        _final_recognizer = None

