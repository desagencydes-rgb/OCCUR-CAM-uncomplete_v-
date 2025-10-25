"""
OCCUR-CAM Face Recognition Engine
Face recognition and embedding generation using InsightFace.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
import time
import json
from pathlib import Path

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
except ImportError:
    logging.error("InsightFace not installed. Please install with: pip install insightface")
    raise

from models.face_models import (
    FaceDetection, FaceRecognition, FaceEmbedding, 
    FaceRecognitionConfig, FrameAnalysis
)
from config.settings import config
from config.database import get_auth_db
from database.schemas.auth_schemas import Employee

class FaceRecognizer:
    """High-accuracy face recognition using InsightFace."""
    
    def __init__(self, config: FaceRecognitionConfig = None):
        """Initialize face recognizer with configuration."""
        self.config = config or FaceRecognitionConfig()
        self.app = None
        self.is_initialized = False
        self.employee_embeddings = {}  # Cache for employee embeddings
        self.employee_metadata = {}    # Cache for employee metadata
        
        # Initialize the recognizer
        self._initialize_recognizer()
    
    def _initialize_recognizer(self):
        """Initialize InsightFace recognizer."""
        try:
            logging.info("Initializing InsightFace recognizer...")
            
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
            logging.info("InsightFace recognizer initialized successfully")
            
            # Load employee embeddings
            self._load_employee_embeddings()
            
        except Exception as e:
            logging.error(f"Failed to initialize InsightFace recognizer: {e}")
            raise
    
    def _load_employee_embeddings(self):
        """Load employee embeddings from database."""
        try:
            with get_auth_db() as db:
                employees = db.query(Employee).filter(
                    Employee.face_embedding.isnot(None),
                    Employee.is_active == True
                ).all()
                
                for employee in employees:
                    try:
                        # Parse embedding from JSON
                        embedding_data = json.loads(employee.face_embedding)
                        embedding_vector = np.array(embedding_data['embedding'])
                        
                        # Store in cache
                        self.employee_embeddings[employee.employee_id] = embedding_vector
                        self.employee_metadata[employee.employee_id] = {
                            'name': f"{employee.first_name} {employee.last_name}",
                            'department': employee.department,
                            'quality_score': employee.face_quality_score,
                            'created_at': embedding_data.get('created_at', '')
                        }
                        
                    except Exception as e:
                        logging.warning(f"Error loading embedding for employee {employee.employee_id}: {e}")
                
                logging.info(f"Loaded {len(self.employee_embeddings)} employee embeddings")
                
        except Exception as e:
            logging.error(f"Error loading employee embeddings: {e}")
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding from face image.
        
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
            
            # Resize image to standard size for better recognition
            if face_image.shape[0] != 112 or face_image.shape[1] != 112:
                face_image = cv2.resize(face_image, (112, 112))
            
            # Convert BGR to RGB for InsightFace
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Ensure image is uint8
            if face_image_rgb.dtype != np.uint8:
                face_image_rgb = face_image_rgb.astype(np.uint8)
            
            # Use the recognition model directly - create a fake detection
            # Since we already have a face region, we'll simulate a detection
            try:
                # Create a larger image with padding to help with detection
                padded_image = np.zeros((224, 224, 3), dtype=np.uint8)
                start_y = (224 - 112) // 2
                start_x = (224 - 112) // 2
                padded_image[start_y:start_y+112, start_x:start_x+112] = face_image_rgb
                
                # Try with the padded image first
                faces = self.app.get(padded_image)
                
                if len(faces) == 0:
                    # If still no faces, try with the original face image
                    faces = self.app.get(face_image_rgb)
                
                if len(faces) == 0:
                    # Last resort: try with a much larger context
                    large_image = np.zeros((448, 448, 3), dtype=np.uint8)
                    start_y = (448 - 112) // 2
                    start_x = (448 - 112) // 2
                    large_image[start_y:start_y+112, start_x:start_x+112] = face_image_rgb
                    faces = self.app.get(large_image)
                
                if len(faces) == 0:
                    logging.warning("No faces found in image for embedding generation")
                    return None
                
                # Use the first (and should be only) face
                face = faces[0]
                
                # Extract embedding
                if hasattr(face, 'embedding') and face.embedding is not None:
                    embedding = face.embedding
                    logging.debug(f"Generated embedding with shape: {embedding.shape}")
                    return embedding
                else:
                    logging.warning("No embedding found in face analysis")
                    return None
                    
            except Exception as analysis_error:
                logging.warning(f"Face analysis error: {analysis_error}")
                return None
                
        except Exception as e:
            logging.error(f"Error generating face embedding: {e}")
            return None
    
    def recognize_face(self, face_image: np.ndarray, detection: FaceDetection) -> FaceRecognition:
        """
        Recognize a face from face image and detection.
        
        Args:
            face_image: Face image as numpy array (BGR format)
            detection: Face detection result
            
        Returns:
            FaceRecognition result
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
            logging.error(f"Error in face recognition: {e}")
            return FaceRecognition(
                employee_id=None,
                confidence=0.0,
                face_detection=detection,
                embedding=None,
                processing_time=time.time() - start_time
            )
    
    def _find_best_match(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Find best matching employee for given embedding."""
        if not self.employee_embeddings:
            return None, 0.0
        
        try:
            best_employee_id = None
            best_confidence = 0.0
            
            # Calculate cosine similarity with all employee embeddings
            for employee_id, stored_embedding in self.employee_embeddings.items():
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, stored_embedding)
                
                # Apply quality factor if available
                if employee_id in self.employee_metadata:
                    quality_score = self.employee_metadata[employee_id].get('quality_score', 1.0)
                    similarity *= quality_score
                
                if similarity > best_confidence:
                    best_confidence = similarity
                    best_employee_id = employee_id
            
            # Apply recognition threshold
            if best_confidence < self.config.recognition_threshold:
                return None, best_confidence
            
            return best_employee_id, best_confidence
            
        except Exception as e:
            logging.error(f"Error finding best match: {e}")
            return None, 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            
            return float(similarity)
            
        except Exception as e:
            logging.warning(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def register_employee_face(self, employee_id: str, face_image: np.ndarray, 
                             face_photo_path: str, quality_score: float) -> bool:
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
            
            # Create face embedding object
            face_embedding = FaceEmbedding(
                employee_id=employee_id,
                embedding_vector=embedding,
                face_photo_path=face_photo_path,
                quality_score=quality_score,
                created_at=time.time()
            )
            
            # Validate embedding
            errors = face_embedding.validate()
            if errors:
                logging.error(f"Face embedding validation failed: {errors}")
                return False
            
            # Save to database
            with get_auth_db() as db:
                employee = db.query(Employee).filter(
                    Employee.employee_id == employee_id
                ).first()
                
                if not employee:
                    logging.error(f"Employee {employee_id} not found")
                    return False
                
                # Update employee record
                employee.face_embedding = face_embedding.to_json()
                employee.face_photo_path = face_photo_path
                employee.face_quality_score = quality_score
                employee.updated_at = time.time()
                
                db.commit()
            
            # Update cache
            self.employee_embeddings[employee_id] = embedding
            self.employee_metadata[employee_id] = {
                'name': f"{employee.first_name} {employee.last_name}",
                'department': employee.department,
                'quality_score': quality_score,
                'created_at': face_embedding.created_at.isoformat()
            }
            
            logging.info(f"Successfully registered face for employee {employee_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error registering employee face: {e}")
            return False
    
    def remove_employee_face(self, employee_id: str) -> bool:
        """Remove employee face from recognition database."""
        try:
            # Remove from database
            with get_auth_db() as db:
                employee = db.query(Employee).filter(
                    Employee.employee_id == employee_id
                ).first()
                
                if employee:
                    employee.face_embedding = None
                    employee.face_photo_path = None
                    employee.face_quality_score = None
                    employee.updated_at = time.time()
                    db.commit()
            
            # Remove from cache
            self.employee_embeddings.pop(employee_id, None)
            self.employee_metadata.pop(employee_id, None)
            
            logging.info(f"Successfully removed face for employee {employee_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error removing employee face: {e}")
            return False
    
    def recognize_faces_batch(self, face_images: List[np.ndarray], 
                            detections: List[FaceDetection]) -> List[FaceRecognition]:
        """Recognize faces in a batch of images."""
        if len(face_images) != len(detections):
            raise ValueError("Number of images must match number of detections")
        
        results = []
        for face_image, detection in zip(face_images, detections):
            recognition = self.recognize_face(face_image, detection)
            results.append(recognition)
        
        return results
    
    def get_employee_info(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Get employee information from cache."""
        return self.employee_metadata.get(employee_id)
    
    def get_recognition_stats(self) -> Dict[str, Any]:
        """Get recognition statistics."""
        return {
            "total_employees": len(self.employee_embeddings),
            "model_name": config.face_recognition.MODEL_NAME,
            "recognition_threshold": self.config.recognition_threshold,
            "embedding_size": self.config.embedding_size,
            "is_initialized": self.is_initialized
        }
    
    def refresh_employee_cache(self):
        """Refresh employee embeddings cache from database."""
        try:
            self.employee_embeddings.clear()
            self.employee_metadata.clear()
            self._load_employee_embeddings()
            logging.info("Employee cache refreshed")
        except Exception as e:
            logging.error(f"Error refreshing employee cache: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.app:
                del self.app
            self.app = None
            self.is_initialized = False
            self.employee_embeddings.clear()
            self.employee_metadata.clear()
            logging.info("Face recognizer cleaned up")
        except Exception as e:
            logging.warning(f"Error during recognizer cleanup: {e}")

# Global recognizer instance
_recognizer_instance = None

def get_face_recognizer(config: FaceRecognitionConfig = None) -> 'WorkingFaceRecognizer':
    """Get global face recognizer instance."""
    global _recognizer_instance
    
    if _recognizer_instance is None:
        from core.working_face_recognizer import WorkingFaceRecognizer
        _recognizer_instance = WorkingFaceRecognizer(config)
    
    return _recognizer_instance

def cleanup_face_recognizer():
    """Cleanup global face recognizer instance."""
    global _recognizer_instance
    
    if _recognizer_instance:
        _recognizer_instance.cleanup()
        _recognizer_instance = None
