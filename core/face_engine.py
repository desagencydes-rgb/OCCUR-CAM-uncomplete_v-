"""
OCCUR-CAM Face Processing Engine
Main face detection and recognition engine that coordinates all face processing.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
import time
import uuid
from datetime import datetime
from pathlib import Path

from models.face_models import (
    FaceDetection, FaceRecognition, FrameAnalysis, 
    FaceRecognitionConfig, FaceRecognitionResult
)
from core.face_detector import FaceDetector, get_face_detector
from core.ultra_simple_recognizer import UltraSimpleRecognizer, get_ultra_simple_recognizer
from config.settings import config

class FaceEngine:
    """Main face processing engine that coordinates detection and recognition."""
    
    def __init__(self, config: FaceRecognitionConfig = None):
        """Initialize face engine with configuration."""
        self.config = config or FaceRecognitionConfig()
        self.detector = None
        self.recognizer = None
        self.is_initialized = False
        
        # Initialize components
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize face detection and recognition components."""
        try:
            logging.info("Initializing face processing engine...")
            
            # Initialize detector
            self.detector = get_face_detector(self.config)
            
            # Initialize recognizer
            self.recognizer = get_ultra_simple_recognizer(self.config)
            
            self.is_initialized = True
            logging.info("Face processing engine initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize face engine: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray, camera_id: str) -> FrameAnalysis:
        """
        Process a single frame for face detection and recognition.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            camera_id: Camera identifier
            
        Returns:
            FrameAnalysis result
        """
        if not self.is_initialized:
            raise RuntimeError("Face engine not initialized")
        
        try:
            start_time = time.time()
            frame_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Detect faces
            detections = self.detector.detect_faces(frame)
            
            # Extract face regions and recognize
            recognitions = []
            for detection in detections:
                # Extract face region
                face_region = self.detector.extract_face_region(frame, detection)
                
                # Always use the original frame for recognition
                # This handles cases where the image is already a face
                if frame.shape[0] > 50 and frame.shape[1] > 50:
                    # Use the entire frame as the face
                    processed_face = self.detector.preprocess_face(frame)
                    recognition = self.recognizer.recognize_face(processed_face, detection)
                    recognitions.append(recognition)
                else:
                    # Create empty recognition for failed extraction
                    recognition = FaceRecognition(
                        employee_id=None,
                        confidence=0.0,
                        face_detection=detection,
                        embedding=None,
                        processing_time=0.0
                    )
                    recognitions.append(recognition)
            
            processing_time = time.time() - start_time
            
            # Create frame analysis
            analysis = FrameAnalysis(
                frame_id=frame_id,
                timestamp=timestamp,
                face_detections=detections,
                face_recognitions=recognitions,
                processing_time=processing_time,
                camera_id=camera_id,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0]
            )
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            # Return empty analysis on error
            return FrameAnalysis(
                frame_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                face_detections=[],
                face_recognitions=[],
                processing_time=0.0,
                camera_id=camera_id,
                frame_width=frame.shape[1] if frame is not None else 0,
                frame_height=frame.shape[0] if frame is not None else 0
            )
    
    def process_frames_batch(self, frames: List[np.ndarray], camera_ids: List[str]) -> List[FrameAnalysis]:
        """Process multiple frames in batch."""
        if len(frames) != len(camera_ids):
            raise ValueError("Number of frames must match number of camera IDs")
        
        results = []
        for frame, camera_id in zip(frames, camera_ids):
            analysis = self.process_frame(frame, camera_id)
            results.append(analysis)
        
        return results
    
    def register_employee_face(self, employee_id: str, face_image: np.ndarray, 
                             face_photo_path: str) -> bool:
        """
        Register a new employee face for recognition.
        
        Args:
            employee_id: Employee ID
            face_image: Face image as numpy array (BGR format)
            face_photo_path: Path to face photo file
            quality_score: Quality score of the face image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Detect faces in the image
            detections = self.detector.detect_faces(face_image)
            
            if not detections:
                logging.error(f"No faces detected in image for employee {employee_id}")
                return False
            
            # Use the best detection (highest confidence)
            best_detection = max(detections, key=lambda x: x.confidence)
            
            # Extract face region
            face_region = self.detector.extract_face_region(face_image, best_detection)
            
            if face_region.size == 0:
                logging.error(f"Failed to extract face region for employee {employee_id}")
                return False
            
            # Preprocess face
            processed_face = self.detector.preprocess_face(face_region)
            
            # Assess face quality
            quality_metrics = self.detector._assess_face_quality(face_region)
            quality_score = quality_metrics.overall_score
            
            # Register with recognizer
            success = self.recognizer.register_employee_face(
                employee_id, processed_face, face_photo_path, quality_score
            )
            
            if success:
                logging.info(f"Successfully registered face for employee {employee_id}")
            else:
                logging.error(f"Failed to register face for employee {employee_id}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error registering employee face: {e}")
            return False
    
    def remove_employee_face(self, employee_id: str) -> bool:
        """Remove employee face from recognition database."""
        try:
            success = self.recognizer.remove_employee_face(employee_id)
            
            if success:
                logging.info(f"Successfully removed face for employee {employee_id}")
            else:
                logging.error(f"Failed to remove face for employee {employee_id}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error removing employee face: {e}")
            return False
    
    def get_recognized_employees(self, analysis: FrameAnalysis, 
                               threshold: float = None) -> List[Dict[str, Any]]:
        """Get list of recognized employees from frame analysis."""
        if threshold is None:
            threshold = self.config.recognition_threshold
        
        recognized = []
        for recognition in analysis.get_recognized_faces(threshold):
            if recognition.employee_id:
                employee_info = self.recognizer.get_employee_info(recognition.employee_id)
                if employee_info:
                    recognized.append({
                        'employee_id': recognition.employee_id,
                        'name': employee_info.get('name', 'Unknown'),
                        'department': employee_info.get('department', 'Unknown'),
                        'confidence': recognition.confidence,
                        'bbox': recognition.face_detection.bbox,
                        'processing_time': recognition.processing_time
                    })
        
        return recognized
    
    def get_unknown_faces(self, analysis: FrameAnalysis, 
                         threshold: float = None) -> List[Dict[str, Any]]:
        """Get list of unknown faces from frame analysis."""
        if threshold is None:
            threshold = self.config.recognition_threshold
        
        unknown = []
        for detection in analysis.get_unknown_faces(threshold):
            unknown.append({
                'face_id': detection.face_id,
                'bbox': detection.bbox,
                'confidence': detection.confidence,
                'center': detection.get_center(),
                'area': detection.get_area()
            })
        
        return unknown
    
    def process_video_stream(self, video_source, camera_id: str, 
                           max_frames: int = None, 
                           callback=None) -> FaceRecognitionResult:
        """
        Process video stream for face recognition.
        
        Args:
            video_source: Video source (camera index, video file, etc.)
            camera_id: Camera identifier
            max_frames: Maximum number of frames to process (None for unlimited)
            callback: Optional callback function for each frame result
            
        Returns:
            FaceRecognitionResult with all processing results
        """
        try:
            # Open video source
            cap = cv2.VideoCapture(video_source)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video source: {video_source}")
            
            # Set video properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.camera.FPS)
            
            result = FaceRecognitionResult()
            result.start_time = datetime.now()
            
            frame_count = 0
            
            try:
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        logging.warning("End of video stream or failed to read frame")
                        break
                    
                    # Process frame
                    analysis = self.process_frame(frame, camera_id)
                    result.add_frame_analysis(analysis)
                    
                    # Call callback if provided
                    if callback:
                        callback(analysis)
                    
                    frame_count += 1
                    
                    # Check max frames limit
                    if max_frames and frame_count >= max_frames:
                        break
                    
                    # Log progress
                    if frame_count % 100 == 0:
                        logging.info(f"Processed {frame_count} frames")
                
            finally:
                cap.release()
            
            result.end_time = datetime.now()
            
            logging.info(f"Video processing completed: {frame_count} frames processed")
            return result
            
        except Exception as e:
            logging.error(f"Error processing video stream: {e}")
            return FaceRecognitionResult()
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics and status."""
        detector_info = self.detector.get_detector_info() if self.detector else {}
        recognizer_stats = self.recognizer.get_recognition_stats() if self.recognizer else {}
        
        return {
            "is_initialized": self.is_initialized,
            "detector": detector_info,
            "recognizer": recognizer_stats,
            "config": self.config.to_dict()
        }
    
    def update_config(self, new_config: FaceRecognitionConfig):
        """Update engine configuration."""
        try:
            self.config = new_config
            
            # Reinitialize components with new config
            if self.detector:
                self.detector.config = new_config
            
            if self.recognizer:
                self.recognizer.config = new_config
            
            logging.info("Face engine configuration updated")
            
        except Exception as e:
            logging.error(f"Error updating engine configuration: {e}")
    
    def refresh_employee_database(self):
        """Refresh employee database from storage."""
        try:
            if self.recognizer:
                self.recognizer.refresh_employee_cache()
            logging.info("Employee database refreshed")
        except Exception as e:
            logging.error(f"Error refreshing employee database: {e}")
    
    def reload_recognizer(self):
        """Reload the face recognizer with updated user data."""
        try:
            if self.recognizer:
                # Cleanup old recognizer
                self.recognizer.cleanup()
                
                # Create new recognizer with updated data
                self.recognizer = get_ultra_simple_recognizer(self.config)
                
                logging.info("Face recognizer reloaded successfully")
        except Exception as e:
            logging.error(f"Error reloading face recognizer: {e}")
    
    def cleanup(self):
        """Cleanup engine resources."""
        try:
            if self.detector:
                self.detector.cleanup()
            if self.recognizer:
                self.recognizer.cleanup()
            
            self.detector = None
            self.recognizer = None
            self.is_initialized = False
            
            logging.info("Face engine cleaned up")
            
        except Exception as e:
            logging.warning(f"Error during engine cleanup: {e}")

# Global engine instance
_engine_instance = None

def get_face_engine(config: FaceRecognitionConfig = None) -> FaceEngine:
    """Get global face engine instance."""
    global _engine_instance
    
    if _engine_instance is None:
        _engine_instance = FaceEngine(config)
    
    return _engine_instance

def cleanup_face_engine():
    """Cleanup global face engine instance."""
    global _engine_instance
    
    if _engine_instance:
        _engine_instance.cleanup()
        _engine_instance = None
