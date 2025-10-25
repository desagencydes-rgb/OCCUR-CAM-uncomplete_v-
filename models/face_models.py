"""
OCCUR-CAM Face Recognition Models
Data models and structures for face detection and recognition.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import json
import numpy as np

@dataclass
class FaceDetection:
    """Face detection result data model."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None
    face_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "landmarks": self.landmarks,
            "face_id": self.face_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaceDetection":
        """Create from dictionary."""
        return cls(**data)
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point of the face bounding box."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)
    
    def get_area(self) -> int:
        """Get area of the face bounding box."""
        x, y, w, h = self.bbox
        return w * h

@dataclass
class FaceRecognition:
    """Face recognition result data model."""
    employee_id: Optional[str]
    confidence: float
    face_detection: FaceDetection
    embedding: Optional[List[float]] = None
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "employee_id": self.employee_id,
            "confidence": self.confidence,
            "face_detection": self.face_detection.to_dict(),
            "embedding": self.embedding,
            "processing_time": self.processing_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaceRecognition":
        """Create from dictionary."""
        face_detection = FaceDetection.from_dict(data["face_detection"])
        return cls(
            employee_id=data["employee_id"],
            confidence=data["confidence"],
            face_detection=face_detection,
            embedding=data.get("embedding"),
            processing_time=data.get("processing_time", 0.0)
        )
    
    def is_recognized(self, threshold: float = 0.7) -> bool:
        """Check if face is recognized above threshold."""
        return self.employee_id is not None and self.confidence >= threshold

@dataclass
class FrameAnalysis:
    """Complete frame analysis result."""
    frame_id: str
    timestamp: datetime
    face_detections: List[FaceDetection]
    face_recognitions: List[FaceRecognition]
    processing_time: float
    camera_id: str
    frame_width: int
    frame_height: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat(),
            "face_detections": [fd.to_dict() for fd in self.face_detections],
            "face_recognitions": [fr.to_dict() for fr in self.face_recognitions],
            "processing_time": self.processing_time,
            "camera_id": self.camera_id,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameAnalysis":
        """Create from dictionary."""
        face_detections = [FaceDetection.from_dict(fd) for fd in data["face_detections"]]
        face_recognitions = [FaceRecognition.from_dict(fr) for fr in data["face_recognitions"]]
        
        return cls(
            frame_id=data["frame_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            face_detections=face_detections,
            face_recognitions=face_recognitions,
            processing_time=data["processing_time"],
            camera_id=data["camera_id"],
            frame_width=data["frame_width"],
            frame_height=data["frame_height"]
        )
    
    def get_recognized_faces(self, threshold: float = 0.7) -> List[FaceRecognition]:
        """Get faces that were successfully recognized."""
        return [fr for fr in self.face_recognitions if fr.is_recognized(threshold)]
    
    def get_unknown_faces(self, threshold: float = 0.7) -> List[FaceDetection]:
        """Get faces that were not recognized."""
        recognized_bboxes = {fr.face_detection.bbox for fr in self.face_recognitions if fr.is_recognized(threshold)}
        return [fd for fd in self.face_detections if fd.bbox not in recognized_bboxes]

@dataclass
class FaceEmbedding:
    """Face embedding data model."""
    employee_id: str
    embedding_vector: np.ndarray
    face_photo_path: str
    quality_score: float
    created_at: datetime
    model_version: str = "buffalo_l"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "employee_id": self.employee_id,
            "embedding": self.embedding_vector.tolist(),
            "face_photo_path": self.face_photo_path,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat(),
            "model_version": self.model_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaceEmbedding":
        """Create from dictionary."""
        return cls(
            employee_id=data["employee_id"],
            embedding_vector=np.array(data["embedding"]),
            face_photo_path=data["face_photo_path"],
            quality_score=data["quality_score"],
            created_at=datetime.fromisoformat(data["created_at"]),
            model_version=data.get("model_version", "buffalo_l")
        )
    
    def to_json(self) -> str:
        """Convert to JSON string for database storage."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "FaceEmbedding":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        """Validate face embedding data."""
        errors = []
        
        if not self.employee_id:
            errors.append("Employee ID is required")
        
        if self.embedding_vector is None or len(self.embedding_vector) == 0:
            errors.append("Embedding vector is required")
        
        if not self.face_photo_path:
            errors.append("Face photo path is required")
        
        if self.quality_score < 0 or self.quality_score > 1:
            errors.append("Quality score must be between 0 and 1")
        
        return errors

@dataclass
class FaceQualityMetrics:
    """Face quality assessment metrics."""
    sharpness: float
    brightness: float
    contrast: float
    symmetry: float
    pose_angle: float
    overall_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "sharpness": self.sharpness,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "symmetry": self.symmetry,
            "pose_angle": self.pose_angle,
            "overall_score": self.overall_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "FaceQualityMetrics":
        """Create from dictionary."""
        return cls(**data)
    
    def is_good_quality(self, threshold: float = 0.7) -> bool:
        """Check if face quality is above threshold."""
        return self.overall_score >= threshold

@dataclass
class FaceRecognitionConfig:
    """Face recognition configuration parameters."""
    detection_threshold: float = 0.6
    recognition_threshold: float = 0.7
    max_faces_per_frame: int = 10
    face_size: Tuple[int, int] = (112, 112)
    embedding_size: int = 512
    model_name: str = "buffalo_l"
    batch_size: int = 4
    enable_landmarks: bool = True
    enable_quality_assessment: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detection_threshold": self.detection_threshold,
            "recognition_threshold": self.recognition_threshold,
            "max_faces_per_frame": self.max_faces_per_frame,
            "face_size": self.face_size,
            "embedding_size": self.embedding_size,
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "enable_landmarks": self.enable_landmarks,
            "enable_quality_assessment": self.enable_quality_assessment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaceRecognitionConfig":
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if not 0 <= self.detection_threshold <= 1:
            errors.append("Detection threshold must be between 0 and 1")
        
        if not 0 <= self.recognition_threshold <= 1:
            errors.append("Recognition threshold must be between 0 and 1")
        
        if self.max_faces_per_frame <= 0:
            errors.append("Max faces per frame must be positive")
        
        if self.face_size[0] <= 0 or self.face_size[1] <= 0:
            errors.append("Face size must be positive")
        
        if self.embedding_size <= 0:
            errors.append("Embedding size must be positive")
        
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        return errors

class FaceRecognitionResult:
    """Container for face recognition results and statistics."""
    
    def __init__(self):
        self.frame_analyses: List[FrameAnalysis] = []
        self.total_frames: int = 0
        self.total_faces_detected: int = 0
        self.total_faces_recognized: int = 0
        self.total_processing_time: float = 0.0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def add_frame_analysis(self, analysis: FrameAnalysis):
        """Add a frame analysis result."""
        self.frame_analyses.append(analysis)
        self.total_frames += 1
        self.total_faces_detected += len(analysis.face_detections)
        self.total_faces_recognized += len(analysis.get_recognized_faces())
        self.total_processing_time += analysis.processing_time
    
    def get_recognition_rate(self) -> float:
        """Get overall face recognition rate."""
        if self.total_faces_detected == 0:
            return 0.0
        return self.total_faces_recognized / self.total_faces_detected
    
    def get_average_processing_time(self) -> float:
        """Get average processing time per frame."""
        if self.total_frames == 0:
            return 0.0
        return self.total_processing_time / self.total_frames
    
    def get_duration(self) -> float:
        """Get total duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_frames": self.total_frames,
            "total_faces_detected": self.total_faces_detected,
            "total_faces_recognized": self.total_faces_recognized,
            "recognition_rate": self.get_recognition_rate(),
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.get_average_processing_time(),
            "duration": self.get_duration(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "frame_analyses": [fa.to_dict() for fa in self.frame_analyses]
        }
