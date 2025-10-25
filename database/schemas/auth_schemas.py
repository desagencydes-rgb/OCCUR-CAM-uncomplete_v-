"""
OCCUR-CAM Authentication Database Schemas
Defines all tables and models for the authentication database.
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, ForeignKey, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from config.database import AuthBase
import uuid

class Employee(AuthBase):
    """Employee information and face data."""
    __tablename__ = "employees"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), unique=True, index=True, nullable=False)
    
    # Personal information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True)
    phone = Column(String(20))
    department = Column(String(100))
    position = Column(String(100))
    
    # Employment status
    is_active = Column(Boolean, default=True, index=True)
    hire_date = Column(DateTime)
    termination_date = Column(DateTime, nullable=True)
    
    # Face recognition data
    face_embedding = Column(Text)  # JSON string of face embedding vector
    face_photo_path = Column(String(500))  # Path to reference photo
    face_quality_score = Column(Float)  # Quality score of reference photo
    
    # System metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(100))
    updated_by = Column(String(100))
    
    # Relationships
    auth_logs = relationship("AuthLog", back_populates="employee")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_employee_active', 'is_active'),
        Index('idx_employee_department', 'department'),
        Index('idx_employee_created', 'created_at'),
    )

class AuthLog(AuthBase):
    """Authentication attempt logs."""
    __tablename__ = "auth_logs"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    
    # Employee reference
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    
    # Authentication details
    auth_type = Column(String(20), nullable=False)  # 'success', 'failure', 'unknown'
    confidence_score = Column(Float)  # Face recognition confidence
    processing_time = Column(Float)  # Time taken for processing in seconds
    
    # Camera and location info
    camera_id = Column(String(100), index=True)
    camera_location = Column(String(200))
    
    # Image data
    snapshot_path = Column(String(500))  # Path to captured snapshot
    face_count = Column(Integer, default=0)  # Number of faces detected
    
    # System info
    timestamp = Column(DateTime, default=func.now(), index=True)
    ip_address = Column(String(45))  # IPv4 or IPv6
    user_agent = Column(String(500))
    
    # Additional metadata
    metadata_json = Column(Text)  # JSON string for additional data
    notes = Column(Text)
    
    # Relationships
    employee = relationship("Employee", back_populates="auth_logs")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_auth_timestamp', 'timestamp'),
        Index('idx_auth_type', 'auth_type'),
        Index('idx_auth_camera', 'camera_id'),
        Index('idx_auth_employee', 'employee_id'),
    )

class CameraConfig(AuthBase):
    """Camera configuration and status."""
    __tablename__ = "camera_configs"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Camera details
    name = Column(String(200), nullable=False)
    source = Column(String(500), nullable=False)  # Camera source (USB index, IP, etc.)
    camera_type = Column(String(50), nullable=False)  # 'usb', 'ip', 'ivcam', 'rtsp'
    
    # Location and description
    location = Column(String(200))
    description = Column(Text)
    
    # Camera settings
    width = Column(Integer, default=640)
    height = Column(Integer, default=480)
    fps = Column(Integer, default=30)
    timeout = Column(Integer, default=5)
    
    # Status and health
    is_enabled = Column(Boolean, default=True, index=True)
    is_online = Column(Boolean, default=False, index=True)
    last_seen = Column(DateTime)
    health_score = Column(Float, default=0.0)  # 0-1 health score
    
    # Error tracking
    error_count = Column(Integer, default=0)
    last_error = Column(Text)
    last_error_time = Column(DateTime)
    
    # System metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_camera_enabled', 'is_enabled'),
        Index('idx_camera_online', 'is_online'),
        Index('idx_camera_type', 'camera_type'),
    )

class SystemLog(AuthBase):
    """System-wide logs and events."""
    __tablename__ = "system_logs"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    
    # Log details
    level = Column(String(20), nullable=False, index=True)  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component = Column(String(100), nullable=False, index=True)  # 'face_engine', 'camera_manager', etc.
    message = Column(Text, nullable=False)
    
    # Additional data
    data = Column(Text)  # JSON string for additional structured data
    exception = Column(Text)  # Exception traceback if applicable
    
    # System info
    timestamp = Column(DateTime, default=func.now(), index=True)
    process_id = Column(Integer)
    thread_id = Column(Integer)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_system_level', 'level'),
        Index('idx_system_component', 'component'),
        Index('idx_system_timestamp', 'timestamp'),
    )

class AuthSession(AuthBase):
    """Active authentication sessions."""
    __tablename__ = "auth_sessions"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    
    # Employee reference
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    
    # Session details
    created_at = Column(DateTime, default=func.now(), index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    last_activity = Column(DateTime, default=func.now())
    
    # Session data
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    camera_id = Column(String(100))
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    
    # Relationships
    employee = relationship("Employee")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_session_employee', 'employee_id'),
        Index('idx_session_active', 'is_active'),
        Index('idx_session_expires', 'expires_at'),
    )
