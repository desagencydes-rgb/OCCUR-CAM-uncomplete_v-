"""
OCCUR-CAM Main Database Schemas
Defines all tables and models for the main application database.
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, ForeignKey, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from config.database import MainBase
import uuid

class Site(MainBase):
    """Physical sites/locations where cameras are deployed."""
    __tablename__ = "sites"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    site_id = Column(String(50), unique=True, index=True, nullable=False)
    
    # Site information
    name = Column(String(200), nullable=False)
    address = Column(Text)
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100))
    postal_code = Column(String(20))
    
    # Contact information
    contact_person = Column(String(200))
    contact_email = Column(String(255))
    contact_phone = Column(String(20))
    
    # Site settings
    timezone = Column(String(50), default="UTC")
    is_active = Column(Boolean, default=True, index=True)
    
    # System metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    cameras = relationship("Camera", back_populates="site")
    employees = relationship("EmployeeSite", back_populates="site")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_site_active', 'is_active'),
        Index('idx_site_city', 'city'),
    )

class Camera(MainBase):
    """Camera hardware and configuration."""
    __tablename__ = "cameras"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Site reference
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=False)
    
    # Camera details
    name = Column(String(200), nullable=False)
    model = Column(String(100))
    manufacturer = Column(String(100))
    serial_number = Column(String(100))
    
    # Technical specifications
    resolution_width = Column(Integer)
    resolution_height = Column(Integer)
    max_fps = Column(Integer)
    night_vision = Column(Boolean, default=False)
    weather_rating = Column(String(20))  # IP65, IP67, etc.
    
    # Installation details
    installation_date = Column(DateTime)
    last_maintenance = Column(DateTime)
    next_maintenance = Column(DateTime)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    status = Column(String(20), default="offline", index=True)  # 'online', 'offline', 'maintenance', 'error'
    
    # System metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    site = relationship("Site", back_populates="cameras")
    camera_configs = relationship("CameraConfig", back_populates="camera")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_camera_site', 'site_id'),
        Index('idx_camera_active', 'is_active'),
        Index('idx_camera_status', 'status'),
    )

class CameraConfig(MainBase):
    """Camera configuration settings and runtime parameters."""
    __tablename__ = "camera_configs"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    config_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    
    # Camera reference
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    
    # Configuration details
    name = Column(String(200), nullable=False)
    is_default = Column(Boolean, default=False, index=True)
    
    # Video settings
    width = Column(Integer, default=640)
    height = Column(Integer, default=480)
    fps = Column(Integer, default=30)
    bitrate = Column(Integer)
    quality = Column(String(20))  # 'low', 'medium', 'high'
    
    # Detection settings
    detection_enabled = Column(Boolean, default=True)
    detection_sensitivity = Column(Float, default=0.5)
    max_faces = Column(Integer, default=10)
    
    # Recording settings
    record_enabled = Column(Boolean, default=False)
    record_duration = Column(Integer)  # seconds
    record_quality = Column(String(20))
    
    # System metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(100))
    
    # Relationships
    camera = relationship("Camera", back_populates="camera_configs")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_config_camera', 'camera_id'),
        Index('idx_config_default', 'is_default'),
    )

class EmployeeSite(MainBase):
    """Employee assignments to specific sites."""
    __tablename__ = "employee_sites"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # References
    employee_id = Column(String(50), nullable=False, index=True)  # References auth.employees.employee_id
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=False)
    
    # Assignment details
    role = Column(String(100))  # 'employee', 'manager', 'admin', 'security'
    access_level = Column(String(20), default="standard")  # 'standard', 'elevated', 'admin'
    
    # Assignment period
    assigned_at = Column(DateTime, default=func.now())
    unassigned_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    
    # System metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    site = relationship("Site", back_populates="employees")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_emp_site_employee', 'employee_id'),
        Index('idx_emp_site_site', 'site_id'),
        Index('idx_emp_site_active', 'is_active'),
    )

class AlertRule(MainBase):
    """Alert rules and notifications configuration."""
    __tablename__ = "alert_rules"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    rule_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    
    # Rule details
    name = Column(String(200), nullable=False)
    description = Column(Text)
    is_enabled = Column(Boolean, default=True, index=True)
    
    # Trigger conditions
    event_type = Column(String(50), nullable=False, index=True)  # 'auth_failure', 'camera_offline', etc.
    threshold_value = Column(Float)
    time_window = Column(Integer)  # seconds
    
    # Notification settings
    notification_method = Column(String(50))  # 'email', 'sms', 'webhook'
    notification_recipients = Column(Text)  # JSON array of recipients
    message_template = Column(Text)
    
    # System metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(100))
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_alert_enabled', 'is_enabled'),
        Index('idx_alert_event_type', 'event_type'),
    )

class AlertLog(MainBase):
    """Alert notifications sent."""
    __tablename__ = "alert_logs"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    
    # Rule reference
    rule_id = Column(Integer, ForeignKey("alert_rules.id"), nullable=False)
    
    # Alert details
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)  # 'low', 'medium', 'high', 'critical'
    message = Column(Text, nullable=False)
    
    # Notification details
    notification_method = Column(String(50))
    recipient = Column(String(255))
    sent_at = Column(DateTime, default=func.now(), index=True)
    delivery_status = Column(String(20), default="pending")  # 'pending', 'sent', 'failed', 'delivered'
    
    # Additional data
    metadata_json = Column(Text)  # JSON string for additional data
    
    # Relationships
    rule = relationship("AlertRule")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_alert_log_rule', 'rule_id'),
        Index('idx_alert_log_sent', 'sent_at'),
        Index('idx_alert_log_status', 'delivery_status'),
    )

class SystemMetrics(MainBase):
    """System performance and health metrics."""
    __tablename__ = "system_metrics"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    metric_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    
    # Metric details
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))  # 'count', 'percentage', 'seconds', 'bytes'
    
    # Context
    component = Column(String(100), index=True)  # 'face_engine', 'camera_manager', etc.
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=True)
    camera_id = Column(String(100), nullable=True, index=True)
    
    # Timestamp
    timestamp = Column(DateTime, default=func.now(), index=True)
    
    # Additional metadata
    metadata_json = Column(Text)  # JSON string for additional context
    
    # Relationships
    site = relationship("Site")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_metrics_name', 'metric_name'),
        Index('idx_metrics_component', 'component'),
        Index('idx_metrics_timestamp', 'timestamp'),
    )
