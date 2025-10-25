"""
OCCUR-CAM Configuration Settings
Central configuration management for the OCCUR-CAM system.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
STORAGE_DIR = BASE_DIR / "storage"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
STORAGE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

class DatabaseConfig:
    """Database configuration settings."""
    
    # Database URLs
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{STORAGE_DIR}/occur_cam.db")
    AUTH_DATABASE_URL = os.getenv("AUTH_DATABASE_URL", f"sqlite:///{STORAGE_DIR}/auth.db")
    
    # Connection settings
    POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
    MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))

class FaceRecognitionConfig:
    """Face recognition configuration settings."""
    
    # Model settings - Use smaller model for CPU
    MODEL_NAME = os.getenv("FACE_RECOGNITION_MODEL", "buffalo_s")  # Smaller model for CPU
    MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.onnx"
    
    # Detection thresholds - Lower for better detection on CPU
    DETECTION_THRESHOLD = float(os.getenv("FACE_DETECTION_THRESHOLD", "0.3"))
    RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.6"))
    
    # Processing settings - Optimized for CPU
    BATCH_SIZE = int(os.getenv("FACE_DETECTION_BATCH_SIZE", "1"))  # Process one at a time
    MAX_FACES_PER_FRAME = int(os.getenv("MAX_FACES_PER_FRAME", "5"))  # Limit for CPU
    
    # Image processing
    FACE_SIZE = (112, 112)  # Standard face size for recognition
    EMBEDDING_SIZE = 512  # Face embedding vector size

class CameraConfig:
    """Camera configuration settings."""
    
    # Default camera settings - Optimized for webcam and CPU
    DEFAULT_SOURCE = os.getenv("DEFAULT_CAMERA_SOURCE", "0")  # Default webcam
    WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))  # Lower resolution for CPU
    HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
    FPS = int(os.getenv("CAMERA_FPS", "15"))  # Lower FPS for CPU
    TIMEOUT = int(os.getenv("CAMERA_TIMEOUT", "5"))
    
    # IVCam integration
    IVCAM_ENABLED = os.getenv("IVCAM_ENABLED", "true").lower() == "true"
    IVCAM_IP = os.getenv("IVCAM_IP", "192.168.1.100")
    IVCAM_PORT = int(os.getenv("IVCAM_PORT", "8080"))
    IVCAM_URL = f"http://{IVCAM_IP}:{IVCAM_PORT}/video"
    
    # Camera health monitoring
    HEALTH_CHECK_INTERVAL = 30  # seconds
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 5  # seconds

class AuthenticationConfig:
    """Authentication configuration settings."""
    
    # Timeout settings
    AUTH_TIMEOUT = int(os.getenv("AUTH_TIMEOUT", "30"))
    MAX_ATTEMPTS = int(os.getenv("MAX_AUTH_ATTEMPTS", "3"))
    COOLDOWN_PERIOD = int(os.getenv("AUTH_COOLDOWN", "60"))
    
    # Security settings
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "default_key_change_in_production")
    JWT_SECRET = os.getenv("JWT_SECRET", "default_jwt_secret_change_in_production")
    
    # Session settings
    SESSION_DURATION = 3600  # 1 hour
    REFRESH_THRESHOLD = 300  # 5 minutes before expiry

class LoggingConfig:
    """Logging configuration settings."""
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "occur_cam.log"))
    LOG_ROTATION = os.getenv("LOG_ROTATION", "1 day")
    LOG_RETENTION = os.getenv("LOG_RETENTION", "30 days")
    
    # Log formats
    CONSOLE_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"

class SystemConfig:
    """System configuration settings."""
    
    # System limits
    MAX_EMPLOYEES = int(os.getenv("MAX_EMPLOYEES", "10000"))
    MAX_CAMERAS = int(os.getenv("MAX_CAMERAS", "50"))
    PROCESSING_THREADS = int(os.getenv("PROCESSING_THREADS", "4"))
    
    # Performance settings
    FRAME_BUFFER_SIZE = 10
    PROCESSING_QUEUE_SIZE = 100
    MAX_CONCURRENT_AUTH = 5
    
    # Storage settings
    MAX_SNAPSHOT_SIZE = 5 * 1024 * 1024  # 5MB
    SNAPSHOT_RETENTION_DAYS = 30
    EMBEDDING_RETENTION_DAYS = 365

class Config:
    """Main configuration class combining all settings."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.face_recognition = FaceRecognitionConfig()
        self.camera = CameraConfig()
        self.authentication = AuthenticationConfig()
        self.logging = LoggingConfig()
        self.system = SystemConfig()
    
    def load_camera_config(self, config_file: Optional[str] = None) -> dict:
        """Load camera configuration from YAML file."""
        if config_file is None:
            config_file = BASE_DIR / "config" / "camera_config.yaml"
        
        if not os.path.exists(config_file):
            return {}
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def save_camera_config(self, config: dict, config_file: Optional[str] = None):
        """Save camera configuration to YAML file."""
        if config_file is None:
            config_file = BASE_DIR / "config" / "camera_config.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

# Global configuration instance
config = Config()
