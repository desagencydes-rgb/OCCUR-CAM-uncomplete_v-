"""
OCCUR-CAM Database Seed Data
Initial data seeding for development and testing.
"""

import logging
from datetime import datetime, timedelta
from config.database import get_auth_db, get_main_db
from database.schemas.auth_schemas import Employee, CameraConfig as AuthCameraConfig, SystemLog
from database.schemas.main_schemas import Site, Camera, CameraConfig as MainCameraConfig, AlertRule

def seed_initial_data():
    """Seed the database with initial data for development."""
    try:
        logging.info("Starting database seeding...")
        
        # Seed auth database
        if not seed_auth_database():
            return False
        
        # Seed main database
        if not seed_main_database():
            return False
        
        logging.info("Database seeding completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error seeding database: {e}")
        return False

def seed_auth_database():
    """Seed authentication database with initial data."""
    try:
        with get_auth_db() as db:
            # Check if data already exists
            if db.query(Employee).count() > 0:
                logging.info("Auth database already has data, skipping seed")
                return True
            
            # Create sample employees
            sample_employees = [
                {
                    "employee_id": "EMP001",
                    "first_name": "John",
                    "last_name": "Doe",
                    "email": "john.doe@company.com",
                    "phone": "+1-555-0101",
                    "department": "Engineering",
                    "position": "Software Engineer",
                    "is_active": True,
                    "hire_date": datetime.now() - timedelta(days=365),
                    "face_quality_score": 0.95,
                    "created_by": "system"
                },
                {
                    "employee_id": "EMP002",
                    "first_name": "Jane",
                    "last_name": "Smith",
                    "email": "jane.smith@company.com",
                    "phone": "+1-555-0102",
                    "department": "HR",
                    "position": "HR Manager",
                    "is_active": True,
                    "hire_date": datetime.now() - timedelta(days=200),
                    "face_quality_score": 0.92,
                    "created_by": "system"
                },
                {
                    "employee_id": "EMP003",
                    "first_name": "Bob",
                    "last_name": "Johnson",
                    "email": "bob.johnson@company.com",
                    "phone": "+1-555-0103",
                    "department": "Security",
                    "position": "Security Officer",
                    "is_active": True,
                    "hire_date": datetime.now() - timedelta(days=100),
                    "face_quality_score": 0.88,
                    "created_by": "system"
                }
            ]
            
            for emp_data in sample_employees:
                employee = Employee(**emp_data)
                db.add(employee)
            
            # Create sample camera configs
            sample_cameras = [
                {
                    "camera_id": "CAM001",
                    "name": "Main Entrance USB Camera",
                    "source": "0",
                    "camera_type": "usb",
                    "location": "Main Entrance",
                    "description": "Primary USB camera for main entrance",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "timeout": 5,
                    "is_enabled": True,
                    "is_online": True,
                    "last_seen": datetime.now(),
                    "health_score": 0.95
                },
                {
                    "camera_id": "CAM002",
                    "name": "IVCam Mobile",
                    "source": "http://192.168.1.100:8080/video",
                    "camera_type": "ivcam",
                    "location": "Mobile Device",
                    "description": "IVCam mobile app camera feed",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "timeout": 10,
                    "is_enabled": True,
                    "is_online": False,
                    "health_score": 0.0
                }
            ]
            
            for cam_data in sample_cameras:
                camera = AuthCameraConfig(**cam_data)
                db.add(camera)
            
            # Create initial system log
            system_log = SystemLog(
                level="INFO",
                component="database",
                message="Database initialized with seed data",
                timestamp=datetime.now()
            )
            db.add(system_log)
            
            db.commit()
            logging.info("Auth database seeded successfully")
            return True
            
    except Exception as e:
        logging.error(f"Error seeding auth database: {e}")
        return False

def seed_main_database():
    """Seed main database with initial data."""
    try:
        with get_main_db() as db:
            # Check if data already exists
            if db.query(Site).count() > 0:
                logging.info("Main database already has data, skipping seed")
                return True
            
            # Create sample site
            site = Site(
                site_id="SITE001",
                name="Main Office",
                address="123 Business St",
                city="New York",
                state="NY",
                country="USA",
                postal_code="10001",
                contact_person="John Manager",
                contact_email="manager@company.com",
                contact_phone="+1-555-0001",
                timezone="America/New_York",
                is_active=True
            )
            db.add(site)
            db.flush()  # Get the site ID
            
            # Create sample cameras
            cameras = [
                {
                    "camera_id": "CAM001",
                    "site_id": site.id,
                    "name": "Main Entrance Camera",
                    "model": "USB-WebCam-Pro",
                    "manufacturer": "Generic",
                    "serial_number": "USB001",
                    "resolution_width": 640,
                    "resolution_height": 480,
                    "max_fps": 30,
                    "night_vision": False,
                    "weather_rating": "IP54",
                    "installation_date": datetime.now() - timedelta(days=30),
                    "is_active": True,
                    "status": "online"
                },
                {
                    "camera_id": "CAM002",
                    "site_id": site.id,
                    "name": "IVCam Mobile",
                    "model": "Mobile App",
                    "manufacturer": "IVCam",
                    "serial_number": "MOBILE001",
                    "resolution_width": 640,
                    "resolution_height": 480,
                    "max_fps": 30,
                    "night_vision": True,
                    "weather_rating": "IP65",
                    "installation_date": datetime.now() - timedelta(days=7),
                    "is_active": True,
                    "status": "offline"
                }
            ]
            
            for cam_data in cameras:
                camera = Camera(**cam_data)
                db.add(camera)
                db.flush()  # Get the camera ID
                
                # Create default camera config
                camera_config = MainCameraConfig(
                    camera_id=camera.id,
                    name="Default Configuration",
                    is_default=True,
                    width=640,
                    height=480,
                    fps=30,
                    quality="medium",
                    detection_enabled=True,
                    detection_sensitivity=0.5,
                    max_faces=10,
                    record_enabled=False,
                    created_by="system"
                )
                db.add(camera_config)
            
            # Create sample alert rules
            alert_rules = [
                {
                    "name": "Authentication Failure Alert",
                    "description": "Alert when multiple authentication failures occur",
                    "is_enabled": True,
                    "event_type": "auth_failure",
                    "threshold_value": 3.0,
                    "time_window": 300,  # 5 minutes
                    "notification_method": "email",
                    "notification_recipients": '["admin@company.com", "security@company.com"]',
                    "message_template": "Multiple authentication failures detected: {count} failures in {time_window} seconds",
                    "created_by": "system"
                },
                {
                    "name": "Camera Offline Alert",
                    "description": "Alert when camera goes offline",
                    "is_enabled": True,
                    "event_type": "camera_offline",
                    "threshold_value": 1.0,
                    "time_window": 60,  # 1 minute
                    "notification_method": "email",
                    "notification_recipients": '["admin@company.com"]',
                    "message_template": "Camera {camera_name} is offline",
                    "created_by": "system"
                }
            ]
            
            for rule_data in alert_rules:
                alert_rule = AlertRule(**rule_data)
                db.add(alert_rule)
            
            db.commit()
            logging.info("Main database seeded successfully")
            return True
            
    except Exception as e:
        logging.error(f"Error seeding main database: {e}")
        return False

def clear_seed_data():
    """Clear all seed data from the database."""
    try:
        logging.warning("Clearing seed data from database...")
        
        with get_auth_db() as db:
            # Clear auth database
            db.query(SystemLog).delete()
            db.query(AuthCameraConfig).delete()
            db.query(Employee).delete()
            db.commit()
        
        with get_main_db() as db:
            # Clear main database
            db.query(AlertRule).delete()
            db.query(MainCameraConfig).delete()
            db.query(Camera).delete()
            db.query(Site).delete()
            db.commit()
        
        logging.info("Seed data cleared successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error clearing seed data: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "seed":
            success = seed_initial_data()
            sys.exit(0 if success else 1)
        elif command == "clear":
            success = clear_seed_data()
            sys.exit(0 if success else 1)
        else:
            print("Usage: python seed_data.py [seed|clear]")
            sys.exit(1)
    else:
        # Default: seed data
        success = seed_initial_data()
        sys.exit(0 if success else 1)
