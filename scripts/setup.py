#!/usr/bin/env python3
"""
OCCUR-CAM System Setup Script
Setup and initialization script for the OCCUR-CAM system.
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import config
from config.database import create_tables, test_connections
from database.migrations import create_all_tables, seed_initial_data
from core.face_engine import get_face_engine
from core.camera_manager import get_camera_manager
from core.auth_engine import get_auth_engine

def setup_logging():
    """Setup logging for setup script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('setup.log')
        ]
    )

def create_directories():
    """Create necessary directories."""
    try:
        logging.info("Creating necessary directories...")
        
        directories = [
            "storage",
            "storage/face_embeddings",
            "storage/reference_photos", 
            "storage/snapshots",
            "logs",
            "models",
            "temp"
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {directory}")
        
        logging.info("Directory creation completed")
        return True
        
    except Exception as e:
        logging.error(f"Error creating directories: {e}")
        return False

def setup_database():
    """Setup database tables and initial data."""
    try:
        logging.info("Setting up database...")
        
        # Test database connections
        if not test_connections():
            logging.error("Database connection test failed")
            return False
        
        # Create tables
        logging.info("Creating database tables...")
        if not create_all_tables():
            logging.error("Failed to create database tables")
            return False
        
        # Seed initial data
        logging.info("Seeding initial data...")
        if not seed_initial_data():
            logging.error("Failed to seed initial data")
            return False
        
        logging.info("Database setup completed")
        return True
        
    except Exception as e:
        logging.error(f"Error setting up database: {e}")
        return False

def test_face_engine():
    """Test face engine initialization."""
    try:
        logging.info("Testing face engine...")
        
        face_engine = get_face_engine()
        if not face_engine.is_initialized:
            logging.error("Face engine initialization failed")
            return False
        
        # Test with a simple operation
        stats = face_engine.get_engine_stats()
        logging.info(f"Face engine stats: {stats}")
        
        logging.info("Face engine test completed")
        return True
        
    except Exception as e:
        logging.error(f"Error testing face engine: {e}")
        return False

def test_camera_manager():
    """Test camera manager initialization."""
    try:
        logging.info("Testing camera manager...")
        
        camera_manager = get_camera_manager()
        if not camera_manager:
            logging.error("Camera manager initialization failed")
            return False
        
        # Test camera status
        status = camera_manager.get_all_cameras_status()
        logging.info(f"Camera manager status: {len(status)} cameras configured")
        
        logging.info("Camera manager test completed")
        return True
        
    except Exception as e:
        logging.error(f"Error testing camera manager: {e}")
        return False

def test_auth_engine():
    """Test authentication engine initialization."""
    try:
        logging.info("Testing authentication engine...")
        
        auth_engine = get_auth_engine()
        if not auth_engine:
            logging.error("Authentication engine initialization failed")
            return False
        
        # Test auth stats
        stats = auth_engine.get_authentication_stats()
        logging.info(f"Auth engine stats: {stats}")
        
        logging.info("Authentication engine test completed")
        return True
        
    except Exception as e:
        logging.error(f"Error testing authentication engine: {e}")
        return False

def create_config_files():
    """Create configuration files if they don't exist."""
    try:
        logging.info("Creating configuration files...")
        
        # Create .env file if it doesn't exist
        env_file = Path(".env")
        if not env_file.exists():
            env_example = Path("env.example")
            if env_example.exists():
                env_file.write_text(env_example.read_text())
                logging.info("Created .env file from template")
            else:
                logging.warning("No env.example file found")
        
        # Create camera config if it doesn't exist
        camera_config = Path("config/camera_config.yaml")
        if not camera_config.exists():
            logging.warning("Camera config file not found")
        
        logging.info("Configuration files check completed")
        return True
        
    except Exception as e:
        logging.error(f"Error creating config files: {e}")
        return False

def run_system_tests():
    """Run comprehensive system tests."""
    try:
        logging.info("Running system tests...")
        
        tests = [
            ("Database Connection", test_connections),
            ("Face Engine", test_face_engine),
            ("Camera Manager", test_camera_manager),
            ("Auth Engine", test_auth_engine)
        ]
        
        results = {}
        for test_name, test_func in tests:
            logging.info(f"Running test: {test_name}")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    logging.info(f"✓ {test_name} passed")
                else:
                    logging.error(f"✗ {test_name} failed")
            except Exception as e:
                logging.error(f"✗ {test_name} failed with error: {e}")
                results[test_name] = False
        
        # Summary
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        logging.info(f"System tests completed: {passed}/{total} passed")
        
        return passed == total
        
    except Exception as e:
        logging.error(f"Error running system tests: {e}")
        return False

def create_setup_report():
    """Create setup report."""
    try:
        logging.info("Creating setup report...")
        
        report = {
            "timestamp": time.time(),
            "setup_version": "1.0.0",
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": str(Path.cwd())
            },
            "configuration": {
                "database_url": config.database.DATABASE_URL,
                "auth_database_url": config.database.AUTH_DATABASE_URL,
                "face_model": config.face_recognition.MODEL_NAME,
                "camera_width": config.camera.WIDTH,
                "camera_height": config.camera.HEIGHT,
                "camera_fps": config.camera.FPS
            },
            "directories_created": [
                "storage",
                "storage/face_embeddings",
                "storage/reference_photos",
                "storage/snapshots",
                "logs",
                "models",
                "temp"
            ]
        }
        
        # Save report
        report_file = Path("setup_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logging.info(f"Setup report saved to: {report_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error creating setup report: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="OCCUR-CAM System Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                    # Full setup
  python setup.py --test-only        # Run tests only
  python setup.py --no-tests         # Setup without tests
  python setup.py --verbose          # Verbose output
        """
    )
    
    parser.add_argument(
        '--test-only', 
        action='store_true', 
        help='Run tests only without setup'
    )
    
    parser.add_argument(
        '--no-tests', 
        action='store_true', 
        help='Skip system tests'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--force', 
        action='store_true', 
        help='Force setup even if already configured'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('setup.log')
        ]
    )
    
    logging.info("=" * 60)
    logging.info("OCCUR-CAM System Setup")
    logging.info("=" * 60)
    
    try:
        if args.test_only:
            # Run tests only
            logging.info("Running system tests only...")
            success = run_system_tests()
            return 0 if success else 1
        
        # Full setup
        logging.info("Starting OCCUR-CAM system setup...")
        
        # Step 1: Create directories
        if not create_directories():
            logging.error("Failed to create directories")
            return 1
        
        # Step 2: Create config files
        if not create_config_files():
            logging.error("Failed to create config files")
            return 1
        
        # Step 3: Setup database
        if not setup_database():
            logging.error("Failed to setup database")
            return 1
        
        # Step 4: Run tests (if not skipped)
        if not args.no_tests:
            if not run_system_tests():
                logging.error("System tests failed")
                return 1
        
        # Step 5: Create setup report
        if not create_setup_report():
            logging.warning("Failed to create setup report")
        
        logging.info("=" * 60)
        logging.info("OCCUR-CAM system setup completed successfully!")
        logging.info("=" * 60)
        logging.info("Next steps:")
        logging.info("1. Configure cameras in config/camera_config.yaml")
        logging.info("2. Add employee data using the management interface")
        logging.info("3. Start the system with: python main.py")
        logging.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logging.error(f"Fatal error in setup: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
