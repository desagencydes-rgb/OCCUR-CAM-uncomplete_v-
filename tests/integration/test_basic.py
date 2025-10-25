#!/usr/bin/env python3
"""
OCCUR-CAM Basic Test
Test basic functionality without InsightFace dependencies.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports without AI dependencies."""
    print("Testing basic imports...")
    
    try:
        # Test config imports
        from config.settings import config
        print("✅ Config import: OK")
        
        # Test database imports
        from config.database import auth_engine, main_engine
        print("✅ Database import: OK")
        
        # Test models
        from models.employee_models import EmployeeProfile
        from models.face_models import FaceDetection
        print("✅ Models import: OK")
        
        # Test core utilities
        from core.utils import setup_logging
        print("✅ Utils import: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_database_creation():
    """Test database creation."""
    print("\nTesting database creation...")
    
    try:
        from config.database import create_tables, test_connections
        
        # Test connections
        if test_connections():
            print("✅ Database connections: OK")
        else:
            print("❌ Database connections: FAILED")
            return False
        
        # Test table creation
        if create_tables():
            print("✅ Database tables: OK")
        else:
            print("❌ Database tables: FAILED")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_camera_detection():
    """Test camera detection."""
    print("\nTesting camera detection...")
    
    try:
        import cv2
        
        # Test camera availability
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Webcam detected: OK")
            cap.release()
            return True
        else:
            print("⚠️  No webcam detected (this is OK for testing)")
            cap.release()
            return True
            
    except Exception as e:
        print(f"❌ Camera test error: {e}")
        return False

def main():
    """Run basic tests."""
    print("=" * 60)
    print("OCCUR-CAM Basic System Test")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Database Creation", test_database_creation),
        ("Camera Detection", test_camera_detection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! System is ready for AI setup.")
        print("\nNext steps:")
        print("1. Install InsightFace: pip install insightface")
        print("2. Run full tests: python -m tests")
        print("3. Start application: python main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
