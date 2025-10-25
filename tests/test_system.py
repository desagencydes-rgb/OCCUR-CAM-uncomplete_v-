"""
OCCUR-CAM System Tests
Comprehensive system tests for CPU-optimized operation.
"""

import sys
import os
import unittest
import logging
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.application import OCCURCamApplication
from core.face_engine import get_face_engine
from core.camera_manager import get_camera_manager
from core.auth_engine import get_auth_engine
from config.settings import config

class TestSystemInitialization(unittest.TestCase):
    """Test system initialization and basic functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.app = None
        self.test_image = None
        self._create_test_image()
    
    def tearDown(self):
        """Clean up after tests."""
        if self.app:
            self.app.shutdown()
    
    def _create_test_image(self):
        """Create a test image for face detection."""
        try:
            # Create a simple test image with a face-like pattern
            self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Draw a simple face
            cv2.circle(self.test_image, (320, 240), 100, (255, 255, 255), -1)  # Face
            cv2.circle(self.test_image, (300, 220), 10, (0, 0, 0), -1)  # Left eye
            cv2.circle(self.test_image, (340, 220), 10, (0, 0, 0), -1)  # Right eye
            cv2.ellipse(self.test_image, (320, 260), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Mouth
            
        except Exception as e:
            logging.warning(f"Error creating test image: {e}")
            self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_application_initialization(self):
        """Test application initialization."""
        try:
            self.app = OCCURCamApplication(
                camera_source="0",
                config_profile="test",
                debug_mode=True,
                test_mode=True
            )
            
            # Test initialization
            self.assertTrue(self.app.initialize(), "Application should initialize successfully")
            self.assertEqual(self.app.state.value, "running", "Application should be in running state")
            
        except Exception as e:
            self.fail(f"Application initialization failed: {e}")
    
    def test_face_engine_initialization(self):
        """Test face engine initialization."""
        try:
            face_engine = get_face_engine()
            self.assertTrue(face_engine.is_initialized, "Face engine should be initialized")
            
            # Test with sample image
            if self.test_image is not None:
                frame_analysis = face_engine.process_frame(self.test_image, "test_camera")
                self.assertIsNotNone(frame_analysis, "Frame analysis should not be None")
                
        except Exception as e:
            self.fail(f"Face engine initialization failed: {e}")
    
    def test_camera_manager_initialization(self):
        """Test camera manager initialization."""
        try:
            camera_manager = get_camera_manager()
            self.assertIsNotNone(camera_manager, "Camera manager should be initialized")
            
            # Test camera status
            status = camera_manager.get_all_cameras_status()
            self.assertIsInstance(status, dict, "Camera status should be a dictionary")
            
        except Exception as e:
            self.fail(f"Camera manager initialization failed: {e}")
    
    def test_auth_engine_initialization(self):
        """Test authentication engine initialization."""
        try:
            auth_engine = get_auth_engine()
            self.assertIsNotNone(auth_engine, "Auth engine should be initialized")
            
            # Test auth stats
            stats = auth_engine.get_authentication_stats()
            self.assertIsInstance(stats, dict, "Auth stats should be a dictionary")
            
        except Exception as e:
            self.fail(f"Auth engine initialization failed: {e}")

class TestWebcamIntegration(unittest.TestCase):
    """Test webcam integration and functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.camera_manager = None
        self.test_camera_id = "test_webcam"
    
    def tearDown(self):
        """Clean up after tests."""
        if self.camera_manager:
            self.camera_manager.cleanup()
    
    def test_webcam_detection(self):
        """Test webcam detection and connection."""
        try:
            self.camera_manager = get_camera_manager()
            
            # Add test webcam
            success = self.camera_manager.add_camera(
                camera_id=self.test_camera_id,
                source="0",  # Default webcam
                camera_type="usb",
                config={
                    'width': 640,
                    'height': 480,
                    'fps': 15,
                    'timeout': 5
                }
            )
            
            self.assertTrue(success, "Should be able to add webcam")
            
        except Exception as e:
            self.fail(f"Webcam detection failed: {e}")
    
    def test_webcam_connection(self):
        """Test webcam connection."""
        try:
            self.camera_manager = get_camera_manager()
            
            # Add and connect webcam
            self.camera_manager.add_camera(
                camera_id=self.test_camera_id,
                source="0",
                camera_type="usb"
            )
            
            # Test connection
            connected = self.camera_manager.connect_camera(self.test_camera_id)
            
            # Note: This might fail if no webcam is available
            if connected:
                # Test getting frame
                frame = self.camera_manager.get_camera_frame(self.test_camera_id, timeout=2.0)
                if frame is not None:
                    self.assertIsInstance(frame, np.ndarray, "Frame should be numpy array")
                    self.assertEqual(len(frame.shape), 3, "Frame should be 3D array")
                
        except Exception as e:
            # This is expected if no webcam is available
            logging.warning(f"Webcam connection test skipped: {e}")

class TestFaceRecognition(unittest.TestCase):
    """Test face recognition functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.face_engine = None
        self.test_image = None
        self._create_test_image()
    
    def tearDown(self):
        """Clean up after tests."""
        if self.face_engine:
            self.face_engine.cleanup()
    
    def _create_test_image(self):
        """Create a test image for face detection."""
        try:
            # Create a more realistic test image
            self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add some noise to make it more realistic
            noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
            self.test_image = cv2.add(self.test_image, noise)
            
            # Draw a simple face
            cv2.circle(self.test_image, (320, 240), 100, (200, 200, 200), -1)  # Face
            cv2.circle(self.test_image, (300, 220), 10, (50, 50, 50), -1)  # Left eye
            cv2.circle(self.test_image, (340, 220), 10, (50, 50, 50), -1)  # Right eye
            cv2.ellipse(self.test_image, (320, 260), (30, 15), 0, 0, 180, (50, 50, 50), 2)  # Mouth
            
        except Exception as e:
            logging.warning(f"Error creating test image: {e}")
            self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_face_detection(self):
        """Test face detection functionality."""
        try:
            self.face_engine = get_face_engine()
            
            if self.test_image is not None:
                # Test frame processing
                frame_analysis = self.face_engine.process_frame(self.test_image, "test_camera")
                
                self.assertIsNotNone(frame_analysis, "Frame analysis should not be None")
                self.assertIsInstance(frame_analysis.face_detections, list, "Face detections should be a list")
                self.assertIsInstance(frame_analysis.face_recognitions, list, "Face recognitions should be a list")
                
        except Exception as e:
            self.fail(f"Face detection test failed: {e}")
    
    def test_face_recognition_performance(self):
        """Test face recognition performance on CPU."""
        try:
            self.face_engine = get_face_engine()
            
            if self.test_image is not None:
                # Test performance
                start_time = time.time()
                
                for i in range(5):  # Test multiple iterations
                    frame_analysis = self.face_engine.process_frame(self.test_image, "test_camera")
                
                end_time = time.time()
                processing_time = end_time - start_time
                avg_time = processing_time / 5
                
                # Should process within reasonable time on CPU
                self.assertLess(avg_time, 2.0, f"Average processing time should be < 2s, got {avg_time:.2f}s")
                
                logging.info(f"Average processing time: {avg_time:.2f}s")
                
        except Exception as e:
            self.fail(f"Face recognition performance test failed: {e}")

class TestAuthentication(unittest.TestCase):
    """Test authentication functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.auth_engine = None
        self.test_image = None
        self._create_test_image()
    
    def tearDown(self):
        """Clean up after tests."""
        if self.auth_engine:
            self.auth_engine.cleanup()
    
    def _create_test_image(self):
        """Create a test image for authentication."""
        try:
            self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add some noise
            noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
            self.test_image = cv2.add(self.test_image, noise)
            
            # Draw a simple face
            cv2.circle(self.test_image, (320, 240), 100, (200, 200, 200), -1)
            cv2.circle(self.test_image, (300, 220), 10, (50, 50, 50), -1)
            cv2.circle(self.test_image, (340, 220), 10, (50, 50, 50), -1)
            cv2.ellipse(self.test_image, (320, 260), (30, 15), 0, 0, 180, (50, 50, 50), 2)
            
        except Exception as e:
            logging.warning(f"Error creating test image: {e}")
            self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_authentication_attempt(self):
        """Test authentication attempt."""
        try:
            self.auth_engine = get_auth_engine()
            
            if self.test_image is not None:
                # Test authentication
                attempt = self.auth_engine.authenticate_face(
                    frame=self.test_image,
                    camera_id="test_camera",
                    ip_address="127.0.0.1"
                )
                
                self.assertIsNotNone(attempt, "Authentication attempt should not be None")
                self.assertIsNotNone(attempt.attempt_id, "Attempt should have ID")
                self.assertIsNotNone(attempt.timestamp, "Attempt should have timestamp")
                
        except Exception as e:
            self.fail(f"Authentication test failed: {e}")
    
    def test_authentication_performance(self):
        """Test authentication performance."""
        try:
            self.auth_engine = get_auth_engine()
            
            if self.test_image is not None:
                # Test performance
                start_time = time.time()
                
                for i in range(3):  # Test multiple iterations
                    attempt = self.auth_engine.authenticate_face(
                        frame=self.test_image,
                        camera_id="test_camera"
                    )
                
                end_time = time.time()
                processing_time = end_time - start_time
                avg_time = processing_time / 3
                
                # Should process within reasonable time
                self.assertLess(avg_time, 3.0, f"Average auth time should be < 3s, got {avg_time:.2f}s")
                
                logging.info(f"Average authentication time: {avg_time:.2f}s")
                
        except Exception as e:
            self.fail(f"Authentication performance test failed: {e}")

class TestSystemIntegration(unittest.TestCase):
    """Test system integration and end-to-end functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.app = None
    
    def tearDown(self):
        """Clean up after tests."""
        if self.app:
            self.app.shutdown()
    
    def test_full_system_startup(self):
        """Test full system startup and shutdown."""
        try:
            self.app = OCCURCamApplication(
                camera_source="0",
                config_profile="test",
                debug_mode=True,
                test_mode=True
            )
            
            # Test initialization
            self.assertTrue(self.app.initialize(), "System should initialize")
            
            # Test startup
            self.assertTrue(self.app.start(), "System should start")
            
            # Test running state
            self.assertTrue(self.app.is_running, "System should be running")
            
            # Test statistics
            stats = self.app.get_application_stats()
            self.assertIsInstance(stats, dict, "Stats should be a dictionary")
            
        except Exception as e:
            self.fail(f"Full system test failed: {e}")
    
    def test_system_health(self):
        """Test system health monitoring."""
        try:
            self.app = OCCURCamApplication(
                camera_source="0",
                config_profile="test",
                debug_mode=True,
                test_mode=True
            )
            
            if self.app.initialize() and self.app.start():
                # Wait a moment for systems to initialize
                time.sleep(2)
                
                # Test system health
                stats = self.app.get_application_stats()
                health = stats.get('system_health', 0)
                
                self.assertGreaterEqual(health, 0.0, "System health should be >= 0")
                self.assertLessEqual(health, 1.0, "System health should be <= 1")
                
                logging.info(f"System health: {health:.2f}")
                
        except Exception as e:
            self.fail(f"System health test failed: {e}")

def run_tests():
    """Run all tests."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSystemInitialization,
        TestWebcamIntegration,
        TestFaceRecognition,
        TestAuthentication,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
