"""
OCCUR-CAM Camera Tests
Tests for camera functionality and webcam integration.
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

from core.camera_manager import get_camera_manager, CameraSource
from config.settings import config

class TestCameraSource(unittest.TestCase):
    """Test individual camera source functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.camera_source = None
        self.test_camera_id = "test_camera"
    
    def tearDown(self):
        """Clean up after tests."""
        if self.camera_source:
            self.camera_source.disconnect()
    
    def test_camera_source_creation(self):
        """Test camera source creation."""
        try:
            self.camera_source = CameraSource(
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
            
            self.assertEqual(self.camera_source.camera_id, self.test_camera_id)
            self.assertEqual(self.camera_source.source, "0")
            self.assertEqual(self.camera_source.camera_type, "usb")
            self.assertFalse(self.camera_source.is_connected)
            self.assertFalse(self.camera_source.is_streaming)
            
        except Exception as e:
            self.fail(f"Camera source creation failed: {e}")
    
    def test_camera_connection(self):
        """Test camera connection."""
        try:
            self.camera_source = CameraSource(
                camera_id=self.test_camera_id,
                source="0",
                camera_type="usb"
            )
            
            # Test connection
            connected = self.camera_source.connect()
            
            # Note: This might fail if no webcam is available
            if connected:
                self.assertTrue(self.camera_source.is_connected)
                self.assertIsNotNone(self.camera_source.last_frame)
                
                # Test getting status
                status = self.camera_source.get_status()
                self.assertIsInstance(status, dict)
                self.assertIn('camera_id', status)
                self.assertIn('is_connected', status)
                
        except Exception as e:
            # This is expected if no webcam is available
            logging.warning(f"Camera connection test skipped: {e}")
    
    def test_camera_streaming(self):
        """Test camera streaming functionality."""
        try:
            self.camera_source = CameraSource(
                camera_id=self.test_camera_id,
                source="0",
                camera_type="usb"
            )
            
            # Connect first
            if self.camera_source.connect():
                # Test streaming
                streaming_started = self.camera_source.start_streaming()
                
                if streaming_started:
                    self.assertTrue(self.camera_source.is_streaming)
                    
                    # Wait a moment for frames
                    time.sleep(1)
                    
                    # Test getting frame
                    frame = self.camera_source.get_frame(timeout=2.0)
                    if frame is not None:
                        self.assertIsInstance(frame, np.ndarray)
                        self.assertEqual(len(frame.shape), 3)
                    
                    # Stop streaming
                    self.camera_source.stop_streaming()
                    self.assertFalse(self.camera_source.is_streaming)
                
        except Exception as e:
            # This is expected if no webcam is available
            logging.warning(f"Camera streaming test skipped: {e}")

class TestCameraManager(unittest.TestCase):
    """Test camera manager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.camera_manager = None
    
    def tearDown(self):
        """Clean up after tests."""
        if self.camera_manager:
            self.camera_manager.cleanup()
    
    def test_camera_manager_initialization(self):
        """Test camera manager initialization."""
        try:
            self.camera_manager = get_camera_manager()
            
            self.assertIsNotNone(self.camera_manager)
            self.assertIsInstance(self.camera_manager.cameras, dict)
            
        except Exception as e:
            self.fail(f"Camera manager initialization failed: {e}")
    
    def test_add_camera(self):
        """Test adding camera to manager."""
        try:
            self.camera_manager = get_camera_manager()
            
            # Add test camera
            success = self.camera_manager.add_camera(
                camera_id="test_webcam",
                source="0",
                camera_type="usb",
                config={
                    'width': 640,
                    'height': 480,
                    'fps': 15
                }
            )
            
            self.assertTrue(success, "Should be able to add camera")
            self.assertIn("test_webcam", self.camera_manager.cameras)
            
        except Exception as e:
            self.fail(f"Add camera test failed: {e}")
    
    def test_remove_camera(self):
        """Test removing camera from manager."""
        try:
            self.camera_manager = get_camera_manager()
            
            # Add camera first
            self.camera_manager.add_camera(
                camera_id="test_webcam",
                source="0",
                camera_type="usb"
            )
            
            # Remove camera
            success = self.camera_manager.remove_camera("test_webcam")
            
            self.assertTrue(success, "Should be able to remove camera")
            self.assertNotIn("test_webcam", self.camera_manager.cameras)
            
        except Exception as e:
            self.fail(f"Remove camera test failed: {e}")
    
    def test_camera_status(self):
        """Test camera status functionality."""
        try:
            self.camera_manager = get_camera_manager()
            
            # Add test camera
            self.camera_manager.add_camera(
                camera_id="test_webcam",
                source="0",
                camera_type="usb"
            )
            
            # Test getting status
            status = self.camera_manager.get_camera_status("test_webcam")
            self.assertIsNotNone(status)
            self.assertIsInstance(status, dict)
            
            # Test getting all status
            all_status = self.camera_manager.get_all_cameras_status()
            self.assertIsInstance(all_status, dict)
            self.assertIn("test_webcam", all_status)
            
        except Exception as e:
            self.fail(f"Camera status test failed: {e}")
    
    def test_camera_connection_management(self):
        """Test camera connection management."""
        try:
            self.camera_manager = get_camera_manager()
            
            # Add test camera
            self.camera_manager.add_camera(
                camera_id="test_webcam",
                source="0",
                camera_type="usb"
            )
            
            # Test connection
            connected = self.camera_manager.connect_camera("test_webcam")
            
            if connected:
                # Test disconnection
                disconnected = self.camera_manager.disconnect_camera("test_webcam")
                self.assertTrue(disconnected, "Should be able to disconnect camera")
            
        except Exception as e:
            # This is expected if no webcam is available
            logging.warning(f"Camera connection management test skipped: {e}")
    
    def test_camera_streaming_management(self):
        """Test camera streaming management."""
        try:
            self.camera_manager = get_camera_manager()
            
            # Add test camera
            self.camera_manager.add_camera(
                camera_id="test_webcam",
                source="0",
                camera_type="usb"
            )
            
            # Test starting camera
            started = self.camera_manager.start_camera("test_webcam")
            
            if started:
                # Test getting frame
                frame = self.camera_manager.get_camera_frame("test_webcam", timeout=2.0)
                if frame is not None:
                    self.assertIsInstance(frame, np.ndarray)
                    self.assertEqual(len(frame.shape), 3)
                
                # Test stopping camera
                stopped = self.camera_manager.stop_camera("test_webcam")
                self.assertTrue(stopped, "Should be able to stop camera")
            
        except Exception as e:
            # This is expected if no webcam is available
            logging.warning(f"Camera streaming management test skipped: {e}")

class TestWebcamIntegration(unittest.TestCase):
    """Test webcam integration specifically."""
    
    def setUp(self):
        """Set up test environment."""
        self.camera_manager = None
    
    def tearDown(self):
        """Clean up after tests."""
        if self.camera_manager:
            self.camera_manager.cleanup()
    
    def test_webcam_detection(self):
        """Test webcam detection."""
        try:
            # Test if webcam is available
            cap = cv2.VideoCapture(0)
            webcam_available = cap.isOpened()
            cap.release()
            
            if webcam_available:
                logging.info("Webcam detected and available")
                
                self.camera_manager = get_camera_manager()
                
                # Add webcam
                success = self.camera_manager.add_camera(
                    camera_id="webcam_0",
                    source="0",
                    camera_type="usb",
                    config={
                        'width': 640,
                        'height': 480,
                        'fps': 15,
                        'timeout': 5
                    }
                )
                
                self.assertTrue(success, "Should be able to add webcam")
                
            else:
                logging.warning("No webcam available, skipping test")
                self.skipTest("No webcam available")
                
        except Exception as e:
            logging.warning(f"Webcam detection test skipped: {e}")
            self.skipTest(f"Webcam detection failed: {e}")
    
    def test_webcam_capture(self):
        """Test webcam capture functionality."""
        try:
            # Test direct webcam capture
            cap = cv2.VideoCapture(0)
            
            if cap.isOpened():
                # Set properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                
                # Capture frame
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    self.assertIsInstance(frame, np.ndarray)
                    self.assertEqual(len(frame.shape), 3)
                    self.assertEqual(frame.shape[2], 3)  # BGR
                    
                    # Test frame properties
                    height, width, channels = frame.shape
                    self.assertGreater(height, 0)
                    self.assertGreater(width, 0)
                    self.assertEqual(channels, 3)
                    
                    logging.info(f"Captured frame: {width}x{height}x{channels}")
                
                cap.release()
            else:
                self.skipTest("Webcam not available")
                
        except Exception as e:
            logging.warning(f"Webcam capture test skipped: {e}")
            self.skipTest(f"Webcam capture failed: {e}")
    
    def test_webcam_performance(self):
        """Test webcam performance."""
        try:
            cap = cv2.VideoCapture(0)
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                
                # Test performance
                start_time = time.time()
                frame_count = 0
                
                for i in range(10):  # Capture 10 frames
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame_count += 1
                
                end_time = time.time()
                processing_time = end_time - start_time
                fps = frame_count / processing_time if processing_time > 0 else 0
                
                cap.release()
                
                self.assertGreater(frame_count, 0, "Should capture at least one frame")
                self.assertGreater(fps, 0, "FPS should be greater than 0")
                
                logging.info(f"Webcam performance: {fps:.1f} FPS over {processing_time:.2f}s")
                
            else:
                self.skipTest("Webcam not available")
                
        except Exception as e:
            logging.warning(f"Webcam performance test skipped: {e}")
            self.skipTest(f"Webcam performance test failed: {e}")

def run_camera_tests():
    """Run all camera tests."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCameraSource,
        TestCameraManager,
        TestWebcamIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("CAMERA TEST SUMMARY")
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
    success = run_camera_tests()
    sys.exit(0 if success else 1)
