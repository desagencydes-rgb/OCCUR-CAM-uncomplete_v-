"""
OCCUR-CAM Face Recognition Tests
Tests for face recognition functionality with CPU optimization.
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

from core.face_engine import get_face_engine
from core.face_detector import get_face_detector
from core.face_recognizer import get_face_recognizer
from core.lighting_optimizer import get_lighting_optimizer
from core.face_quality_enhancer import get_face_quality_enhancer
from config.settings import config

class TestFaceDetection(unittest.TestCase):
    """Test face detection functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.face_detector = None
        self.test_images = []
        self._create_test_images()
    
    def tearDown(self):
        """Clean up after tests."""
        if self.face_detector:
            self.face_detector.cleanup()
    
    def _create_test_images(self):
        """Create test images for face detection."""
        try:
            # Create simple test images
            for i in range(3):
                # Create base image
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Add noise
                noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
                img = cv2.add(img, noise)
                
                # Draw face-like pattern
                center_x = 320 + (i - 1) * 100
                center_y = 240
                
                # Face circle
                cv2.circle(img, (center_x, center_y), 80, (200, 200, 200), -1)
                
                # Eyes
                cv2.circle(img, (center_x - 30, center_y - 20), 8, (50, 50, 50), -1)
                cv2.circle(img, (center_x + 30, center_y - 20), 8, (50, 50, 50), -1)
                
                # Nose
                cv2.circle(img, (center_x, center_y), 5, (150, 150, 150), -1)
                
                # Mouth
                cv2.ellipse(img, (center_x, center_y + 20), (25, 10), 0, 0, 180, (50, 50, 50), 2)
                
                self.test_images.append(img)
                
        except Exception as e:
            logging.warning(f"Error creating test images: {e}")
            # Create fallback images
            for i in range(3):
                self.test_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
    
    def test_face_detector_initialization(self):
        """Test face detector initialization."""
        try:
            self.face_detector = get_face_detector()
            
            self.assertIsNotNone(self.face_detector)
            self.assertTrue(self.face_detector.is_initialized)
            
            # Test detector info
            info = self.face_detector.get_detector_info()
            self.assertIsInstance(info, dict)
            self.assertIn('is_initialized', info)
            self.assertIn('model_name', info)
            
        except Exception as e:
            self.fail(f"Face detector initialization failed: {e}")
    
    def test_face_detection(self):
        """Test face detection on test images."""
        try:
            self.face_detector = get_face_detector()
            
            for i, test_image in enumerate(self.test_images):
                detections = self.face_detector.detect_faces(test_image)
                
                self.assertIsInstance(detections, list)
                logging.info(f"Image {i}: Detected {len(detections)} faces")
                
                # Each detection should have required attributes
                for detection in detections:
                    self.assertIsInstance(detection.bbox, tuple)
                    self.assertEqual(len(detection.bbox), 4)
                    self.assertIsInstance(detection.confidence, float)
                    self.assertGreaterEqual(detection.confidence, 0.0)
                    self.assertLessEqual(detection.confidence, 1.0)
                
        except Exception as e:
            self.fail(f"Face detection test failed: {e}")
    
    def test_face_detection_performance(self):
        """Test face detection performance on CPU."""
        try:
            self.face_detector = get_face_detector()
            
            if self.test_images:
                # Test performance
                start_time = time.time()
                
                for i in range(5):  # Test multiple iterations
                    for test_image in self.test_images:
                        detections = self.face_detector.detect_faces(test_image)
                
                end_time = time.time()
                processing_time = end_time - start_time
                total_detections = 5 * len(self.test_images)
                avg_time = processing_time / total_detections
                
                # Should process within reasonable time on CPU
                self.assertLess(avg_time, 1.0, f"Average detection time should be < 1s, got {avg_time:.2f}s")
                
                logging.info(f"Average detection time: {avg_time:.2f}s per image")
                
        except Exception as e:
            self.fail(f"Face detection performance test failed: {e}")
    
    def test_face_region_extraction(self):
        """Test face region extraction."""
        try:
            self.face_detector = get_face_detector()
            
            if self.test_images:
                test_image = self.test_images[0]
                detections = self.face_detector.detect_faces(test_image)
                
                for detection in detections:
                    # Extract face region
                    face_region = self.face_detector.extract_face_region(test_image, detection)
                    
                    self.assertIsInstance(face_region, np.ndarray)
                    self.assertGreater(face_region.size, 0)
                    
                    # Test preprocessing
                    processed_face = self.face_detector.preprocess_face(face_region)
                    
                    self.assertIsInstance(processed_face, np.ndarray)
                    self.assertGreater(processed_face.size, 0)
                
        except Exception as e:
            self.fail(f"Face region extraction test failed: {e}")

class TestFaceRecognition(unittest.TestCase):
    """Test face recognition functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.face_recognizer = None
        self.test_images = []
        self._create_test_images()
    
    def tearDown(self):
        """Clean up after tests."""
        if self.face_recognizer:
            self.face_recognizer.cleanup()
    
    def _create_test_images(self):
        """Create test images for face recognition."""
        try:
            # Create test images similar to detection test
            for i in range(2):
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Add noise
                noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
                img = cv2.add(img, noise)
                
                # Draw face
                center_x = 320
                center_y = 240
                
                cv2.circle(img, (center_x, center_y), 80, (200, 200, 200), -1)
                cv2.circle(img, (center_x - 30, center_y - 20), 8, (50, 50, 50), -1)
                cv2.circle(img, (center_x + 30, center_y - 20), 8, (50, 50, 50), -1)
                cv2.ellipse(img, (center_x, center_y + 20), (25, 10), 0, 0, 180, (50, 50, 50), 2)
                
                self.test_images.append(img)
                
        except Exception as e:
            logging.warning(f"Error creating test images: {e}")
            self.test_images = [np.zeros((480, 640, 3), dtype=np.uint8)]
    
    def test_face_recognizer_initialization(self):
        """Test face recognizer initialization."""
        try:
            self.face_recognizer = get_face_recognizer()
            
            self.assertIsNotNone(self.face_recognizer)
            self.assertTrue(self.face_recognizer.is_initialized)
            
            # Test recognizer stats
            stats = self.face_recognizer.get_recognition_stats()
            self.assertIsInstance(stats, dict)
            self.assertIn('is_initialized', stats)
            
        except Exception as e:
            self.fail(f"Face recognizer initialization failed: {e}")
    
    def test_embedding_generation(self):
        """Test face embedding generation."""
        try:
            self.face_recognizer = get_face_recognizer()
            
            if self.test_images:
                test_image = self.test_images[0]
                
                # Generate embedding
                embedding = self.face_recognizer.generate_embedding(test_image)
                
                if embedding is not None:
                    self.assertIsInstance(embedding, np.ndarray)
                    self.assertGreater(len(embedding), 0)
                    self.assertEqual(embedding.dtype, np.float32)
                    
                    logging.info(f"Generated embedding with {len(embedding)} dimensions")
                else:
                    logging.warning("No embedding generated (no face detected)")
                
        except Exception as e:
            self.fail(f"Embedding generation test failed: {e}")
    
    def test_face_recognition(self):
        """Test face recognition."""
        try:
            self.face_recognizer = get_face_recognizer()
            
            if self.test_images:
                test_image = self.test_images[0]
                
                # Create a mock detection
                from models.face_models import FaceDetection
                detection = FaceDetection(
                    bbox=(200, 150, 200, 200),
                    confidence=0.8
                )
                
                # Test recognition
                recognition = self.face_recognizer.recognize_face(test_image, detection)
                
                self.assertIsNotNone(recognition)
                self.assertIsInstance(recognition.employee_id, (str, type(None)))
                self.assertIsInstance(recognition.confidence, float)
                self.assertGreaterEqual(recognition.confidence, 0.0)
                self.assertLessEqual(recognition.confidence, 1.0)
                
        except Exception as e:
            self.fail(f"Face recognition test failed: {e}")
    
    def test_face_recognition_performance(self):
        """Test face recognition performance on CPU."""
        try:
            self.face_recognizer = get_face_recognizer()
            
            if self.test_images:
                start_time = time.time()
                
                for i in range(3):  # Test multiple iterations
                    test_image = self.test_images[0]
                    
                    # Generate embedding
                    embedding = self.face_recognizer.generate_embedding(test_image)
                    
                    if embedding is not None:
                        # Test recognition
                        from models.face_models import FaceDetection
                        detection = FaceDetection(
                            bbox=(200, 150, 200, 200),
                            confidence=0.8
                        )
                        
                        recognition = self.face_recognizer.recognize_face(test_image, detection)
                
                end_time = time.time()
                processing_time = end_time - start_time
                avg_time = processing_time / 3
                
                # Should process within reasonable time on CPU
                self.assertLess(avg_time, 2.0, f"Average recognition time should be < 2s, got {avg_time:.2f}s")
                
                logging.info(f"Average recognition time: {avg_time:.2f}s")
                
        except Exception as e:
            self.fail(f"Face recognition performance test failed: {e}")

class TestLightingOptimization(unittest.TestCase):
    """Test lighting optimization functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.lighting_optimizer = None
        self.test_images = []
        self._create_test_images()
    
    def _create_test_images(self):
        """Create test images with different lighting conditions."""
        try:
            # Create images with different brightness levels
            brightness_levels = [0.3, 0.7, 1.0, 1.5]  # Dark, normal, bright, very bright
            
            for brightness in brightness_levels:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Add base pattern
                cv2.circle(img, (320, 240), 80, (200, 200, 200), -1)
                cv2.circle(img, (300, 220), 8, (50, 50, 50), -1)
                cv2.circle(img, (340, 220), 8, (50, 50, 50), -1)
                cv2.ellipse(img, (320, 260), (25, 10), 0, 0, 180, (50, 50, 50), 2)
                
                # Apply brightness
                img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
                
                self.test_images.append(img)
                
        except Exception as e:
            logging.warning(f"Error creating test images: {e}")
            self.test_images = [np.zeros((480, 640, 3), dtype=np.uint8)]
    
    def test_lighting_optimizer_initialization(self):
        """Test lighting optimizer initialization."""
        try:
            self.lighting_optimizer = get_lighting_optimizer()
            
            self.assertIsNotNone(self.lighting_optimizer)
            
            # Test optimizer stats
            stats = self.lighting_optimizer.get_optimization_stats()
            self.assertIsInstance(stats, dict)
            
        except Exception as e:
            self.fail(f"Lighting optimizer initialization failed: {e}")
    
    def test_lighting_analysis(self):
        """Test lighting condition analysis."""
        try:
            self.lighting_optimizer = get_lighting_optimizer()
            
            for i, test_image in enumerate(self.test_images):
                analysis = self.lighting_optimizer.analyze_lighting(test_image)
                
                self.assertIsNotNone(analysis)
                self.assertIsInstance(analysis.brightness, float)
                self.assertIsInstance(analysis.contrast, float)
                self.assertIsInstance(analysis.sharpness, float)
                self.assertIsInstance(analysis.quality_score, float)
                
                logging.info(f"Image {i}: Brightness={analysis.brightness:.1f}, Quality={analysis.quality_score:.2f}")
                
        except Exception as e:
            self.fail(f"Lighting analysis test failed: {e}")
    
    def test_image_optimization(self):
        """Test image optimization for face recognition."""
        try:
            self.lighting_optimizer = get_lighting_optimizer()
            
            for i, test_image in enumerate(self.test_images):
                optimized = self.lighting_optimizer.optimize_for_face_recognition(test_image)
                
                self.assertIsNotNone(optimized)
                self.assertIsInstance(optimized, np.ndarray)
                self.assertEqual(optimized.shape, test_image.shape)
                
                # Check if optimization actually changed the image
                if not np.array_equal(optimized, test_image):
                    logging.info(f"Image {i}: Optimization applied")
                
        except Exception as e:
            self.fail(f"Image optimization test failed: {e}")
    
    def test_optimization_performance(self):
        """Test optimization performance on CPU."""
        try:
            self.lighting_optimizer = get_lighting_optimizer()
            
            if self.test_images:
                start_time = time.time()
                
                for i in range(3):  # Test multiple iterations
                    for test_image in self.test_images:
                        optimized = self.lighting_optimizer.optimize_for_face_recognition(test_image)
                
                end_time = time.time()
                processing_time = end_time - start_time
                total_optimizations = 3 * len(self.test_images)
                avg_time = processing_time / total_optimizations
                
                # Should process within reasonable time on CPU
                self.assertLess(avg_time, 0.5, f"Average optimization time should be < 0.5s, got {avg_time:.2f}s")
                
                logging.info(f"Average optimization time: {avg_time:.2f}s per image")
                
        except Exception as e:
            self.fail(f"Optimization performance test failed: {e}")

class TestFaceQualityEnhancement(unittest.TestCase):
    """Test face quality enhancement functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.quality_enhancer = None
        self.test_images = []
        self._create_test_images()
    
    def _create_test_images(self):
        """Create test images with different quality levels."""
        try:
            # Create images with different quality levels
            for i in range(3):
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Add base pattern
                cv2.circle(img, (320, 240), 80, (200, 200, 200), -1)
                cv2.circle(img, (300, 220), 8, (50, 50, 50), -1)
                cv2.circle(img, (340, 220), 8, (50, 50, 50), -1)
                cv2.ellipse(img, (320, 260), (25, 10), 0, 0, 180, (50, 50, 50), 2)
                
                # Add different levels of blur/noise
                if i == 0:  # High quality
                    pass
                elif i == 1:  # Medium quality
                    img = cv2.GaussianBlur(img, (5, 5), 0)
                else:  # Low quality
                    img = cv2.GaussianBlur(img, (15, 15), 0)
                    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
                    img = cv2.add(img, noise)
                
                self.test_images.append(img)
                
        except Exception as e:
            logging.warning(f"Error creating test images: {e}")
            self.test_images = [np.zeros((480, 640, 3), dtype=np.uint8)]
    
    def test_quality_enhancer_initialization(self):
        """Test quality enhancer initialization."""
        try:
            self.quality_enhancer = get_face_quality_enhancer()
            
            self.assertIsNotNone(self.quality_enhancer)
            
            # Test enhancer stats
            stats = self.quality_enhancer.get_enhancement_stats()
            self.assertIsInstance(stats, dict)
            
        except Exception as e:
            self.fail(f"Quality enhancer initialization failed: {e}")
    
    def test_quality_analysis(self):
        """Test face quality analysis."""
        try:
            self.quality_enhancer = get_face_quality_enhancer()
            
            for i, test_image in enumerate(self.test_images):
                metrics = self.quality_enhancer.analyze_face_quality(test_image)
                
                self.assertIsNotNone(metrics)
                self.assertIsInstance(metrics.sharpness, float)
                self.assertIsInstance(metrics.brightness, float)
                self.assertIsInstance(metrics.contrast, float)
                self.assertIsInstance(metrics.overall_score, float)
                
                logging.info(f"Image {i}: Quality={metrics.overall_score:.2f}, Sharpness={metrics.sharpness:.2f}")
                
        except Exception as e:
            self.fail(f"Quality analysis test failed: {e}")
    
    def test_quality_enhancement(self):
        """Test face quality enhancement."""
        try:
            self.quality_enhancer = get_face_quality_enhancer()
            
            for i, test_image in enumerate(self.test_images):
                enhanced, enhanced_metrics = self.quality_enhancer.enhance_face_quality(test_image)
                
                self.assertIsNotNone(enhanced)
                self.assertIsInstance(enhanced, np.ndarray)
                self.assertEqual(enhanced.shape, test_image.shape)
                
                self.assertIsNotNone(enhanced_metrics)
                self.assertIsInstance(enhanced_metrics.overall_score, float)
                
                logging.info(f"Image {i}: Original quality={enhanced_metrics.overall_score:.2f}")
                
        except Exception as e:
            self.fail(f"Quality enhancement test failed: {e}")
    
    def test_recognition_enhancement(self):
        """Test enhancement specifically for recognition."""
        try:
            self.quality_enhancer = get_face_quality_enhancer()
            
            for i, test_image in enumerate(self.test_images):
                enhanced = self.quality_enhancer.enhance_for_recognition(test_image)
                
                self.assertIsNotNone(enhanced)
                self.assertIsInstance(enhanced, np.ndarray)
                self.assertEqual(enhanced.shape, test_image.shape)
                
        except Exception as e:
            self.fail(f"Recognition enhancement test failed: {e}")

def run_face_recognition_tests():
    """Run all face recognition tests."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestFaceDetection,
        TestFaceRecognition,
        TestLightingOptimization,
        TestFaceQualityEnhancement
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("FACE RECOGNITION TEST SUMMARY")
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
    success = run_face_recognition_tests()
    sys.exit(0 if success else 1)
