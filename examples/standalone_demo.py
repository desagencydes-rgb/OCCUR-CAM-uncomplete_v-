#!/usr/bin/env python3
"""
OCCUR-CAM Standalone Demo
Standalone version without any background processes or monitoring.
"""

import sys
import os
import logging
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import core modules
from core.face_detector import FaceDetector
from core.face_recognizer import FaceRecognizer
from core.camera_manager import CameraManager
from config.settings import config
from core.utils import setup_logging

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    sys.exit(0)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def display_system_info():
    """Display system information."""
    try:
        print("\n" + "=" * 60)
        print("🎬 OCCUR-CAM STANDALONE DEMO")
        print("=" * 60)
        print(f"System Status: Running")
        print(f"Python Version: {sys.version.split()[0]}")
        print(f"OpenCV Version: {cv2.__version__}")
        print(f"Camera Source: {config.camera.DEFAULT_SOURCE}")
        print(f"Face Model: {config.face_recognition.MODEL_NAME}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error displaying system info: {e}")

def test_camera():
    """Test camera functionality."""
    try:
        print("\n" + "=" * 60)
        print("📹 CAMERA TEST")
        print("=" * 60)
        
        import cv2
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No camera available")
            return False
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera connected: {width}x{height} @ {fps:.1f} FPS")
        
        # Test a few frames
        frame_count = 0
        start_time = time.time()
        
        print("📹 Testing camera feed (5 seconds)...")
        print("Press 'q' to quit early")
        
        while frame_count < 150:  # About 5 seconds at 30 FPS
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Add test info
            cv2.putText(frame, f"Camera Test - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Camera Test", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Show results
        elapsed = time.time() - start_time
        print(f"✅ Camera test completed: {frame_count} frames in {elapsed:.1f}s")
        print(f"   Average FPS: {frame_count/elapsed:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Camera test error: {e}")
        return False

def test_face_detection():
    """Test face detection functionality."""
    try:
        print("\n" + "=" * 60)
        print("🧪 FACE DETECTION TEST")
        print("=" * 60)
        
        # Initialize face detector
        print("Initializing face detector...")
        detector = FaceDetector()
        
        if not detector.is_initialized:
            print("❌ Face detector initialization failed")
            return False
        
        print("✅ Face detector initialized")
        
        # Test with camera
        import cv2
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No camera available")
            return False
        
        print("📹 Starting face detection test...")
        print("Look at the camera and press 'q' to quit")
        
        frame_count = 0
        faces_detected = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces
            detections = detector.detect_faces(frame)
            
            if detections:
                faces_detected += 1
                # Draw bounding boxes
                for detection in detections:
                    bbox = detection.bbox
                    cv2.rectangle(frame, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), 
                                (0, 255, 0), 2)
                    cv2.putText(frame, f"Face {detection.confidence:.2f}", 
                               (int(bbox[0]), int(bbox[1]) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add test info
            cv2.putText(frame, f"Face Detection Test - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces detected: {faces_detected}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Face Detection Test", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Show results
        elapsed = time.time() - start_time
        print(f"✅ Face detection test completed:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Faces detected: {faces_detected}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Average FPS: {frame_count/elapsed:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Face detection test error: {e}")
        return False

def interactive_menu():
    """Interactive menu system."""
    while True:
        print("\n" + "=" * 60)
        print("🎬 OCCUR-CAM STANDALONE MENU")
        print("=" * 60)
        print("Commands:")
        print("  i - Show system info")
        print("  c - Test camera")
        print("  f - Test face detection")
        print("  q - Quit")
        print("=" * 60)
        
        try:
            choice = input("Enter command: ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                break
            elif choice == 'i':
                display_system_info()
            elif choice == 'c':
                test_camera()
            elif choice == 'f':
                test_face_detection()
            else:
                print("❌ Invalid command. Try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="OCCUR-CAM Standalone Demo (No Background Processes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python standalone_demo.py                # Start interactive mode
  python standalone_demo.py --camera       # Test camera only
  python standalone_demo.py --face         # Test face detection only
  python standalone_demo.py --debug        # Start in debug mode
        """
    )
    
    parser.add_argument(
        '--camera', 
        action='store_true', 
        help='Test camera only'
    )
    
    parser.add_argument(
        '--face', 
        action='store_true', 
        help='Test face detection only'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='OCCUR-CAM v1.0.0 (Standalone)'
    )
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logging(log_level)
        
        # Setup signal handlers
        setup_signal_handlers()
        
        logging.info("=" * 60)
        logging.info("OCCUR-CAM Standalone Demo v1.0.0")
        logging.info("=" * 60)
        logging.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Debug mode: {args.debug}")
        
        # Handle specific test modes
        if args.camera:
            print("🎬 OCCUR-CAM Camera Test")
            success = test_camera()
            return 0 if success else 1
        
        if args.face:
            print("🎬 OCCUR-CAM Face Detection Test")
            success = test_face_detection()
            return 0 if success else 1
        
        # Show initial info
        display_system_info()
        
        # Start interactive menu
        interactive_menu()
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        return 0
    except Exception as e:
        logging.error(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    # Import cv2 here to avoid issues if not available
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV not available. Please install: pip install opencv-python")
        sys.exit(1)
    
    sys.exit(main())
