#!/usr/bin/env python3
"""
OCCUR-CAM No Monitoring Main Application
Version without any background monitoring or database spam.
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
from core.auth_engine import AuthenticationEngine
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

def display_status(face_engine, camera_manager, auth_engine):
    """Display current system status."""
    try:
        print("\n" + "=" * 60)
        print("🎬 OCCUR-CAM STATUS")
        print("=" * 60)
        print(f"System Status: Running")
        print(f"Face Engine: {'Ready' if face_engine else 'Not Available'}")
        print(f"Camera Manager: {'Ready' if camera_manager else 'Not Available'}")
        print(f"Auth Engine: {'Ready' if auth_engine else 'Not Available'}")
        
        # Camera status
        if camera_manager:
            camera_status = camera_manager.get_all_cameras_status()
            print(f"\nCamera Status:")
            for camera_id, status in camera_status.items():
                print(f"  {camera_id}: {'Connected' if status.get('is_connected') else 'Disconnected'}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error displaying status: {e}")

def show_camera_details(camera_manager):
    """Show detailed camera information."""
    try:
        if not camera_manager:
            print("❌ Camera manager not available")
            return
        
        camera_status = camera_manager.get_all_cameras_status()
        
        print("\n" + "=" * 60)
        print("📹 CAMERA DETAILS")
        print("=" * 60)
        
        for camera_id, status in camera_status.items():
            print(f"\nCamera: {camera_id}")
            print(f"  Type: {status.get('camera_type', 'Unknown')}")
            print(f"  Connected: {'Yes' if status.get('is_connected') else 'No'}")
            print(f"  Streaming: {'Yes' if status.get('is_streaming') else 'No'}")
            print(f"  FPS: {status.get('fps', 'N/A')}")
            print(f"  Resolution: {status.get('width', 'N/A')}x{status.get('height', 'N/A')}")
            print(f"  Last Frame: {status.get('last_frame_time', 'Never')}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error showing camera details: {e}")

def show_auth_details(auth_engine):
    """Show authentication details."""
    try:
        if not auth_engine:
            print("❌ Authentication engine not available")
            return
        
        auth_stats = auth_engine.get_authentication_stats()
        
        print("\n" + "=" * 60)
        print("🔐 AUTHENTICATION DETAILS")
        print("=" * 60)
        print(f"Total Attempts: {auth_stats.get('total_attempts', 0)}")
        print(f"Successful: {auth_stats.get('successful_attempts', 0)}")
        print(f"Failed: {auth_stats.get('failed_attempts', 0)}")
        print(f"Success Rate: {auth_stats.get('success_rate', 0):.1%}")
        print(f"Average Processing Time: {auth_stats.get('avg_processing_time', 0):.2f}s")
        print(f"Last Attempt: {auth_stats.get('last_attempt_time', 'Never')}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error showing auth details: {e}")

def test_face_recognition(face_engine, camera_manager):
    """Test face recognition functionality."""
    try:
        print("\n" + "=" * 60)
        print("🧪 FACE RECOGNITION TEST")
        print("=" * 60)
        print("This will test face recognition on the current camera feed.")
        print("Press 'q' to quit the test, 's' to save a test image.")
        print("=" * 60)
        
        if not camera_manager:
            print("❌ Camera manager not available")
            return
        
        # Get first available camera
        camera_status = camera_manager.get_all_cameras_status()
        active_cameras = [cid for cid, status in camera_status.items() if status.get('is_connected')]
        
        if not active_cameras:
            print("❌ No active cameras available")
            return
        
        camera_id = active_cameras[0]
        print(f"Using camera: {camera_id}")
        
        # Test face recognition
        import cv2
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return
        
        print("📹 Starting face recognition test...")
        print("Look at the camera and press 'q' to quit")
        
        frame_count = 0
        faces_detected = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces if face engine is available
            if face_engine:
                detections = face_engine.detect_faces(frame)
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
            cv2.putText(frame, f"Face Recognition Test - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces detected: {faces_detected}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw face detection area
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detection Area", 
                       (w//4, h//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Face Recognition Test", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"face_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Test image saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Show test results
        elapsed = time.time() - start_time
        print(f"\n📊 Test Results:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Faces detected: {faces_detected}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Average FPS: {frame_count/elapsed:.1f}")
        print("✅ Face recognition test completed")
        
    except Exception as e:
        print(f"❌ Error during face recognition test: {e}")

def interactive_menu(face_engine, camera_manager, auth_engine):
    """Interactive menu system."""
    while True:
        print("\n" + "=" * 60)
        print("🎬 OCCUR-CAM INTERACTIVE MENU")
        print("=" * 60)
        print("Commands:")
        print("  s - Show status")
        print("  c - Show camera details")
        print("  a - Show authentication details")
        print("  t - Test face recognition")
        print("  q - Quit")
        print("=" * 60)
        
        try:
            choice = input("Enter command: ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                break
            elif choice == 's':
                display_status(face_engine, camera_manager, auth_engine)
            elif choice == 'c':
                show_camera_details(camera_manager)
            elif choice == 'a':
                show_auth_details(auth_engine)
            elif choice == 't':
                test_face_recognition(face_engine, camera_manager)
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
        description="OCCUR-CAM AI Authentication System (No Monitoring Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_no_monitoring.py                # Start interactive mode
  python main_no_monitoring.py --camera 0     # Start with specific camera
  python main_no_monitoring.py --debug        # Start in debug mode
        """
    )
    
    parser.add_argument(
        '--camera', 
        type=str, 
        default='0', 
        help='Camera source (USB index, IP address, or camera ID)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='OCCUR-CAM v1.0.0 (No Monitoring)'
    )
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logging(log_level)
        
        # Setup signal handlers
        setup_signal_handlers()
        
        logging.info("=" * 60)
        logging.info("OCCUR-CAM AI Authentication System v1.0.0 (No Monitoring)")
        logging.info("=" * 60)
        logging.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Debug mode: {args.debug}")
        logging.info(f"Camera source: {args.camera}")
        
        # Initialize components directly (no background monitoring)
        logging.info("Initializing face processing engine...")
        face_engine = None
        try:
            from core.face_engine import FaceEngine
            face_engine = FaceEngine()
            if face_engine.initialize():
                logging.info("Face engine initialized successfully")
            else:
                logging.warning("Face engine initialization failed")
        except Exception as e:
            logging.warning(f"Face engine not available: {e}")
        
        # Initialize camera manager
        logging.info("Initializing camera manager...")
        camera_manager = None
        try:
            camera_manager = CameraManager()
            if camera_manager.initialize():
                logging.info("Camera manager initialized successfully")
                # Connect to camera
                if camera_manager.connect_camera("CAM001", "usb", args.camera):
                    logging.info(f"Camera connected: {args.camera}")
                else:
                    logging.warning("Failed to connect camera")
            else:
                logging.warning("Camera manager initialization failed")
        except Exception as e:
            logging.warning(f"Camera manager not available: {e}")
        
        # Initialize auth engine
        logging.info("Initializing authentication engine...")
        auth_engine = None
        try:
            auth_engine = AuthenticationEngine(face_engine, camera_manager)
            if auth_engine.initialize():
                logging.info("Authentication engine initialized successfully")
            else:
                logging.warning("Authentication engine initialization failed")
        except Exception as e:
            logging.warning(f"Authentication engine not available: {e}")
        
        # Show initial status
        display_status(face_engine, camera_manager, auth_engine)
        
        # Start interactive menu
        interactive_menu(face_engine, camera_manager, auth_engine)
        
        # Cleanup
        logging.info("Cleaning up...")
        if camera_manager:
            camera_manager.cleanup()
        if face_engine:
            face_engine.cleanup()
        if auth_engine:
            auth_engine.cleanup()
        
        logging.info("OCCUR-CAM application shutdown complete")
        return 0
        
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        return 0
    except Exception as e:
        logging.error(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
