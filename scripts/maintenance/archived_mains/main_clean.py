#!/usr/bin/env python3
"""
OCCUR-CAM Clean Main Application
Clean version without any background refreshing or monitoring.
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
from core.application import OCCURCamApplication
from config.settings import config
from core.utils import setup_logging

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    if hasattr(signal_handler, 'app') and signal_handler.app:
        signal_handler.app.shutdown()
    sys.exit(0)

def setup_signal_handlers(app: OCCURCamApplication):
    """Setup signal handlers for graceful shutdown."""
    signal_handler.app = app
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def display_status(app: OCCURCamApplication):
    """Display current system status."""
    try:
        stats = app.get_application_stats()
        
        print("\n" + "=" * 60)
        print("🎬 OCCUR-CAM STATUS")
        print("=" * 60)
        print(f"System Status: {stats.get('state', 'unknown')}")
        print(f"Uptime: {stats.get('uptime', 0):.1f} seconds")
        print(f"Health: {stats.get('system_health', 0):.1%}")
        print(f"Cameras: {stats.get('active_cameras', 0)}")
        print(f"Auth Attempts: {stats.get('total_auth_attempts', 0)}")
        
        # Camera status
        if app.camera_manager:
            camera_status = app.camera_manager.get_all_cameras_status()
            print(f"\nCamera Status:")
            for camera_id, status in camera_status.items():
                print(f"  {camera_id}: {'Connected' if status.get('is_connected') else 'Disconnected'}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error displaying status: {e}")

def interactive_menu(app: OCCURCamApplication):
    """Interactive menu system."""
    while True:
        print("\n" + "=" * 60)
        print("🎬 OCCUR-CAM INTERACTIVE MENU")
        print("=" * 60)
        print("Commands:")
        print("  s - Show status")
        print("  c - Show camera details")
        print("  a - Show authentication stats")
        print("  t - Test face recognition")
        print("  q - Quit")
        print("=" * 60)
        
        try:
            choice = input("Enter command: ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                break
            elif choice == 's':
                display_status(app)
            elif choice == 'c':
                show_camera_details(app)
            elif choice == 'a':
                show_auth_stats(app)
            elif choice == 't':
                test_face_recognition(app)
            else:
                print("❌ Invalid command. Try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def show_camera_details(app: OCCURCamApplication):
    """Show detailed camera information."""
    try:
        if not app.camera_manager:
            print("❌ Camera manager not available")
            return
        
        camera_status = app.camera_manager.get_all_cameras_status()
        
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

def show_auth_stats(app: OCCURCamApplication):
    """Show authentication statistics."""
    try:
        if not app.auth_engine:
            print("❌ Authentication engine not available")
            return
        
        auth_stats = app.auth_engine.get_authentication_stats()
        
        print("\n" + "=" * 60)
        print("🔐 AUTHENTICATION STATISTICS")
        print("=" * 60)
        print(f"Total Attempts: {auth_stats.get('total_attempts', 0)}")
        print(f"Successful: {auth_stats.get('successful_attempts', 0)}")
        print(f"Failed: {auth_stats.get('failed_attempts', 0)}")
        print(f"Success Rate: {auth_stats.get('success_rate', 0):.1%}")
        print(f"Average Processing Time: {auth_stats.get('avg_processing_time', 0):.2f}s")
        print(f"Last Attempt: {auth_stats.get('last_attempt_time', 'Never')}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error showing auth stats: {e}")

def test_face_recognition(app: OCCURCamApplication):
    """Test face recognition functionality."""
    try:
        print("\n" + "=" * 60)
        print("🧪 FACE RECOGNITION TEST")
        print("=" * 60)
        print("This will test face recognition on the current camera feed.")
        print("Press 'q' to quit the test, 's' to save a test image.")
        print("=" * 60)
        
        if not app.camera_manager:
            print("❌ Camera manager not available")
            return
        
        # Get first available camera
        camera_status = app.camera_manager.get_all_cameras_status()
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
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Add test info
            cv2.putText(frame, f"Face Recognition Test - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
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
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Average FPS: {frame_count/elapsed:.1f}")
        print("✅ Face recognition test completed")
        
    except Exception as e:
        print(f"❌ Error during face recognition test: {e}")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="OCCUR-CAM AI Authentication System (Clean Interactive Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_clean.py                # Start interactive mode
  python main_clean.py --setup       # Setup system and exit
  python main_clean.py --camera 0    # Start with specific camera
  python main_clean.py --debug       # Start in debug mode
        """
    )
    
    parser.add_argument(
        '--setup', 
        action='store_true', 
        help='Setup system (create databases, load initial data)'
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
        '--config', 
        type=str, 
        default='default', 
        help='Configuration profile to use'
    )
    
    parser.add_argument(
        '--test-mode', 
        action='store_true', 
        help='Run in test mode with sample data'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='OCCUR-CAM v1.0.0 (Clean Interactive)'
    )
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logging(log_level)
        
        logging.info("=" * 60)
        logging.info("OCCUR-CAM AI Authentication System v1.0.0 (Clean Interactive)")
        logging.info("=" * 60)
        logging.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Configuration: {args.config}")
        logging.info(f"Debug mode: {args.debug}")
        logging.info(f"Camera source: {args.camera}")
        
        # Create application instance
        app = OCCURCamApplication(
            camera_source=args.camera,
            config_profile=args.config,
            debug_mode=args.debug,
            test_mode=args.test_mode
        )
        
        # Setup signal handlers
        setup_signal_handlers(app)
        
        # Handle setup mode
        if args.setup:
            logging.info("Running system setup...")
            success = app.setup_system()
            if success:
                logging.info("System setup completed successfully!")
                return 0
            else:
                logging.error("System setup failed!")
                return 1
        
        # Initialize application
        logging.info("Initializing OCCUR-CAM application...")
        if not app.initialize():
            logging.error("Failed to initialize application!")
            return 1
        
        # Start application
        logging.info("Starting OCCUR-CAM application...")
        if not app.start():
            logging.error("Failed to start application!")
            return 1
        
        # Show initial status
        display_status(app)
        
        # Start interactive menu
        interactive_menu(app)
        
        # Shutdown
        logging.info("Shutting down OCCUR-CAM application...")
        app.shutdown()
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        return 0
    except Exception as e:
        logging.error(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
