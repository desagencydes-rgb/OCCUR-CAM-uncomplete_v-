#!/usr/bin/env python3
"""
OCCUR-CAM Fixed Main Application
Fixed version without terminal interface looping issues.
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

def simple_status_display(app: OCCURCamApplication):
    """Display simple status without terminal interface."""
    try:
        print("\n" + "=" * 60)
        print("🎬 OCCUR-CAM STATUS")
        print("=" * 60)
        
        # Get application stats
        stats = app.get_application_stats()
        
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
        
        print("\n" + "=" * 60)
        print("Press Ctrl+C to quit")
        print("=" * 60)
        
        # Simple loop without terminal interface
        while True:
            time.sleep(5)  # Update every 5 seconds
            
            # Update stats
            stats = app.get_application_stats()
            uptime = stats.get('uptime', 0)
            
            # Clear screen and redraw (simple approach)
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n" + "=" * 60)
            print("🎬 OCCUR-CAM LIVE STATUS")
            print("=" * 60)
            print(f"System Status: {stats.get('state', 'unknown')}")
            print(f"Uptime: {uptime:.1f} seconds")
            print(f"Health: {stats.get('system_health', 0):.1%}")
            print(f"Cameras: {stats.get('active_cameras', 0)}")
            print(f"Auth Attempts: {stats.get('total_auth_attempts', 0)}")
            
            # Camera status
            if app.camera_manager:
                camera_status = app.camera_manager.get_all_cameras_status()
                print(f"\nCamera Status:")
                for camera_id, status in camera_status.items():
                    print(f"  {camera_id}: {'Connected' if status.get('is_connected') else 'Disconnected'}")
            
            print("\n" + "=" * 60)
            print("Press Ctrl+C to quit")
            print("=" * 60)
            
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    except Exception as e:
        print(f"❌ Error in status display: {e}")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="OCCUR-CAM AI Authentication System (Fixed Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_fixed.py                    # Start with default settings
  python main_fixed.py --setup            # Setup system and exit
  python main_fixed.py --camera 0         # Start with specific camera
  python main_fixed.py --debug            # Start in debug mode
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
        '--no-cameras', 
        action='store_true', 
        help='Start without camera initialization'
    )
    
    parser.add_argument(
        '--test-mode', 
        action='store_true', 
        help='Run in test mode with sample data'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='OCCUR-CAM v1.0.0 (Fixed)'
    )
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logging(log_level)
        
        logging.info("=" * 60)
        logging.info("OCCUR-CAM AI Authentication System v1.0.0 (Fixed)")
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
        
        # Display simple status instead of terminal interface
        simple_status_display(app)
        
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
