"""
OCCUR-CAM System Monitor Dashboard
Terminal-based monitoring dashboard for system health.
"""

import sys
import os
import logging
import time
import threading
import curses
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from system_monitor.monitor_daemon import get_system_monitor
from system_monitor.camera_monitor import get_camera_monitor
from system_monitor.face_monitor import get_face_recognition_monitor

# Import core components to monitor
from core.face_engine import get_face_engine
from core.camera_manager import get_camera_manager
from core.auth_engine import get_auth_engine

class MonitorDashboard:
    def __init__(self, screen):
        self.screen = screen
        self.running = False
        self.update_interval = 1.0  # seconds
        
        # Initialize monitors
        self.system_monitor = get_system_monitor()
        self.camera_monitor = get_camera_monitor()
        self.face_monitor = get_face_recognition_monitor()
        
        # Setup display areas
        self.setup_screen()
    
    def setup_screen(self):
        """Setup curses screen."""
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.curs_set(0)  # Hide cursor
        self.screen.nodelay(1)  # Non-blocking input
    
    def start(self):
        """Start the dashboard."""
        self.running = True
        
        # Start monitors
        self.system_monitor.start()
        self.camera_monitor.start()
        self.face_monitor.start()
        
        # Register core components
        self._register_components()
        
        # Main display loop
        while self.running:
            try:
                self.update_display()
                time.sleep(self.update_interval)
                
                # Check for keyboard input
                c = self.screen.getch()
                if c == ord('q'):
                    self.stop()
                elif c == ord('r'):
                    self._handle_refresh()
                elif c == ord('c'):
                    self._handle_cleanup()
            except Exception as e:
                self.stop()
                raise e
    
    def stop(self):
        """Stop the dashboard."""
        self.running = False
        
        # Stop monitors
        self.system_monitor.stop()
        self.camera_monitor.stop()
        self.face_monitor.stop()
    
    def _register_components(self):
        """Register core components for monitoring."""
        try:
            # Get core components
            face_engine = get_face_engine()
            camera_manager = get_camera_manager()
            auth_engine = get_auth_engine()
            
            # Register with system monitor
            if face_engine:
                self.system_monitor.register_component('face_engine', face_engine)
                self.face_monitor.register_recognizer('main', face_engine)
            
            if camera_manager:
                self.system_monitor.register_component('camera_manager', camera_manager)
                for camera_id, camera in camera_manager.cameras.items():
                    self.camera_monitor.register_camera(camera_id, camera)
            
            if auth_engine:
                self.system_monitor.register_component('auth_engine', auth_engine)
            
        except Exception as e:
            logging.error(f"Error registering components: {e}")
    
    def update_display(self):
        """Update the dashboard display."""
        try:
            self.screen.clear()
            max_y, max_x = self.screen.getmaxyx()
            
            # Title
            title = "OCCUR-CAM System Monitor"
            self.screen.addstr(0, (max_x - len(title)) // 2, title, curses.A_BOLD)
            
            # System status
            self._draw_system_status(2, 0, max_x)
            
            # Component status
            self._draw_component_status(8, 0, max_x)
            
            # Camera status
            self._draw_camera_status(15, 0, max_x)
            
            # Face recognition status
            self._draw_face_recognition_status(22, 0, max_x)
            
            # Controls
            controls = "Controls: (q)uit  (r)efresh  (c)leanup"
            self.screen.addstr(max_y-1, (max_x - len(controls)) // 2, controls)
            
            self.screen.refresh()
            
        except curses.error:
            pass  # Ignore curses errors from terminal resizing
    
    def _draw_system_status(self, y: int, x: int, max_x: int):
        """Draw system resource status."""
        self.screen.addstr(y, x, "System Status:", curses.A_BOLD)
        
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        color = self._get_resource_color(memory.percent)
        self.screen.addstr(y+1, x+2, f"Memory: {memory.percent:.1f}% used", color)
        
        color = self._get_resource_color(cpu_percent)
        self.screen.addstr(y+2, x+2, f"CPU: {cpu_percent:.1f}% used", color)
    
    def _draw_component_status(self, y: int, x: int, max_x: int):
        """Draw component status."""
        self.screen.addstr(y, x, "Component Status:", curses.A_BOLD)
        
        status = self.system_monitor.get_system_status()
        row = y + 1
        for name, health in status['components'].items():
            if name != 'system':
                color = self._get_status_color(health['status'])
                self.screen.addstr(row, x+2, f"{name}: {health['status']}", color)
                row += 1
    
    def _draw_camera_status(self, y: int, x: int, max_x: int):
        """Draw camera status."""
        self.screen.addstr(y, x, "Camera Status:", curses.A_BOLD)
        
        status = self.camera_monitor.get_all_cameras_status()
        row = y + 1
        for camera_id, health in status.items():
            color = self._get_status_color(health['status'])
            self.screen.addstr(row, x+2, 
                f"Camera {camera_id}: {health['status']} "
                f"(Frames: {health['frame_count']} Errors: {health['error_count']})",
                color)
            row += 1
    
    def _draw_face_recognition_status(self, y: int, x: int, max_x: int):
        """Draw face recognition status."""
        self.screen.addstr(y, x, "Face Recognition Status:", curses.A_BOLD)
        
        status = self.face_monitor.get_all_recognizers_status()
        stats = self.face_monitor.get_performance_stats()
        
        row = y + 1
        for name, health in status.items():
            color = self._get_status_color(health['status'])
            self.screen.addstr(row, x+2, 
                f"{name}: {health['status']} "
                f"(Memory: {health['memory_usage']:.1f}%)",
                color)
            row += 1
        
        self.screen.addstr(row+1, x+2, 
            f"Performance: {stats['processed_frames']} frames "
            f"({stats['detection_rate']:.1f}% detection rate)")
    
    def _get_status_color(self, status: str) -> int:
        """Get color pair for status."""
        if status == 'healthy':
            return curses.color_pair(1)  # Green
        elif status in ['warning', 'unknown']:
            return curses.color_pair(2)  # Yellow
        else:
            return curses.color_pair(3)  # Red
    
    def _get_resource_color(self, value: float) -> int:
        """Get color pair for resource value."""
        if value < 70:
            return curses.color_pair(1)  # Green
        elif value < 90:
            return curses.color_pair(2)  # Yellow
        else:
            return curses.color_pair(3)  # Red
    
    def _handle_refresh(self):
        """Handle refresh command."""
        # Force immediate status update
        self.update_display()
    
    def _handle_cleanup(self):
        """Handle cleanup command."""
        # Trigger cleanup procedures
        self.face_monitor._enforce_aggressive_cleanup()
        self.update_display()

def main():
    """Main entry point."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('system_monitor.log'),
                logging.StreamHandler()
            ]
        )
        
        # Start dashboard
        curses.wrapper(lambda screen: MonitorDashboard(screen).start())
        
    except Exception as e:
        logging.error(f"Monitor dashboard error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())