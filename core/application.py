"""
OCCUR-CAM Core Application Manager
Main application controller that coordinates all system components.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import signal
import sys

from config.settings import config
from config.database import create_tables, test_connections
from database.migrations import create_all_tables, seed_initial_data
from core.face_engine import get_face_engine, cleanup_face_engine
from core.camera_manager import get_camera_manager, cleanup_camera_manager
from core.auth_engine import get_auth_engine, cleanup_auth_engine
from core.camera_monitor import get_camera_monitor, cleanup_camera_monitor
from core.ivcam_integration import get_third_party_camera_manager, cleanup_third_party_camera_manager
from core.terminal_interface import TerminalInterface

class ApplicationState(Enum):
    """Application state enumeration."""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"

@dataclass
class ApplicationStats:
    """Application statistics."""
    start_time: datetime
    uptime: float
    total_authentications: int
    successful_authentications: int
    failed_authentications: int
    active_cameras: int
    active_sessions: int
    system_health: float
    memory_usage: float
    cpu_usage: float

class OCCURCamApplication:
    """Main OCCUR-CAM application controller."""
    
    def __init__(self, camera_source: str = "0", config_profile: str = "default", 
                 debug_mode: bool = False, test_mode: bool = False):
        """Initialize OCCUR-CAM application."""
        self.camera_source = camera_source
        self.config_profile = config_profile
        self.debug_mode = debug_mode
        self.test_mode = test_mode
        
        # Application state
        self.state = ApplicationState.STOPPED
        self.start_time = None
        self.is_running = False
        
        # Core components
        self.face_engine = None
        self.camera_manager = None
        self.auth_engine = None
        self.camera_monitor = None
        self.third_party_manager = None
        self.terminal_interface = None
        
        # Threading
        self.main_thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = ApplicationStats(
            start_time=datetime.now(),
            uptime=0.0,
            total_authentications=0,
            successful_authentications=0,
            failed_authentications=0,
            active_cameras=0,
            active_sessions=0,
            system_health=0.0,
            memory_usage=0.0,
            cpu_usage=0.0
        )
        
        # Callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'startup': [],
            'shutdown': [],
            'error': [],
            'auth_success': [],
            'auth_failure': [],
            'camera_connected': [],
            'camera_disconnected': []
        }
    
    def initialize(self) -> bool:
        """Initialize the application and all components."""
        try:
            if self.state != ApplicationState.STOPPED:
                logging.warning("Application already initialized or running")
                return False
            
            self.state = ApplicationState.INITIALIZING
            logging.info("Initializing OCCUR-CAM application...")
            
            # 1. Test database connections
            if not self._test_database_connections():
                return False
            
            # 2. Initialize core components
            if not self._initialize_core_components():
                return False
            
            # 3. Setup terminal interface
            if not self._setup_terminal_interface():
                return False
            
            # 4. Setup event handlers
            self._setup_event_handlers()
            
            # 5. Initialize camera system
            if not self._initialize_camera_system():
                return False
            
            # 6. Start monitoring systems
            if not self._start_monitoring_systems():
                return False
            
            self.state = ApplicationState.RUNNING
            logging.info("OCCUR-CAM application initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing application: {e}")
            self.state = ApplicationState.ERROR
            return False
    
    def start(self) -> bool:
        """Start the application."""
        try:
            if self.state != ApplicationState.RUNNING:
                logging.error("Application not in running state")
                return False
            
            logging.info("Starting OCCUR-CAM application...")
            
            # Start main application thread
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            self.is_running = True
            self.start_time = datetime.now()
            
            # Call startup callbacks
            self._call_callbacks('startup')
            
            logging.info("OCCUR-CAM application started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error starting application: {e}")
            self.state = ApplicationState.ERROR
            return False
    
    def run(self):
        """Run the main application loop."""
        try:
            if not self.is_running:
                logging.error("Application not running")
                return
            
            # Start terminal interface
            if self.terminal_interface:
                self.terminal_interface.start()
            
            # Wait for stop event
            while not self.stop_event.is_set():
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt")
        except Exception as e:
            logging.error(f"Error in main run loop: {e}")
            self._call_callbacks('error', e)
    
    def shutdown(self):
        """Shutdown the application gracefully."""
        try:
            if self.state == ApplicationState.SHUTTING_DOWN:
                return
            
            self.state = ApplicationState.SHUTTING_DOWN
            logging.info("Shutting down OCCUR-CAM application...")
            
            # Stop main loop
            self.stop_event.set()
            
            # Stop terminal interface
            if self.terminal_interface:
                self.terminal_interface.stop()
            
            # Stop monitoring systems
            self._stop_monitoring_systems()
            
            # Stop camera systems
            self._stop_camera_systems()
            
            # Cleanup components
            self._cleanup_components()
            
            # Call shutdown callbacks
            self._call_callbacks('shutdown')
            
            self.state = ApplicationState.STOPPED
            self.is_running = False
            
            logging.info("OCCUR-CAM application shutdown complete")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
    
    def setup_system(self) -> bool:
        """Setup the system (create databases, load initial data)."""
        try:
            logging.info("Setting up OCCUR-CAM system...")
            
            # 1. Create database tables
            logging.info("Creating database tables...")
            if not create_all_tables():
                logging.error("Failed to create database tables")
                return False
            
            # 2. Seed initial data
            logging.info("Seeding initial data...")
            if not seed_initial_data():
                logging.error("Failed to seed initial data")
                return False
            
            # 3. Test system components (optional during setup)
            logging.info("Testing system components...")
            try:
                if not self._test_system_components():
                    logging.warning("System component tests failed (this is OK during setup)")
            except Exception as e:
                logging.warning(f"System component tests skipped: {e}")
            
            logging.info("OCCUR-CAM system setup completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error setting up system: {e}")
            return False
    
    def _test_database_connections(self) -> bool:
        """Test database connections."""
        try:
            logging.info("Testing database connections...")
            
            if not test_connections():
                logging.error("Database connection test failed")
                return False
            
            logging.info("Database connections successful")
            return True
            
        except Exception as e:
            logging.error(f"Error testing database connections: {e}")
            return False
    
    def _initialize_core_components(self) -> bool:
        """Initialize core system components."""
        try:
            logging.info("Initializing core components...")
            
            # Initialize face engine
            self.face_engine = get_face_engine()
            if not self.face_engine.is_initialized:
                logging.error("Failed to initialize face engine")
                return False
            
            # Initialize camera manager
            self.camera_manager = get_camera_manager()
            
            # Initialize auth engine
            self.auth_engine = get_auth_engine()
            
            # Initialize camera monitor
            self.camera_monitor = get_camera_monitor()
            
            # Initialize third-party camera manager
            self.third_party_manager = get_third_party_camera_manager()
            
            logging.info("Core components initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing core components: {e}")
            return False
    
    def _setup_terminal_interface(self) -> bool:
        """Setup terminal interface."""
        try:
            logging.info("Setting up terminal interface...")
            
            self.terminal_interface = TerminalInterface(
                app=self,
                debug_mode=self.debug_mode
            )
            
            logging.info("Terminal interface setup complete")
            return True
            
        except Exception as e:
            logging.error(f"Error setting up terminal interface: {e}")
            return False
    
    def _setup_event_handlers(self):
        """Setup event handlers for system components."""
        try:
            # Auth engine callbacks
            if self.auth_engine:
                self.auth_engine.add_auth_callback(self._on_authentication)
                self.auth_engine.add_session_callback(self._on_session_change)
            
            # Camera manager callbacks
            if self.camera_manager:
                # Add camera event callbacks here
                pass
            
            # Camera monitor callbacks
            if self.camera_monitor:
                self.camera_monitor.add_health_callback(self._on_camera_health_change)
                self.camera_monitor.add_alert_callback(self._on_camera_alert)
            
            logging.info("Event handlers setup complete")
            
        except Exception as e:
            logging.error(f"Error setting up event handlers: {e}")
    
    def _initialize_camera_system(self) -> bool:
        """Initialize camera system."""
        try:
            logging.info("Initializing camera system...")
            
            # Start camera manager
            if not self.camera_manager.start_all_cameras():
                logging.warning("Some cameras failed to start")
            
            # Start camera monitoring
            self.camera_monitor.start_monitoring()
            
            logging.info("Camera system initialized")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing camera system: {e}")
            return False
    
    def _start_monitoring_systems(self) -> bool:
        """Start monitoring systems."""
        try:
            logging.info("Starting monitoring systems...")
            
            # Camera health monitoring is already started in camera system init
            
            logging.info("Monitoring systems started")
            return True
            
        except Exception as e:
            logging.error(f"Error starting monitoring systems: {e}")
            return False
    
    def _stop_monitoring_systems(self):
        """Stop monitoring systems."""
        try:
            if self.camera_monitor:
                self.camera_monitor.stop_monitoring()
            
        except Exception as e:
            logging.error(f"Error stopping monitoring systems: {e}")
    
    def _stop_camera_systems(self):
        """Stop camera systems."""
        try:
            if self.camera_manager:
                self.camera_manager.stop_all_cameras()
            
        except Exception as e:
            logging.error(f"Error stopping camera systems: {e}")
    
    def _cleanup_components(self):
        """Cleanup all components."""
        try:
            cleanup_face_engine()
            cleanup_camera_manager()
            cleanup_auth_engine()
            cleanup_camera_monitor()
            cleanup_third_party_camera_manager()
            
        except Exception as e:
            logging.error(f"Error cleaning up components: {e}")
    
    def _test_system_components(self) -> bool:
        """Test system components."""
        try:
            logging.info("Testing system components...")
            
            # Test face engine
            if self.face_engine and not self.face_engine.is_initialized:
                logging.error("Face engine not initialized")
                return False
            
            # Test camera manager
            if not self.camera_manager:
                logging.error("Camera manager not initialized")
                return False
            
            # Test auth engine
            if not self.auth_engine:
                logging.error("Auth engine not initialized")
                return False
            
            logging.info("System component tests passed")
            return True
            
        except Exception as e:
            logging.error(f"Error testing system components: {e}")
            return False
    
    def _main_loop(self):
        """Main application loop."""
        try:
            while not self.stop_event.is_set():
                # Update statistics
                self._update_statistics()
                
                # Process any pending tasks
                self._process_tasks()
                
                # Sleep briefly
                time.sleep(1.0)
                
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            self._call_callbacks('error', e)
    
    def _update_statistics(self):
        """Update application statistics."""
        try:
            current_time = datetime.now()
            
            if self.start_time:
                self.stats.uptime = (current_time - self.start_time).total_seconds()
            
            # Update auth stats
            if self.auth_engine:
                auth_stats = self.auth_engine.get_authentication_stats()
                self.stats.total_authentications = auth_stats.get('total_attempts', 0)
                self.stats.successful_authentications = auth_stats.get('successful_authentications', 0)
                self.stats.failed_authentications = auth_stats.get('failed_authentications', 0)
                self.stats.active_sessions = auth_stats.get('active_sessions', 0)
            
            # Update camera stats
            if self.camera_manager:
                camera_status = self.camera_manager.get_all_cameras_status()
                self.stats.active_cameras = sum(1 for status in camera_status.values() 
                                              if status.get('is_connected', False))
            
            # Update system health
            self.stats.system_health = self._calculate_system_health()
            
        except Exception as e:
            logging.warning(f"Error updating statistics: {e}")
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health."""
        try:
            health_factors = []
            
            # Camera health
            if self.camera_monitor:
                all_health = self.camera_monitor.get_all_camera_health()
                if all_health:
                    avg_camera_health = sum(h.health_score for h in all_health.values()) / len(all_health)
                    health_factors.append(avg_camera_health)
            
            # Auth engine health
            if self.auth_engine:
                auth_stats = self.auth_engine.get_authentication_stats()
                if auth_stats.get('total_attempts', 0) > 0:
                    success_rate = auth_stats.get('successful_authentications', 0) / auth_stats.get('total_attempts', 1)
                    health_factors.append(success_rate)
            
            # Database health
            if test_connections():
                health_factors.append(1.0)
            else:
                health_factors.append(0.0)
            
            # Calculate average health
            if health_factors:
                return sum(health_factors) / len(health_factors)
            else:
                return 0.5
                
        except Exception as e:
            logging.warning(f"Error calculating system health: {e}")
            return 0.5
    
    def _process_tasks(self):
        """Process any pending tasks."""
        try:
            # This is where you would process any pending tasks
            # For now, it's empty but can be extended
            pass
            
        except Exception as e:
            logging.warning(f"Error processing tasks: {e}")
    
    def _on_authentication(self, attempt):
        """Handle authentication event."""
        try:
            if attempt.status.value == "success":
                self._call_callbacks('auth_success', attempt)
            else:
                self._call_callbacks('auth_failure', attempt)
                
        except Exception as e:
            logging.warning(f"Error handling authentication event: {e}")
    
    def _on_session_change(self, session):
        """Handle session change event."""
        try:
            # Handle session changes
            pass
            
        except Exception as e:
            logging.warning(f"Error handling session change: {e}")
    
    def _on_camera_health_change(self, health_status):
        """Handle camera health change event."""
        try:
            # Handle camera health changes
            pass
            
        except Exception as e:
            logging.warning(f"Error handling camera health change: {e}")
    
    def _on_camera_alert(self, alert):
        """Handle camera alert event."""
        try:
            # Handle camera alerts
            logging.warning(f"Camera alert: {alert.message}")
            
        except Exception as e:
            logging.warning(f"Error handling camera alert: {e}")
    
    def _call_callbacks(self, event_type: str, *args, **kwargs):
        """Call event callbacks."""
        try:
            if event_type in self.event_callbacks:
                for callback in self.event_callbacks[event_type]:
                    try:
                        callback(*args, **kwargs)
                    except Exception as e:
                        logging.warning(f"Error in callback: {e}")
                        
        except Exception as e:
            logging.warning(f"Error calling callbacks: {e}")
    
    def add_event_callback(self, event_type: str, callback: Callable):
        """Add event callback."""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    def get_application_stats(self) -> Dict[str, Any]:
        """Get application statistics."""
        return {
            "state": self.state.value,
            "is_running": self.is_running,
            "uptime": self.stats.uptime,
            "start_time": self.stats.start_time.isoformat() if self.stats.start_time else None,
            "total_authentications": self.stats.total_authentications,
            "successful_authentications": self.stats.successful_authentications,
            "failed_authentications": self.stats.failed_authentications,
            "active_cameras": self.stats.active_cameras,
            "active_sessions": self.stats.active_sessions,
            "system_health": self.stats.system_health,
            "debug_mode": self.debug_mode,
            "test_mode": self.test_mode
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status."""
        status = {
            "application": self.get_application_stats(),
            "face_engine": self.face_engine.get_engine_stats() if self.face_engine else {},
            "camera_manager": self.camera_manager.get_all_cameras_status() if self.camera_manager else {},
            "auth_engine": self.auth_engine.get_authentication_stats() if self.auth_engine else {},
            "camera_monitor": self.camera_monitor.get_monitoring_stats() if self.camera_monitor else {}
        }
        
        return status
