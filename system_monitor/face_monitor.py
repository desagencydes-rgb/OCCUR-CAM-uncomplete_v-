"""
OCCUR-CAM Face Recognition Monitor
Monitors face recognition system health and manages resource usage.
"""

import logging
import threading
import time
import psutil
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from queue import Queue
import gc

class FaceRecognitionMonitor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.recognizers = {}
        self.recognizer_health = {}
        self.running = False
        
        # Monitoring settings
        self.memory_threshold = self.config.get('memory_threshold', 85.0)  # percentage
        self.cpu_threshold = self.config.get('cpu_threshold', 90.0)  # percentage
        self.degradation_threshold = self.config.get('degradation_threshold', 95.0)  # percentage
        self.check_interval = self.config.get('check_interval', 1.0)  # seconds
        
        # Performance tracking
        self.performance_stats = {
            'processed_frames': 0,
            'failed_frames': 0,
            'avg_processing_time': 0.0,
            'detection_rate': 0.0
        }
        
        # Initialize monitoring thread
        self.monitor_thread = None
    
    def start(self):
        """Start recognition monitoring."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        logging.info("Face recognition monitor started")
    
    def stop(self):
        """Stop recognition monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logging.info("Face recognition monitor stopped")
    
    def register_recognizer(self, name: str, recognizer: Any):
        """Register a face recognizer for monitoring."""
        self.recognizers[name] = recognizer
        self.recognizer_health[name] = {
            'status': 'unknown',
            'last_check': None,
            'error_count': 0,
            'avg_processing_time': 0.0,
            'memory_usage': 0.0,
            'last_error': None
        }
        logging.info(f"Registered face recognizer for monitoring: {name}")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_system_resources()
                for name, recognizer in self.recognizers.items():
                    self._check_recognizer_health(name, recognizer)
                time.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"Error in face recognition monitor loop: {e}")
    
    def _check_system_resources(self):
        """Check system resource usage and manage degradation."""
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.degradation_threshold:
                self._enforce_aggressive_cleanup()
            elif memory.percent > self.memory_threshold:
                self._enforce_memory_optimization()
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.cpu_threshold:
                self._enforce_cpu_optimization()
            
        except Exception as e:
            logging.error(f"Error checking system resources: {e}")
    
    def _check_recognizer_health(self, name: str, recognizer: Any):
        """Check health of a specific recognizer."""
        try:
            health = self.recognizer_health[name]
            health['last_check'] = datetime.now()
            
            # Check initialization
            if hasattr(recognizer, 'is_initialized') and not recognizer.is_initialized:
                self._handle_recognizer_error(name, "Recognizer not initialized")
                return
            
            # Monitor memory usage
            process = psutil.Process()
            health['memory_usage'] = process.memory_percent()
            
            # Update status
            health['status'] = 'healthy'
            
        except Exception as e:
            logging.error(f"Error checking recognizer {name} health: {e}")
            self._handle_recognizer_error(name, str(e))
    
    def _enforce_memory_optimization(self):
        """Enforce memory optimization measures."""
        logging.warning("Enforcing memory optimization")
        gc.collect()  # Force garbage collection
        
        for name, recognizer in self.recognizers.items():
            try:
                # Clear any caches if available
                if hasattr(recognizer, 'clear_cache'):
                    recognizer.clear_cache()
                
                # Reduce batch sizes if available
                if hasattr(recognizer, 'set_batch_size'):
                    recognizer.set_batch_size(1)
            except Exception as e:
                logging.error(f"Error optimizing recognizer {name}: {e}")
    
    def _enforce_cpu_optimization(self):
        """Enforce CPU usage optimization."""
        logging.warning("Enforcing CPU optimization")
        
        for name, recognizer in self.recognizers.items():
            try:
                # Reduce processing quality if available
                if hasattr(recognizer, 'set_quality_level'):
                    recognizer.set_quality_level('low')
                
                # Increase processing interval if available
                if hasattr(recognizer, 'set_processing_interval'):
                    recognizer.set_processing_interval(2.0)  # Reduce frequency
            except Exception as e:
                logging.error(f"Error optimizing recognizer {name} CPU usage: {e}")
    
    def _enforce_aggressive_cleanup(self):
        """Enforce aggressive cleanup under high memory pressure."""
        logging.warning("Enforcing aggressive cleanup")
        
        # Force garbage collection
        gc.collect()
        
        for name, recognizer in self.recognizers.items():
            try:
                # Reset recognizer if available
                if hasattr(recognizer, 'reset'):
                    recognizer.reset()
                
                # Clear all caches
                if hasattr(recognizer, 'clear_all_caches'):
                    recognizer.clear_all_caches()
                
                # Minimize memory footprint
                if hasattr(recognizer, 'minimize_memory'):
                    recognizer.minimize_memory()
            except Exception as e:
                logging.error(f"Error cleaning up recognizer {name}: {e}")
    
    def _handle_recognizer_error(self, name: str, error_msg: str):
        """Handle recognizer error."""
        health = self.recognizer_health[name]
        health['error_count'] += 1
        health['last_error'] = error_msg
        health['status'] = 'error'
        
        logging.warning(f"Face recognizer {name} error: {error_msg}")
    
    def get_recognizer_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of specific recognizer."""
        return self.recognizer_health.get(name)
    
    def get_all_recognizers_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all recognizers."""
        return self.recognizer_health.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.copy()

# Global monitor instance
_face_monitor_instance = None

def get_face_recognition_monitor(config: Dict[str, Any] = None) -> FaceRecognitionMonitor:
    """Get global face recognition monitor instance."""
    global _face_monitor_instance
    
    if _face_monitor_instance is None:
        _face_monitor_instance = FaceRecognitionMonitor(config)
    
    return _face_monitor_instance

def cleanup_face_recognition_monitor():
    """Cleanup global face recognition monitor instance."""
    global _face_monitor_instance
    
    if _face_monitor_instance:
        _face_monitor_instance.stop()
        _face_monitor_instance = None