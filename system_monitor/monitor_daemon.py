"""
OCCUR-CAM System Monitor Daemon
Monitors system health and manages component recovery.
"""

import psutil
import logging
import time
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from queue import Queue
import cv2
import numpy as np

class SystemMonitor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.running = False
        self.components = {}
        self.health_data = {}
        self.alert_queue = Queue()
        self.recovery_queue = Queue()
        
        # Monitoring thresholds
        self.memory_threshold = self.config.get('memory_threshold', 85.0)  # percentage
        self.cpu_threshold = self.config.get('cpu_threshold', 90.0)  # percentage
        self.frame_timeout = self.config.get('frame_timeout', 5.0)  # seconds
        self.reconnect_attempts = self.config.get('reconnect_attempts', 3)
        
        # Initialize monitoring threads
        self.monitor_thread = None
        self.recovery_thread = None
        self.alert_thread = None
    
    def start(self):
        """Start the monitoring system."""
        self.running = True
        
        # Start monitoring threads
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.recovery_thread = threading.Thread(target=self._recovery_loop)
        self.alert_thread = threading.Thread(target=self._alert_loop)
        
        self.monitor_thread.start()
        self.recovery_thread.start()
        self.alert_thread.start()
        
        logging.info("System monitor started")
    
    def stop(self):
        """Stop the monitoring system."""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join()
        if self.recovery_thread:
            self.recovery_thread.join()
        if self.alert_thread:
            self.alert_thread.join()
        
        logging.info("System monitor stopped")
    
    def register_component(self, name: str, component: Any):
        """Register a component for monitoring."""
        self.components[name] = component
        self.health_data[name] = {
            'status': 'unknown',
            'last_check': datetime.now(),
            'failures': 0,
            'recoveries': 0
        }
        logging.info(f"Registered component for monitoring: {name}")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Check system resources
                self._check_system_resources()
                
                # Check component health
                for name, component in self.components.items():
                    self._check_component_health(name, component)
                
                time.sleep(1)  # Check interval
            except Exception as e:
                logging.error(f"Error in monitor loop: {e}")
                self.alert_queue.put(('error', f"Monitor loop error: {e}"))
    
    def _recovery_loop(self):
        """Component recovery loop."""
        while self.running:
            try:
                if not self.recovery_queue.empty():
                    component_name = self.recovery_queue.get()
                    self._attempt_recovery(component_name)
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Error in recovery loop: {e}")
    
    def _alert_loop(self):
        """Alert processing loop."""
        while self.running:
            try:
                if not self.alert_queue.empty():
                    level, message = self.alert_queue.get()
                    self._handle_alert(level, message)
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Error in alert loop: {e}")
    
    def _check_system_resources(self):
        """Check system resource usage."""
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold:
                self.alert_queue.put(('warning', f"High memory usage: {memory.percent}%"))
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.cpu_threshold:
                self.alert_queue.put(('warning', f"High CPU usage: {cpu_percent}%"))
            
            # Update health data
            self.health_data['system'] = {
                'memory_usage': memory.percent,
                'cpu_usage': cpu_percent,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logging.error(f"Error checking system resources: {e}")
    
    def _check_component_health(self, name: str, component: Any):
        """Check health of a specific component."""
        try:
            if hasattr(component, 'is_healthy'):
                is_healthy = component.is_healthy()
            else:
                # Default health check based on component type
                is_healthy = self._default_health_check(component)
            
            self.health_data[name]['status'] = 'healthy' if is_healthy else 'unhealthy'
            self.health_data[name]['last_check'] = datetime.now()
            
            if not is_healthy:
                self.health_data[name]['failures'] += 1
                self.recovery_queue.put(name)
                
        except Exception as e:
            logging.error(f"Error checking component health - {name}: {e}")
            self.health_data[name]['status'] = 'error'
    
    def _default_health_check(self, component: Any) -> bool:
        """Default health check based on component type."""
        try:
            if hasattr(component, 'is_initialized'):
                return component.is_initialized
            
            if hasattr(component, 'is_connected'):
                return component.is_connected
            
            return True  # Assume healthy if no checks available
            
        except Exception:
            return False
    
    def _attempt_recovery(self, component_name: str):
        """Attempt to recover a failed component."""
        try:
            component = self.components.get(component_name)
            if not component:
                return
            
            logging.info(f"Attempting recovery of {component_name}")
            
            # Try standard recovery methods
            if hasattr(component, 'restart'):
                component.restart()
            elif hasattr(component, 'reconnect'):
                component.reconnect()
            elif hasattr(component, 'initialize'):
                component.initialize()
            
            # Update recovery stats
            self.health_data[component_name]['recoveries'] += 1
            
        except Exception as e:
            logging.error(f"Recovery failed for {component_name}: {e}")
            self.alert_queue.put(('error', f"Recovery failed - {component_name}: {e}"))
    
    def _handle_alert(self, level: str, message: str):
        """Handle system alerts."""
        if level == 'error':
            logging.error(message)
        elif level == 'warning':
            logging.warning(message)
        else:
            logging.info(message)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'timestamp': datetime.now(),
            'components': self.health_data,
            'system': self.health_data.get('system', {}),
            'alerts': list(self.alert_queue.queue),
            'recovery_queue': list(self.recovery_queue.queue)
        }

# Global monitor instance
_monitor_instance = None

def get_system_monitor(config: Dict[str, Any] = None) -> SystemMonitor:
    """Get global system monitor instance."""
    global _monitor_instance
    
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor(config)
    
    return _monitor_instance

def cleanup_system_monitor():
    """Cleanup global system monitor instance."""
    global _monitor_instance
    
    if _monitor_instance:
        _monitor_instance.stop()
        _monitor_instance = None