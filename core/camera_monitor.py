"""
OCCUR-CAM Camera Monitor
Camera health monitoring and alerting system.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from config.settings import config
from config.database import get_auth_db, get_main_db
from database.schemas.auth_schemas import SystemLog
from database.schemas.main_schemas import AlertRule, AlertLog, SystemMetrics

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class CameraHealthStatus:
    """Camera health status information."""
    camera_id: str
    is_online: bool
    health_score: float
    fps: float
    error_count: int
    last_error: Optional[str]
    last_frame_time: Optional[datetime]
    uptime: float
    status: str  # 'online', 'offline', 'degraded', 'error'

@dataclass
class HealthAlert:
    """Health alert information."""
    alert_id: str
    camera_id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    data: Dict[str, Any]

class CameraMonitor:
    """Camera health monitoring and alerting system."""
    
    def __init__(self):
        """Initialize camera monitor."""
        self.is_running = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # Health tracking
        self.camera_health: Dict[str, CameraHealthStatus] = {}
        self.alert_history: List[HealthAlert] = []
        
        # Alert rules
        self.alert_rules: List[AlertRule] = []
        self.load_alert_rules()
        
        # Callbacks
        self.health_callbacks: List[Callable[[CameraHealthStatus], None]] = []
        self.alert_callbacks: List[Callable[[HealthAlert], None]] = []
        
        # Monitoring settings
        self.check_interval = 30  # seconds
        self.offline_threshold = 60  # seconds
        self.degraded_threshold = 0.7  # health score threshold
        
    def load_alert_rules(self):
        """Load alert rules from database."""
        try:
            with get_main_db() as db:
                rules = db.query(AlertRule).filter(
                    AlertRule.is_enabled == True
                ).all()
                
                self.alert_rules = rules
                logging.info(f"Loaded {len(rules)} alert rules")
                
        except Exception as e:
            logging.error(f"Error loading alert rules: {e}")
            self.alert_rules = []
    
    def start_monitoring(self):
        """Start camera health monitoring."""
        try:
            if self.is_running:
                logging.warning("Camera monitoring already running")
                return
            
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.is_running = True
            
            logging.info("Started camera health monitoring")
            
        except Exception as e:
            logging.error(f"Error starting camera monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop camera health monitoring."""
        try:
            self.is_running = False
            self.stop_event.set()
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            logging.info("Stopped camera health monitoring")
            
        except Exception as e:
            logging.error(f"Error stopping camera monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set() and self.is_running:
            try:
                # Update camera health status
                self._update_camera_health()
                
                # Check for alerts
                self._check_alerts()
                
                # Update system metrics
                self._update_system_metrics()
                
                # Sleep for check interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _update_camera_health(self):
        """Update health status for all cameras."""
        try:
            from core.camera_manager import get_camera_manager
            
            camera_manager = get_camera_manager()
            all_status = camera_manager.get_all_cameras_status()
            
            for camera_id, status in all_status.items():
                self._update_single_camera_health(camera_id, status)
                
        except Exception as e:
            logging.error(f"Error updating camera health: {e}")
    
    def _update_single_camera_health(self, camera_id: str, status: Dict[str, Any]):
        """Update health status for single camera."""
        try:
            # Determine if camera is online
            is_online = status.get('is_connected', False) and status.get('is_streaming', False)
            
            # Calculate uptime
            last_frame_time = status.get('last_frame_time')
            if last_frame_time:
                try:
                    last_frame_dt = datetime.fromisoformat(last_frame_time)
                    uptime = (datetime.now() - last_frame_dt).total_seconds()
                except:
                    uptime = 0
            else:
                uptime = 0
            
            # Determine status
            health_score = status.get('health_score', 0.0)
            if not is_online:
                camera_status = 'offline'
            elif health_score < self.degraded_threshold:
                camera_status = 'degraded'
            elif status.get('error_count', 0) > 0:
                camera_status = 'error'
            else:
                camera_status = 'online'
            
            # Create health status
            health_status = CameraHealthStatus(
                camera_id=camera_id,
                is_online=is_online,
                health_score=health_score,
                fps=status.get('fps', 0.0),
                error_count=status.get('error_count', 0),
                last_error=status.get('last_error'),
                last_frame_time=last_frame_time,
                uptime=uptime,
                status=camera_status
            )
            
            # Update health tracking
            self.camera_health[camera_id] = health_status
            
            # Call health callbacks
            for callback in self.health_callbacks:
                try:
                    callback(health_status)
                except Exception as e:
                    logging.warning(f"Error in health callback: {e}")
            
        except Exception as e:
            logging.error(f"Error updating health for camera {camera_id}: {e}")
    
    def _check_alerts(self):
        """Check for alerts based on current health status."""
        try:
            for rule in self.alert_rules:
                self._check_alert_rule(rule)
                
        except Exception as e:
            logging.error(f"Error checking alerts: {e}")
    
    def _check_alert_rule(self, rule: AlertRule):
        """Check individual alert rule."""
        try:
            if rule.event_type == "camera_offline":
                self._check_camera_offline_alert(rule)
            elif rule.event_type == "camera_degraded":
                self._check_camera_degraded_alert(rule)
            elif rule.event_type == "camera_error":
                self._check_camera_error_alert(rule)
            elif rule.event_type == "low_fps":
                self._check_low_fps_alert(rule)
            elif rule.event_type == "high_error_rate":
                self._check_high_error_rate_alert(rule)
                
        except Exception as e:
            logging.error(f"Error checking alert rule {rule.rule_id}: {e}")
    
    def _check_camera_offline_alert(self, rule: AlertRule):
        """Check for camera offline alerts."""
        try:
            offline_cameras = [
                camera_id for camera_id, health in self.camera_health.items()
                if health.status == 'offline'
            ]
            
            if offline_cameras:
                for camera_id in offline_cameras:
                    self._create_alert(
                        camera_id=camera_id,
                        level=AlertLevel.WARNING,
                        message=f"Camera {camera_id} is offline",
                        rule=rule,
                        data={"offline_cameras": offline_cameras}
                    )
                    
        except Exception as e:
            logging.error(f"Error checking camera offline alert: {e}")
    
    def _check_camera_degraded_alert(self, rule: AlertRule):
        """Check for camera degraded alerts."""
        try:
            degraded_cameras = [
                camera_id for camera_id, health in self.camera_health.items()
                if health.status == 'degraded' and health.health_score < rule.threshold_value
            ]
            
            if degraded_cameras:
                for camera_id in degraded_cameras:
                    health = self.camera_health[camera_id]
                    self._create_alert(
                        camera_id=camera_id,
                        level=AlertLevel.WARNING,
                        message=f"Camera {camera_id} is degraded (health: {health.health_score:.2f})",
                        rule=rule,
                        data={"health_score": health.health_score}
                    )
                    
        except Exception as e:
            logging.error(f"Error checking camera degraded alert: {e}")
    
    def _check_camera_error_alert(self, rule: AlertRule):
        """Check for camera error alerts."""
        try:
            error_cameras = [
                camera_id for camera_id, health in self.camera_health.items()
                if health.status == 'error' and health.error_count >= rule.threshold_value
            ]
            
            if error_cameras:
                for camera_id in error_cameras:
                    health = self.camera_health[camera_id]
                    self._create_alert(
                        camera_id=camera_id,
                        level=AlertLevel.ERROR,
                        message=f"Camera {camera_id} has errors (count: {health.error_count})",
                        rule=rule,
                        data={"error_count": health.error_count, "last_error": health.last_error}
                    )
                    
        except Exception as e:
            logging.error(f"Error checking camera error alert: {e}")
    
    def _check_low_fps_alert(self, rule: AlertRule):
        """Check for low FPS alerts."""
        try:
            low_fps_cameras = [
                camera_id for camera_id, health in self.camera_health.items()
                if health.is_online and health.fps < rule.threshold_value
            ]
            
            if low_fps_cameras:
                for camera_id in low_fps_cameras:
                    health = self.camera_health[camera_id]
                    self._create_alert(
                        camera_id=camera_id,
                        level=AlertLevel.WARNING,
                        message=f"Camera {camera_id} has low FPS ({health.fps:.1f})",
                        rule=rule,
                        data={"fps": health.fps, "threshold": rule.threshold_value}
                    )
                    
        except Exception as e:
            logging.error(f"Error checking low FPS alert: {e}")
    
    def _check_high_error_rate_alert(self, rule: AlertRule):
        """Check for high error rate alerts."""
        try:
            high_error_cameras = [
                camera_id for camera_id, health in self.camera_health.items()
                if health.is_online and health.error_count >= rule.threshold_value
            ]
            
            if high_error_cameras:
                for camera_id in high_error_cameras:
                    health = self.camera_health[camera_id]
                    self._create_alert(
                        camera_id=camera_id,
                        level=AlertLevel.ERROR,
                        message=f"Camera {camera_id} has high error rate ({health.error_count} errors)",
                        rule=rule,
                        data={"error_count": health.error_count}
                    )
                    
        except Exception as e:
            logging.error(f"Error checking high error rate alert: {e}")
    
    def _create_alert(self, camera_id: str, level: AlertLevel, message: str, 
                     rule: AlertRule, data: Dict[str, Any]):
        """Create and process alert."""
        try:
            alert_id = f"{camera_id}_{int(time.time())}"
            
            alert = HealthAlert(
                alert_id=alert_id,
                camera_id=camera_id,
                level=level,
                message=message,
                timestamp=datetime.now(),
                data=data
            )
            
            # Add to history
            self.alert_history.append(alert)
            
            # Keep only last 1000 alerts
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
            
            # Log alert
            logging.warning(f"ALERT [{level.value.upper()}] {message}")
            
            # Save to database
            self._save_alert_to_database(alert, rule)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logging.warning(f"Error in alert callback: {e}")
                    
        except Exception as e:
            logging.error(f"Error creating alert: {e}")
    
    def _save_alert_to_database(self, alert: HealthAlert, rule: AlertRule):
        """Save alert to database."""
        try:
            with get_main_db() as db:
                alert_log = AlertLog(
                    rule_id=rule.id,
                    event_type=rule.event_type,
                    severity=alert.level.value,
                    message=alert.message,
                    notification_method=rule.notification_method,
                    recipient=rule.notification_recipients,
                    metadata=json.dumps(alert.data)
                )
                
                db.add(alert_log)
                db.commit()
                
        except Exception as e:
            logging.error(f"Error saving alert to database: {e}")
    
    def _update_system_metrics(self):
        """Update system metrics."""
        try:
            # Calculate overall system health
            total_cameras = len(self.camera_health)
            online_cameras = sum(1 for h in self.camera_health.values() if h.is_online)
            healthy_cameras = sum(1 for h in self.camera_health.values() if h.health_score > 0.8)
            
            # Calculate average health score
            if total_cameras > 0:
                avg_health = sum(h.health_score for h in self.camera_health.values()) / total_cameras
                online_ratio = online_cameras / total_cameras
                healthy_ratio = healthy_cameras / total_cameras
            else:
                avg_health = 0.0
                online_ratio = 0.0
                healthy_ratio = 0.0
            
            # Save metrics to database
            self._save_metrics_to_database({
                "total_cameras": total_cameras,
                "online_cameras": online_cameras,
                "healthy_cameras": healthy_cameras,
                "avg_health_score": avg_health,
                "online_ratio": online_ratio,
                "healthy_ratio": healthy_ratio
            })
            
        except Exception as e:
            logging.error(f"Error updating system metrics: {e}")
    
    def _save_metrics_to_database(self, metrics: Dict[str, Any]):
        """Save metrics to database."""
        try:
            with get_main_db() as db:
                for metric_name, value in metrics.items():
                    metric = SystemMetrics(
                        metric_name=metric_name,
                        metric_value=value,
                        metric_unit="count" if "cameras" in metric_name else "ratio",
                        component="camera_monitor",
                        metadata=json.dumps({"timestamp": datetime.now().isoformat()})
                    )
                    db.add(metric)
                
                db.commit()
                
        except Exception as e:
            logging.error(f"Error saving metrics to database: {e}")
    
    def get_camera_health(self, camera_id: str) -> Optional[CameraHealthStatus]:
        """Get health status for specific camera."""
        return self.camera_health.get(camera_id)
    
    def get_all_camera_health(self) -> Dict[str, CameraHealthStatus]:
        """Get health status for all cameras."""
        return self.camera_health.copy()
    
    def get_recent_alerts(self, limit: int = 100) -> List[HealthAlert]:
        """Get recent alerts."""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def get_alerts_by_camera(self, camera_id: str, limit: int = 50) -> List[HealthAlert]:
        """Get alerts for specific camera."""
        camera_alerts = [a for a in self.alert_history if a.camera_id == camera_id]
        return camera_alerts[-limit:] if camera_alerts else []
    
    def add_health_callback(self, callback: Callable[[CameraHealthStatus], None]):
        """Add health status callback."""
        self.health_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        total_cameras = len(self.camera_health)
        online_cameras = sum(1 for h in self.camera_health.values() if h.is_online)
        healthy_cameras = sum(1 for h in self.camera_health.values() if h.health_score > 0.8)
        
        return {
            "is_running": self.is_running,
            "total_cameras": total_cameras,
            "online_cameras": online_cameras,
            "healthy_cameras": healthy_cameras,
            "total_alerts": len(self.alert_history),
            "check_interval": self.check_interval,
            "alert_rules": len(self.alert_rules)
        }
    
    def cleanup(self):
        """Cleanup monitoring resources."""
        try:
            self.stop_monitoring()
            self.health_callbacks.clear()
            self.alert_callbacks.clear()
            self.camera_health.clear()
            self.alert_history.clear()
            
            logging.info("Camera monitor cleaned up")
            
        except Exception as e:
            logging.error(f"Error during camera monitor cleanup: {e}")

# Global camera monitor instance
_camera_monitor = None

def get_camera_monitor() -> CameraMonitor:
    """Get global camera monitor instance."""
    global _camera_monitor
    
    if _camera_monitor is None:
        _camera_monitor = CameraMonitor()
    
    return _camera_monitor

def cleanup_camera_monitor():
    """Cleanup global camera monitor instance."""
    global _camera_monitor
    
    if _camera_monitor:
        _camera_monitor.cleanup()
        _camera_monitor = None
