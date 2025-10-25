"""
OCCUR-CAM Terminal Interface
Terminal-based interface for system monitoring and management.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import os
import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class TerminalMode(Enum):
    """Terminal interface modes."""
    DASHBOARD = "dashboard"
    MONITOR = "monitor"
    CONFIG = "config"
    LOGS = "logs"
    HELP = "help"

@dataclass
class TerminalCommand:
    """Terminal command definition."""
    name: str
    description: str
    handler: Callable
    args: List[str] = None

class TerminalInterface:
    """Terminal-based interface for OCCUR-CAM system."""
    
    def __init__(self, app, debug_mode: bool = False):
        """Initialize terminal interface."""
        self.app = app
        self.debug_mode = debug_mode
        self.is_running = False
        self.current_mode = TerminalMode.DASHBOARD
        self.refresh_interval = 10.0  # seconds - much longer to prevent constant refreshing
        
        # Rich console
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
            logging.warning("Rich library not available. Using basic terminal interface.")
        
        # Commands
        self.commands = self._setup_commands()
        
        # Threading
        self.interface_thread = None
        self.stop_event = threading.Event()
        
        # Display data
        self.last_update = datetime.now()
        self.display_data = {}
    
    def start(self):
        """Start the terminal interface."""
        try:
            if self.is_running:
                return
            
            self.is_running = True
            self.stop_event.clear()
            
            # Start interface thread
            self.interface_thread = threading.Thread(target=self._interface_loop, daemon=True)
            self.interface_thread.start()
            
            logging.info("Terminal interface started")
            
        except Exception as e:
            logging.error(f"Error starting terminal interface: {e}")
    
    def stop(self):
        """Stop the terminal interface."""
        try:
            if not self.is_running:
                return
            
            self.is_running = False
            self.stop_event.set()
            
            if self.interface_thread and self.interface_thread.is_alive():
                self.interface_thread.join(timeout=5)
            
            logging.info("Terminal interface stopped")
            
        except Exception as e:
            logging.error(f"Error stopping terminal interface: {e}")
    
    def _interface_loop(self):
        """Main interface loop."""
        try:
            if RICH_AVAILABLE:
                self._rich_interface_loop()
            else:
                self._basic_interface_loop()
                
        except Exception as e:
            logging.error(f"Error in interface loop: {e}")
    
    def _rich_interface_loop(self):
        """Rich-based interface loop."""
        try:
            # Display initial dashboard
            dashboard = self._create_dashboard()
            self.console.print(dashboard)
            
            # Interactive loop
            while not self.stop_event.is_set():
                try:
                    # Get user input
                    print("\n" + "=" * 60)
                    print("Commands: h=help q=quit r=refresh c=cameras a=auth")
                    print("=" * 60)
                    
                    try:
                        choice = input("Enter command: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        break
                    
                    if choice == 'q':
                        break
                    elif choice == 'h':
                        self._show_help()
                    elif choice == 'r':
                        self._update_display_data()
                        dashboard = self._create_dashboard()
                        self.console.print(dashboard)
                    elif choice == 'c':
                        self._show_camera_details()
                    elif choice == 'a':
                        self._show_auth_details()
                    else:
                        print("❌ Invalid command. Type 'h' for help.")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logging.warning(f"Error in rich interface loop: {e}")
                    time.sleep(1)
                        
        except Exception as e:
            logging.error(f"Error in rich interface: {e}")
    
    def _show_help(self):
        """Show help information."""
        print("\n" + "=" * 60)
        print("🎬 OCCUR-CAM HELP")
        print("=" * 60)
        print("Commands:")
        print("  h - Show this help")
        print("  q - Quit application")
        print("  r - Refresh dashboard")
        print("  c - Show camera details")
        print("  a - Show authentication details")
        print("=" * 60)
    
    def _show_camera_details(self):
        """Show camera details."""
        try:
            if not self.app.camera_manager:
                print("❌ Camera manager not available")
                return
            
            camera_status = self.app.camera_manager.get_all_cameras_status()
            
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
    
    def _show_auth_details(self):
        """Show authentication details."""
        try:
            if not self.app.auth_engine:
                print("❌ Authentication engine not available")
                return
            
            auth_stats = self.app.auth_engine.get_authentication_stats()
            
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
    
    def _basic_interface_loop(self):
        """Basic terminal interface loop."""
        try:
            # Display initial dashboard
            self._display_basic_dashboard()
            
            # Interactive loop
            while not self.stop_event.is_set():
                try:
                    # Get user input
                    print("\n" + "=" * 60)
                    print("Commands: h=help q=quit r=refresh c=cameras a=auth")
                    print("=" * 60)
                    
                    try:
                        choice = input("Enter command: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        break
                    
                    if choice == 'q':
                        break
                    elif choice == 'h':
                        self._show_help()
                    elif choice == 'r':
                        self._update_display_data()
                        self._display_basic_dashboard()
                    elif choice == 'c':
                        self._show_camera_details()
                    elif choice == 'a':
                        self._show_auth_details()
                    else:
                        print("❌ Invalid command. Type 'h' for help.")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logging.warning(f"Error in basic interface loop: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            logging.error(f"Error in basic interface: {e}")
    
    def _create_dashboard(self):
        """Create rich dashboard layout."""
        try:
            layout = Layout()
            
            # Split into sections
            layout.split_column(
                Layout(self._create_header(), size=3),
                Layout(self._create_main_content(), name="main"),
                Layout(self._create_footer(), size=3)
            )
            
            # Split main content
            layout["main"].split_row(
                Layout(self._create_status_panel(), size=40),
                Layout(self._create_camera_panel()),
                Layout(self._create_auth_panel())
            )
            
            return layout
            
        except Exception as e:
            logging.warning(f"Error creating dashboard: {e}")
            return Panel("Error creating dashboard", title="OCCUR-CAM")
    
    def _create_header(self):
        """Create header panel."""
        try:
            stats = self.app.get_application_stats()
            uptime = timedelta(seconds=int(stats.get('uptime', 0)))
            
            header_text = Text()
            header_text.append("OCCUR-CAM AI Authentication System", style="bold blue")
            header_text.append(f" | Uptime: {uptime}", style="green")
            header_text.append(f" | Health: {stats.get('system_health', 0):.1%}", style="yellow")
            header_text.append(f" | State: {stats.get('state', 'unknown')}", style="cyan")
            
            return Panel(header_text, box=box.DOUBLE)
            
        except Exception as e:
            logging.warning(f"Error creating header: {e}")
            return Panel("OCCUR-CAM System", box=box.DOUBLE)
    
    def _create_main_content(self):
        """Create main content layout."""
        try:
            # Create a simple main content area
            main_text = Text()
            main_text.append("OCCUR-CAM Main Dashboard", style="bold blue")
            main_text.append("\n\nSystem Status: Running", style="green")
            main_text.append("\nCamera Status: Active", style="yellow")
            main_text.append("\nAuthentication: Ready", style="cyan")
            
            return Panel(main_text, title="Main Dashboard", box=box.ROUNDED)
            
        except Exception as e:
            logging.warning(f"Error creating main content: {e}")
            return Panel("Main Content", title="Dashboard")
    
    def _create_footer(self):
        """Create footer panel."""
        try:
            footer_text = Text()
            footer_text.append("Commands: ", style="bold")
            footer_text.append("h=help ", style="cyan")
            footer_text.append("q=quit ", style="red")
            footer_text.append("r=refresh ", style="green")
            footer_text.append("c=cameras ", style="yellow")
            footer_text.append("a=auth ", style="magenta")
            
            return Panel(footer_text, box=box.ROUNDED)
            
        except Exception as e:
            logging.warning(f"Error creating footer: {e}")
            return Panel("Commands: h=help, q=quit", box=box.ROUNDED)
    
    def _create_status_panel(self):
        """Create system status panel."""
        try:
            stats = self.app.get_application_stats()
            
            table = Table(title="System Status", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("State", stats.get('state', 'unknown'))
            table.add_row("Running", "Yes" if stats.get('is_running', False) else "No")
            table.add_row("Uptime", str(timedelta(seconds=int(stats.get('uptime', 0)))))
            table.add_row("Health", f"{stats.get('system_health', 0):.1%}")
            table.add_row("Cameras", str(stats.get('active_cameras', 0)))
            table.add_row("Sessions", str(stats.get('active_sessions', 0)))
            table.add_row("Auth Total", str(stats.get('total_authentications', 0)))
            table.add_row("Auth Success", str(stats.get('successful_authentications', 0)))
            table.add_row("Auth Failed", str(stats.get('failed_authentications', 0)))
            
            return Panel(table, title="System Status", box=box.ROUNDED)
            
        except Exception as e:
            logging.warning(f"Error creating status panel: {e}")
            return Panel("Error loading status", title="System Status")
    
    def _create_camera_panel(self):
        """Create camera status panel."""
        try:
            if not self.app.camera_manager:
                return Panel("Camera manager not available", title="Cameras")
            
            camera_status = self.app.camera_manager.get_all_cameras_status()
            
            table = Table(title="Camera Status", box=box.ROUNDED)
            table.add_column("ID", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Status", style="green")
            table.add_column("FPS", style="blue")
            table.add_column("Health", style="magenta")
            
            for camera_id, status in camera_status.items():
                status_text = "Online" if status.get('is_connected', False) else "Offline"
                fps = f"{status.get('fps', 0):.1f}"
                health = f"{status.get('health_score', 0):.1%}"
                
                table.add_row(
                    camera_id,
                    status.get('camera_type', 'unknown'),
                    status_text,
                    fps,
                    health
                )
            
            return Panel(table, title="Cameras", box=box.ROUNDED)
            
        except Exception as e:
            logging.warning(f"Error creating camera panel: {e}")
            return Panel("Error loading cameras", title="Cameras")
    
    def _create_auth_panel(self):
        """Create authentication panel."""
        try:
            if not self.app.auth_engine:
                return Panel("Auth engine not available", title="Authentication")
            
            auth_stats = self.app.auth_engine.get_authentication_stats()
            
            table = Table(title="Authentication", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Attempts", str(auth_stats.get('total_attempts', 0)))
            table.add_row("Successful", str(auth_stats.get('successful_authentications', 0)))
            table.add_row("Failed", str(auth_stats.get('failed_authentications', 0)))
            table.add_row("Success Rate", f"{auth_stats.get('success_rate', 0):.1%}")
            table.add_row("Active Sessions", str(auth_stats.get('active_sessions', 0)))
            table.add_row("Recognition Threshold", f"{auth_stats.get('recognition_threshold', 0):.2f}")
            table.add_row("Detection Threshold", f"{auth_stats.get('detection_threshold', 0):.2f}")
            
            return Panel(table, title="Authentication", box=box.ROUNDED)
            
        except Exception as e:
            logging.warning(f"Error creating auth panel: {e}")
            return Panel("Error loading auth stats", title="Authentication")
    
    def _display_basic_dashboard(self):
        """Display basic dashboard without rich."""
        try:
            stats = self.app.get_application_stats()
            
            print("=" * 80)
            print("OCCUR-CAM AI Authentication System")
            print("=" * 80)
            print(f"State: {stats.get('state', 'unknown')}")
            print(f"Uptime: {timedelta(seconds=int(stats.get('uptime', 0)))}")
            print(f"Health: {stats.get('system_health', 0):.1%}")
            print(f"Active Cameras: {stats.get('active_cameras', 0)}")
            print(f"Active Sessions: {stats.get('active_sessions', 0)}")
            print(f"Total Authentications: {stats.get('total_authentications', 0)}")
            print(f"Successful: {stats.get('successful_authentications', 0)}")
            print(f"Failed: {stats.get('failed_authentications', 0)}")
            print("=" * 80)
            print("Commands: h=help, q=quit, r=refresh, c=cameras, a=auth")
            print("=" * 80)
            
        except Exception as e:
            logging.warning(f"Error displaying basic dashboard: {e}")
            print("Error displaying dashboard")
    
    def _update_display_data(self):
        """Update display data."""
        try:
            self.display_data = {
                'timestamp': datetime.now(),
                'app_stats': self.app.get_application_stats(),
                'camera_status': self.app.camera_manager.get_all_cameras_status() if self.app.camera_manager else {},
                'auth_stats': self.app.auth_engine.get_authentication_stats() if self.app.auth_engine else {}
            }
            self.last_update = datetime.now()
            
        except Exception as e:
            logging.warning(f"Error updating display data: {e}")
    
    def _check_user_input(self) -> bool:
        """Check for user input (non-blocking)."""
        try:
            import select
            import tty
            import termios
            
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                return self._handle_user_input(char)
            
            return False
            
        except ImportError:
            # Windows or systems without select
            return False
        except Exception as e:
            logging.warning(f"Error checking user input: {e}")
            return False
    
    def _handle_user_input(self, char: str) -> bool:
        """Handle user input character."""
        try:
            char = char.lower()
            
            if char == 'q':
                return True  # Quit
            elif char == 'h':
                self._show_help()
            elif char == 'r':
                self._refresh_display()
            elif char == 'c':
                self._show_camera_details()
            elif char == 'a':
                self._show_auth_details()
            elif char == 's':
                self._show_system_status()
            
            return False
            
        except Exception as e:
            logging.warning(f"Error handling user input: {e}")
            return False
    
    def _show_help(self):
        """Show help information."""
        try:
            if RICH_AVAILABLE:
                help_text = Text()
                help_text.append("OCCUR-CAM Terminal Commands\n", style="bold blue")
                help_text.append("h - Show this help\n", style="cyan")
                help_text.append("q - Quit application\n", style="red")
                help_text.append("r - Refresh display\n", style="green")
                help_text.append("c - Show camera details\n", style="yellow")
                help_text.append("a - Show authentication details\n", style="magenta")
                help_text.append("s - Show system status\n", style="blue")
                
                self.console.print(Panel(help_text, title="Help", box=box.ROUNDED))
            else:
                print("\nOCCUR-CAM Terminal Commands:")
                print("h - Show help")
                print("q - Quit application")
                print("r - Refresh display")
                print("c - Show camera details")
                print("a - Show authentication details")
                print("s - Show system status")
                
        except Exception as e:
            logging.warning(f"Error showing help: {e}")
    
    def _refresh_display(self):
        """Refresh display."""
        try:
            self._update_display_data()
            if RICH_AVAILABLE:
                self.console.print("[green]Display refreshed[/green]")
            else:
                print("Display refreshed")
                
        except Exception as e:
            logging.warning(f"Error refreshing display: {e}")
    
    def _show_camera_details(self):
        """Show detailed camera information."""
        try:
            if not self.app.camera_manager:
                return
            
            camera_status = self.app.camera_manager.get_all_cameras_status()
            
            if RICH_AVAILABLE:
                table = Table(title="Detailed Camera Status", box=box.ROUNDED)
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")
                
                for camera_id, status in camera_status.items():
                    table.add_row("Camera ID", camera_id)
                    table.add_row("  Type", status.get('camera_type', 'unknown'))
                    table.add_row("  Connected", "Yes" if status.get('is_connected', False) else "No")
                    table.add_row("  Streaming", "Yes" if status.get('is_streaming', False) else "No")
                    table.add_row("  FPS", f"{status.get('fps', 0):.1f}")
                    table.add_row("  Health", f"{status.get('health_score', 0):.1%}")
                    table.add_row("  Frames", str(status.get('frame_count', 0)))
                    table.add_row("  Errors", str(status.get('error_count', 0)))
                    table.add_row("", "")  # Empty row for spacing
                
                self.console.print(table)
            else:
                print("\nDetailed Camera Status:")
                for camera_id, status in camera_status.items():
                    print(f"Camera {camera_id}:")
                    print(f"  Type: {status.get('camera_type', 'unknown')}")
                    print(f"  Connected: {'Yes' if status.get('is_connected', False) else 'No'}")
                    print(f"  Streaming: {'Yes' if status.get('is_streaming', False) else 'No'}")
                    print(f"  FPS: {status.get('fps', 0):.1f}")
                    print(f"  Health: {status.get('health_score', 0):.1%}")
                    print(f"  Frames: {status.get('frame_count', 0)}")
                    print(f"  Errors: {status.get('error_count', 0)}")
                    print()
                
        except Exception as e:
            logging.warning(f"Error showing camera details: {e}")
    
    def _show_auth_details(self):
        """Show detailed authentication information."""
        try:
            if not self.app.auth_engine:
                return
            
            auth_stats = self.app.auth_engine.get_authentication_stats()
            recent_attempts = self.app.auth_engine.get_recent_attempts(10)
            
            if RICH_AVAILABLE:
                # Stats table
                stats_table = Table(title="Authentication Statistics", box=box.ROUNDED)
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="green")
                
                for key, value in auth_stats.items():
                    if isinstance(value, float):
                        stats_table.add_row(key.replace('_', ' ').title(), f"{value:.2f}")
                    else:
                        stats_table.add_row(key.replace('_', ' ').title(), str(value))
                
                self.console.print(stats_table)
                
                # Recent attempts table
                if recent_attempts:
                    attempts_table = Table(title="Recent Authentication Attempts", box=box.ROUNDED)
                    attempts_table.add_column("Time", style="cyan")
                    attempts_table.add_column("Employee", style="yellow")
                    attempts_table.add_column("Status", style="green")
                    attempts_table.add_column("Confidence", style="blue")
                    
                    for attempt in recent_attempts[-10:]:  # Last 10 attempts
                        time_str = attempt.timestamp.strftime("%H:%M:%S")
                        employee = attempt.employee_id or "Unknown"
                        status = attempt.status.value
                        confidence = f"{attempt.confidence:.2f}"
                        
                        attempts_table.add_row(time_str, employee, status, confidence)
                    
                    self.console.print(attempts_table)
                
            else:
                print("\nAuthentication Statistics:")
                for key, value in auth_stats.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
                
                print("\nRecent Authentication Attempts:")
                for attempt in recent_attempts[-10:]:
                    time_str = attempt.timestamp.strftime("%H:%M:%S")
                    employee = attempt.employee_id or "Unknown"
                    status = attempt.status.value
                    confidence = f"{attempt.confidence:.2f}"
                    print(f"{time_str} - {employee} - {status} - {confidence}")
                
        except Exception as e:
            logging.warning(f"Error showing auth details: {e}")
    
    def _show_system_status(self):
        """Show detailed system status."""
        try:
            system_status = self.app.get_system_status()
            
            if RICH_AVAILABLE:
                # Convert to JSON for display
                status_text = json.dumps(system_status, indent=2, default=str)
                self.console.print(Panel(status_text, title="System Status", box=box.ROUNDED))
            else:
                print("\nSystem Status:")
                print(json.dumps(system_status, indent=2, default=str))
                
        except Exception as e:
            logging.warning(f"Error showing system status: {e}")
    
    def _setup_commands(self) -> Dict[str, TerminalCommand]:
        """Setup terminal commands."""
        commands = {}
        
        # Add commands here
        commands['help'] = TerminalCommand(
            name='help',
            description='Show help information',
            handler=self._show_help
        )
        
        commands['quit'] = TerminalCommand(
            name='quit',
            description='Quit application',
            handler=lambda: True
        )
        
        commands['refresh'] = TerminalCommand(
            name='refresh',
            description='Refresh display',
            handler=self._refresh_display
        )
        
        return commands
