#!/usr/bin/env python3
"""
OCCUR-CAM Dashboard
Real-time dashboard for monitoring authentication, user management, and system status.
"""

import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
from pathlib import Path
import json
from PIL import Image, ImageTk

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.main import OCCURCamSystem  # Updated import path

class OCCURCamDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OCCUR-CAM Dashboard v2.0.0")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # System instance
        self.system = None
        self.is_running = False
        self.update_thread = None
        
        # Variables
        self.stats_vars = {
            'frames_processed': tk.StringVar(value="0"),
            'faces_detected': tk.StringVar(value="0"),
            'authentications': tk.StringVar(value="0"),
            'registrations': tk.StringVar(value="0"),
            'known_users': tk.StringVar(value="0"),
            'system_status': tk.StringVar(value="Stopped"),
            'uptime': tk.StringVar(value="00:00:00")
        }
        
        self.setup_ui()
        self.start_time = None
        
        # Load static users on startup
        self.load_static_users()
        
    def setup_ui(self):
        """Setup the dashboard UI."""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = tk.Label(title_frame, text="🎬 OCCUR-CAM Dashboard", 
                              font=('Arial', 20, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        title_label.pack()
        
        # Control panel
        self.setup_control_panel()
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#2c3e50')
        content_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Statistics
        self.setup_stats_panel(content_frame)
        
        # Right panel - Live feed and logs
        self.setup_live_panel(content_frame)
        
        # Bottom panel - User management
        self.setup_user_panel()
        
    def setup_control_panel(self):
        """Setup control buttons."""
        control_frame = tk.Frame(self.root, bg='#34495e', relief='raised', bd=2)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Start/Stop button
        self.start_btn = tk.Button(control_frame, text="▶️ START SYSTEM", 
                                  command=self.toggle_system, font=('Arial', 12, 'bold'),
                                  bg='#27ae60', fg='white', padx=20, pady=5)
        self.start_btn.pack(side='left', padx=5, pady=5)
        
        # Camera selection
        tk.Label(control_frame, text="Camera:", bg='#34495e', fg='white', font=('Arial', 10)).pack(side='left', padx=(20,5))
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(control_frame, textvariable=self.camera_var, 
                                   values=["0", "1", "2"], width=5)
        camera_combo.pack(side='left', padx=5)
        
        # Debug mode
        self.debug_var = tk.BooleanVar()
        debug_check = tk.Checkbutton(control_frame, text="Debug Mode", 
                                    variable=self.debug_var, bg='#34495e', fg='white')
        debug_check.pack(side='left', padx=20)
        
        # Status indicator
        self.status_label = tk.Label(control_frame, text="● STOPPED", 
                                    font=('Arial', 12, 'bold'), fg='#e74c3c', bg='#34495e')
        self.status_label.pack(side='right', padx=10)
        
    def setup_stats_panel(self, parent):
        """Setup statistics panel."""
        stats_frame = tk.LabelFrame(parent, text="📊 System Statistics", 
                                   font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        stats_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Statistics grid
        stats_grid = tk.Frame(stats_frame, bg='#2c3e50')
        stats_grid.pack(fill='both', expand=True, padx=10, pady=10)
        
        stats_data = [
            ("System Status:", "system_status"),
            ("Uptime:", "uptime"),
            ("Frames Processed:", "frames_processed"),
            ("Faces Detected:", "faces_detected"),
            ("Authentications:", "authentications"),
            ("New Registrations:", "registrations"),
            ("Known Users:", "known_users")
        ]
        
        for i, (label, var_key) in enumerate(stats_data):
            row = i // 2
            col = (i % 2) * 2
            
            tk.Label(stats_grid, text=label, font=('Arial', 10, 'bold'), 
                   fg='#bdc3c7', bg='#2c3e50').grid(row=row, column=col, sticky='w', padx=5, pady=2)
            
            value_label = tk.Label(stats_grid, textvariable=self.stats_vars[var_key], 
                                 font=('Arial', 10), fg='#ecf0f1', bg='#2c3e50')
            value_label.grid(row=row, column=col+1, sticky='w', padx=5, pady=2)
        
        # Real-time notifications
        notification_frame = tk.LabelFrame(stats_frame, text="🔔 Face Detection Notifications", 
                                         font=('Arial', 10, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        notification_frame.pack(fill='x', padx=10, pady=5)
        
        self.notification_text = tk.Text(notification_frame, height=6, width=40, 
                                       font=('Consolas', 9), bg='#34495e', fg='#ecf0f1')
        self.notification_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollbar for notifications
        notif_scrollbar = tk.Scrollbar(notification_frame, orient="vertical", command=self.notification_text.yview)
        notif_scrollbar.pack(side="right", fill="y")
        self.notification_text.configure(yscrollcommand=notif_scrollbar.set)
        
        # Real-time activity
        activity_frame = tk.LabelFrame(stats_frame, text="🔄 System Activity", 
                                     font=('Arial', 10, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        activity_frame.pack(fill='x', padx=10, pady=5)
        
        self.activity_text = tk.Text(activity_frame, height=4, width=40, 
                                   font=('Consolas', 9), bg='#34495e', fg='#ecf0f1')
        self.activity_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollbar for activity
        scrollbar = tk.Scrollbar(activity_frame, orient="vertical", command=self.activity_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.activity_text.configure(yscrollcommand=scrollbar.set)
        
    def setup_live_panel(self, parent):
        """Setup live feed panel."""
        live_frame = tk.LabelFrame(parent, text="📹 Live Camera Feed", 
                                  font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        live_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Video display
        self.video_label = tk.Label(live_frame, text="Camera Feed\nNot Available", 
                                   font=('Arial', 14), fg='#7f8c8d', bg='#34495e',
                                   width=40, height=15)
        self.video_label.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Controls
        controls_frame = tk.Frame(live_frame, bg='#2c3e50')
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(controls_frame, text="📸 Capture", command=self.capture_photo,
                 bg='#3498db', fg='white', font=('Arial', 10)).pack(side='left', padx=5)
        
        tk.Button(controls_frame, text="🔍 Test Recognition", command=self.test_recognition,
                 bg='#9b59b6', fg='white', font=('Arial', 10)).pack(side='left', padx=5)
        
    def setup_user_panel(self):
        """Setup user management panel."""
        user_frame = tk.LabelFrame(self.root, text="👥 User Management", 
                                  font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        user_frame.pack(fill='x', padx=10, pady=5)
        
        # User list
        list_frame = tk.Frame(user_frame, bg='#2c3e50')
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Treeview for users
        columns = ('ID', 'Name', 'Email', 'Department', 'Last Seen', 'Status')
        self.user_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=4)
        
        for col in columns:
            self.user_tree.heading(col, text=col)
            self.user_tree.column(col, width=100)
        
        # Scrollbar for user list
        user_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.user_tree.yview)
        self.user_tree.configure(yscrollcommand=user_scrollbar.set)
        
        self.user_tree.pack(side='left', fill='both', expand=True)
        user_scrollbar.pack(side='right', fill='y')
        
        # User actions - better layout
        actions_frame = tk.Frame(user_frame, bg='#2c3e50')
        actions_frame.pack(fill='x', padx=10, pady=5)
        
        # First row of buttons
        row1 = tk.Frame(actions_frame, bg='#2c3e50')
        row1.pack(fill='x', pady=2)
        
        tk.Button(row1, text="➕ Add User", command=self.add_user,
                 bg='#27ae60', fg='white', font=('Arial', 9), padx=10, pady=3).pack(side='left', padx=2)
        
        tk.Button(row1, text="✏️ Edit User", command=self.edit_user,
                 bg='#f39c12', fg='white', font=('Arial', 9), padx=10, pady=3).pack(side='left', padx=2)
        
        tk.Button(row1, text="🗑️ Delete User", command=self.delete_user,
                 bg='#e74c3c', fg='white', font=('Arial', 9), padx=10, pady=3).pack(side='left', padx=2)
        
        tk.Button(row1, text="🔄 Refresh", command=self.refresh_users,
                 bg='#3498db', fg='white', font=('Arial', 9), padx=10, pady=3).pack(side='left', padx=2)
        
    def toggle_system(self):
        """Start or stop the system."""
        if not self.is_running:
            self.start_system()
        else:
            self.stop_system()
    
    def start_system(self):
        """Start the OCCUR-CAM system."""
        try:
            camera_source = self.camera_var.get()
            debug_mode = self.debug_var.get()
            
            self.system = OCCURCamSystem(camera_source=camera_source, debug=debug_mode, dashboard_callback=self)
            
            if self.system.initialize():
                if self.system.start_monitoring():
                    self.is_running = True
                    self.start_time = datetime.now()
                    self.start_btn.config(text="⏹️ STOP SYSTEM", bg='#e74c3c')
                    self.status_label.config(text="● RUNNING", fg='#27ae60')
                    self.add_activity("System started successfully")
                    
                    # Start update thread
                    self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
                    self.update_thread.start()
                else:
                    messagebox.showerror("Error", "Failed to start monitoring")
            else:
                messagebox.showerror("Error", "Failed to initialize system")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start system: {str(e)}")
    
    def stop_system(self):
        """Stop the OCCUR-CAM system."""
        try:
            if self.system:
                self.system.stop()
                self.is_running = False
                self.start_btn.config(text="▶️ START SYSTEM", bg='#27ae60')
                self.status_label.config(text="● STOPPED", fg='#e74c3c')
                self.add_activity("System stopped")
                
                # Clear system reference to allow reinitialization
                self.system = None
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop system: {str(e)}")
    
    def update_loop(self):
        """Update dashboard data."""
        user_list_counter = 0
        camera_feed_counter = 0
        
        while self.is_running and self.system:
            try:
                # Update statistics
                status = self.system.get_system_status()
                
                self.stats_vars['frames_processed'].set(str(status['statistics']['total_frames_processed']))
                self.stats_vars['faces_detected'].set(str(status['statistics']['total_faces_detected']))
                self.stats_vars['authentications'].set(str(status['statistics']['total_authentications']))
                self.stats_vars['registrations'].set(str(status['statistics']['total_registrations']))
                self.stats_vars['known_users'].set(str(status['known_users_count']))
                self.stats_vars['system_status'].set(status['status'].upper())
                
                # Update uptime
                if self.start_time:
                    uptime = datetime.now() - self.start_time
                    hours, remainder = divmod(uptime.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    self.stats_vars['uptime'].set(f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
                
                # Update user list every 5 seconds (less frequent)
                user_list_counter += 1
                if user_list_counter >= 5:
                    self.update_user_list()
                    user_list_counter = 0
                
                # Update camera feed every 2 seconds
                camera_feed_counter += 1
                if camera_feed_counter >= 2:
                    self.update_camera_feed()
                    camera_feed_counter = 0
                
                time.sleep(1)
                
            except Exception as e:
                self.add_activity(f"Update error: {str(e)}")
                time.sleep(1)
    
    def update_user_list(self):
        """Update the user list display."""
        try:
            # Clear existing items
            for item in self.user_tree.get_children():
                self.user_tree.delete(item)
            
            # Add users from system if running, otherwise show static list
            if self.system and hasattr(self.system, 'known_users') and self.is_running:
                for user_id, user_data in self.system.known_users.items():
                    self.user_tree.insert('', 'end', values=(
                        user_id,
                        f"{user_data.first_name} {user_data.last_name}",
                        user_data.email or "N/A",
                        user_data.department or "N/A",
                        "Online" if hasattr(user_data, 'last_seen') else "Unknown",
                        "Active"
                    ))
            else:
                # Show static user list when system is stopped
                self.load_static_users()
                    
        except Exception as e:
            self.add_activity(f"User list update error: {str(e)}")
    
    def update_camera_feed(self):
        """Update the live camera feed display."""
        try:
            if self.system and self.system.camera_manager:
                # Get current frame from camera
                camera = self.system.camera_manager.get_camera("CAM001")
                if camera and hasattr(camera, 'get_frame'):
                    frame = camera.get_frame()
                    if frame is not None:
                        # Resize frame for display
                        height, width = frame.shape[:2]
                        max_width, max_height = 400, 300
                        
                        if width > max_width or height > max_height:
                            scale = min(max_width/width, max_height/height)
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Convert BGR to RGB for tkinter
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Convert to PhotoImage
                        image = Image.fromarray(frame_rgb)
                        photo = ImageTk.PhotoImage(image)
                        
                        # Update label
                        self.video_label.configure(image=photo, text="")
                        self.video_label.image = photo  # Keep a reference
                    else:
                        self.video_label.configure(image="", text="Camera Feed\nNot Available")
                else:
                    self.video_label.configure(image="", text="Camera Feed\nNot Available")
        except Exception as e:
            self.video_label.configure(image="", text="Camera Feed\nError")
    
    def add_activity(self, message):
        """Add activity message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.activity_text.see(tk.END)
        
        # Limit log size
        lines = self.activity_text.get("1.0", tk.END).split('\n')
        if len(lines) > 100:
            self.activity_text.delete("1.0", f"{len(lines)-100}.0")
    
    def add_notification(self, message, notification_type="info"):
        """Add face detection notification."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding based on notification type
        if notification_type == "unknown":
            color = "#e74c3c"  # Red for unknown faces
            icon = "❓"
        elif notification_type == "recognized":
            color = "#27ae60"  # Green for recognized faces
            icon = "✅"
        elif notification_type == "registered":
            color = "#f39c12"  # Orange for new registrations
            icon = "🆕"
        else:
            color = "#3498db"  # Blue for info
            icon = "ℹ️"
        
        # Insert with color
        self.notification_text.insert(tk.END, f"[{timestamp}] {icon} {message}\n")
        
        # Apply color to the last line
        start_line = self.notification_text.index("end-2l")
        end_line = self.notification_text.index("end-1l")
        self.notification_text.tag_add("colored", start_line, end_line)
        self.notification_text.tag_configure("colored", foreground=color)
        
        self.notification_text.see(tk.END)
        
        # Keep only last 20 notifications
        lines = self.notification_text.get("1.0", tk.END).split('\n')
        if len(lines) > 20:
            self.notification_text.delete("1.0", f"{len(lines)-20}.0")
    
    def notify_unknown_face(self, user_id):
        """Notify about unknown face detection."""
        message = f"Unknown face detected! Generating user ID: {user_id}"
        self.add_notification(message, "unknown")
    
    def notify_recognized_face(self, user_id, confidence, user_info=None):
        """Notify about recognized face."""
        if user_info:
            name = f"{user_info.get('first_name', 'Unknown')} {user_info.get('last_name', 'User')}"
            message = f"Recognized: {name} (ID: {user_id}) - Confidence: {confidence:.2f}"
        else:
            message = f"Recognized user: {user_id} - Confidence: {confidence:.2f}"
        self.add_notification(message, "recognized")
    
    def notify_new_registration(self, user_id, user_info=None):
        """Notify about new user registration."""
        if user_info:
            name = f"{user_info.get('first_name', 'Unknown')} {user_info.get('last_name', 'User')}"
            message = f"New user registered: {name} (ID: {user_id})"
        else:
            message = f"New user registered: {user_id}"
        self.add_notification(message, "registered")
    
    def capture_photo(self):
        """Capture a photo from the camera."""
        if self.is_running and self.system:
            self.add_activity("Photo capture requested")
            # Implementation would capture current frame
        else:
            messagebox.showwarning("Warning", "System is not running")
    
    def test_recognition(self):
        """Test face recognition."""
        if self.is_running and self.system:
            self.add_activity("Face recognition test requested")
            # Implementation would trigger recognition test
        else:
            messagebox.showwarning("Warning", "System is not running")
    
    def add_user(self):
        """Add a new user with face capture."""
        dialog = UserDialog(self.root, "Add User")
        if dialog.result:
            try:
                # Show face capture dialog
                face_capture_dialog = FaceCaptureDialog(self.root, "Capture Face")
                if face_capture_dialog.result:
                    # Save user to database with face data
                    self.save_user_to_database(dialog.result, face_capture_dialog.result)
                    first_name = dialog.result.get('first_name', '')
                    last_name = dialog.result.get('last_name', '')
                    name = f"{first_name} {last_name}".strip()
                    self.add_activity(f"User added successfully: {name}")
                    
                    # Reload face recognizer if system is running
                    if self.is_running and self.system:
                        self.reload_face_recognizer()
                    
                    self.refresh_users()
                else:
                    self.add_activity("User registration cancelled - no face captured")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add user: {str(e)}")
                self.add_activity(f"Error adding user: {str(e)}")
    
    def edit_user(self):
        """Edit selected user."""
        selection = self.user_tree.selection()
        if selection:
            item = self.user_tree.item(selection[0])
            user_id = item['values'][0]
            dialog = UserDialog(self.root, "Edit User", user_id)
            if dialog.result:
                try:
                    # Update user in database
                    self.update_user_in_database(user_id, dialog.result)
                    first_name = dialog.result.get('first_name', '')
                    last_name = dialog.result.get('last_name', '')
                    name = f"{first_name} {last_name}".strip()
                    self.add_activity(f"User edited: {name}")
                    self.refresh_users()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to edit user: {str(e)}")
                    self.add_activity(f"Error editing user: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please select a user to edit")
    
    def delete_user(self):
        """Delete selected user."""
        selection = self.user_tree.selection()
        if selection:
            item = self.user_tree.item(selection[0])
            user_id = item['values'][0]
            if messagebox.askyesno("Confirm", f"Delete user {user_id}?"):
                try:
                    # Delete user from database
                    self.delete_user_from_database(user_id)
                    self.add_activity(f"User deleted: {user_id}")
                    self.refresh_users()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete user: {str(e)}")
                    self.add_activity(f"Error deleting user: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please select a user to delete")
    
    def refresh_users(self):
        """Refresh user list."""
        self.update_user_list()
        self.add_activity("User list refreshed")
    
    def load_static_users(self):
        """Load users from database when system is stopped."""
        try:
            from database.schemas.auth_schemas import Employee
            from config.database import get_auth_db
            
            with get_auth_db() as db:
                employees = db.query(Employee).filter(Employee.is_active == True).all()
                
                for employee in employees:
                    self.user_tree.insert('', 'end', values=(
                        employee.employee_id,
                        f"{employee.first_name} {employee.last_name}",
                        employee.email or "N/A",
                        employee.department or "N/A",
                        "Offline",
                        "Active" if employee.is_active else "Inactive"
                    ))
                    
        except Exception as e:
            self.add_activity(f"Error loading static users: {str(e)}")
    
    def save_user_to_database(self, user_data, face_data=None):
        """Save user to database with face data."""
        from database.schemas.auth_schemas import Employee
        from config.database import get_auth_db
        from datetime import datetime
        import json
        import os
        
        with get_auth_db() as db:
            # Generate unique user ID with microsecond precision
            user_id = f"USER_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Handle face data
            face_embedding = None
            face_photo_path = None
            
            if face_data and 'face_image' in face_data:
                # Save face image
                face_photo_path = f"storage/reference_photos/{user_id}.jpg"
                os.makedirs(os.path.dirname(face_photo_path), exist_ok=True)
                cv2.imwrite(face_photo_path, face_data['face_image'])
                
                # Generate face embedding
                try:
                    # Try to use system face engine if available
                    if self.is_running and self.system and hasattr(self.system, 'face_engine'):
                        face_embedding = self.system.face_engine.recognizer.generate_embedding(face_data['face_image'])
                    else:
                        # Create a temporary face recognizer for embedding generation
                        from core.ultra_simple_recognizer import UltraSimpleRecognizer
                        temp_recognizer = UltraSimpleRecognizer()
                        face_embedding = temp_recognizer.generate_embedding(face_data['face_image'])
                        temp_recognizer.cleanup()
                    
                    face_embedding = json.dumps(face_embedding.tolist()) if face_embedding is not None else None
                except Exception as e:
                    self.add_activity(f"Warning: Could not generate face embedding: {str(e)}")
                    face_embedding = None
            
            # Generate unique email - always make it unique
            user_email = user_data.get('email', '')
            if not user_email or user_email.strip() == '':
                user_email = f"{user_id}@example.com"
            else:
                # Always make email unique by appending timestamp
                timestamp = datetime.now().strftime('%H%M%S')
                user_email = f"{timestamp}_{user_email}"
            
            employee = Employee(
                employee_id=user_id,
                first_name=user_data.get('first_name', ''),
                last_name=user_data.get('last_name', ''),
                email=user_email,
                phone=user_data.get('phone', ''),
                department=user_data.get('department', ''),
                position=user_data.get('position', ''),
                is_active=True,
                face_embedding=face_embedding,
                face_photo_path=face_photo_path,
                created_at=datetime.now()
            )
            
            db.add(employee)
            db.commit()
            
            # Return success status
            return True
    
    def update_user_in_database(self, user_id, user_data):
        """Update user in database."""
        from database.schemas.auth_schemas import Employee
        from config.database import get_auth_db
        from datetime import datetime
        
        with get_auth_db() as db:
            employee = db.query(Employee).filter(Employee.employee_id == user_id).first()
            if employee:
                # Handle email uniqueness
                new_email = user_data.get('email', employee.email)
                if new_email and new_email != employee.email:
                    # Check if email already exists for another user
                    existing = db.query(Employee).filter(Employee.email == new_email, Employee.id != employee.id).first()
                    if existing:
                        # Make email unique by appending timestamp
                        timestamp = datetime.now().strftime('%H%M%S')
                        new_email = f"{timestamp}_{new_email}"
                
                employee.first_name = user_data.get('first_name', employee.first_name)
                employee.last_name = user_data.get('last_name', employee.last_name)
                employee.email = new_email
                employee.phone = user_data.get('phone', employee.phone)
                employee.department = user_data.get('department', employee.department)
                employee.position = user_data.get('position', employee.position)
                employee.updated_at = datetime.now()
                
                db.commit()
            else:
                raise Exception(f"User {user_id} not found")
    
    def delete_user_from_database(self, user_id):
        """Delete user from database."""
        from database.schemas.auth_schemas import Employee
        from config.database import get_auth_db
        
        with get_auth_db() as db:
            employee = db.query(Employee).filter(Employee.employee_id == user_id).first()
            if employee:
                employee.is_active = False
                db.commit()
            else:
                raise Exception(f"User {user_id} not found")
    
    def reload_face_recognizer(self):
        """Reload the face recognizer with updated user data."""
        try:
            if self.system and hasattr(self.system, 'face_engine'):
                # Reload the face recognizer
                self.system.face_engine.reload_recognizer()
                self.add_activity("Face recognizer reloaded with new users")
        except Exception as e:
            self.add_activity(f"Error reloading face recognizer: {str(e)}")
    
    def run(self):
        """Start the dashboard."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing."""
        if self.is_running:
            self.stop_system()
        self.root.destroy()

class UserDialog:
    def __init__(self, parent, title, user_id=None):
        self.result = None
        self.user_id = user_id
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x300")
        self.dialog.configure(bg='#2c3e50')
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        self.setup_ui()
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def setup_ui(self):
        """Setup dialog UI."""
        # Title
        title_label = tk.Label(self.dialog, text=self.dialog.title(), 
                              font=('Arial', 14, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        title_label.pack(pady=10)
        
        # Form frame
        form_frame = tk.Frame(self.dialog, bg='#2c3e50')
        form_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Form fields
        fields = [
            ("First Name:", "first_name"),
            ("Last Name:", "last_name"),
            ("Email:", "email"),
            ("Phone:", "phone"),
            ("Department:", "department"),
            ("Position:", "position")
        ]
        
        self.vars = {}
        # Load existing user data if editing
        existing_data = {}
        if self.user_id:
            try:
                from database.schemas.auth_schemas import Employee
                from config.database import get_auth_db
                with get_auth_db() as db:
                    employee = db.query(Employee).filter(Employee.employee_id == self.user_id).first()
                    if employee:
                        existing_data = {
                            'first_name': employee.first_name or '',
                            'last_name': employee.last_name or '',
                            'email': employee.email or '',
                            'phone': employee.phone or '',
                            'department': employee.department or '',
                            'position': employee.position or ''
                        }
            except Exception as e:
                print(f"Error loading user data: {e}")
        
        for i, (label, key) in enumerate(fields):
            tk.Label(form_frame, text=label, font=('Arial', 10), 
                   fg='#bdc3c7', bg='#2c3e50').grid(row=i, column=0, sticky='w', pady=5)
            
            var = tk.StringVar()
            # Set existing value if available
            if key in existing_data:
                var.set(existing_data[key])
            entry = tk.Entry(form_frame, textvariable=var, font=('Arial', 10), width=30)
            entry.grid(row=i, column=1, sticky='ew', pady=5, padx=(10, 0))
            self.vars[key] = var
        
        # Buttons
        button_frame = tk.Frame(self.dialog, bg='#2c3e50')
        button_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Button(button_frame, text="Save", command=self.save_user,
                 bg='#27ae60', fg='white', font=('Arial', 10), padx=20).pack(side='right', padx=5)
        
        tk.Button(button_frame, text="Cancel", command=self.cancel,
                 bg='#95a5a6', fg='white', font=('Arial', 10), padx=20).pack(side='right')
        
        # Configure grid
        form_frame.columnconfigure(1, weight=1)
    
    def save_user(self):
        """Save user data."""
        self.result = {key: var.get() for key, var in self.vars.items()}
        self.dialog.destroy()
    
    def cancel(self):
        """Cancel dialog."""
        self.dialog.destroy()

class FaceCaptureDialog:
    """Dialog for capturing or uploading face images."""
    
    def __init__(self, parent, title):
        self.result = None
        self.face_image = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("500x400")
        self.dialog.configure(bg='#2c3e50')
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        self.setup_ui()
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def setup_ui(self):
        """Setup dialog UI."""
        # Title
        title_label = tk.Label(self.dialog, text="Face Capture", 
                              font=('Arial', 14, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        title_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(self.dialog, 
                               text="Choose how to capture the user's face:",
                               font=('Arial', 10), fg='#bdc3c7', bg='#2c3e50')
        instructions.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(self.dialog, bg='#2c3e50')
        button_frame.pack(pady=20)
        
        # Capture from camera button
        tk.Button(button_frame, text="📷 Capture from Camera", 
                 command=self.capture_from_camera,
                 bg='#3498db', fg='white', font=('Arial', 12), 
                 padx=20, pady=10).pack(pady=10)
        
        # Upload image button
        tk.Button(button_frame, text="📁 Upload Image File", 
                 command=self.upload_image,
                 bg='#9b59b6', fg='white', font=('Arial', 12), 
                 padx=20, pady=10).pack(pady=10)
        
        # Cancel button
        tk.Button(button_frame, text="❌ Cancel", 
                 command=self.cancel,
                 bg='#e74c3c', fg='white', font=('Arial', 12), 
                 padx=20, pady=10).pack(pady=10)
    
    def capture_from_camera(self):
        """Capture face from camera with live feed."""
        try:
            # Create camera capture window - much larger size for better face capture
            self.camera_window = tk.Toplevel(self.dialog)
            self.camera_window.title("Camera Capture - Position your face in the frame")
            self.camera_window.geometry("1200x900")
            self.camera_window.configure(bg='#2c3e50')
            self.camera_window.transient(self.dialog)
            self.camera_window.grab_set()
            
            # Center the window
            self.camera_window.geometry("+%d+%d" % (self.dialog.winfo_rootx() + 50, self.dialog.winfo_rooty() + 50))
            
            # Initialize camera
            self.cap = None
            self.captured_image = None
            self.is_capturing = False
            
            self.setup_camera_ui()
            self.start_camera_feed()
            
        except Exception as e:
            messagebox.showerror("Error", f"Camera capture failed: {str(e)}")
    
    def setup_camera_ui(self):
        """Setup camera capture UI."""
        # Title
        title_label = tk.Label(self.camera_window, text="Camera Capture - Click 'Capture' to take photo", 
                              font=('Arial', 14, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        title_label.pack(pady=10)
        
        # Camera feed frame
        self.camera_frame = tk.Frame(self.camera_window, bg='#2c3e50')
        self.camera_frame.pack(pady=10)
        
        # Video label - proper size for video display
        self.video_label = tk.Label(self.camera_frame, bg='#34495e')
        self.video_label.pack(pady=10, padx=10)
        
        # Button frame
        button_frame = tk.Frame(self.camera_window, bg='#2c3e50')
        button_frame.pack(pady=10)
        
        # Capture button
        self.capture_btn = tk.Button(button_frame, text="📷 Capture Photo", 
                                   command=self.capture_photo,
                                   bg='#27ae60', fg='white', font=('Arial', 12), 
                                   padx=20, pady=10)
        self.capture_btn.pack(side=tk.LEFT, padx=10)
        
        # Cancel button
        tk.Button(button_frame, text="❌ Cancel", 
                 command=self.cancel_camera,
                 bg='#e74c3c', fg='white', font=('Arial', 12), 
                 padx=20, pady=10).pack(side=tk.LEFT, padx=10)
    
    def start_camera_feed(self):
        """Start camera feed."""
        try:
            # Try different camera backends
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            self.cap = None
            
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(0, backend)
                    if self.cap.isOpened():
                        # Set camera properties for better quality
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                        break
                except Exception as e:
                    print(f"Camera backend {backend} failed: {e}")
                    continue
            
            if self.cap is None or not self.cap.isOpened():
                raise Exception("Could not open camera. Please check if camera is connected and not being used by another application.")
            
            self.is_capturing = True
            self.update_camera_feed()
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Could not start camera: {str(e)}\n\nPlease ensure:\n1. Camera is connected\n2. No other application is using the camera\n3. Camera drivers are installed")
            self.cancel_camera()
    
    def update_camera_feed(self):
        """Update camera feed display."""
        if not self.is_capturing or self.cap is None:
            return
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Resize frame for display - proper size for camera capture
                height, width = frame.shape[:2]
                max_width = 800
                max_height = 600
                
                if width > max_width or height > max_height:
                    scale = min(max_width/width, max_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                
                # Update label
                self.video_label.configure(image=photo)
                self.video_label.image = photo  # Keep a reference
                
                # Store current frame for capture
                self.current_frame = frame.copy()
            
            # Schedule next update
            if self.is_capturing:
                self.camera_window.after(30, self.update_camera_feed)
                
        except Exception as e:
            print(f"Camera feed error: {e}")
            self.cancel_camera()
    
    def capture_photo(self):
        """Capture current frame."""
        try:
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                # Resize to standard face size
                face_resized = cv2.resize(self.current_frame, (112, 112))
                self.captured_image = face_resized
                self.face_image = face_resized
                self.result = {'face_image': self.face_image}
                
                # Show success message
                messagebox.showinfo("Success", "Photo captured successfully!")
                
                # Close camera window
                self.cancel_camera()
                
                # Close main dialog
                self.dialog.destroy()
            else:
                messagebox.showwarning("Warning", "No frame available for capture")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture photo: {str(e)}")
    
    def cancel_camera(self):
        """Cancel camera capture."""
        self.is_capturing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'camera_window'):
            self.camera_window.destroy()
    
    def upload_image(self):
        """Upload image file."""
        try:
            from tkinter import filedialog
            
            file_path = filedialog.askopenfilename(
                title="Select Face Image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Load and process image
                image = cv2.imread(file_path)
                if image is not None:
                    # Resize to standard face size
                    face_resized = cv2.resize(image, (112, 112))
                    self.face_image = face_resized
                    self.result = {'face_image': self.face_image}
                    
                    # Show success message
                    messagebox.showinfo("Success", "Image uploaded successfully!")
                    
                    # Close main dialog
                    self.dialog.destroy()
                else:
                    messagebox.showerror("Error", "Could not load image file. Please try a different format.")
            else:
                self.cancel()
                
        except Exception as e:
            messagebox.showerror("Error", f"Image upload failed: {str(e)}")
    
    def cancel(self):
        """Cancel dialog."""
        self.dialog.destroy()

def main():
    """Main entry point."""
    try:
        dashboard = OCCURCamDashboard()
        dashboard.run()
    except Exception as e:
        print(f"Dashboard error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
