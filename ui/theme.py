"""
UI Theme Configuration for OCCUR-CAM
Provides consistent theme and style settings across all UI components.
"""

import tkinter as tk
from tkinter import ttk
import logging

class OCCURCamTheme:
    # Color scheme
    COLORS = {
        'primary': '#2c3e50',      # Dark blue-gray
        'secondary': '#34495e',    # Lighter blue-gray
        'accent': '#3498db',       # Blue
        'success': '#27ae60',      # Green
        'warning': '#f39c12',      # Orange
        'danger': '#e74c3c',       # Red
        'light': '#ecf0f1',       # Light gray
        'text': '#bdc3c7',        # Text gray
        'bg': '#2c3e50'           # Background
    }

    # Font configurations
    FONTS = {
        'title': ('Arial', 20, 'bold'),
        'heading': ('Arial', 14, 'bold'),
        'subheading': ('Arial', 12, 'bold'),
        'normal': ('Arial', 10),
        'small': ('Arial', 9),
        'mono': ('Consolas', 9),
        'status': ('Arial', 12, 'bold'),  # For status indicators
        'stats_label': ('Arial', 10, 'bold'),  # For statistics labels
        'stats_value': ('Arial', 10)  # For statistics values
    }

    @staticmethod
    def setup_theme(root):
        """Initialize theme for the application."""
        try:
            # Configure root window
            root.configure(bg=OCCURCamTheme.COLORS['bg'])
            
            # Create and configure ttk style
            style = ttk.Style(root)
            
            # Try to use 'clam' theme as base (more customizable)
            try:
                style.theme_use('clam')
            except tk.TclError:
                logging.warning("Clam theme not available, using default")
                style.theme_use('default')
            
            # Configure ttk widgets
            style.configure('TFrame', background=OCCURCamTheme.COLORS['bg'])
            style.configure('TLabel', 
                          background=OCCURCamTheme.COLORS['bg'],
                          foreground=OCCURCamTheme.COLORS['light'])
            style.configure('TButton',
                          background=OCCURCamTheme.COLORS['accent'],
                          foreground=OCCURCamTheme.COLORS['light'])
            style.configure('Success.TButton',
                          background=OCCURCamTheme.COLORS['success'],
                          foreground=OCCURCamTheme.COLORS['light'])
            style.configure('Danger.TButton',
                          background=OCCURCamTheme.COLORS['danger'],
                          foreground=OCCURCamTheme.COLORS['light'])
            
            # Configure ttk entry fields
            style.configure('TEntry', 
                          fieldbackground=OCCURCamTheme.COLORS['secondary'],
                          foreground=OCCURCamTheme.COLORS['light'])
            
            return style
            
        except Exception as e:
            logging.error(f"Error setting up theme: {e}")
            return None

    @staticmethod
    def create_label(parent, text, font_type='normal', **kwargs):
        """Create a themed label."""
        return tk.Label(
            parent,
            text=text,
            font=OCCURCamTheme.FONTS[font_type],
            fg=OCCURCamTheme.COLORS['light'],
            bg=OCCURCamTheme.COLORS['bg'],
            **kwargs
        )

    @staticmethod
    def create_button(parent, text, command, style='primary', **kwargs):
        """Create a themed button."""
        if style == 'success':
            return ttk.Button(
                parent,
                text=text,
                command=command,
                style='Success.TButton',
                **kwargs
            )
        elif style == 'danger':
            return ttk.Button(
                parent,
                text=text,
                command=command,
                style='Danger.TButton',
                **kwargs
            )
        else:
            return ttk.Button(
                parent,
                text=text,
                command=command,
                **kwargs
            )

    @staticmethod
    def create_frame(parent, **kwargs):
        """Create a themed frame."""
        frame = ttk.Frame(parent, **kwargs)
        return frame

    @staticmethod
    def create_text_area(parent, height=10, width=40, **kwargs):
        """Create a themed text area."""
        text_area = tk.Text(
            parent,
            height=height,
            width=width,
            font=OCCURCamTheme.FONTS['mono'],
            bg=OCCURCamTheme.COLORS['secondary'],
            fg=OCCURCamTheme.COLORS['light'],
            **kwargs
        )
        return text_area