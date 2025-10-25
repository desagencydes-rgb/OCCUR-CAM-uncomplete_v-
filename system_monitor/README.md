# OCCUR-CAM System Monitor

A standalone monitoring and health management system for the OCCUR-CAM face recognition system.

## Features

- Real-time system resource monitoring
- Camera health tracking and automatic recovery
- Face recognition performance monitoring
- Automatic resource optimization
- Interactive terminal dashboard
- Comprehensive error logging

## Installation

1. Make sure you have Python 3.8+ installed
2. Run the setup script:
```bash
python setup.py
```

## Usage

Start the monitoring dashboard:
```bash
python monitor_dashboard.py
```

### Dashboard Controls

- `q` - Quit the dashboard
- `r` - Refresh display
- `c` - Trigger cleanup procedures

## Components

- `monitor_daemon.py` - Core system monitoring
- `camera_monitor.py` - Camera health monitoring
- `face_monitor.py` - Face recognition monitoring
- `monitor_dashboard.py` - Terminal UI

## Dependencies

See `requirements.txt` for the full list of dependencies. Key requirements:

- psutil - System monitoring
- opencv-python - Camera frame processing
- numpy - Array operations
- windows-curses (Windows only) - Terminal UI