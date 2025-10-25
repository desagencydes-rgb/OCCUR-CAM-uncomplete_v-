## OCCUR-CAM — Project Documentation

This repository contains a face-recognition and multi-camera monitoring system. The codebase mixes a Python backend API, multiple recognizer implementations, camera managers, and demos. All rights are reserved.

This README is intentionally comprehensive and points to deeper documentation in the `docs/` folder. It contains an installation quickstart, high-level architecture, risks and mitigations, and pointers to tests and modelization diagrams.

High-level contents
- Overview and purpose
- Quick installation (Windows + Linux)
- Requirements (software, hardware, build tools)
- Running demos and services
- Troubleshooting and known issues
- Links to detailed docs (architecture, API, tests, risks)

Important note: This README and the docs directory are documentation-only artifacts. No source code was modified.

Quickstart (Windows PowerShell)

1. Create and activate a Python virtual environment (Python 3.8+ recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install build tools and dependencies (see `docs/INSTALLATION.md` for full guidance). On Windows you'll usually need Visual C++ Build Tools (see docs), and optionally CUDA/toolkit for GPU builds.

3. Install Python dependencies (choose CPU or GPU set as appropriate):

```powershell
pip install -r requirements.txt
# or for more complete dev/runtime requirements
pip install -r requirements-full.txt
```

4. Configure `config/camera_config.yaml` and `config/settings.py` for your environment. The repository includes example camera configuration in `config/camera_config.yaml`.

5. Run a demo locally:

```powershell
python examples/standalone_demo.py
```

Where to read more
- Architecture and component diagrams: `docs/ARCHITECTURE.md`
- Full installation instructions: `docs/INSTALLATION.md`
- API and codebase overview: `docs/API_DOCUMENTATION.md`
- Tests, modelization and test artifacts: `docs/TESTS_AND_MODELS.md`
- Risks, mitigations and recommendations: `docs/RISKS_AND_RECOMMENDATIONS.md`

Policies and safety

This project performs face detection and recognition. Before deploying, ensure you understand local laws and privacy regulations (e.g., GDPR, CCPA). The docs contain a dedicated section covering data retention, consent, bias and model performance considerations.

Developer context and empathy

During audit we observed many pragmatic choices in the codebase that are typical for time-pressed, first-time efforts (see `docs/RISKS_AND_RECOMMENDATIONS.md`). The goal of this documentation is to explain the current shape, defend the developer's choices with context, and provide prioritized, practical remediation steps.

How you can help
- If you want the repository to contain automated diagrams or packaged docs (Sphinx, mkdocs), I can add a docs scaffold and scripts. This initial pass intentionally avoids code changes.

License & contact
Refer to `Git/LICENSE.txt` and `face-service/LICENSE` for license information. For questions please open an issue in your tracker or contact the original maintainers.

---
Generated docs folder with deeper guides is in `docs/`.
# OCCUR-CAM AI Authentication System

A robust, enterprise-grade face recognition authentication system designed for industrial environments with multiple entry points and varying lighting conditions.

## 🚀 Features

- **High-Accuracy Face Recognition**: Powered by InsightFace with CPU optimization
- **Multi-Camera Support**: USB webcams, IP cameras, IVCam mobile app, RTSP streams
- **All Lighting Conditions**: Automatic optimization for very dark, low-light, normal, bright, and mixed lighting
- **Real-Time Processing**: Optimized for CPU-only operation with 15 FPS processing
- **Enterprise Scale**: Support for 10,000+ employees across multiple sites
- **Terminal Interface**: Beautiful Rich-based terminal UI for monitoring and management
- **Health Monitoring**: Real-time camera and system health tracking
- **Comprehensive Logging**: Detailed authentication logs and system metrics

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or compatible camera
- 4GB+ RAM (8GB recommended)
- CPU with AVX support (for optimal performance)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd OCCUR-CAM
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup the system**
   ```bash
   python scripts/setup.py
   ```

4. **Start the application**
   ```bash
   python main.py
   ```

### Advanced Setup

1. **Custom configuration**
   ```bash
   # Copy environment template
   cp env.example .env
   
   # Edit configuration
   nano .env
   ```

2. **Database setup**
   ```bash
   # Create database tables
   python -c "from database.migrations import create_all_tables; create_all_tests()"
   
   # Seed initial data
   python -c "from database.migrations import seed_initial_data; seed_initial_data()"
   ```

## 📖 Usage

### Basic Usage

```bash
# Start with default webcam
python main.py

# Start with specific camera
python main.py --camera 1

# Start in debug mode
python main.py --debug

# Setup system only
python main.py --setup
```

### Terminal Interface

The system provides a beautiful terminal interface with real-time monitoring:

- **Dashboard**: Live system status and statistics
- **Camera Monitor**: Real-time camera health and performance
- **Authentication Logs**: Live authentication attempts and results
- **System Health**: Overall system health and performance metrics

#### Terminal Commands

- `h` - Show help
- `q` - Quit application
- `r` - Refresh display
- `c` - Show camera details
- `a` - Show authentication details
- `s` - Show system status

### Configuration

#### Camera Configuration

Edit `config/camera_config.yaml` to configure cameras:

```yaml
cameras:
  webcam_0:
    name: "Main Webcam"
    source: 0
    type: "usb"
    enabled: true
    width: 640
    height: 480
    fps: 15
    location: "Main Entrance"
```

#### Environment Variables

Key environment variables in `.env`:

```bash
# Face Recognition
FACE_RECOGNITION_MODEL=buffalo_s
FACE_DETECTION_THRESHOLD=0.5
FACE_RECOGNITION_THRESHOLD=0.6

# Camera Settings
DEFAULT_CAMERA_SOURCE=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=15

# System Settings
MAX_EMPLOYEES=10000
PROCESSING_THREADS=4
```

## 🧪 Testing

### Run All Tests

```bash
python -m tests
```

### Run Specific Test Suites

```bash
# System tests
python tests/test_system.py

# Camera tests
python tests/test_camera.py

# Face recognition tests
python tests/test_face_recognition.py
```

### Test Coverage

The test suite covers:
- System initialization and startup
- Camera detection and connection
- Face detection and recognition
- Lighting optimization
- Quality enhancement
- Performance benchmarks
- Error handling and recovery

## 📊 Performance

### CPU Optimization

The system is optimized for CPU-only operation:

- **Model**: Uses `buffalo_s` (smaller, faster model)
- **Detection Size**: 320x320 (reduced from 640x640)
- **Batch Size**: 1 (process one frame at a time)
- **FPS**: 15 FPS (optimized for CPU)
- **Resolution**: 640x480 (balanced quality/performance)

### Performance Benchmarks

On a typical CPU (Intel i5-8400):
- Face detection: ~200ms per frame
- Face recognition: ~300ms per frame
- Total processing: ~500ms per frame
- Memory usage: ~2GB
- CPU usage: ~60-80%

## 🏗️ Architecture

### Core Components

- **Face Engine**: InsightFace-based detection and recognition
- **Camera Manager**: Multi-camera support and health monitoring
- **Auth Engine**: Authentication logic and session management
- **Lighting Optimizer**: Automatic lighting condition detection and correction
- **Quality Enhancer**: Face quality assessment and improvement
- **Terminal Interface**: Rich-based monitoring and control interface

### Database Schema

- **Auth Database**: Employee data, authentication logs, sessions
- **Main Database**: Sites, cameras, alerts, system metrics
- **Separate Storage**: Face embeddings, reference photos, snapshots

## 🔧 Development

### Project Structure

```
OCCUR-CAM/
├── main.py                 # Main application entry point
├── core/                   # Core system components
│   ├── application.py      # Main application manager
│   ├── face_engine.py      # Face detection and recognition
│   ├── camera_manager.py   # Camera management
│   ├── auth_engine.py      # Authentication engine
│   └── terminal_interface.py # Terminal UI
├── config/                 # Configuration files
├── database/               # Database schemas and migrations
├── models/                 # Data models
├── tests/                  # Test suite
└── scripts/                # Setup and utility scripts
```

### Adding New Features

1. **New Camera Types**: Extend `CameraSource` class in `core/camera_manager.py`
2. **New Authentication Methods**: Extend `AuthenticationEngine` in `core/auth_engine.py`
3. **New UI Components**: Extend `TerminalInterface` in `core/terminal_interface.py`

## 🐛 Troubleshooting

### Common Issues

1. **No Camera Detected**
   ```bash
   # Check camera availability
   python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
   ```

2. **Face Recognition Not Working**
   ```bash
   # Check InsightFace installation
   python -c "import insightface; print('InsightFace OK')"
   ```

3. **Performance Issues**
   - Reduce camera resolution in config
   - Lower FPS setting
   - Check CPU usage and memory

4. **Database Errors**
   ```bash
   # Recreate database
   python scripts/setup.py --force
   ```

### Debug Mode

Enable debug mode for detailed logging:

```bash
python main.py --debug
```

Check logs in `logs/occur_cam.log` for detailed information.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For support and questions:
- Check the troubleshooting section
- Review the test suite for examples
- Open an issue on GitHub

## 🔄 Updates

### Version 1.0.0
- Initial release
- CPU-optimized face recognition
- Multi-camera support
- Terminal interface
- Comprehensive test suite
- Enterprise-grade logging and monitoring

---

**OCCUR-CAM** - Reliable face recognition authentication for industrial environments.
All rights are reserved.
