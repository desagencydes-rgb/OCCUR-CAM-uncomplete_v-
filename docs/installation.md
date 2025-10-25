## Installation and Setup

This guide covers common installation paths: Windows and Linux. It explains build tool requirements, GPU options, and environment configuration. Follow the quickstart in `README.md` for a simple run, and this page for deeper details.

Prerequisites

- Python 3.8 or 3.9 (3.10+ may work but verify before upgrading)
- Git
- C/C++ build tools (required to compile some native wheels)
  - Windows: Install "Build Tools for Visual Studio" (Desktop development with C++) — include MSVC v142 or newer, Windows SDK.
  - Ubuntu/Debian: `sudo apt-get install build-essential cmake pkg-config` and `sudo apt-get install libjpeg-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev` for some image libs.
- Optional (GPU): NVIDIA GPU + CUDA toolkit + cuDNN (match supported versions of the ML backends). Use official vendor install docs.

Windows-specific install (PowerShell)

1. Install Visual C++ Build Tools

- Download and run: https://visualstudio.microsoft.com/downloads/ -> "Build Tools for Visual Studio" -> select "C++ build tools" workload.

2. Create and activate a venv

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

3. Upgrade pip and wheel

```powershell
pip install --upgrade pip setuptools wheel
```

4. Install Python packages

```powershell
pip install -r requirements.txt
# If you need full options (examples, dev extras):
pip install -r requirements-full.txt
```

Notes on heavy dependencies

- Packages such as dlib, face-recognition, or compiled versions of insightface can require MSVC and long compile times. Consider using pre-built wheels or a conda environment where available.
- If using GPU-accelerated models, ensure the installed package version is compatible with your CUDA/cuDNN versions.

Linux quick install (Ubuntu example)

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential cmake git pkg-config libjpeg-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev
python3 -m venv .venv; source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Database

- The project includes `database/` helpers. Determine the database backend you will use (SQLite for local testing or PostgreSQL/MySQL for production). The repo likely assumes a simple local DB for demos.

Configuration

- Edit `config/camera_config.yaml` to list RTSP/HTTP camera streams and camera IDs.
- Edit `config/settings.py` for database URLs, logging, and feature flags.

Running the project

- Demos: `examples/standalone_demo.py`, `examples/demo.py`, `examples/simple_demo.py`
- Service: `face-service/app.py` can be used as a packaged service; the repo also contains `docker-compose.yml` for multi-container setups.

Troubleshooting common issues

- Build failures for native libs: Ensure Visual C++ Build Tools on Windows or build-essential on Linux.
- ImportError: missing compiled extension — try installing package wheels from PyPI or use conda.
- Camera not streaming: verify RTSP/HTTP feed via VLC or ffmpeg; check firewall and credentials.
- GPU vs CPU mismatch: use the CPU-only requirements files if you don't have a supported GPU.

Recommendations

- For Windows development, use the Visual Studio Build Tools and install pre-built wheels when available.
- For reproducible environments, consider using conda (especially if using dlib/insightface with GPUs) or Docker with prebuilt images.
