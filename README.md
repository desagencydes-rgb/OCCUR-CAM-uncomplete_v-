# Accurency: Enterprise Biometric Authentication System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![InsightFace](https://img.shields.io/badge/InsightFace-ResNet50-green?style=for-the-badge&logo=opencv)
![Status](https://img.shields.io/badge/Status-Production%20Ready-orange?style=for-the-badge)

**Accurency** is a robust, enterprise-grade face recognition system designed for industrial environments. Unlike standard implementations, it utilizes a strictly typed, CPU-optimized architecture capable of handling multi-camera streams in varying lighting conditions without GPU dependencies.

---

## 🚀 Key Features

* **High-Accuracy Recognition:** Powered by **InsightFace (Buffalo_S)** with custom quantization for sub-second CPU inference.
* **Multi-Camera Orchestration:** Concurrent support for USB webcams, IP cameras (RTSP), and IVCam mobile streams.
* **Adaptive Lighting Engine:** Automatically optimizes image parameters for dark, bright, or mixed lighting environments using histogram equalization.
* **Terminal & GUI Interfaces:** Includes both a **Rich-based CLI** for headless servers and a **Tkinter Dashboard** for security personnel.
* **Enterprise Scale:** Architected to support **10,000+ employee identities** with <500ms lookup time.

---

## 🏗 System Architecture

The system implements a modular design pattern to decouple detection logic from stream management.

| Component | Responsibility |
| :--- | :--- |
| **Face Engine** | Wraps InsightFace for detection/recognition. Implements the **Strategy Pattern** for interchangeable models. |
| **Camera Manager** | Handles concurrent streams, health monitoring, and automatic reconnection. |
| **Auth Engine** | Manages session validity, anti-spoofing checks, and access logging. |
| **Lighting Optimizer** | Real-time image enhancement pipeline using LBP/HOG features. |

### Performance Benchmarks (Intel i5-8400 / CPU Only)

* **Face Detection:** ~200ms
* **Vector Recognition:** ~300ms
* **Total Latency:** ~500ms/frame
* **Memory Footprint:** ~2GB (Optimized)

---

## 🛠 Installation

### Prerequisites

* Python 3.11+
* CPU with AVX support (Standard on most modern processors)
* 4GB RAM minimum

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/desagencydes-rgb/Accurency-Biometric-Security.git
cd Accurency-Biometric-Security

# 2. Install dependencies (CPU Optimized)
pip install -r requirements.txt

# 3. Initialize the Database
python scripts/setup.py

# 4. Run the System
python main.py
```

---

## 📖 Usage

### Running the Dashboard (GUI)

For security desk monitoring:

```bash
python dashboard.py
```

### Running Headless (Server Mode)

For deployment on edge devices or servers:

```bash
python main.py --camera 1 --debug
```

### Terminal Commands (CLI)

* `c` - Show Camera Telemetry
* `a` - Live Authentication Logs
* `s` - System Health Status

---

## 📂 Project Structure

```plaintext
Accurency/
├── core/
│   ├── face_engine.py      # Detection & Recognition Logic
│   ├── camera_manager.py   # Multi-thread Stream Handling
│   └── lighting.py         # Image Enhancement Algorithms
├── api/                    # REST API for remote management
├── config/                 # YAML Configuration & Environmental Variables
├── database/               # SQLite/SQLAlchemy Schemas
├── dashboard.py            # Tkinter GUI Entry Point
└── main.py                 # CLI Application Entry Point
```

---

## ⚙️ Configuration

Edit `config/camera_config.yaml` to define your video sources:

```yaml
cameras:
  entrance_cam:
    source: "rtsp://192.168.1.55:554/stream1"
    type: "ip"
    location: "North Gate"
    fps: 15
```

---

## 🧪 Testing

The repository includes a comprehensive test suite covering unit logic and integration benchmarks.

```bash
# Run full test suite
python -m tests

# Run specific subsystem tests
python tests/test_face_recognition.py
```

---

## 📄 License

Proprietary Software. Developed by **D.E.S Agency R&D**. All rights reserved.

For licensing inquiries, contact: desagencydes@gmail.com

