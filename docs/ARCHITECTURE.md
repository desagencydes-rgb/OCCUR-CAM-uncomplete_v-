## Architecture Overview

This file explains the major components, responsibilities, and the data/control flows across the repository. It is written to give engineers a quick mental model before they read the code.

Top-level components

- api/ — Flask (or similar) API layer exposing endpoints used by the dashboard and external clients. Key files: `api/main.py`, `api/auth.py`, `api/cameras.py`, `api/employees.py`.
- core/ — Recognition engines, camera managers, detectors, and orchestrators. This is the heart of the system with multiple recognizer implementations (simple, advanced, production, insightface, etc.). Key files: `core/face_recognizer.py`, `core/face_engine.py`, `core/camera_manager.py`, `core/camera_monitor.py`, and many recognizer variants.
- config/ — Central place for settings and camera configuration (e.g., `config/camera_config.yaml`, `config/settings.py`, `config/database.py`).
- models/ — Persistent data models and schemas used for employees and faces. For potential DB schema mapping and ORM usage.
- examples/ and face-service/ — Demos, standalone apps, and an alternate service packaging with its own `pyproject.toml`.
- docs/ — Documentation (this folder).

Control flow (request -> recognition)

1. Client or dashboard triggers an API endpoint in `api/` (rest call or websocket).
2. API code authenticates (see `api/auth.py`) and passes the request to camera manager or to recognizer via `core` modules.
3. `core/camera_manager.py` or `core/camera_monitor.py` fetches frames from camera(s) using OpenCV or other capture backends.
4. Frames are passed to face detectors (`core/face_detector.py`) which locate faces and crop them.
5. Crops are sent to a recognizer implementation in `core/` (many variants available). Recognition is typically a two-step process: embed (feature extraction) then compare to known embeddings (database in `models/` or persistent store).
6. Results are returned to API which logs events and notifies connected clients or stores events in DB (`database/` folder contains helpers).

Data flow (simplified ASCII)

Client ---> api/ ----> core/camera_manager ---> face_detector ---> recognizer ---> database/models

Component responsibility notes

- Recognizer implementations: Many recognizers exist (e.g., `production_face_recognizer.py`, `insightface_recognizer.py`, `simple_face_recognizer.py`) — they implement different trade-offs (accuracy, speed, GPU support). Each recognizer typically exposes a common method set (detect, embed, match) but the interface is not enforced by a typed abstract base class.
- Camera manager: handles capture, reconnection, and multiple streams. Real-time behaviour depends on thread/process model used in `core/camera_manager.py`.
- API layer: lightweight; authentication is in `api/auth.py`. Many endpoints are thin wrappers to core services.

Design observations

- Strengths: clear separation of API vs recognition code; multiple recognizer backends mean the project explored alternative approaches; example scripts and `face-service/` package show intent for deployment.
- Weaknesses: there is duplication across recognizer implementations and few enforced interfaces; global config access patterns and hard-coded paths appear in places; limited test coverage observed.

Next steps

- Add an interface/abstract base class for recognizers to standardize usage across the project.
- Consider a single CameraStream abstraction with lifecycle management and connection backoff policies.
- Add sequence diagrams for capture -> recognition -> notification flows.
