## API & Codebase Overview

This document gives a quick reference to the public-facing API (files under `api/`) and the main code modules in `core/` and `config/`. It is not a full automated doc but a high-value human-oriented map.

API Endpoints (high level)

- `api/main.py` — likely entry point to mount routes and run the API server. Look for app run logic.
- `api/auth.py` — authentication helpers, session/token handling, and decorators used to protect endpoints.
- `api/cameras.py` — endpoints to list camera streams, add/remove cameras, and to fetch snapshots or status.
- `api/employees.py` — endpoints for CRUD operations on employees (create employee, upload face samples, list known identities).

Core modules

- `core/camera_manager.py` — manages camera lifecycle, capture threads, connection retries and frame buffering.
- `core/camera_monitor.py` — monitoring utilities and possible health-checks for streams.
- `core/face_detector.py` — pre-processing and face detection logic that extracts bounding boxes.
- Recognizers in `core/`: multiple implementations: `simple_face_recognizer.py`, `proper_face_recognizer.py`, `production_face_recognizer.py`, `insightface_recognizer.py`, `final_face_recognizer.py`, `ultra_simple_recognizer.py`, etc. Each provides different detection/recognition algorithms.
- `core/face_engine.py` — orchestration between detector and recognizer; provides higher-level methods consumed by the API.
- `core/database_manager.py` — DB operations for storing embeddings, face metadata, and event logs.

Database folder

- `database/auth_db.py` — authentication DB helpers.

Scripts and demos

- `examples/standalone_demo.py` — quick standalone demo that uses local camera/video and shows recognized faces.
- `examples/simple_accuracy_test.py` — testing harness for recognition accuracy (run if you want to benchmark models).

How to locate function/class responsibilities

- Search for common exported names: `match`, `recognize`, `detect`, `get_frame`, `start_stream` to find entry points.
- Many recognizers implement similar named methods — reading `core/face_engine.py` is a good way to find the canonical usage pattern used by the API.

Notes on observability and logging

- The project relies on simple logging; consider adding structured logging (JSON) and standardized event schema for identity events.

Extending the API

- To add a new endpoint, follow existing `api/` file patterns. Use `api/auth.py` to protect the route and call core engine methods to process frames/requests.
