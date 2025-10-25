## Risks, Code Smells, and Recommendations

This document lists risks discovered during a documentation-only audit, the likely causes (often time pressure and early learning), and prioritized recommendations. The goal is to be constructive and practical.

High-level risks

1. Privacy, legal and ethical risk
   - Face recognition systems can violate privacy laws and civil liberties if used without consent or proper controls. In many jurisdictions, processing biometric data requires explicit consent and strong protection.
   - Recommendation: Add a compliance checklist for target jurisdiction(s), anonymize or encrypt stored images/embeddings, and minimize retention.

2. Security risk
   - Exposed APIs or default credentials can leak identity data. The project contains an `api/` layer — ensure authentication and rate-limiting are enforced in production.
   - Recommendation: Harden APIs, add HTTPS/TLS, rotate keys, and restrict network access to cameras.

3. Build and dependency fragility
   - Native builds (dlib, OpenCV, insightface) can be brittle across OSes and Python versions.
   - Recommendation: Provide pinned dependency sets, pre-built wheels or Docker images, and document required compiler toolchains.

Code smells and maintainability issues

- Duplication of recognizer logic across multiple files. This increases maintenance burden and chance of inconsistent bug fixes.
  - Fix: Introduce an abstract Recognizer interface and refactor common code into a base class.

- Global configuration access and hard-coded paths. Makes tests and deployment fragile.
  - Fix: Centralize configuration with a single loader that supports environment-variable overrides and validation (use pydantic or similar for schema validation).

- Low or missing test coverage. Many core paths appear to lack unit tests, especially for error conditions (camera disconnects, partial frames).
  - Fix: Add unit and integration tests as suggested in `docs/TESTS_AND_MODELS.md`.

- Large, monolithic modules (some `core/` files contain many responsibilities).
  - Fix: Extract small classes (CameraStream, Detector, Recognizer) with single responsibility.

Defending the original developer

From inspection the repository shows typical signs of a project created under tight time constraints:

- Evidence: multiple example scripts, many recognizer variants (experimentation), incomplete tests, pragmatic inline config.
- Reasonable defense: The developer prioritized delivering a working prototype across hardware variants (CPU/GPU) and multiple recognizer approaches. That often requires ad-hoc code duplication and selective optimization.
- Practical empathy: The developer likely worked under stress, learning new libraries (OpenCV, insightface, dlib, ML stacks), and had to prioritize run-time correctness over architectural purity.

Prioritized remediation (practical order)

1. Stabilize the environment: pin dependencies and publish a reproducible Docker image or conda env. Add a smoke test that verifies camera capture and a single recognition call.
2. Add basic tests for camera reconnection and for the primary recognizer used in production.
3. Centralize config and logging. Add structured logs and a runtime health endpoint.
4. Gradually refactor by extracting an AbstractRecognizer interface and move shared logic into a common package.

Operational recommendations

- Add monitoring and alerts (latency, error rate, camera offline count).
- Maintain a data-retention policy for face embeddings and raw images. Prefer storing only hashed/embedded data if legally acceptable.

When things break: recommended steps

1. If cameras fail: check network, verify RTSP with VLC/ffmpeg, and check camera credentials.
2. If native wheel build fails: switch to a known-good Python version or use the Docker image with prebuilt dependencies.
3. If recognition accuracy drops: check lighting conditions, alignment pipeline in `core/face_detector.py`, and embedding model versions.

Final note

This repo contains many good building blocks; the practical improvements suggested above will make it production-safe and easier to maintain without demanding a full rewrite.
