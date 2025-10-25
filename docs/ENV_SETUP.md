# Environment setup and requirements

This document explains how to create and activate a Python virtual environment, and how to install the project's dependencies on Windows (PowerShell) and Unix (bash). It also points to the `requirements-full.txt` and `requirements-dev.txt` files.

## Overview
- `requirements-full.txt` — combined runtime requirements for the project and the `face-service` (CPU variant). Review this file for potential version conflicts if you install additional packages.
- `requirements-dev.txt` — development tools (linters, test runners, formatters).
- `face-service/requirements-cpu.txt` and `face-service/requirements-gpu.txt` — alternate dependency sets for that subproject. Choose one depending on whether you have GPU support.

## Windows (PowerShell) — recommended for local development
Open PowerShell (use an elevated prompt only if changing execution policy). Run the following commands from the repository root.

1) Create a virtual environment (use the built-in venv module):

```powershell
# create a .venv folder in the project root
python -m venv .venv

# If PowerShell blocks script execution, allow activation for this session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Sanity checks
python --version
pip --version
```

2) Upgrade pip and install requirements

```powershell
python -m pip install --upgrade pip setuptools wheel
# Install runtime requirements
python -m pip install -r requirements-full.txt

# If you plan development work (linters/tests), also install dev requirements
python -m pip install -r requirements-dev.txt
```

3) Face-service GPU note
- If you need GPU support for `face-service`, install CUDA-compatible Python packages and then replace the `onnxruntime` entry with `onnxruntime-gpu` or use `face-service/requirements-gpu.txt`:

```powershell
# example: install face-service GPU requirements (after ensuring CUDA drivers are present)
python -m pip install -r face-service\requirements-gpu.txt
```

4) Optional: freeze installed packages

```powershell
python -m pip freeze > requirements-locked.txt
```

## Unix / macOS (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-full.txt
python -m pip install -r requirements-dev.txt
```

## Creating reproducible environments
- For deterministic installs, create a lock file with `pip freeze` as shown above, or consider using `pip-tools` (pip-compile) / `poetry` / `pipenv` for lockfile management.

## Troubleshooting
- Version conflicts: if pip reports dependency conflicts, try to align versions between `requirements-full.txt` and any subproject files (for example `face-service/requirements-*.txt`).
- GPU runtime issues: ensure the proper CUDA toolkit and drivers are installed for your GPU and the `onnxruntime-gpu` build.
- If activation fails on Windows: re-run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force` in the same PowerShell session.

## Developer tips
- Use separate virtual environments for `face-service` if you need different packages (GPU vs CPU) to avoid conflicts.
- Keep `requirements-full.txt` minimal for runtime; prefer `requirements-dev.txt` for tools that contributors use.

## Next steps
- After setting up the environment run the test suite:

```powershell
# From project root, with venv activated
python -m pytest -q
```

If you want, I can now:
- Run `python -m pip install -r requirements-full.txt` in the activated environment (I cannot run it here without creating a venv), or
- Produce a `requirements-locked.txt` by installing in a temporary environment and freezing exact versions (this will take time and network access).

