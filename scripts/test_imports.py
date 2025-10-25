import sys
import importlib

# A small list of modules and packages to sanity-check after installation.
MODULES = [
    # core libs
    'numpy', 'cv2', 'PIL', 'yaml', 'sqlalchemy', 'alembic',
    # ai/face
    'insightface', 'onnxruntime', 'torch',
    # utils
    'psutil', 'pandas',
    # web
    'fastapi',
]

errors = []
for mod in MODULES:
    try:
        importlib.import_module(mod)
        print(f'OK: imported {mod}')
    except Exception as e:
        print(f'ERROR: could not import {mod}: {e}')
        errors.append((mod, str(e)))

# Also attempt to import the application's top-level packages where possible
TOP_PACKAGES = ['core', 'api', 'config', 'database', 'models']
for pkg in TOP_PACKAGES:
    try:
        importlib.import_module(pkg)
        print(f'OK: imported package {pkg}')
    except Exception as e:
        print(f'WARN: could not import package {pkg}: {e}')
        # Don't fail on package import errors yet — they might depend on moved scripts

if errors:
    print('\nSome runtime imports failed. See errors above.')
    sys.exit(3)

print('\nAll critical imports succeeded.')
sys.exit(0)
