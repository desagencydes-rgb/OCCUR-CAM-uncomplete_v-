#!/usr/bin/env python3
"""
Run Script - Ensures proper Python path and environment setup before running any script
"""

import os
import sys
from pathlib import Path
import subprocess
from dotenv import load_dotenv

def setup_environment():
    """Setup the Python environment properly."""
    # Get the project root directory (parent of this script)
    project_root = Path(__file__).parent.parent.absolute()
    
    # Add project root to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Load environment variables from .env file
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
    
    # Ensure PYTHONPATH includes project root
    python_path = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    if str(project_root) not in python_path:
        python_path.insert(0, str(project_root))
        os.environ['PYTHONPATH'] = os.pathsep.join(python_path)

def run_script(script_path: str):
    """Run a Python script with proper environment setup."""
    setup_environment()
    
    # Convert script path to absolute path
    script_path = Path(script_path).absolute()
    
    if not script_path.exists():
        print(f"Error: Script {script_path} not found!")
        sys.exit(1)
    
    print(f"\nRunning script: {script_path}")
    print("=" * 80)
    
    try:
        # Use subprocess to run the script with real-time output
        import subprocess
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        while True:
            # Read stdout
            output = process.stdout.readline()
            if output:
                print(output.rstrip())
            
            # Read stderr
            error = process.stderr.readline()
            if error:
                print(error.rstrip(), file=sys.stderr)
            
            # Check if process has finished
            if output == '' and error == '' and process.poll() is not None:
                break
        
        # Get return code
        return_code = process.poll()
        if return_code != 0:
            print(f"\nScript failed with return code: {return_code}")
            sys.exit(return_code)
            
    except Exception as e:
        print(f"Error running {script_path}:")
        print(e)
        sys.exit(1)
    finally:
        print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <script_path>")
        sys.exit(1)
    
    run_script(sys.argv[1])