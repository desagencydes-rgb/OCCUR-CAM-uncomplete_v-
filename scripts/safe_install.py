#!/usr/bin/env python3
"""
Safe Installation Script for OCCUR-CAM
Handles virtualenv creation and dependencies installation with proper error handling.
"""

import sys
import os
import subprocess
import tempfile
import shutil
import venv
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import time

def ensure_base_requirements():
    """Install base requirements needed for this script."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", 
             str(Path(__file__).parent / "safe_install_requirements.txt")],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print("Failed to install base requirements:")
        print(e.stderr.decode())
        return False

def create_venv(venv_path: Path) -> bool:
    """Create a virtual environment if it doesn't exist."""
    if venv_path.exists():
        logging.info(f"Virtual environment already exists at {venv_path}")
        return True
        
    try:
        logging.info(f"Creating virtual environment at {venv_path}")
        venv.create(venv_path, with_pip=True)
        return True
    except Exception as e:
        logging.error(f"Failed to create virtual environment: {e}")
        return False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('install.log', mode='w', encoding='utf-8')
    ]
)

class SafeInstaller:
    def __init__(self, venv_path: Path):
        self.venv_path = venv_path
        self.python_exe = venv_path / "Scripts" / "python.exe"
        self.pip_exe = venv_path / "Scripts" / "pip.exe"
        self.temp_dir = Path("C:/tmp/occucam_install")
        self.original_temp = os.environ.get('TEMP')
        self.original_tmp = os.environ.get('TMP')
        
    def setup(self):
        """Setup installation environment."""
        # Create short temp directory if needed
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set short temp paths
        os.environ['TEMP'] = str(self.temp_dir)
        os.environ['TMP'] = str(self.temp_dir)
        
        logging.info(f"Using temporary directory: {self.temp_dir}")
        
    def cleanup(self):
        """Restore original environment."""
        if self.original_temp:
            os.environ['TEMP'] = self.original_temp
        if self.original_tmp:
            os.environ['TMP'] = self.original_tmp
            
        # Cleanup temp directory
        try:
            shutil.rmtree(self.temp_dir)
            logging.info("Cleaned up temporary directory")
        except Exception as e:
            logging.warning(f"Could not clean temporary directory: {e}")
    
    def run_pip(self, args: List[str], description: str) -> Tuple[int, str]:
        """Run pip with given arguments and return status and output."""
        cmd = [str(self.python_exe), "-m", "pip"] + args
        logging.info(f"Running: {description}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode, result.stdout + result.stderr
            
        except Exception as e:
            return 1, str(e)
    
    def install_package(self, package: str, description: str) -> bool:
        """Install a single package."""
        code, output = self.run_pip(
            ["install", "--no-cache-dir", "-v", package],
            f"Installing {description}"
        )
        
        if code != 0:
            logging.error(f"Failed to install {package}:")
            logging.error(output)
            return False
            
        logging.info(f"Successfully installed {description}")
        return True
    
    def install_requirements(self, req_file: Path) -> bool:
        """Install from requirements file."""
        code, output = self.run_pip(
            ["install", "--no-cache-dir", "-r", str(req_file)],
            "Installing remaining requirements"
        )
        
        if code != 0:
            logging.error("Failed to install requirements:")
            logging.error(output)
            return False
            
        logging.info("Successfully installed requirements")
        return True
    
    def verify_imports(self) -> bool:
        """Run import verification script."""
        script = Path(__file__).parent / "test_imports.py"
        
        if not script.exists():
            logging.error("test_imports.py not found")
            return False
        
        try:
            result = subprocess.run(
                [str(self.python_exe), str(script)],
                capture_output=True,
                text=True,
                check=False
            )
            
            logging.info("Import test results:")
            logging.info(result.stdout)
            
            if result.returncode != 0:
                logging.error("Import tests failed:")
                logging.error(result.stderr)
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error running import tests: {e}")
            return False

def main():
    """Main installation process."""
    # Ensure we have base requirements for this script
    if not ensure_base_requirements():
        return 1

    # Get project root and venv path
    project_root = Path(__file__).parent.parent
    venv_path = project_root / ".venv"
    
    # Create virtual environment if it doesn't exist
    if not create_venv(venv_path):
        return 1
    
    installer = SafeInstaller(venv_path)
    
    try:
        # Setup safe installation environment
        installer.setup()
        
        # Update pip first
        installer.run_pip(["install", "--upgrade", "pip"], "Upgrading pip")
        
        # Critical dependencies first (installed separately to avoid path issues)
        critical_packages = [
            ("onnxruntime>=1.18.0", "onnxruntime"),
            ("insightface==0.7.3", "insightface"),
            ("numpy==1.24.3", "numpy"),
            ("opencv-python==4.8.1.78", "opencv-python")
        ]
        
        for package, desc in critical_packages:
            if not installer.install_package(package, desc):
                logging.error(f"Failed to install {desc}. Aborting.")
                return 1
            time.sleep(1)  # Brief pause between installs
        
        # Install remaining requirements
        req_file = project_root / "requirements-full.txt"
        if not installer.install_requirements(req_file):
            return 1
        
        # Verify imports
        logging.info("\nVerifying imports...")
        if not installer.verify_imports():
            logging.warning("Some imports failed - check install.log for details")
        
        logging.info("\nInstallation completed!")
        return 0
        
    except Exception as e:
        logging.error(f"Installation failed: {e}")
        return 1
        
    finally:
        installer.cleanup()

if __name__ == "__main__":
    sys.exit(main())