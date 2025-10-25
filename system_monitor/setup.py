#!/usr/bin/env python3
"""
Install script for OCCUR-CAM System Monitor.
This script ensures all required dependencies are installed.
"""

import sys
import subprocess
import pkg_resources
from pathlib import Path

def install_requirements():
    """Install system monitor requirements."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements_file)
        ])
        print("Successfully installed system monitor dependencies")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def verify_installation():
    """Verify all required packages are installed."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        # Read requirements
        with open(requirements_file) as f:
            requirements = pkg_resources.parse_requirements(f)
        
        # Check each requirement
        missing = []
        for req in requirements:
            try:
                pkg_resources.require(str(req))
            except (pkg_resources.DistributionNotFound, 
                    pkg_resources.VersionConflict):
                missing.append(str(req))
        
        if missing:
            print("Missing required packages:")
            for pkg in missing:
                print(f"  - {pkg}")
            return False
            
        print("All required packages are installed")
        return True
        
    except Exception as e:
        print(f"Error verifying installation: {e}")
        return False

def main():
    """Main entry point."""
    print("Installing OCCUR-CAM System Monitor dependencies...")
    
    if install_requirements() and verify_installation():
        print("\nSystem Monitor setup complete!")
        print("You can now run the monitor with: python monitor_dashboard.py")
        return 0
    else:
        print("\nSystem Monitor setup failed!")
        print("Please check the error messages above and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())