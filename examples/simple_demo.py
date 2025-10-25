#!/usr/bin/env python3
"""
OCCUR-CAM Simple Demo
A simple, non-looping demo of the OCCUR-CAM system.
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def simple_camera_demo():
    """Simple camera demo without terminal interface."""
    print("=" * 60)
    print("🎬 OCCUR-CAM Simple Demo")
    print("=" * 60)
    print("Press 'q' to quit, 's' to save snapshot")
    print("=" * 60)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No camera available")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    print("✅ Camera initialized")
    print("📹 Starting camera feed...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Add timestamp and info
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"OCCUR-CAM Demo - {timestamp}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw face detection area
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detection Area", 
                       (w//4, h//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("OCCUR-CAM Simple Demo", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save snapshot
                filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Snapshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        elapsed = time.time() - start_time
        if elapsed > 0:
            avg_fps = frame_count / elapsed
            print(f"\n📊 Demo Statistics:")
            print(f"   Frames captured: {frame_count}")
            print(f"   Duration: {elapsed:.1f}s")
            print(f"   Average FPS: {avg_fps:.1f}")
        
        return True

def simple_system_info():
    """Display simple system information."""
    print("\n💻 System Information")
    print("-" * 40)
    
    try:
        import platform
        import psutil
        
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        print(f"CPU: {psutil.cpu_count()} cores")
        print(f"RAM: {psutil.virtual_memory().total // (1024**3)} GB")
        print(f"OpenCV: {cv2.__version__}")
        
        # Check camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera: {width}x{height} @ {fps:.1f} FPS")
            cap.release()
        else:
            print("Camera: Not available")
        
        return True
        
    except Exception as e:
        print(f"❌ System info error: {e}")
        return False

def simple_database_test():
    """Simple database test."""
    print("\n🗄️  Database Test")
    print("-" * 40)
    
    try:
        from config.database import test_connections, check_database_health
        
        # Test connections
        if test_connections():
            print("✅ Database connections: OK")
        else:
            print("❌ Database connections: FAILED")
            return False
        
        # Check health
        health = check_database_health()
        print(f"📊 Database Health: {health}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def main():
    """Run simple OCCUR-CAM demo."""
    print("=" * 60)
    print("🎬 OCCUR-CAM SIMPLE DEMO")
    print("=" * 60)
    print("This demo shows OCCUR-CAM without the terminal interface")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("System Information", simple_system_info),
        ("Database Test", simple_database_test),
        ("Camera Demo", simple_camera_demo)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🚀 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: COMPLETED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEMO SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} demos passed")
    
    if passed == total:
        print("\n🎉 All demos completed successfully!")
        print("OCCUR-CAM is working perfectly!")
    else:
        print(f"\n⚠️  {total - passed} demo(s) failed.")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Use this simple demo: python simple_demo.py")
    print("2. Or run the full system: python main.py --test-mode")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
