#!/usr/bin/env python3
"""
OCCUR-CAM Demo
Demonstrates the system without AI dependencies.
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

def demo_camera():
    """Demonstrate camera functionality."""
    print("🎥 Camera Demo")
    print("-" * 40)
    
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
    print("📹 Press 'q' to quit, 's' to save snapshot")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Add timestamp
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
            
            # Draw face detection placeholder
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detection Area", 
                       (w//4, h//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("OCCUR-CAM Demo", frame)
            
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

def demo_database():
    """Demonstrate database functionality."""
    print("\n🗄️  Database Demo")
    print("-" * 40)
    
    try:
        from config.database import test_connections, check_database_health
        from database.migrations import create_all_tables, seed_initial_data
        
        # Test connections
        if test_connections():
            print("✅ Database connections: OK")
        else:
            print("❌ Database connections: FAILED")
            return False
        
        # Check health
        health = check_database_health()
        print(f"📊 Database Health: {health}")
        
        # Create tables
        if create_all_tables():
            print("✅ Database tables: OK")
        else:
            print("❌ Database tables: FAILED")
            return False
        
        # Seed data
        if seed_initial_data():
            print("✅ Sample data: OK")
        else:
            print("❌ Sample data: FAILED")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def demo_system_info():
    """Display system information."""
    print("\n💻 System Information")
    print("-" * 40)
    
    try:
        import platform
        import psutil
        
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        print(f"CPU: {psutil.cpu_count()} cores")
        print(f"RAM: {psutil.virtual_memory().total // (1024**3)} GB")
        
        # Check OpenCV
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

def main():
    """Run OCCUR-CAM demo."""
    print("=" * 60)
    print("🎬 OCCUR-CAM DEMO")
    print("=" * 60)
    print("Demonstrating OCCUR-CAM system capabilities")
    print("(Without AI dependencies)")
    print("=" * 60)
    
    demos = [
        ("System Information", demo_system_info),
        ("Database Operations", demo_database),
        ("Camera Feed", demo_camera)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        print(f"\n🚀 Running: {demo_name}")
        try:
            result = demo_func()
            results.append((demo_name, result))
            if result:
                print(f"✅ {demo_name}: COMPLETED")
            else:
                print(f"❌ {demo_name}: FAILED")
        except Exception as e:
            print(f"❌ {demo_name}: ERROR - {e}")
            results.append((demo_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEMO SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for demo_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {demo_name}")
    
    print(f"\nOverall: {passed}/{total} demos passed")
    
    if passed == total:
        print("\n🎉 All demos completed successfully!")
        print("OCCUR-CAM is ready for AI integration.")
    else:
        print(f"\n⚠️  {total - passed} demo(s) failed.")
        print("Check the errors above for troubleshooting.")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Install AI dependencies: pip install insightface")
    print("2. Run full system: python main.py")
    print("3. Run tests: python -m tests")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
