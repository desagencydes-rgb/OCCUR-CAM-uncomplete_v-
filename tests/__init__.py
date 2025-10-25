"""
OCCUR-CAM Test Suite
Comprehensive test suite for the OCCUR-CAM system.
"""

from .test_system import run_tests as run_system_tests
from .test_camera import run_camera_tests
from .test_face_recognition import run_face_recognition_tests

def run_all_tests():
    """Run all test suites."""
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("OCCUR-CAM COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Run all test suites
    results = []
    
    print("\n1. Running System Tests...")
    results.append(("System Tests", run_system_tests()))
    
    print("\n2. Running Camera Tests...")
    results.append(("Camera Tests", run_camera_tests()))
    
    print("\n3. Running Face Recognition Tests...")
    results.append(("Face Recognition Tests", run_face_recognition_tests()))
    
    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL TEST SUMMARY")
    print("=" * 80)
    
    total_passed = 0
    total_tests = len(results)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
        if passed:
            total_passed += 1
    
    print(f"\nOverall: {total_passed}/{total_tests} test suites passed")
    print(f"Success rate: {(total_passed / total_tests * 100):.1f}%")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! OCCUR-CAM is ready for use.")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test suite(s) failed. Please check the logs.")
    
    print("=" * 80)
    
    return total_passed == total_tests

if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
