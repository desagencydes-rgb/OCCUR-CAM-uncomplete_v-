#!/usr/bin/env python3
"""
Debug face extraction to find the real issue.
"""

def debug_face_extraction():
    """Debug face extraction to find the real issue."""
    try:
        print("🔍 DEBUGGING FACE EXTRACTION")
        print("=" * 40)
        
        from core.face_detector import get_face_detector
        from core.ultra_simple_recognizer import UltraSimpleRecognizer
        from models.face_models import FaceRecognitionConfig
        import cv2
        
        config = FaceRecognitionConfig()
        detector = get_face_detector(config)
        recognizer = UltraSimpleRecognizer()
        
        # Load a real face image
        from database.schemas.auth_schemas import Employee
        from config.database import get_auth_db
        
        with get_auth_db() as db:
            employee = db.query(Employee).filter(
                Employee.is_active == True,
                Employee.face_photo_path.isnot(None)
            ).first()
        
        if employee and employee.face_photo_path:
            face_image = cv2.imread(employee.face_photo_path)
            if face_image is not None:
                print(f"✅ Loaded face image: {employee.face_photo_path}")
                print(f"   Image shape: {face_image.shape}")
                
                # Step 1: Detect faces
                detections = detector.detect_faces(face_image)
                print(f"✅ Face detections: {len(detections)}")
                
                for i, detection in enumerate(detections):
                    print(f"   Face {i+1}: confidence = {detection.confidence:.4f}")
                    print(f"   Bbox: {detection.bbox}")
                
                if detections:
                    # Step 2: Extract face region
                    face_region = detector.extract_face_region(face_image, detections[0])
                    print(f"✅ Face region extracted: {face_region.shape}")
                    print(f"   Face region size: {face_region.size}")
                    
                    if face_region.size > 0:
                        # Step 3: Preprocess face
                        processed_face = detector.preprocess_face(face_region)
                        print(f"✅ Face preprocessed: {processed_face.shape}")
                        
                        # Step 4: Recognize face
                        recognition = recognizer.recognize_face(processed_face, detections[0])
                        print(f"✅ Recognition result:")
                        print(f"   Employee ID: {recognition.employee_id}")
                        print(f"   Confidence: {recognition.confidence:.4f}")
                        print(f"   Processing time: {recognition.processing_time:.4f}")
                        
                        if recognition.employee_id:
                            print(f"   🎉 RECOGNITION SUCCESS!")
                        else:
                            print(f"   ❌ RECOGNITION FAILED!")
                    else:
                        print(f"❌ Face region extraction failed - size = {face_region.size}")
                else:
                    print(f"❌ No face detections")
            else:
                print(f"❌ Could not load face image")
        else:
            print(f"❌ No employee with face photo found")
        
        print(f"\n🎯 FACE EXTRACTION DEBUG COMPLETE!")
        
        # Cleanup
        recognizer.cleanup()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_face_extraction()

