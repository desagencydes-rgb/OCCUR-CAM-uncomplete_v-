#!/usr/bin/env python3
"""
Debug face engine step by step.
"""

def debug_face_engine_step_by_step():
    """Debug face engine step by step."""
    try:
        print("🔍 DEBUGGING FACE ENGINE STEP BY STEP")
        print("=" * 50)
        
        from core.face_detector import get_face_detector
        from core.ultra_simple_recognizer import UltraSimpleRecognizer
        from models.face_models import FaceRecognitionConfig, FaceDetection, FaceRecognition
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
            frame = cv2.imread(employee.face_photo_path)
            if frame is not None:
                print(f"✅ Loaded face image: {employee.face_photo_path}")
                print(f"   Image shape: {frame.shape}")
                
                # Step 1: Detect faces
                detections = detector.detect_faces(frame)
                print(f"✅ Face detections: {len(detections)}")
                
                for i, detection in enumerate(detections):
                    print(f"   Face {i+1}: confidence = {detection.confidence:.4f}")
                    print(f"   Bbox: {detection.bbox}")
                
                if detections:
                    detection = detections[0]
                    
                    # Step 2: Extract face region
                    face_region = detector.extract_face_region(frame, detection)
                    print(f"✅ Face region extracted: {face_region.shape}")
                    print(f"   Face region size: {face_region.size}")
                    
                    # Step 3: Always use original frame
                    print(f"✅ Always using original frame for recognition")
                    print(f"   frame.shape[0] > 50: {frame.shape[0] > 50}")
                    print(f"   frame.shape[1] > 50: {frame.shape[1] > 50}")
                    
                    if frame.shape[0] > 50 and frame.shape[1] > 50:
                        print(f"   ✅ Using original frame")
                        processed_face = detector.preprocess_face(frame)
                    else:
                        print(f"   ❌ Frame too small, creating empty recognition")
                        recognition = FaceRecognition(
                            employee_id=None,
                            confidence=0.0,
                            face_detection=detection,
                            embedding=None,
                            processing_time=0.0
                        )
                        print(f"   Empty recognition created: {recognition.employee_id}")
                        return
                    
                    # Step 4: Recognize face
                    print(f"✅ Preprocessed face: {processed_face.shape}")
                    recognition = recognizer.recognize_face(processed_face, detection)
                    print(f"✅ Recognition result:")
                    print(f"   Employee ID: {recognition.employee_id}")
                    print(f"   Confidence: {recognition.confidence:.4f}")
                    print(f"   Processing time: {recognition.processing_time:.4f}")
                    
                    if recognition.employee_id:
                        print(f"   🎉 RECOGNITION SUCCESS!")
                    else:
                        print(f"   ❌ RECOGNITION FAILED!")
                else:
                    print(f"❌ No face detections")
            else:
                print(f"❌ Could not load face image")
        else:
            print(f"❌ No employee with face photo found")
        
        print(f"\n🎯 FACE ENGINE STEP BY STEP DEBUG COMPLETE!")
        
        # Cleanup
        recognizer.cleanup()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_face_engine_step_by_step()
