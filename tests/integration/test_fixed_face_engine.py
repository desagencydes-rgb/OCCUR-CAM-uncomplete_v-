#!/usr/bin/env python3
"""
Test the fixed face engine.
"""

def test_fixed_face_engine():
    """Test the fixed face engine."""
    try:
        print("🧪 TESTING FIXED FACE ENGINE")
        print("=" * 40)
        
        from core.face_engine import FaceEngine
        from models.face_models import FaceRecognitionConfig
        import cv2
        
        config = FaceRecognitionConfig()
        face_engine = FaceEngine(config)
        
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
                
                # Process with face engine
                analysis = face_engine.process_frame(face_image, "TEST_CAM")
                print(f"✅ Face detections: {len(analysis.face_detections)}")
                print(f"✅ Face recognitions: {len(analysis.face_recognitions)}")
                
                # Check recognition results
                recognized_faces = analysis.get_recognized_faces(0.3)
                unknown_faces = analysis.get_unknown_faces(0.3)
                
                print(f"✅ Recognized faces: {len(recognized_faces)}")
                print(f"✅ Unknown faces: {len(unknown_faces)}")
                
                for recognition in recognized_faces:
                    print(f"   🎉 RECOGNIZED: {recognition.employee_id} (confidence: {recognition.confidence:.4f})")
                
                for detection in unknown_faces:
                    print(f"   ❓ UNKNOWN: {detection.face_id}")
                
                if recognized_faces:
                    print(f"\n🎉 SUCCESS! Face recognition is now working!")
                else:
                    print(f"\n❌ Still not working. Need further investigation.")
            else:
                print(f"❌ Could not load face image")
        else:
            print(f"❌ No employee with face photo found")
        
        print(f"\n🎯 FIXED FACE ENGINE TEST COMPLETE!")
        
        # Cleanup
        face_engine.cleanup()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_face_engine()

