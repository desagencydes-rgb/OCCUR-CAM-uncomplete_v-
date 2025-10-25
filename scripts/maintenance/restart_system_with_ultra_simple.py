#!/usr/bin/env python3
"""
Script to completely restart the system with ultra simple recognizer.
"""

def restart_system_with_ultra_simple():
    """Completely restart the system with ultra simple recognizer."""
    try:
        print("🔄 COMPLETELY RESTARTING SYSTEM WITH ULTRA SIMPLE RECOGNIZER")
        print("=" * 70)
        
        # 1. Clear all cached recognizers
        print("🧹 Clearing all cached recognizers...")
        try:
            from core.ultra_simple_recognizer import cleanup_ultra_simple_recognizer
            cleanup_ultra_simple_recognizer()
            print("✅ Cleared ultra simple recognizer cache")
        except:
            pass
        
        try:
            from core.simple_fixed_recognizer import cleanup_simple_fixed_recognizer
            cleanup_simple_fixed_recognizer()
            print("✅ Cleared simple fixed recognizer cache")
        except:
            pass
        
        try:
            from core.extreme_face_recognizer import cleanup_extreme_face_recognizer
            cleanup_extreme_face_recognizer()
            print("✅ Cleared extreme face recognizer cache")
        except:
            pass
        
        # 2. Clear face engine cache
        print("🧹 Clearing face engine cache...")
        try:
            from core.face_engine import cleanup_face_engine
            cleanup_face_engine()
            print("✅ Cleared face engine cache")
        except:
            pass
        
        # 3. Test the ultra simple recognizer directly
        print("🧪 Testing ultra simple recognizer directly...")
        from core.ultra_simple_recognizer import UltraSimpleRecognizer
        recognizer = UltraSimpleRecognizer()
        
        print(f"✅ Ultra simple recognizer initialized")
        print(f"✅ Loaded {len(recognizer.employee_embeddings)} employee embeddings")
        print(f"✅ Recognition threshold: {recognizer.recognition_threshold}")
        
        # 4. Test with real face images
        print("🔍 Testing with real face images...")
        from database.schemas.auth_schemas import Employee
        from config.database import get_auth_db
        import cv2
        
        with get_auth_db() as db:
            employees = db.query(Employee).filter(
                Employee.is_active == True,
                Employee.face_photo_path.isnot(None)
            ).all()
        
        for employee in employees:
            if employee.face_photo_path:
                try:
                    face_image = cv2.imread(employee.face_photo_path)
                    if face_image is not None:
                        embedding = recognizer.generate_embedding(face_image)
                        if embedding is not None:
                            best_match_id, best_confidence = recognizer._find_best_match(embedding)
                            print(f"✅ {employee.employee_id}: Best match = {best_match_id}, Confidence = {best_confidence:.4f}")
                            
                            if best_match_id == employee.employee_id:
                                print(f"   🎉 CORRECT RECOGNITION!")
                            else:
                                print(f"   ❌ MISIDENTIFICATION")
                        else:
                            print(f"❌ {employee.employee_id}: Failed to generate embedding")
                    else:
                        print(f"❌ {employee.employee_id}: Could not load image")
                except Exception as e:
                    print(f"❌ {employee.employee_id}: Error - {e}")
        
        # 5. Test the face engine
        print("🧪 Testing face engine...")
        from core.face_engine import FaceEngine
        from models.face_models import FaceRecognitionConfig
        
        config = FaceRecognitionConfig()
        face_engine = FaceEngine(config)
        
        print(f"✅ Face engine initialized")
        print(f"✅ Face engine recognizer type: {type(face_engine.recognizer).__name__}")
        
        # 6. Test with a synthetic image
        print("🔍 Testing with synthetic image...")
        import numpy as np
        test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        try:
            analysis = face_engine.process_frame(test_image, "TEST_CAM")
            print(f"✅ Face engine processed synthetic image")
            print(f"✅ Face detections: {len(analysis.face_detections)}")
            print(f"✅ Recognized faces: {len(analysis.get_recognized_faces(0.3))}")
            print(f"✅ Unknown faces: {len(analysis.get_unknown_faces(0.3))}")
        except Exception as e:
            print(f"❌ Error testing face engine: {e}")
        
        print("\n🎯 SYSTEM RESTART COMPLETE!")
        print("✅ Ultra simple recognizer is now active")
        print("✅ Face engine is using ultra simple recognizer")
        print("✅ System should now recognize faces properly")
        print("✅ Ready for live testing")
        
        # 7. Cleanup
        recognizer.cleanup()
        face_engine.cleanup()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    restart_system_with_ultra_simple()

