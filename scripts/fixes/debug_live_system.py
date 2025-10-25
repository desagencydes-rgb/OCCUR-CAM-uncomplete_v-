#!/usr/bin/env python3
"""
Debug the live system to understand why faces are still unknown.
"""

def debug_live_system():
    """Debug the live system to find the real issue."""
    try:
        print("🔍 DEBUGGING LIVE SYSTEM")
        print("=" * 40)
        
        # 1. Check what's actually in the database
        from database.schemas.auth_schemas import Employee
        from config.database import get_auth_db
        import json
        
        with get_auth_db() as db:
            employees = db.query(Employee).filter(
                Employee.is_active == True,
                Employee.face_embedding.isnot(None)
            ).all()
            
            print(f"📋 Database has {len(employees)} employees with embeddings")
            
            for employee in employees:
                print(f"   • {employee.employee_id}: {employee.first_name} {employee.last_name}")
                if employee.face_embedding:
                    try:
                        embedding_data = json.loads(employee.face_embedding)
                        print(f"     Embedding shape: {len(embedding_data)}")
                        print(f"     Embedding range: [{min(embedding_data):.4f}, {max(embedding_data):.4f}]")
                    except:
                        print(f"     Embedding: Invalid format")
        
        # 2. Test the face engine directly with a real image
        print(f"\n🧪 Testing face engine with real image...")
        from core.face_engine import FaceEngine
        from models.face_models import FaceRecognitionConfig
        import cv2
        
        config = FaceRecognitionConfig()
        face_engine = FaceEngine(config)
        
        # Test with the first employee's face
        if employees and employees[0].face_photo_path:
            face_image = cv2.imread(employees[0].face_photo_path)
            if face_image is not None:
                print(f"✅ Loaded face image: {employees[0].face_photo_path}")
                print(f"   Image shape: {face_image.shape}")
                
                # Process with face engine
                analysis = face_engine.process_frame(face_image, "DEBUG_CAM")
                print(f"✅ Face detections: {len(analysis.face_detections)}")
                
                if analysis.face_detections:
                    print(f"✅ Face detection worked")
                    
                    # Check recognition
                    recognized_faces = analysis.get_recognized_faces(0.3)
                    unknown_faces = analysis.get_unknown_faces(0.3)
                    
                    print(f"✅ Recognized faces: {len(recognized_faces)}")
                    print(f"✅ Unknown faces: {len(unknown_faces)}")
                    
                    for recognition in recognized_faces:
                        print(f"   🎉 RECOGNIZED: {recognition.employee_id} (confidence: {recognition.confidence:.4f})")
                    
                    for detection in unknown_faces:
                        print(f"   ❓ UNKNOWN: {detection.face_id}")
                else:
                    print(f"❌ NO FACE DETECTED - This is the problem!")
            else:
                print(f"❌ Could not load face image")
        
        # 3. Test face detection separately
        print(f"\n🧪 Testing face detection separately...")
        from core.face_detector import get_face_detector
        
        detector = get_face_detector(config)
        
        if employees and employees[0].face_photo_path:
            face_image = cv2.imread(employees[0].face_photo_path)
            if face_image is not None:
                detections = detector.detect_faces(face_image)
                print(f"✅ Direct face detection: {len(detections)} faces")
                
                for i, detection in enumerate(detections):
                    print(f"   Face {i+1}: confidence = {detection.confidence:.4f}")
        
        # 4. Test recognition separately
        print(f"\n🧪 Testing recognition separately...")
        from core.ultra_simple_recognizer import UltraSimpleRecognizer
        
        recognizer = UltraSimpleRecognizer()
        
        if employees and employees[0].face_photo_path:
            face_image = cv2.imread(employees[0].face_photo_path)
            if face_image is not None:
                embedding = recognizer.generate_embedding(face_image)
                if embedding is not None:
                    best_match_id, best_confidence = recognizer._find_best_match(embedding)
                    print(f"✅ Direct recognition: {best_match_id} (confidence: {best_confidence:.4f})")
                else:
                    print(f"❌ Failed to generate embedding")
        
        print(f"\n🎯 DEBUG COMPLETE!")
        print(f"Check the results above to identify the real issue.")
        
        # Cleanup
        face_engine.cleanup()
        recognizer.cleanup()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_live_system()

