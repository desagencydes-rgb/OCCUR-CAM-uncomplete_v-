#!/usr/bin/env python3
"""
Test direct recognition without face extraction.
"""

def test_direct_recognition():
    """Test direct recognition without face extraction."""
    try:
        print("🧪 TESTING DIRECT RECOGNITION")
        print("=" * 40)
        
        from core.ultra_simple_recognizer import UltraSimpleRecognizer
        import cv2
        
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
                
                # Test direct recognition
                from models.face_models import FaceDetection
                detection = FaceDetection(
                    face_id="test_face",
                    bbox=[0, 0, face_image.shape[1], face_image.shape[0]],
                    confidence=1.0
                )
                
                recognition = recognizer.recognize_face(face_image, detection)
                print(f"✅ Direct recognition result:")
                print(f"   Employee ID: {recognition.employee_id}")
                print(f"   Confidence: {recognition.confidence:.4f}")
                print(f"   Processing time: {recognition.processing_time:.4f}")
                
                if recognition.employee_id:
                    print(f"   🎉 DIRECT RECOGNITION SUCCESS!")
                else:
                    print(f"   ❌ DIRECT RECOGNITION FAILED!")
                    
                    # Debug the embedding generation
                    print(f"\n🔍 Debugging embedding generation...")
                    embedding = recognizer.generate_embedding(face_image)
                    if embedding is not None:
                        print(f"   Embedding shape: {embedding.shape}")
                        print(f"   Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
                        
                        # Test similarity with stored embeddings
                        for emp_id, stored_embedding in recognizer.employee_embeddings.items():
                            similarity = recognizer._find_best_match(embedding)
                            print(f"   Similarity with {emp_id}: {similarity[1]:.4f}")
                    else:
                        print(f"   ❌ Failed to generate embedding")
            else:
                print(f"❌ Could not load face image")
        else:
            print(f"❌ No employee with face photo found")
        
        print(f"\n🎯 DIRECT RECOGNITION TEST COMPLETE!")
        
        # Cleanup
        recognizer.cleanup()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_recognition()

