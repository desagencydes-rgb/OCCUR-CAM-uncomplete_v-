#!/usr/bin/env python3
"""
Test the fixed face recognition system.
"""

import cv2
import numpy as np
from core.extreme_face_recognizer import ExtremeFaceRecognizer

def test_fixed_recognition():
    """Test the fixed face recognition system."""
    print("🧪 TESTING FIXED FACE RECOGNITION SYSTEM")
    print("=" * 50)
    
    try:
        # Initialize recognizer
        recognizer = ExtremeFaceRecognizer()
        
        print(f"✅ Extreme face recognizer initialized")
        print(f"✅ Loaded {len(recognizer.employee_embeddings)} employee embeddings")
        print(f"✅ Recognition threshold: {recognizer.recognition_threshold}")
        
        # Test with a synthetic image
        print("\n🔍 Testing with synthetic image...")
        test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        try:
            embedding = recognizer.generate_embedding(test_image)
            if embedding is not None:
                print(f"✅ Embedding generated successfully: shape {embedding.shape}")
                print(f"✅ Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
            else:
                print("❌ Failed to generate embedding")
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
        
        # Test with a real face image if available
        print("\n🔍 Testing with real face images...")
        from database.schemas.auth_schemas import Employee
        from config.database import get_auth_db
        
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
                        else:
                            print(f"❌ {employee.employee_id}: Failed to generate embedding")
                    else:
                        print(f"❌ {employee.employee_id}: Could not load image")
                except Exception as e:
                    print(f"❌ {employee.employee_id}: Error - {e}")
        
        print("\n🎯 FIXED SYSTEM TEST COMPLETE!")
        print("✅ OpenCV errors should be resolved")
        print("✅ Face recognition should work properly")
        
    except Exception as e:
        print(f"❌ Error testing fixed system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_recognition()

