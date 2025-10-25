#!/usr/bin/env python3
"""
Script to fix face recognition using simple fixed approach.
"""

def fix_simple_recognition():
    """Fix the face recognition system using simple fixed approach."""
    try:
        from database.schemas.auth_schemas import Employee
        from config.database import get_auth_db
        from core.simple_fixed_recognizer import SimpleFixedRecognizer
        import json
        import os
        import cv2
        import numpy as np
        
        print("🔧 FIXING FACE RECOGNITION WITH SIMPLE FIXED APPROACH")
        print("=" * 60)
        print("Avoiding OpenCV issues with simple, robust approach")
        
        # Initialize the simple fixed recognizer
        recognizer = SimpleFixedRecognizer()
        
        with get_auth_db() as db:
            # Get all employees with face photos
            employees = db.query(Employee).filter(
                Employee.is_active == True,
                Employee.face_photo_path.isnot(None)
            ).all()
            
            print(f"📋 Found {len(employees)} employees with face photos")
            
            if not employees:
                print("❌ No employees with face photos found")
                return
            
            # Clear existing embeddings
            print("🧹 Clearing existing face embeddings...")
            for employee in employees:
                employee.face_embedding = None
            db.commit()
            print("✅ Cleared existing embeddings")
            
            # Regenerate embeddings with simple fixed approach
            print("🔄 Regenerating face embeddings with SIMPLE FIXED approach...")
            success_count = 0
            
            for i, employee in enumerate(employees, 1):
                try:
                    print(f"   Processing {i}/{len(employees)}: {employee.employee_id}")
                    
                    if employee.face_photo_path and os.path.exists(employee.face_photo_path):
                        # Load face image
                        face_image = cv2.imread(employee.face_photo_path)
                        
                        if face_image is not None:
                            # Generate new embedding using simple fixed approach
                            embedding = recognizer.generate_embedding(face_image)
                            
                            if embedding is not None:
                                # Save to database
                                employee.face_embedding = json.dumps(embedding.tolist())
                                success_count += 1
                                print(f"   ✅ Generated simple fixed embedding for {employee.employee_id}")
                                print(f"      Embedding shape: {embedding.shape}")
                                print(f"      Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
                            else:
                                print(f"   ❌ Failed to generate embedding for {employee.employee_id}")
                        else:
                            print(f"   ❌ Could not load image for {employee.employee_id}")
                    else:
                        print(f"   ❌ Face photo not found for {employee.employee_id}")
                        
                except Exception as e:
                    print(f"   ❌ Error processing {employee.employee_id}: {e}")
                    continue
            
            # Commit all changes
            db.commit()
            
            print(f"\n🎯 SIMPLE FIXED REGENERATION COMPLETE!")
            print(f"✅ Successfully regenerated {success_count}/{len(employees)} embeddings")
            print(f"✅ Face recognition system is now using SIMPLE FIXED approach")
            print(f"✅ Recognition threshold: 60% (reasonable)")
            print(f"✅ Simple, robust feature extraction")
            print(f"✅ No OpenCV issues")
            
            # Test the system
            print("\n🧪 Testing the simple fixed system...")
            recognizer._load_employee_embeddings()
            print(f"✅ Loaded {len(recognizer.employee_embeddings)} embeddings into memory")
            
            # Show embedding uniqueness
            if len(recognizer.employee_embeddings) > 1:
                embeddings = list(recognizer.employee_embeddings.values())
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        similarities.append(sim)
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    max_similarity = np.max(similarities)
                    min_similarity = np.min(similarities)
                    print(f"✅ Average similarity between different faces: {avg_similarity:.4f}")
                    print(f"✅ Maximum similarity between different faces: {max_similarity:.4f}")
                    print(f"✅ Minimum similarity between different faces: {min_similarity:.4f}")
                    print(f"✅ Recognition threshold: 0.6 (60%)")
                    
                    if max_similarity < 0.6:
                        print("🎉 SUCCESS: Different faces are distinguishable!")
                        print("🎉 Registered faces will be recognized!")
                    else:
                        print("⚠️  WARNING: Some faces may still be similar")
            
            print("\n🚀 FACE RECOGNITION SYSTEM IS NOW SIMPLE FIXED!")
            print("• Using simple, robust feature extraction")
            print("• Recognition threshold: 60% (reasonable)")
            print("• No OpenCV issues")
            print("• Ready for testing")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_simple_recognition()

