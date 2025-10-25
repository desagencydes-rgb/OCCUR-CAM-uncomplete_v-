#!/usr/bin/env python3
"""
Simple Face Recognition Accuracy Test
"""

import cv2
import numpy as np
import json
import os
from core.extreme_face_recognizer import ExtremeFaceRecognizer
from database.schemas.auth_schemas import Employee
from config.database import get_auth_db

def test_face_recognition_accuracy():
    """Test face recognition accuracy."""
    print("🧪 FACE RECOGNITION ACCURACY TEST")
    print("=" * 50)
    
    try:
        # Initialize recognizer
        recognizer = ExtremeFaceRecognizer()
        
        with get_auth_db() as db:
            employees = db.query(Employee).filter(
                Employee.is_active == True,
                Employee.face_photo_path.isnot(None)
            ).all()
        
        print(f"📋 Found {len(employees)} employees with face photos")
        
        if len(employees) < 1:
            print("❌ No employees found for testing")
            return 0.0
        
        # Test 1: Verification (1:1 matching)
        print("\n🔍 TEST 1: VERIFICATION (1:1 Matching)")
        print("-" * 30)
        
        correct_verification = 0
        total_verification = 0
        
        for employee in employees:
            if employee.face_photo_path and os.path.exists(employee.face_photo_path):
                try:
                    # Load face image
                    face_image = cv2.imread(employee.face_photo_path)
                    
                    if face_image is not None:
                        # Generate embedding
                        embedding = recognizer.generate_embedding(face_image)
                        
                        if embedding is not None:
                            # Test against stored embedding
                            stored_embedding = recognizer.employee_embeddings.get(employee.employee_id)
                            
                            if stored_embedding is not None:
                                # Calculate similarity
                                similarity = np.dot(embedding, stored_embedding) / (
                                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding) + 1e-8
                                )
                                
                                # Check if it matches (above threshold)
                                if similarity > recognizer.recognition_threshold:
                                    correct_verification += 1
                                    print(f"   ✅ {employee.employee_id}: {similarity:.4f} (MATCH)")
                                else:
                                    print(f"   ❌ {employee.employee_id}: {similarity:.4f} (NO MATCH)")
                                
                                total_verification += 1
                except Exception as e:
                    print(f"   ❌ Error testing {employee.employee_id}: {e}")
                    continue
        
        verification_accuracy = (correct_verification / total_verification * 100) if total_verification > 0 else 0.0
        print(f"✅ Verification Accuracy: {verification_accuracy:.2f}% ({correct_verification}/{total_verification})")
        
        # Test 2: Identification (1:N matching)
        print("\n🔍 TEST 2: IDENTIFICATION (1:N Matching)")
        print("-" * 30)
        
        correct_identification = 0
        total_identification = 0
        
        for employee in employees:
            if employee.face_photo_path and os.path.exists(employee.face_photo_path):
                try:
                    # Load face image
                    face_image = cv2.imread(employee.face_photo_path)
                    
                    if face_image is not None:
                        # Generate embedding
                        embedding = recognizer.generate_embedding(face_image)
                        
                        if embedding is not None:
                            # Find best match
                            best_match_id, best_confidence = recognizer._find_best_match(embedding)
                            
                            # Check if correct identification
                            if best_match_id == employee.employee_id:
                                correct_identification += 1
                                print(f"   ✅ {employee.employee_id}: Correctly identified (Confidence: {best_confidence:.4f})")
                            else:
                                print(f"   ❌ {employee.employee_id}: Misidentified as {best_match_id} (Confidence: {best_confidence:.4f})")
                            
                            total_identification += 1
                except Exception as e:
                    print(f"   ❌ Error testing {employee.employee_id}: {e}")
                    continue
        
        identification_accuracy = (correct_identification / total_identification * 100) if total_identification > 0 else 0.0
        print(f"✅ Identification Accuracy: {identification_accuracy:.2f}% ({correct_identification}/{total_identification})")
        
        # Test 3: False Acceptance Rate (FAR)
        print("\n🔍 TEST 3: FALSE ACCEPTANCE RATE (FAR)")
        print("-" * 30)
        
        synthetic_tests = 20
        false_acceptances = 0
        
        for i in range(synthetic_tests):
            # Generate random image
            random_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            
            # Generate embedding
            embedding = recognizer.generate_embedding(random_image)
            
            if embedding is not None:
                # Check if it matches any stored embedding
                best_match_id, best_confidence = recognizer._find_best_match(embedding)
                
                if best_match_id is not None:
                    false_acceptances += 1
                    print(f"   ❌ Synthetic test {i+1}: FALSE ACCEPTANCE (Confidence: {best_confidence:.4f})")
                else:
                    print(f"   ✅ Synthetic test {i+1}: Correctly rejected")
        
        far_rate = (false_acceptances / synthetic_tests * 100) if synthetic_tests > 0 else 0.0
        print(f"✅ False Acceptance Rate (FAR): {far_rate:.2f}% ({false_acceptances}/{synthetic_tests})")
        
        # Test 4: False Rejection Rate (FRR)
        print("\n🔍 TEST 4: FALSE REJECTION RATE (FRR)")
        print("-" * 30)
        
        false_rejections = 0
        total_frr_tests = 0
        
        for employee in employees:
            if employee.face_photo_path and os.path.exists(employee.face_photo_path):
                try:
                    # Load face image
                    face_image = cv2.imread(employee.face_photo_path)
                    
                    if face_image is not None:
                        # Test with slight variations
                        for variation in range(3):
                            if variation == 0:
                                test_image = face_image.copy()
                            elif variation == 1:
                                # Brightness adjustment
                                test_image = cv2.convertScaleAbs(face_image, alpha=1.1, beta=10)
                            else:
                                # Contrast adjustment
                                test_image = cv2.convertScaleAbs(face_image, alpha=1.2, beta=0)
                            
                            # Generate embedding
                            embedding = recognizer.generate_embedding(test_image)
                            
                            if embedding is not None:
                                # Check if it matches
                                best_match_id, best_confidence = recognizer._find_best_match(embedding)
                                
                                if best_match_id != employee.employee_id:
                                    false_rejections += 1
                                    print(f"   ❌ {employee.employee_id} (Variation {variation+1}): FALSE REJECTION")
                                else:
                                    print(f"   ✅ {employee.employee_id} (Variation {variation+1}): Correctly accepted")
                                
                                total_frr_tests += 1
                except Exception as e:
                    print(f"   ❌ Error testing {employee.employee_id}: {e}")
                    continue
        
        frr_rate = (false_rejections / total_frr_tests * 100) if total_frr_tests > 0 else 0.0
        print(f"✅ False Rejection Rate (FRR): {frr_rate:.2f}% ({false_rejections}/{total_frr_tests})")
        
        # Calculate Overall Accuracy
        print("\n🎯 OVERALL ACCURACY CALCULATION")
        print("=" * 40)
        
        # Weighted average: Verification 40%, Identification 40%, FAR 10%, FRR 10%
        verification_score = verification_accuracy
        identification_score = identification_accuracy
        far_score = 100 - far_rate  # Lower FAR is better
        frr_score = 100 - frr_rate  # Lower FRR is better
        
        overall_accuracy = (
            verification_score * 0.40 +
            identification_score * 0.40 +
            far_score * 0.10 +
            frr_score * 0.10
        )
        
        print(f"📊 ACCURACY METRICS:")
        print(f"   • Verification Accuracy: {verification_accuracy:.2f}%")
        print(f"   • Identification Accuracy: {identification_accuracy:.2f}%")
        print(f"   • False Acceptance Rate (FAR): {far_rate:.2f}%")
        print(f"   • False Rejection Rate (FRR): {frr_rate:.2f}%")
        
        print(f"\n🎯 OVERALL SYSTEM ACCURACY: {overall_accuracy:.2f}%")
        
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        if overall_accuracy >= 95:
            print("   🏆 EXCELLENT: System meets industry standards")
        elif overall_accuracy >= 90:
            print("   ✅ VERY GOOD: System performs well")
        elif overall_accuracy >= 80:
            print("   ⚠️  GOOD: System is acceptable")
        elif overall_accuracy >= 70:
            print("   ⚠️  FAIR: System needs improvement")
        else:
            print("   ❌ POOR: System requires significant improvement")
        
        print(f"\n🔧 SYSTEM SPECIFICATIONS:")
        print(f"   • Recognition Threshold: {recognizer.recognition_threshold:.2f}")
        print(f"   • Embedding Dimension: 2048")
        print(f"   • Feature Extraction: Ultra-aggressive")
        print(f"   • Noise Sources: Image hash, position, time, UUID")
        
        print(f"\n🚀 CONCLUSION:")
        print(f"   The face recognition system achieves {overall_accuracy:.2f}% overall accuracy")
        print(f"   with a {recognizer.recognition_threshold:.2f} recognition threshold.")
        
        if overall_accuracy >= 90:
            print(f"   🎉 SYSTEM IS PRODUCTION READY!")
        else:
            print(f"   ⚠️  SYSTEM NEEDS FURTHER OPTIMIZATION")
        
        return overall_accuracy
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    accuracy = test_face_recognition_accuracy()
    print(f"\n🎯 FINAL ACCURACY PERCENTAGE: {accuracy:.2f}%")

