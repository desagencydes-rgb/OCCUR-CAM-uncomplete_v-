"""
OCCUR-CAM Face Quality Enhancer
Advanced face quality enhancement for low-light and poor quality conditions.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import time

class QualityLevel(Enum):
    """Face quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"

@dataclass
class FaceQualityMetrics:
    """Face quality metrics."""
    sharpness: float
    brightness: float
    contrast: float
    symmetry: float
    pose_angle: float
    noise_level: float
    blur_level: float
    overall_score: float
    quality_level: QualityLevel

class FaceQualityEnhancer:
    """Advanced face quality enhancement system."""
    
    def __init__(self):
        """Initialize face quality enhancer."""
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.9,
            QualityLevel.GOOD: 0.8,
            QualityLevel.FAIR: 0.6,
            QualityLevel.POOR: 0.4,
            QualityLevel.VERY_POOR: 0.0
        }
        
        # Enhancement parameters for different quality levels
        self.enhancement_params = {
            QualityLevel.EXCELLENT: {
                "sharpen": False,
                "denoise": False,
                "brightness": 0,
                "contrast": 1.0,
                "gamma": 1.0
            },
            QualityLevel.GOOD: {
                "sharpen": False,
                "denoise": False,
                "brightness": 0,
                "contrast": 1.0,
                "gamma": 1.0
            },
            QualityLevel.FAIR: {
                "sharpen": True,
                "denoise": True,
                "brightness": 10,
                "contrast": 1.1,
                "gamma": 0.9
            },
            QualityLevel.POOR: {
                "sharpen": True,
                "denoise": True,
                "brightness": 20,
                "contrast": 1.2,
                "gamma": 0.8
            },
            QualityLevel.VERY_POOR: {
                "sharpen": True,
                "denoise": True,
                "brightness": 30,
                "contrast": 1.3,
                "gamma": 0.7
            }
        }
    
    def analyze_face_quality(self, face_image: np.ndarray) -> FaceQualityMetrics:
        """
        Analyze face quality metrics.
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            FaceQualityMetrics with detailed analysis
        """
        try:
            # Convert to grayscale for analysis
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image.copy()
            
            # Calculate individual metrics
            sharpness = self._calculate_sharpness(gray)
            brightness = self._calculate_brightness(gray)
            contrast = self._calculate_contrast(gray)
            symmetry = self._calculate_symmetry(gray)
            pose_angle = self._calculate_pose_angle(gray)
            noise_level = self._calculate_noise_level(gray)
            blur_level = self._calculate_blur_level(gray)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                sharpness, brightness, contrast, symmetry, 
                pose_angle, noise_level, blur_level
            )
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            return FaceQualityMetrics(
                sharpness=sharpness,
                brightness=brightness,
                contrast=contrast,
                symmetry=symmetry,
                pose_angle=pose_angle,
                noise_level=noise_level,
                blur_level=blur_level,
                overall_score=overall_score,
                quality_level=quality_level
            )
            
        except Exception as e:
            logging.error(f"Error analyzing face quality: {e}")
            return FaceQualityMetrics(
                sharpness=0.5,
                brightness=128.0,
                contrast=64.0,
                symmetry=0.5,
                pose_angle=0.0,
                noise_level=0.5,
                blur_level=0.5,
                overall_score=0.5,
                quality_level=QualityLevel.POOR
            )
    
    def enhance_face_quality(self, face_image: np.ndarray) -> Tuple[np.ndarray, FaceQualityMetrics]:
        """
        Enhance face quality for better recognition.
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Tuple of (enhanced_image, quality_metrics)
        """
        try:
            # Analyze current quality
            quality_metrics = self.analyze_face_quality(face_image)
            
            # Apply enhancement based on quality level
            enhanced = self._apply_enhancement(face_image, quality_metrics.quality_level)
            
            # Re-analyze enhanced image
            enhanced_metrics = self.analyze_face_quality(enhanced)
            
            return enhanced, enhanced_metrics
            
        except Exception as e:
            logging.error(f"Error enhancing face quality: {e}")
            return face_image, quality_metrics
    
    def _calculate_sharpness(self, gray_image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        try:
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            # Normalize to 0-1 range
            return min(laplacian_var / 1000.0, 1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating sharpness: {e}")
            return 0.5
    
    def _calculate_brightness(self, gray_image: np.ndarray) -> float:
        """Calculate image brightness."""
        try:
            return float(np.mean(gray_image))
            
        except Exception as e:
            logging.warning(f"Error calculating brightness: {e}")
            return 128.0
    
    def _calculate_contrast(self, gray_image: np.ndarray) -> float:
        """Calculate image contrast."""
        try:
            return float(np.std(gray_image))
            
        except Exception as e:
            logging.warning(f"Error calculating contrast: {e}")
            return 64.0
    
    def _calculate_symmetry(self, gray_image: np.ndarray) -> float:
        """Calculate face symmetry."""
        try:
            h, w = gray_image.shape
            
            # Split image into left and right halves
            left_half = gray_image[:, :w//2]
            right_half = cv2.flip(gray_image[:, w//2:], 1)
            
            # Ensure both halves have the same width
            min_width = min(left_half.shape[1], right_half.shape[1])
            if min_width > 0:
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
                
                # Calculate difference
                diff = np.abs(left_half.astype(float) - right_half.astype(float))
                symmetry = 1.0 - np.mean(diff) / 255.0
                
                return max(0.0, min(1.0, symmetry))
            else:
                return 0.5
                
        except Exception as e:
            logging.warning(f"Error calculating symmetry: {e}")
            return 0.5
    
    def _calculate_pose_angle(self, gray_image: np.ndarray) -> float:
        """Calculate face pose angle (simplified)."""
        try:
            # This is a simplified implementation
            # In a real system, you would use facial landmarks
            
            # Calculate horizontal gradient
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate angle
            angle = np.arctan2(np.mean(grad_y), np.mean(grad_x))
            angle_degrees = np.degrees(angle)
            
            # Normalize to 0-1 range (0 = frontal, 1 = profile)
            normalized_angle = abs(angle_degrees) / 90.0
            
            return min(1.0, normalized_angle)
            
        except Exception as e:
            logging.warning(f"Error calculating pose angle: {e}")
            return 0.0
    
    def _calculate_noise_level(self, gray_image: np.ndarray) -> float:
        """Calculate noise level in image."""
        try:
            # Use Laplacian to detect noise
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            noise_level = np.var(laplacian)
            
            # Normalize to 0-1 range
            return min(noise_level / 10000.0, 1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating noise level: {e}")
            return 0.5
    
    def _calculate_blur_level(self, gray_image: np.ndarray) -> float:
        """Calculate blur level in image."""
        try:
            # Use gradient magnitude to detect blur
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            blur_level = 1.0 - (np.mean(gradient_magnitude) / 1000.0)
            
            return max(0.0, min(1.0, blur_level))
            
        except Exception as e:
            logging.warning(f"Error calculating blur level: {e}")
            return 0.5
    
    def _calculate_overall_score(self, sharpness: float, brightness: float, 
                               contrast: float, symmetry: float, pose_angle: float,
                               noise_level: float, blur_level: float) -> float:
        """Calculate overall quality score."""
        try:
            # Brightness score (optimal around 128)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            
            # Contrast score (higher is better, up to a point)
            contrast_score = min(contrast / 80.0, 1.0)
            
            # Pose score (frontal is better)
            pose_score = 1.0 - pose_angle
            
            # Noise score (lower is better)
            noise_score = 1.0 - noise_level
            
            # Blur score (lower is better)
            blur_score = 1.0 - blur_level
            
            # Weighted combination
            overall_score = (
                sharpness * 0.25 +
                brightness_score * 0.15 +
                contrast_score * 0.15 +
                symmetry * 0.15 +
                pose_score * 0.15 +
                noise_score * 0.075 +
                blur_score * 0.075
            )
            
            return max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            logging.warning(f"Error calculating overall score: {e}")
            return 0.5
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level based on overall score."""
        try:
            if overall_score >= self.quality_thresholds[QualityLevel.EXCELLENT]:
                return QualityLevel.EXCELLENT
            elif overall_score >= self.quality_thresholds[QualityLevel.GOOD]:
                return QualityLevel.GOOD
            elif overall_score >= self.quality_thresholds[QualityLevel.FAIR]:
                return QualityLevel.FAIR
            elif overall_score >= self.quality_thresholds[QualityLevel.POOR]:
                return QualityLevel.POOR
            else:
                return QualityLevel.VERY_POOR
                
        except Exception as e:
            logging.warning(f"Error determining quality level: {e}")
            return QualityLevel.POOR
    
    def _apply_enhancement(self, face_image: np.ndarray, quality_level: QualityLevel) -> np.ndarray:
        """Apply enhancement based on quality level."""
        try:
            if quality_level not in self.enhancement_params:
                return face_image
            
            params = self.enhancement_params[quality_level]
            enhanced = face_image.copy()
            
            # Apply gamma correction
            if params["gamma"] != 1.0:
                enhanced = self._apply_gamma_correction(enhanced, params["gamma"])
            
            # Apply brightness and contrast
            if params["brightness"] != 0 or params["contrast"] != 1.0:
                enhanced = cv2.convertScaleAbs(
                    enhanced, 
                    alpha=params["contrast"], 
                    beta=params["brightness"]
                )
            
            # Apply denoising
            if params["denoise"]:
                enhanced = self._apply_denoising(enhanced)
            
            # Apply sharpening
            if params["sharpen"]:
                enhanced = self._apply_sharpening(enhanced)
            
            return enhanced
            
        except Exception as e:
            logging.error(f"Error applying enhancement: {e}")
            return face_image
    
    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction to image."""
        try:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            return cv2.LUT(image, table)
            
        except Exception as e:
            logging.warning(f"Error applying gamma correction: {e}")
            return image
    
    def _apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to image."""
        try:
            if len(image.shape) == 3:
                return cv2.bilateralFilter(image, 9, 75, 75)
            else:
                return cv2.fastNlMeansDenoising(image)
                
        except Exception as e:
            logging.warning(f"Error applying denoising: {e}")
            return image
    
    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening to image."""
        try:
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            
            # Apply sharpening
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend with original to avoid over-sharpening
            return cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
            
        except Exception as e:
            logging.warning(f"Error applying sharpening: {e}")
            return image
    
    def enhance_for_recognition(self, face_image: np.ndarray) -> np.ndarray:
        """
        Enhance face image specifically for recognition.
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Enhanced face image optimized for recognition
        """
        try:
            # Analyze quality
            quality_metrics = self.analyze_face_quality(face_image)
            
            # Apply recognition-specific enhancements
            enhanced = face_image.copy()
            
            # 1. Normalize lighting
            enhanced = self._normalize_lighting(enhanced)
            
            # 2. Enhance contrast
            enhanced = self._enhance_contrast(enhanced)
            
            # 3. Reduce noise
            enhanced = self._reduce_noise(enhanced)
            
            # 4. Sharpen if needed
            if quality_metrics.sharpness < 0.5:
                enhanced = self._apply_sharpening(enhanced)
            
            # 5. Normalize face orientation
            enhanced = self._normalize_orientation(enhanced)
            
            return enhanced
            
        except Exception as e:
            logging.error(f"Error enhancing for recognition: {e}")
            return face_image
    
    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """Normalize lighting in face image."""
        try:
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                
                # Apply CLAHE to L channel
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
                
                # Convert back to BGR
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
                
        except Exception as e:
            logging.warning(f"Error normalizing lighting: {e}")
            return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast in face image."""
        try:
            # Apply adaptive histogram equalization
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image)
                
        except Exception as e:
            logging.warning(f"Error enhancing contrast: {e}")
            return image
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Reduce noise in face image."""
        try:
            if len(image.shape) == 3:
                return cv2.bilateralFilter(image, 9, 75, 75)
            else:
                return cv2.fastNlMeansDenoising(image)
                
        except Exception as e:
            logging.warning(f"Error reducing noise: {e}")
            return image
    
    def _normalize_orientation(self, image: np.ndarray) -> np.ndarray:
        """Normalize face orientation."""
        try:
            # This is a simplified implementation
            # In a real system, you would use facial landmarks to detect and correct orientation
            
            # For now, just return the original image
            return image
            
        except Exception as e:
            logging.warning(f"Error normalizing orientation: {e}")
            return image
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get enhancement statistics."""
        return {
            "quality_levels": list(QualityLevel),
            "quality_thresholds": self.quality_thresholds,
            "enhancement_params": self.enhancement_params
        }

# Global face quality enhancer instance
_face_quality_enhancer = None

def get_face_quality_enhancer() -> FaceQualityEnhancer:
    """Get global face quality enhancer instance."""
    global _face_quality_enhancer
    
    if _face_quality_enhancer is None:
        _face_quality_enhancer = FaceQualityEnhancer()
    
    return _face_quality_enhancer
