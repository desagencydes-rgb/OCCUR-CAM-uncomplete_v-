"""
OCCUR-CAM Lighting Optimizer
Advanced lighting and image quality optimization for all lighting conditions.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time

class LightingCondition(Enum):
    """Lighting condition types."""
    VERY_DARK = "very_dark"
    DARK = "dark"
    LOW_LIGHT = "low_light"
    NORMAL = "normal"
    BRIGHT = "bright"
    VERY_BRIGHT = "very_bright"
    MIXED = "mixed"
    BACKLIT = "backlit"

@dataclass
class LightingAnalysis:
    """Lighting analysis results."""
    condition: LightingCondition
    brightness: float
    contrast: float
    sharpness: float
    noise_level: float
    color_balance: Tuple[float, float, float]
    quality_score: float
    recommendations: List[str]

class LightingOptimizer:
    """Advanced lighting and image quality optimizer."""
    
    def __init__(self):
        """Initialize lighting optimizer."""
        self.brightness_thresholds = {
            LightingCondition.VERY_DARK: (0, 30),
            LightingCondition.DARK: (30, 60),
            LightingCondition.LOW_LIGHT: (60, 100),
            LightingCondition.NORMAL: (100, 180),
            LightingCondition.BRIGHT: (180, 220),
            LightingCondition.VERY_BRIGHT: (220, 255)
        }
        
        # Optimization parameters
        self.optimization_params = {
            LightingCondition.VERY_DARK: {
                "gamma": 0.6,
                "brightness": 50,
                "contrast": 1.5,
                "denoise": True,
                "histogram_eq": True,
                "sharpen": True
            },
            LightingCondition.DARK: {
                "gamma": 0.7,
                "brightness": 30,
                "contrast": 1.3,
                "denoise": True,
                "histogram_eq": True,
                "sharpen": True
            },
            LightingCondition.LOW_LIGHT: {
                "gamma": 0.8,
                "brightness": 20,
                "contrast": 1.2,
                "denoise": True,
                "histogram_eq": True,
                "sharpen": False
            },
            LightingCondition.NORMAL: {
                "gamma": 1.0,
                "brightness": 0,
                "contrast": 1.0,
                "denoise": False,
                "histogram_eq": False,
                "sharpen": False
            },
            LightingCondition.BRIGHT: {
                "gamma": 1.2,
                "brightness": -20,
                "contrast": 0.9,
                "denoise": False,
                "histogram_eq": False,
                "sharpen": False
            },
            LightingCondition.VERY_BRIGHT: {
                "gamma": 1.4,
                "brightness": -40,
                "contrast": 0.8,
                "denoise": False,
                "histogram_eq": False,
                "sharpen": False
            }
        }
    
    def analyze_lighting(self, image: np.ndarray) -> LightingAnalysis:
        """
        Analyze lighting conditions in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            LightingAnalysis with condition and recommendations
        """
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = np.std(gray)
            
            # Calculate sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 1000.0, 1.0)
            
            # Calculate noise level
            noise_level = self._calculate_noise_level(gray)
            
            # Calculate color balance
            color_balance = self._calculate_color_balance(image)
            
            # Determine lighting condition
            condition = self._determine_lighting_condition(brightness, contrast, noise_level)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(brightness, contrast, sharpness, noise_level)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(condition, brightness, contrast, sharpness, noise_level)
            
            return LightingAnalysis(
                condition=condition,
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness,
                noise_level=noise_level,
                color_balance=color_balance,
                quality_score=quality_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logging.error(f"Error analyzing lighting: {e}")
            return LightingAnalysis(
                condition=LightingCondition.NORMAL,
                brightness=128.0,
                contrast=64.0,
                sharpness=0.5,
                noise_level=0.5,
                color_balance=(1.0, 1.0, 1.0),
                quality_score=0.5,
                recommendations=["Error in analysis"]
            )
    
    def optimize_for_face_recognition(self, image: np.ndarray) -> np.ndarray:
        """
        Optimize image for face recognition.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Optimized image
        """
        try:
            # Analyze lighting conditions
            analysis = self.analyze_lighting(image)
            
            # Apply optimization based on lighting condition
            optimized = self._apply_optimization(image, analysis.condition)
            
            # Additional face-specific optimizations
            optimized = self._apply_face_optimizations(optimized, analysis)
            
            return optimized
            
        except Exception as e:
            logging.error(f"Error optimizing image for face recognition: {e}")
            return image
    
    def _calculate_noise_level(self, gray_image: np.ndarray) -> float:
        """Calculate noise level in image."""
        try:
            # Use Laplacian to detect edges and noise
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            noise_level = np.var(laplacian)
            
            # Normalize to 0-1 range
            return min(noise_level / 10000.0, 1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating noise level: {e}")
            return 0.5
    
    def _calculate_color_balance(self, image: np.ndarray) -> Tuple[float, float, float]:
        """Calculate color balance ratios."""
        try:
            if len(image.shape) != 3:
                return (1.0, 1.0, 1.0)
            
            # Calculate mean values for each channel
            b_mean = np.mean(image[:, :, 0])
            g_mean = np.mean(image[:, :, 1])
            r_mean = np.mean(image[:, :, 2])
            
            # Calculate overall mean
            overall_mean = (b_mean + g_mean + r_mean) / 3.0
            
            # Calculate balance ratios
            b_balance = overall_mean / b_mean if b_mean > 0 else 1.0
            g_balance = overall_mean / g_mean if g_mean > 0 else 1.0
            r_balance = overall_mean / r_mean if r_mean > 0 else 1.0
            
            # Normalize to reasonable range
            b_balance = max(0.5, min(2.0, b_balance))
            g_balance = max(0.5, min(2.0, g_balance))
            r_balance = max(0.5, min(2.0, r_balance))
            
            return (b_balance, g_balance, r_balance)
            
        except Exception as e:
            logging.warning(f"Error calculating color balance: {e}")
            return (1.0, 1.0, 1.0)
    
    def _determine_lighting_condition(self, brightness: float, contrast: float, noise_level: float) -> LightingCondition:
        """Determine lighting condition based on image analysis."""
        try:
            # Check for very dark conditions
            if brightness < 30:
                return LightingCondition.VERY_DARK
            
            # Check for dark conditions
            if brightness < 60:
                return LightingCondition.DARK
            
            # Check for low light conditions
            if brightness < 100:
                return LightingCondition.LOW_LIGHT
            
            # Check for very bright conditions
            if brightness > 220:
                return LightingCondition.VERY_BRIGHT
            
            # Check for bright conditions
            if brightness > 180:
                return LightingCondition.BRIGHT
            
            # Check for mixed lighting (high contrast)
            if contrast > 80:
                return LightingCondition.MIXED
            
            # Check for backlit conditions (high brightness with low contrast)
            if brightness > 150 and contrast < 40:
                return LightingCondition.BACKLIT
            
            # Normal lighting
            return LightingCondition.NORMAL
            
        except Exception as e:
            logging.warning(f"Error determining lighting condition: {e}")
            return LightingCondition.NORMAL
    
    def _calculate_quality_score(self, brightness: float, contrast: float, 
                               sharpness: float, noise_level: float) -> float:
        """Calculate overall image quality score."""
        try:
            # Brightness score (optimal around 128)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            
            # Contrast score (higher is better, up to a point)
            contrast_score = min(contrast / 80.0, 1.0)
            
            # Sharpness score
            sharpness_score = sharpness
            
            # Noise score (lower is better)
            noise_score = 1.0 - noise_level
            
            # Weighted combination
            quality_score = (
                brightness_score * 0.3 +
                contrast_score * 0.3 +
                sharpness_score * 0.2 +
                noise_score * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logging.warning(f"Error calculating quality score: {e}")
            return 0.5
    
    def _generate_recommendations(self, condition: LightingCondition, brightness: float,
                                contrast: float, sharpness: float, noise_level: float) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        try:
            if condition == LightingCondition.VERY_DARK:
                recommendations.extend([
                    "Apply strong gamma correction",
                    "Increase brightness significantly",
                    "Apply denoising",
                    "Use histogram equalization",
                    "Consider additional lighting"
                ])
            elif condition == LightingCondition.DARK:
                recommendations.extend([
                    "Apply moderate gamma correction",
                    "Increase brightness",
                    "Apply denoising",
                    "Use histogram equalization"
                ])
            elif condition == LightingCondition.LOW_LIGHT:
                recommendations.extend([
                    "Apply light gamma correction",
                    "Slightly increase brightness",
                    "Apply light denoising"
                ])
            elif condition == LightingCondition.BRIGHT:
                recommendations.extend([
                    "Reduce brightness",
                    "Apply gamma correction",
                    "Check for overexposure"
                ])
            elif condition == LightingCondition.VERY_BRIGHT:
                recommendations.extend([
                    "Significantly reduce brightness",
                    "Apply strong gamma correction",
                    "Check for severe overexposure"
                ])
            elif condition == LightingCondition.MIXED:
                recommendations.extend([
                    "Apply local histogram equalization",
                    "Use adaptive contrast enhancement",
                    "Consider HDR processing"
                ])
            elif condition == LightingCondition.BACKLIT:
                recommendations.extend([
                    "Apply backlight compensation",
                    "Use local contrast enhancement",
                    "Consider face detection for local adjustment"
                ])
            
            # Add specific recommendations based on metrics
            if noise_level > 0.7:
                recommendations.append("Apply strong denoising")
            elif noise_level > 0.4:
                recommendations.append("Apply moderate denoising")
            
            if sharpness < 0.3:
                recommendations.append("Apply sharpening")
            
            if contrast < 30:
                recommendations.append("Increase contrast")
            elif contrast > 100:
                recommendations.append("Reduce contrast")
            
        except Exception as e:
            logging.warning(f"Error generating recommendations: {e}")
            recommendations.append("Error in analysis")
        
        return recommendations
    
    def _apply_optimization(self, image: np.ndarray, condition: LightingCondition) -> np.ndarray:
        """Apply optimization based on lighting condition."""
        try:
            if condition not in self.optimization_params:
                return image
            
            params = self.optimization_params[condition]
            optimized = image.copy()
            
            # Apply gamma correction
            if params["gamma"] != 1.0:
                optimized = self._apply_gamma_correction(optimized, params["gamma"])
            
            # Apply brightness and contrast
            if params["brightness"] != 0 or params["contrast"] != 1.0:
                optimized = cv2.convertScaleAbs(
                    optimized, 
                    alpha=params["contrast"], 
                    beta=params["brightness"]
                )
            
            # Apply denoising
            if params["denoise"]:
                optimized = self._apply_denoising(optimized)
            
            # Apply histogram equalization
            if params["histogram_eq"]:
                optimized = self._apply_histogram_equalization(optimized)
            
            # Apply sharpening
            if params["sharpen"]:
                optimized = self._apply_sharpening(optimized)
            
            return optimized
            
        except Exception as e:
            logging.error(f"Error applying optimization: {e}")
            return image
    
    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction to image."""
        try:
            # Create gamma correction lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            
            # Apply gamma correction
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
    
    def _apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to image."""
        try:
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
                
        except Exception as e:
            logging.warning(f"Error applying histogram equalization: {e}")
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
    
    def _apply_face_optimizations(self, image: np.ndarray, analysis: LightingAnalysis) -> np.ndarray:
        """Apply face-specific optimizations."""
        try:
            optimized = image.copy()
            
            # Apply color balance correction
            if len(optimized.shape) == 3:
                b_balance, g_balance, r_balance = analysis.color_balance
                
                # Apply color balance
                optimized[:, :, 0] = np.clip(optimized[:, :, 0] * b_balance, 0, 255).astype(np.uint8)
                optimized[:, :, 1] = np.clip(optimized[:, :, 1] * g_balance, 0, 255).astype(np.uint8)
                optimized[:, :, 2] = np.clip(optimized[:, :, 2] * r_balance, 0, 255).astype(np.uint8)
            
            # Apply local contrast enhancement for face regions
            if analysis.condition in [LightingCondition.MIXED, LightingCondition.BACKLIT]:
                optimized = self._apply_local_contrast_enhancement(optimized)
            
            return optimized
            
        except Exception as e:
            logging.warning(f"Error applying face optimizations: {e}")
            return image
    
    def _apply_local_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply local contrast enhancement."""
        try:
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                
                # Apply CLAHE to L channel
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
                
                # Convert back to BGR
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image)
                
        except Exception as e:
            logging.warning(f"Error applying local contrast enhancement: {e}")
            return image
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "supported_conditions": list(LightingCondition),
            "optimization_params": self.optimization_params,
            "brightness_thresholds": self.brightness_thresholds
        }

# Global lighting optimizer instance
_lighting_optimizer = None

def get_lighting_optimizer() -> LightingOptimizer:
    """Get global lighting optimizer instance."""
    global _lighting_optimizer
    
    if _lighting_optimizer is None:
        _lighting_optimizer = LightingOptimizer()
    
    return _lighting_optimizer
