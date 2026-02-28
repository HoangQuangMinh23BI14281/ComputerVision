"""
Image preprocessing utilities for receipt OCR.
"""
import cv2
import numpy as np


def preprocess_for_detection(image, target_size=640):
    """
    Preprocess image for detection model input.
    
    Args:
        image: BGR numpy array (H, W, C)
        target_size: resize to this size (square)
    
    Returns:
        processed: float32 numpy array (target_size, target_size, 3), normalized
        scale_w, scale_h: scale factors for mapping back to original size
    """
    h, w = image.shape[:2]
    scale_w = target_size / w
    scale_h = target_size / h

    resized = cv2.resize(image, (target_size, target_size))

    # Normalize to [0, 1] then ImageNet stats
    processed = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    processed = (processed - mean) / std

    return processed, scale_w, scale_h


def preprocess_for_recognition(image, target_h=32, target_w=100):
    """
    Preprocess a cropped text region for recognition model.
    
    Args:
        image: BGR numpy array of cropped text region
        target_h: target height
        target_w: target width
    
    Returns:
        processed: float32 numpy array (target_h, target_w, 1), normalized
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Resize keeping aspect ratio, pad if needed
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w), dtype=np.float32)

    ratio = target_h / h
    new_w = min(int(w * ratio), target_w)
    resized = cv2.resize(gray, (new_w, target_h))

    # Pad to target_w
    padded = np.zeros((target_h, target_w), dtype=np.uint8)
    padded[:, :new_w] = resized

    # Normalize to [0, 1]
    processed = padded.astype(np.float32) / 255.0

    return processed


def enhance_image(image):
    """
    Apply contrast enhancement to a receipt image.
    
    Args:
        image: BGR numpy array
    Returns:
        enhanced: BGR numpy array
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    lab[:, :, 0] = l_enhanced

    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced
