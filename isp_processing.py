"""
Simple ISP Tuning Tool - Image Processing Module
Provides functions for converting OpenCV images to Qt format, generating histograms,
computing frequency domain representations, and applying Fourier filters for real-time
image processing in a PyQt5 application.

Author: Ido Okashi
Date: February 24, 2025
"""

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap

def convert_cv_qt(cv_img):
    """Convert an OpenCV image to a QPixmap for display in Qt.

    Args:
        cv_img (np.ndarray): Input image in BGR format from OpenCV.

    Returns:
        QPixmap: The converted image ready for Qt display in RGB format.
    """
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QPixmap.fromImage(QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888))

def generate_histogram(image):
    """Generate a histogram visualization of an image as a QPixmap.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        QPixmap: A pixmap representing the grayscale histogram of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])  # Compute histogram
    hist_img = np.zeros((150, 256, 3), dtype=np.uint8)  # Create blank image for histogram
    cv2.normalize(hist, hist, 0, 150, cv2.NORM_MINMAX)  # Normalize to fit height
    for x, y in enumerate(hist):
        cv2.line(hist_img, (x, 150), (x, 150 - int(y[0])), (255, 255, 255))  # Draw bars
    return convert_cv_qt(hist_img)

def compute_frequency_image(img, inner=None, outer=None):
    """Compute a frequency domain visualization with optional filter mask overlay.

    Args:
        img (np.ndarray): Input image in BGR format.
        inner (int, optional): Inner radius of the filter mask (0-100). If provided with outer, overlays mask.
        outer (int, optional): Outer radius of the filter mask (0-100). If provided with inner, overlays mask.

    Returns:
        np.ndarray: BGR image of the frequency magnitude spectrum, with filter mask overlaid if specified.
    """
    if max(img.shape[:2]) > 1080:  # Resize if too large
        scale = 1080 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute frequency magnitude spectrum
    f = np.fft.fft2(gray)  # Compute 2D FFT
    fshift = np.fft.fftshift(f)  # Shift zero frequency to center
    magnitude = 20 * np.log(np.abs(fshift) + 1)  # Log scale for visibility
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    magnitude_bgr = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    if inner is not None and outer is not None:
        # Create filter mask
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        distances = np.sqrt((np.arange(cols) - ccol)**2 + (np.arange(rows)[:, None] - crow)**2)
        diagonal = int(np.ceil((cols**2 + rows**2)**0.5))
        inner_r, outer_r = inner * diagonal // 200, outer * diagonal // 200
        mask = np.ones((rows, cols), np.float32)
        mask[(distances >= inner_r) & (distances <= outer_r)] = 0  # Filtered areas

        # Overlay mask on magnitude (e.g., red tint for filtered areas)
        mask_bgr = np.zeros_like(magnitude_bgr, dtype=np.uint8)
        mask_bgr[..., 2] = (1 - mask) * 255  # Red channel for filtered areas
        magnitude_bgr = cv2.addWeighted(magnitude_bgr, 0.7, mask_bgr, 0.3, 0)  # Blend with transparency
    
    return magnitude_bgr

def apply_fourier_filter(img, inner, outer):
    """Apply a band-pass Fourier filter to remove specific frequency components.

    Args:
        img (np.ndarray): Input image in BGR format.
        inner (int): Inner radius of the frequency band to filter out (0-100).
        outer (int): Outer radius of the frequency band to filter out (0-100).

    Returns:
        np.ndarray: Filtered image in BGR format with specified frequencies removed.
    """
    if inner >= outer:  # Skip if invalid range
        return img
    h, w = img.shape[:2]
    if h > 1080:  # Downscale if height exceeds 1080
        scale = 1080 / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    
    # Pad image to square with diagonal size for FFT
    diagonal = int(np.ceil((w**2 + h**2)**0.5))
    pad_y, pad_x = (diagonal - h) // 2, (diagonal - w) // 2
    img_padded = cv2.copyMakeBorder(img, pad_y, diagonal - h - pad_y, pad_x, diagonal - w - pad_x, cv2.BORDER_REFLECT)
    
    # Convert to YCrCb and process luminance channel
    ycrcb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    f = np.fft.fft2(np.float32(y))  # FFT on luminance
    fshift = np.fft.fftshift(f)
    
    # Create annular mask to filter frequencies
    rows, cols = y.shape
    crow, ccol = rows // 2, cols // 2
    distances = np.sqrt((np.arange(cols) - ccol)**2 + (np.arange(rows)[:, None] - crow)**2)
    inner_r, outer_r = inner * diagonal // 200, outer * diagonal // 200
    mask = np.ones((rows, cols), np.float32)
    mask[(distances >= inner_r) & (distances <= outer_r)] = 0  # Zero out band
    
    # Apply mask and inverse FFT
    fshift *= mask
    y_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    y_filtered = cv2.normalize(y_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Reconstruct and crop image
    img_filtered = cv2.cvtColor(cv2.merge([y_filtered, cr, cb]), cv2.COLOR_YCrCb2BGR)
    img_filtered = img_filtered[pad_y:pad_y+h, pad_x:pad_x+w]
    return cv2.resize(img_filtered, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)