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

def resize_image_1080p(img):
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or invalid")
        
    h, w = img.shape[:2]
    if h > 1080:  # Downscale if height exceeds 1080
        scale = 1080 / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    return img, h, w 

def convert_cv_qt(cv_img):
    """Convert an OpenCV image to a QPixmap for display in Qt.

    Args:
        cv_img (np.ndarray): Input image in BGR format from OpenCV.

    Returns:
        QPixmap: The converted image ready for Qt display in RGB format.
    """
    if cv_img is None or cv_img.size == 0:
        raise ValueError("Cannot convert empty or invalid image to QPixmap")

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
    """Compute a frequency domain visualization with optional black filter mask overlay."""
    if img is None or img.size == 0:
        raise ValueError("Input image to compute_frequency_image is empty or invalid")
        
    img, h, w = resize_image_1080p(img)
    diagonal = int(np.ceil((w**2 + h**2)**0.5))
    pad_y, pad_x = (diagonal - h) // 2, (diagonal - w) // 2
    img_padded = cv2.copyMakeBorder(img, pad_y, diagonal - h - pad_y, pad_x, diagonal - w - pad_x, cv2.BORDER_REFLECT)
    gray = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)
    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    if not np.all(np.isfinite(magnitude)):
        magnitude[~np.isfinite(magnitude)] = 20 * np.log(np.finfo(np.float64).max)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    magnitude_bgr = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    if inner is not None and outer is not None:
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        distances = np.sqrt((np.arange(cols) - ccol)**2 + (np.arange(rows)[:, None] - crow)**2)
        # Set minimum inner radius if needed, but include center when inner = 0
        inner_r = inner * diagonal // 200 if inner != 0 else 0  # Start at center when inner = 0
        outer_r = outer * diagonal // 200
        
        # Directly set ring pixels to black, including center when inner = 0
        magnitude_bgr[(distances >= inner_r) & (distances <= outer_r)] = [0, 0, 0]
    
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

    img, h, w = resize_image_1080p(img)
    
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
    
def apply_sharpening(slider, image):
    # Unsharp masking: sharpen by subtracting a blurred version
    strength = slider * 0.05  # Scale 0-5
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)  # Fixed kernel and sigma
    sharpened_image = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened_image

def apply_lens_shading(strength, img):
    """Apply lens shading correction by adjusting edge intensity based on strength.
    
    Args:
        strength (float): Strength of the effect (-1.0 to 1.0). Negative darkens edges, positive brightens edges.
        img (np.ndarray): Input image in BGR format.
    
    Returns:
        np.ndarray: Image with lens shading correction applied in BGR format.
    """
    if strength == 0:  # No effect at 0 strength
        return img.copy()
    
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    diagonal = np.sqrt(w**2 + h**2)

    y, x = np.indices((h, w))
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Normalize distance (0 at center, 1 at max edge)
    normalized_distance = distances / (diagonal / 2)
    
    # Mask: 1 at center, adjusts based on strength (negative darkens, positive brightens)
    mask = 1 + strength * normalized_distance
    mask = np.clip(mask, 0, None)  # No upper clip to allow brightening above 1
    
    return cv2.convertScaleAbs(img * mask[..., np.newaxis])

def adjust_rgb_gain(img, gain, channel):
    """Adjust gain for a specific RGB channel.
    
    Args:
        img (np.ndarray): Input image in BGR format.
        gain (float): Gain factor (e.g., 0.0 to 2.0).
        channel (str): Channel to adjust ('R', 'G', 'B').
    
    Returns:
        np.ndarray: Image with adjusted channel gain in BGR format.
    """
    channels = list(cv2.split(img))  # Convert tuple to list for mutability
    if channel == 'B':
        channels[0] = cv2.convertScaleAbs(channels[0], alpha=gain)
    elif channel == 'G':
        channels[1] = cv2.convertScaleAbs(channels[1], alpha=gain)
    elif channel == 'R':
        channels[2] = cv2.convertScaleAbs(channels[2], alpha=gain)
    return cv2.merge(channels)
    
def adjust_white_balance(img, temperature, tint):
    """Adjust white balance using temperature (warm/cool) and tint (green/magenta).
    
    Args:
        img (np.ndarray): Input image in BGR format.
        temperature (float): Temperature adjustment (-1.0 cool to 1.0 warm).
        tint (float): Tint adjustment (-1.0 magenta to 1.0 green).
    
    Returns:
        np.ndarray: White-balanced image in BGR format.
    """
    if temperature == 0 and tint == 0:
        return img.copy()
    
    # Simplified white balance: adjust RGB gains based on temperature and tint
    r_gain = 1.0 + temperature * 0.5  # Warm increases red
    b_gain = 1.0 - temperature * 0.5  # Cool increases blue
    g_gain = 1.0 + tint * 0.5         # Green increases with positive tint

    channels = list(cv2.split(img))
    channels[0] = cv2.convertScaleAbs(channels[0], alpha=b_gain)  # Blue
    channels[1] = cv2.convertScaleAbs(channels[1], alpha=g_gain)  # Green
    channels[2] = cv2.convertScaleAbs(channels[2], alpha=r_gain)  # Red
    return cv2.merge(channels)

def apply_noise_reduction(strength, img):
    """Apply noise reduction using bilateral filtering.
    
    Args:
        strength (float): Strength of noise reduction (0.0 to 1.0).
        img (np.ndarray): Input image in BGR format.
    
    Returns:
        np.ndarray: Denoised image in BGR format.
    """
    if strength <= 0:
        return img.copy()
    
    # Bilateral filter: preserves edges while reducing noise
    sigma = int(strength * 50)  # Scale strength to sigma values (0-50)
    return cv2.bilateralFilter(img, d=9, sigmaColor=sigma, sigmaSpace=sigma)
    
def apply_gamma_correction(gamma, img):
    """Apply gamma correction to adjust image brightness and contrast.
    
    Args:
        gamma (float): Gamma value (0.1 to 5.0, where <1 darkens, >1 brightens).
        img (np.ndarray): Input image in BGR format.
    
    Returns:
        np.ndarray: Gamma-corrected image in BGR format.
    """
    if gamma <= 0.1:  # Minimum gamma to avoid extreme darkening
        gamma = 0.1
    
    # Create a lookup table for gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply the lookup table to the image
    return cv2.LUT(img, table)