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

def resize_image_720p(img):
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or invalid")
        
    h, w = img.shape[:2]
    if max(h, w) > 720:
        scale = 720 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
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
        return QPixmap()
        
    h, w = cv_img.shape[:2]
    if len(cv_img.shape) == 2:  # Grayscale
        qimg = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
    else:  # BGR
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_img.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

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
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or invalid")
    
    # Downscale for consistency
    img, h, w = resize_image_720p(img)
    diagonal = int(np.ceil((w**2 + h**2)**0.5))
    pad_y, pad_x = (diagonal - h) // 2, (diagonal - w) // 2
    # Use constant padding (black) instead of reflect to avoid artifacts
    img_padded = cv2.copyMakeBorder(img, pad_y, diagonal - h - pad_y, pad_x, diagonal - w - pad_x, cv2.BORDER_CONSTANT, value=0)
    gray = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV's DFT
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)  # Shift zero frequency to center
    
    # Compute magnitude spectrum
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    magnitude_bgr = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
    
    # Apply inner/outer filter if provided
    if inner is not None and outer is not None:
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        # Create distance map from center
        y, x = np.ogrid[:rows, :cols]
        distances = np.sqrt((x - ccol)**2 + (y - crow)**2)
        # Scale radii based on diagonal
        inner_r = inner * diagonal // 200 if inner != 0 else 0  # Start at center if inner = 0
        outer_r = outer * diagonal // 200
        
        # Create annular mask
        mask = (distances >= inner_r) & (distances <= outer_r)
        # Dim the selected region
        magnitude_bgr[mask] = magnitude_bgr[mask] * 0.3  # 70% dimming
    
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
    if inner >= outer:
        return img
    
    # Downscale for consistency
    img, h, w = resize_image_720p(img)
    
    # Pad image
    diagonal = int(np.ceil((w**2 + h**2)**0.5))
    pad_y, pad_x = (diagonal - h) // 2, (diagonal - w) // 2
    img_padded = cv2.copyMakeBorder(img, pad_y, diagonal - h - pad_y, pad_x, diagonal - w - pad_x, cv2.BORDER_CONSTANT)
    
    # Convert to YCrCb and split channels
    ycrcb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # Use OpenCV's DFT
    dft = cv2.dft(np.float32(y), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Create and apply mask
    rows, cols = y.shape
    crow, ccol = rows // 2, cols // 2
    distances = np.sqrt((np.arange(cols) - ccol)**2 + (np.arange(rows)[:, None] - crow)**2)
    inner_r, outer_r = inner * diagonal // 200, outer * diagonal // 200
    mask = np.ones((rows, cols, 2), np.float32)
    mask[(distances >= inner_r) & (distances <= outer_r), :] = 0
    dft_shift *= mask
    
    # Inverse DFT
    f_ishift = np.fft.ifftshift(dft_shift)
    y_filtered = cv2.idft(f_ishift)
    y_filtered = cv2.magnitude(y_filtered[:,:,0], y_filtered[:,:,1])
    y_filtered = cv2.normalize(y_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Reconstruct and crop
    img_filtered = cv2.cvtColor(cv2.merge([y_filtered, cr, cb]), cv2.COLOR_YCrCb2BGR)
    img_filtered = img_filtered[pad_y:pad_y+h, pad_x:pad_x+w]
    return cv2.resize(img_filtered, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    
def apply_sharpening(slider, image):
    # Unsharp masking: sharpen by subtracting a blurred version
    edges = cv2.Canny(image, 100, 200)
    edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    sigma = 0.5 + edge_density  # Range 0.5 to 1.5
    blurred = cv2.GaussianBlur(image, (3, 3), sigma)
    sharpened_image = cv2.addWeighted(image, 1 + slider, blurred, -slider, 0)
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
    """Apply noise reduction using fast NL means denoising with adaptive strength.
    
    Args:
        strength (float): Strength of noise reduction (0.0 to 1.0).
        img (np.ndarray): Input image in BGR format.
    
    Returns:
        np.ndarray: Denoised image in BGR format.
    """
    if strength <= 0:
        return img.copy()
    
    downscaled, _, _ = resize_image_720p(img)
    sigma = int(strength * 50)
    denoised = cv2.bilateralFilter(downscaled, d=5, sigmaColor=sigma, sigmaSpace=sigma)
    if downscaled.shape != img.shape:
        denoised = cv2.resize(denoised, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return denoised
    
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
    
def apply_tone_mapping(strength, img):
    """Apply simple tone mapping to simulate HDR adjustment."""
    if strength <= 0:
        return img.copy()
    # Normalize to float32 for processing
    img_float = img.astype(np.float32) / 255.0
    # Logarithmic tone mapping
    tone_mapped = np.log1p(img_float * strength) / np.log1p(strength)
    return cv2.normalize(tone_mapped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
def apply_edge_detection(img, low_threshold, high_threshold, apply=False):
    """Apply Canny edge detection and compute edge density."""
    if not apply:
        return img.copy(), 0.0
    edges = cv2.Canny(img, low_threshold, high_threshold)
    edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])  # Proportion of edge pixels
    img_with_edges = img.copy()
    img_with_edges[edges != 0] = [255, 255, 255]  # White edges
    return img_with_edges, edge_density

def apply_orb_keypoints(img, max_keypoints, apply=False):
    """Apply ORB keypoint detection and compute average strength."""
    if not apply:
        return img.copy(), 0, 0.0
    orb = cv2.ORB_create(nfeatures=max_keypoints)
    keypoints = orb.detect(img, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    avg_strength = sum(kp.response for kp in keypoints) / len(keypoints) if keypoints else 0.0
    return img_with_keypoints, len(keypoints), avg_strength
    
def apply_clahe(img, apply, clip_limit):
    if not apply or clip_limit == 0:
        return img.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)