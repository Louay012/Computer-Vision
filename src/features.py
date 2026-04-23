"""Feature extraction helpers used by the ML pipeline.

Exports `extract_features(rgb, mask=None)` which returns (vec, details).
The returned vector matches the columns in `results/combined_run/features/features.csv`:
  [h_mean,s_mean,v_mean,h_std,s_std,v_std,contrast,energy,homogeneity,area,perimeter,circularity]

This is a lightweight implementation intended to be compatible with the project's
existing CSV features and classical models.
"""
from typing import Tuple, Dict
import numpy as np
import cv2

try:
    from skimage.feature import greycomatrix, greycoprops
    from skimage import img_as_ubyte
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

from .preprocessing import generate_auto_mask_from_bgr


def _texture_props(gray: np.ndarray) -> Tuple[float, float, float]:
    # Compute simple GLCM texture features (contrast, energy, homogeneity)
    # Fallback to simple statistics if skimage is not available.
    try:
        if not _HAS_SKIMAGE:
            raise ImportError
        # quantize to 64 levels for speed
        gray_u8 = img_as_ubyte(gray)
        levels = 64
        # rescale to the levels
        bins = np.linspace(0, 256, levels + 1)
        gray_q = np.digitize(gray_u8, bins) - 1
        gray_q = (gray_q * (255 // (levels - 1))).astype('uint8')
        glcm = greycomatrix(gray_q, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
        contrast = float(greycoprops(glcm, 'contrast')[0, 0])
        energy = float(greycoprops(glcm, 'energy')[0, 0])
        homogeneity = float(greycoprops(glcm, 'homogeneity')[0, 0])
        return contrast, energy, homogeneity
    except Exception:
        # Fallback: use variance, inverse variance and normalized Laplacian energy
        contrast = float(np.var(gray.astype('float32')))
        # energy ~ sum of squared normalized histogram
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-9)
        energy = float(np.sum(hist ** 2))
        homogeneity = float(1.0 / (1.0 + contrast))
        return contrast, energy, homogeneity


def extract_features(rgb: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """Extract a compact feature vector from an RGB image.

    Parameters
    - rgb: HxWx3 uint8 RGB image (0-255)
    - mask: optional HxW uint8 mask (0 or 255) where 255 indicates foreground.

    Returns (vec, details) where vec is a 1D numpy array matching the project's
    features CSV, and details is a dict with extra values (mask, shape stats).
    """
    if rgb is None:
        raise ValueError('rgb is None')
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype('uint8') if rgb.max() <= 1.0 else rgb.astype('uint8')

    # Ensure mask
    if mask is None:
        # generate mask expects BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = generate_auto_mask_from_bgr(bgr)
    # normalize mask to boolean
    mask_bool = (mask > 0)
    if mask_bool.sum() == 0:
        # fallback to whole image
        mask_bool = np.ones(rgb.shape[:2], dtype=bool)

    # HSV stats (OpenCV ranges H:0-179, S:0-255, V:0-255)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype('float32')
    s = hsv[:, :, 1].astype('float32')
    v = hsv[:, :, 2].astype('float32')
    h_mean = float(np.mean(h[mask_bool]))
    s_mean = float(np.mean(s[mask_bool]))
    v_mean = float(np.mean(v[mask_bool]))
    h_std = float(np.std(h[mask_bool]))
    s_std = float(np.std(s[mask_bool]))
    v_std = float(np.std(v[mask_bool]))

    # Texture features on grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_masked = gray.copy()
    gray_masked[~mask_bool] = 0
    contrast, energy, homogeneity = _texture_props(gray_masked)

    # Shape features: area, perimeter, circularity
    area = float(int(mask_bool.sum()))
    # perimeter: find largest contour
    contours, _ = cv2.findContours((mask_bool.astype('uint8') * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0.0
    if contours:
        # choose the largest contour by area
        c = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(c, True))
    # circularity: 4*pi*area / perimeter^2
    circularity = 0.0
    if perimeter > 1e-6:
        circularity = float(4.0 * np.pi * area / (perimeter ** 2))

    vec = np.array([
        h_mean, s_mean, v_mean,
        h_std, s_std, v_std,
        contrast, energy, homogeneity,
        area, perimeter, circularity
    ], dtype='float32')

    details = {
        'mask': mask,
        'shape': {'area': area, 'perimeter': perimeter, 'circularity': circularity},
    }
    return vec, details
