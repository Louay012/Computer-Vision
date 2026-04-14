"""Feature extraction helpers: couleur, texture (GLCM), forme.

Fonctions principales:
- color_histogram_rgb / color_histogram_hsv: histogrammes normalisés
- glcm_features: propriétés GLCM (contrast, homogeneity, energy, ...)
- shape_features: surface, périmètre, circularité (sur un masque binaire)
- extract_features: wrapper qui concatène les features

Assume les images en format RGB (uint8) pour les entrées courantes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Import GLCM functions from scikit-image with compatibility across versions
try:
    # modern/US-spelling exports
    from skimage.feature import graycomatrix as greycomatrix, graycoprops as greycoprops
except Exception:
    try:
        # older/british-spelling exports
        from skimage.feature import greycomatrix, greycoprops
    except Exception:
        # attempt submodule path
        from skimage.feature.texture import greycomatrix, greycoprops


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        arr = np.clip(img, 0.0, 1.0) * 255.0
        return arr.astype(np.uint8)
    # other integer types: scale down if necessary
    info = np.iinfo(img.dtype)
    if info.max <= 255:
        return img.astype(np.uint8)
    return ((img.astype(np.float32) / info.max) * 255.0).astype(np.uint8)


def color_histogram_rgb(image: np.ndarray, bins: int = 32, mask: Optional[np.ndarray] = None, normalize: bool = True) -> np.ndarray:
    """Return a concatenated RGB histogram vector (R,G,B).

    image: RGB uint8 image (H,W,3)
    mask: optional binary mask (same HxW) to compute hist over object
    """
    img = _to_uint8(image)
    chans = cv2.split(img)
    feats: List[np.ndarray] = []
    for ch in chans:
        if mask is not None:
            vals = ch[mask.astype(bool)]
            hist, _ = np.histogram(vals, bins=bins, range=(0, 256))
        else:
            hist, _ = np.histogram(ch, bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        if normalize:
            s = hist.sum()
            if s > 0:
                hist /= s
        feats.append(hist)
    return np.concatenate(feats)


def color_histogram_hsv(image: np.ndarray, bins: int = 32, mask: Optional[np.ndarray] = None, normalize: bool = True) -> np.ndarray:
    """Return a concatenated HSV histogram vector (H,S,V).

    Assumes input `image` is RGB; conversion to HSV est faite en interne.
    """
    img = _to_uint8(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    feats: List[np.ndarray] = []
    for ch in cv2.split(hsv):
        if mask is not None:
            vals = ch[mask.astype(bool)]
            hist, _ = np.histogram(vals, bins=bins, range=(0, 256))
        else:
            hist, _ = np.histogram(ch, bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        if normalize:
            s = hist.sum()
            if s > 0:
                hist /= s
        feats.append(hist)
    return np.concatenate(feats)


def glcm_features(image: np.ndarray, distances: List[int] = [1], angles: List[float] = [0.0], levels: int = 8, properties: Optional[List[str]] = None, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute a set of GLCM texture properties on the grayscale image.

    Returns a dict with keys = properties and scalar values (mean over distances/angles).
    """
    if properties is None:
        properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

    img = _to_uint8(image)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    if mask is not None:
        gray = gray.copy()
        gray[~mask.astype(bool)] = 0

    # Quantize to `levels` gray levels required by skimage.greycomatrix
    gray_q = np.floor((gray.astype(np.float32) * (levels - 1) / 255.0)).astype(np.uint8)

    glcm = greycomatrix(gray_q, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    feat_vals: Dict[str, float] = {}
    for prop in properties:
        try:
            vals = greycoprops(glcm, prop)
            feat_vals[prop] = float(np.nanmean(vals))
        except Exception:
            feat_vals[prop] = float("nan")
    return feat_vals


def shape_features(mask: np.ndarray) -> Dict[str, float]:
    """Compute area, perimeter and circularity from a binary mask.

    If multiple contours exist, the largest by area is used.
    """
    if mask is None:
        return {"area": 0.0, "perimeter": 0.0, "circularity": 0.0}

    m = (mask > 0).astype(np.uint8) * 255
    contours_info = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if not contours:
        return {"area": 0.0, "perimeter": 0.0, "circularity": 0.0}

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, True))
    circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
    return {"area": area, "perimeter": perimeter, "circularity": circularity}


def extract_features(image: np.ndarray, mask: Optional[np.ndarray] = None, bins: int = 32, glcm_distances: List[int] = [1], glcm_angles: List[float] = [0.0], glcm_levels: int = 8) -> Tuple[np.ndarray, Dict]:
    """Compute and return a concatenated feature vector and a details dict.

    Returns (vector, details) where `vector` is a 1D numpy array and `details` contains the
    individual components (`rgb_hist`, `hsv_hist`, `glcm`, `shape`).
    """
    rgb_hist = color_histogram_rgb(image, bins=bins, mask=mask)
    hsv_hist = color_histogram_hsv(image, bins=bins, mask=mask)
    glcm = glcm_features(image, distances=glcm_distances, angles=glcm_angles, levels=glcm_levels, mask=mask)
    shape = shape_features(mask)

    # Keep a stable order for GLCM properties
    glcm_order = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    glcm_vec = np.array([glcm.get(k, np.nan) for k in glcm_order], dtype=np.float32)
    shape_vec = np.array([shape["area"], shape["perimeter"], shape["circularity"]], dtype=np.float32)

    vector = np.concatenate([rgb_hist.astype(np.float32), hsv_hist.astype(np.float32), glcm_vec, shape_vec])
    details = {"rgb_hist": rgb_hist, "hsv_hist": hsv_hist, "glcm": glcm, "shape": shape}
    return vector, details


__all__ = [
    "color_histogram_rgb",
    "color_histogram_hsv",
    "glcm_features",
    "shape_features",
    "extract_features",
]
