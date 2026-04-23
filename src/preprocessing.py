import cv2
import numpy as np

def generate_auto_mask_from_bgr(bgr: np.ndarray) -> np.ndarray:
    """Generate a binary mask isolating the foreground (leaf) from a BGR image.

    Uses Otsu thresholding on the blurred grayscale image and some morphology
    to remove noise. Returns a uint8 mask with values 0 or 255.
    """
    if bgr is None:
        raise ValueError('bgr image is None')
    if len(bgr.shape) == 3 and bgr.shape[2] == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # In some images background is white — try inverting if mask covers most of image
    h, w = mask.shape[:2]
    if mask.mean() > 250:
        mask = cv2.bitwise_not(mask)

    # Morphological open + close to clean small holes/noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Ensure mask is binary 0/255
    mask = (mask > 0).astype('uint8') * 255
    return mask
