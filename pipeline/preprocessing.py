"""Faithful P2 inference pipeline from notebook cells 66, 75, 78 and 107.

No augmentation. Input is RGB uint8; output is ImageNet-normalized [1,3,260,260].
"""
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

EPS = 1e-6
CLASS_NAMES = ["Normal", "DR", "ARMD", "Drusen", "ODC"]
CLASS_TO_IDX = {name:i for i,name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = 260

def get_retinal_roi(image, threshold=10, padding=10):
    """
    Conservative retinal-field ROI extraction.

    Parameters:
        image: RGB uint8 image
        threshold: threshold for detecting non-black retinal pixels
        padding: extra pixels retained around detected retinal field

    Returns:
        roi: cropped RGB image
        bbox: (x1, y1, x2, y2)
    """

    # Convert RGB -> grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect non-black region
    mask = (gray > threshold).astype(np.uint8) * 255

    # Remove tiny noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image.copy(), (0, 0, image.shape[1], image.shape[0])

    # Largest connected component
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add conservative padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    roi = image[y1:y2, x1:x2]

    return roi, (x1, y1, x2, y2)

def p2_retinex(image, sigma=30, strength=0.5):
    """
    P2: Conservative Single-Scale Retinex (SSR)

    Corrects illumination variation while retaining
    the natural retinal appearance.
    """

    # Convert RGB -> LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    L, A, B = cv2.split(lab)

    # Normalize luminance
    L_float = L.astype(np.float32) / 255.0
    L_safe = np.clip(L_float, EPS, 1.0)

    # Estimate illumination
    illumination = cv2.GaussianBlur(
        L_safe,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    illumination = np.clip(
        illumination,
        EPS,
        1.0
    )

    # Single-scale Retinex
    retinex = (
        np.log(L_safe)
        - np.log(illumination)
    )

    # Robust normalization
    low, high = np.percentile(
        retinex,
        [2, 98]
    )

    retinex_norm = np.clip(
        (retinex - low) /
        (high - low + EPS),
        0,
        1
    )

    # Blend Retinex result with original luminance
    L_corrected = (
        (1.0 - strength) * L_float
        + strength * retinex_norm
    )

    L_corrected = np.clip(
        L_corrected * 255,
        0,
        255
    ).astype(np.uint8)

    # Reconstruct LAB
    result_lab = cv2.merge([
        L_corrected,
        A,
        B
    ])

    # Convert back to RGB
    result = cv2.cvtColor(
        result_lab,
        cv2.COLOR_LAB2RGB
    )

    return result

def resize_with_padding(image, target_size=260):
    """
    Resize image while preserving aspect ratio.

    The shorter dimension is padded using reflection
    instead of introducing large black borders.
    """

    h, w = image.shape[:2]

    scale = target_size / max(h, w)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Calculate padding
    pad_h = target_size - new_h
    pad_w = target_size - new_w

    top = pad_h // 2
    bottom = pad_h - top

    left = pad_w // 2
    right = pad_w - left

    # Reflection padding
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_REFLECT_101
    )

    return padded


def prepare_image(image: Image.Image):
    """Returns tensor, enhanced PIL image, and display-only retinal FOV mask.

    Mask is obtained from the original ROI prior to Retinex; it only limits
    visual display of attribution and does not alter the model input.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    roi, bbox = get_retinal_roi(rgb)
    enhanced = p2_retinex(roi)
    resized = resize_with_padding(enhanced, target_size=IMAGE_SIZE)
    tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(transforms.ToTensor()(Image.fromarray(resized))).unsqueeze(0)

    # Visualization mask only. Resize original RGB ROI with identical geometry;
    # threshold removes the dark camera background, where practical.
    original_resized = resize_with_padding(roi, target_size=IMAGE_SIZE)
    gray = cv2.cvtColor(original_resized, cv2.COLOR_RGB2GRAY)
    mask = (gray > 10).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return tensor, Image.fromarray(resized), mask.astype(bool)
