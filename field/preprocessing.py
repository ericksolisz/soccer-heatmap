

import cv2
import numpy as np
from typing import List


def apply_clahe_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to the L channel in LAB space and return a BGR image.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    lab_clahe = cv2.merge((cl, a, b))
    clahe_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return clahe_bgr


def build_line_emphasis_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Use Canny + HoughLinesP to detect strong lines (e.g., pitch lines)
    and return a 3-channel mask highlighting those lines.

    The mask has white lines on black background (BGR).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Canny edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

    # Probabilistic Hough transform to get line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=80,
        maxLineGap=10
    )

    # Create a blank mask
    mask = np.zeros_like(frame_bgr, dtype=np.uint8)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)

    return mask


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for field keypoints:

    1. CLAHE to enhance contrast.
    2. Canny + HoughLinesP to detect strong lines.
    3. Blend the line mask with the CLAHE image to emphasize pitch lines.

    Returns a BGR image suitable for YOLOv8 pose.
    """
    # 1) CLAHE
    clahe_frame = apply_clahe_bgr(frame_bgr)

    # 2) Line mask (white lines on black)
    line_mask = build_line_emphasis_mask(clahe_frame)

    # 3) Blend mask with the CLAHE image
    #    We make lines brighter by adding a weighted version of the mask
    alpha = 0.8  # weight for CLAHE image
    beta = 0.7   # weight for line mask
    gamma = 0.0  # scalar added

    enhanced = cv2.addWeighted(clahe_frame, alpha, line_mask, beta, gamma)

    return enhanced


def preprocess_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply the preprocessing pipeline to a list of frames.
    """
    return [preprocess_frame(f) for f in frames]
