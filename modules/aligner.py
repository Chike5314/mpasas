"""
MPASAS – Aligner Module
Straightens student script images to match the blank template
using ORB feature matching and perspective homography.
"""
import cv2
import numpy as np
from typing import Optional, Tuple


def load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        # Try with PIL as fallback
        try:
            from PIL import Image
            pil = Image.open(path).convert('RGB')
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img


def preprocess_for_matching(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    # CLAHE to improve contrast under varying lighting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def align_to_template(
    template_path: str,
    script_path: str,
    min_good_matches: int = 8,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Warp *script_path* so it aligns pixel-perfectly with *template_path*.

    Returns
    -------
    aligned_img : np.ndarray or None
        BGR image the same size as the template, or None on hard failure.
    note : str
        Human-readable description of what happened.
    """
    template_bgr = load_image(template_path)
    script_bgr   = load_image(script_path)

    if template_bgr is None:
        return None, "Could not load template image"
    if script_bgr is None:
        return None, "Could not load script image"

    tH, tW = template_bgr.shape[:2]

    template_gray = preprocess_for_matching(template_bgr)
    script_gray   = preprocess_for_matching(script_bgr)

    # ── ORB detection ───────────────────────────────────────────────────────
    orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8)
    kp_t, des_t = orb.detectAndCompute(template_gray, None)
    kp_s, des_s = orb.detectAndCompute(script_gray, None)

    if des_t is None or des_s is None or len(kp_t) < 4 or len(kp_s) < 4:
        aligned = _resize_fallback(script_bgr, tW, tH)
        return aligned, f"Not enough ORB keypoints (template={len(kp_t)}, script={len(kp_s)}); used resize fallback"

    # ── Matching ─────────────────────────────────────────────────────────────
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_t, des_s, k=2)

    # Lowe ratio test
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < min_good_matches:
        aligned = _resize_fallback(script_bgr, tW, tH)
        return aligned, f"Weak match ({len(good)} good matches < {min_good_matches}); used resize fallback"

    # ── Homography ────────────────────────────────────────────────────────────
    src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if H is None:
        aligned = _resize_fallback(script_bgr, tW, tH)
        return aligned, "Homography failed; used resize fallback"

    inliers = int(np.sum(mask))
    aligned = cv2.warpPerspective(script_bgr, H, (tW, tH),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    return aligned, f"Aligned OK ({len(good)} matches, {inliers} inliers)"


def _resize_fallback(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Simple resize – better than nothing when matching fails."""
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
