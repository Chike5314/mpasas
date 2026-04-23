"""
MPASAS – Extractor Module
Relative-coordinate OMR bubble detection using pixel-density analysis.
Works with any template layout defined in the calibrator.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Zone ROI helper ──────────────────────────────────────────────────────────

def extract_roi(image: np.ndarray, zone: dict) -> np.ndarray:
    """Crop the zone ROI from the image using relative (0-1) coords."""
    h, w = image.shape[:2]
    x  = max(0, int(zone['xPct'] * w))
    y  = max(0, int(zone['yPct'] * h))
    x2 = min(w, int((zone['xPct'] + zone['wPct']) * w))
    y2 = min(h, int((zone['yPct'] + zone['hPct']) * h))
    return image[y:y2, x:x2]


# ── Pre-processing ────────────────────────────────────────────────────────────

def _binarise(roi: np.ndarray) -> np.ndarray:
    """Convert ROI to a binary (dark marks = 255) image."""
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # CLAHE then adaptive threshold
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq      = clahe.apply(gray)
    blurred = cv2.GaussianBlur(eq, (5, 5), 0)
    binary  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=6,
    )
    return binary


# ── Core bubble-grid analysis ────────────────────────────────────────────────

def analyse_grid(
    roi: np.ndarray,
    rows: int,
    cols: int,
    labels: Optional[List[str]] = None,
    fill_threshold: float = 0.10,
) -> Dict[int, dict]:
    """
    Divide *roi* into a rows×cols grid and detect filled bubbles.

    Parameters
    ----------
    roi             : BGR or grayscale zone image (already cropped)
    rows            : number of questions (rows)
    cols            : number of options per question (columns)
    labels          : option labels; defaults to A B C D …
    fill_threshold  : minimum dark-pixel ratio to count as 'marked'

    Returns
    -------
    {1: {'answer': 'B', 'confidence': 0.82, 'fill_ratios': [...], 'status': 'ok'}, …}
    """
    if labels is None:
        labels = [chr(65 + i) for i in range(cols)]

    binary = _binarise(roi)
    rH, rW = binary.shape

    cell_h = rH / rows
    cell_w = rW / cols

    results: Dict[int, dict] = {}

    for r in range(rows):
        q_num = r + 1
        fill_ratios = []

        for c in range(cols):
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)

            cell = binary[y1:y2, x1:x2]

            # Inset by ~8% to ignore printed grid lines / box borders
            margin_y = max(2, int((y2 - y1) * 0.08))
            margin_x = max(2, int((x2 - x1) * 0.08))
            inner = cell[margin_y:-margin_y, margin_x:-margin_x]

            if inner.size == 0:
                fill_ratios.append(0.0)
                continue

            ratio = float(np.sum(inner > 127)) / float(inner.size)
            fill_ratios.append(round(ratio, 4))

        max_r = max(fill_ratios) if fill_ratios else 0.0

        if max_r < fill_threshold:
            results[q_num] = {
                'answer':      None,
                'confidence':  0.0,
                'fill_ratios': fill_ratios,
                'status':      'blank',
            }
            continue

        # How many options are 'significantly' marked?
        threshold_for_multi = fill_threshold * 0.55
        marked_cols = [i for i, rat in enumerate(fill_ratios) if rat >= threshold_for_multi]

        best_col = fill_ratios.index(max_r)
        label    = labels[best_col] if best_col < len(labels) else str(best_col + 1)

        if len(marked_cols) > 1:
            status     = 'ambiguous'
            confidence = max_r / (sum(fill_ratios) + 1e-9)
        else:
            status     = 'ok'
            confidence = max_r

        results[q_num] = {
            'answer':      label,
            'confidence':  round(confidence, 4),
            'fill_ratios': fill_ratios,
            'status':      status,
        }

    return results


# ── High-level extraction ─────────────────────────────────────────────────────

def extract_all_answers(
    aligned_image: np.ndarray,
    zones: List[dict],
) -> dict:
    """
    Run the full extraction pipeline for one aligned student image.

    Returns
    -------
    {
      'answers':  {'Q1': 'A', 'Q2': 'C', ...},   # None if blank/ambiguous
      'metadata': {'Q1': {...grid_result...}, ...},
      'warnings': ['Q3 ambiguous', ...]
    }
    """
    answers:  Dict[str, Optional[str]] = {}
    metadata: Dict[str, dict]          = {}
    warnings: List[str]                = []

    q_offset = 0

    for zone in zones:
        if zone.get('type') != 'omr':
            continue

        roi = extract_roi(aligned_image, zone)
        if roi.size == 0:
            warnings.append(f"Zone '{zone.get('name', '?')}' produced an empty ROI – check coordinates")
            continue

        rows   = int(zone.get('rows', 20))
        cols   = int(zone.get('cols', 5))
        labels = zone.get('labels') or [chr(65 + i) for i in range(cols)]

        grid   = analyse_grid(roi, rows, cols, labels)

        for local_q, result in grid.items():
            global_q = f"Q{local_q + q_offset}"
            answers[global_q]  = result['answer']
            metadata[global_q] = result

            if result['status'] == 'ambiguous':
                warnings.append(f"{global_q}: multiple marks detected – took strongest")
            elif result['status'] == 'blank':
                warnings.append(f"{global_q}: no answer detected (blank)")

        q_offset += rows

    return {'answers': answers, 'metadata': metadata, 'warnings': warnings}


# ── Name extraction (OCR optional) ───────────────────────────────────────────

def extract_name(aligned_image: np.ndarray, zones: List[dict]) -> str:
    """Extract student name from the first 'text' zone via pytesseract."""
    for zone in zones:
        if zone.get('type') != 'text':
            continue
        roi = extract_roi(aligned_image, zone)
        if roi.size == 0:
            continue
        try:
            import pytesseract
            gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            scale    = 3
            enlarged = cv2.resize(gray, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(enlarged, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, config='--psm 7').strip()
            if text:
                return text
        except (ImportError, Exception):
            pass
    return ''
