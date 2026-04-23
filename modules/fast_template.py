"""
MPASAS – Fast Template Module
Generates a printable A4 answer-sheet PNG with L-shaped corner registration
marks. The auto-detect function locates those marks in a photograph and
returns the zones dict without the user drawing anything.
"""
import cv2
import numpy as np
import json
from typing import List, Optional, Tuple

# ── Sheet dimensions (A4 at 96 dpi = 794 × 1123 px) ───────────────
SHEET_W, SHEET_H = 794, 1123

# ── Registration mark geometry ────────────────────────────────────
MARK_OUTER = 36   # outer side of the L  (px)
MARK_THICK = 10   # thickness of the L stroke
MARK_MARGIN = 18  # distance from sheet edge

# ── Layout constants ──────────────────────────────────────────────
NAME_BOX_X  = 55
NAME_BOX_Y  = 88
NAME_BOX_W  = SHEET_W - 110
NAME_BOX_H  = 38

GRID_X      = 55
GRID_Y      = 160
OPTION_LBLS = ['A', 'B', 'C', 'D', 'E']


def _draw_reg_mark(img: np.ndarray, corner: str, mx: int, my: int,
                   size: int, thick: int):
    """Draw an L-shaped registration mark at (mx, my) for the given corner."""
    clr = (0, 0, 0)
    if corner == 'TL':
        cv2.rectangle(img, (mx, my), (mx + size, my + thick), clr, -1)
        cv2.rectangle(img, (mx, my), (mx + thick, my + size), clr, -1)
    elif corner == 'TR':
        cv2.rectangle(img, (mx - size, my), (mx, my + thick), clr, -1)
        cv2.rectangle(img, (mx - thick, my), (mx, my + size), clr, -1)
    elif corner == 'BL':
        cv2.rectangle(img, (mx, my - size), (mx + thick, my), clr, -1)
        cv2.rectangle(img, (mx, my - thick), (mx + size, my), clr, -1)
    elif corner == 'BR':
        cv2.rectangle(img, (mx - thick, my - size), (mx, my), clr, -1)
        cv2.rectangle(img, (mx - size, my - thick), (mx, my), clr, -1)


def generate_fast_template(
    n_questions: int = 20,
    n_options: int = 5,
    two_columns: bool = True,
    sheet_title: str = "MPASAS Quick Answer Sheet",
    exam_info_lines: int = 2,
) -> Tuple[np.ndarray, dict, dict]:
    """
    Build the A4 template image.

    Returns
    -------
    img   : BGR image (794 × 1123)
    zones : list of zone dicts (relative coords, ready to save)
    meta  : extra layout metadata for auto-detection
    """
    img = np.ones((SHEET_H, SHEET_W, 3), dtype=np.uint8) * 252

    # ── Border ───────────────────────────────────────────────────
    cv2.rectangle(img, (12, 12), (SHEET_W - 12, SHEET_H - 12),
                  (180, 180, 180), 1)

    # ── Registration marks (4 corners) ───────────────────────────
    m = MARK_MARGIN
    s = MARK_OUTER
    t = MARK_THICK
    _draw_reg_mark(img, 'TL', m,            m,            s, t)
    _draw_reg_mark(img, 'TR', SHEET_W - m,  m,            s, t)
    _draw_reg_mark(img, 'BL', m,            SHEET_H - m,  s, t)
    _draw_reg_mark(img, 'BR', SHEET_W - m,  SHEET_H - m,  s, t)

    # ── Header ───────────────────────────────────────────────────
    cv2.putText(img, sheet_title, (55, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2)
    cv2.line(img, (55, 68), (SHEET_W - 55, 68), (150, 150, 150), 1)

    # ── Name / info boxes ────────────────────────────────────────
    # Name
    cv2.rectangle(img,
                  (NAME_BOX_X, NAME_BOX_Y),
                  (NAME_BOX_X + NAME_BOX_W, NAME_BOX_Y + NAME_BOX_H),
                  (180, 180, 180), 1)
    cv2.putText(img, "Name:", (NAME_BOX_X + 6, NAME_BOX_Y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)

    # Extra info lines (Reg No, Date …)
    info_labels = ["Reg. No.:", "Date:"]
    info_y = NAME_BOX_Y + NAME_BOX_H + 8
    info_w = (NAME_BOX_W - 8) // max(exam_info_lines, 1)
    for i in range(min(exam_info_lines, 2)):
        bx = NAME_BOX_X + i * info_w
        cv2.rectangle(img,
                      (bx, info_y), (bx + info_w - 6, info_y + 30),
                      (180, 180, 180), 1)
        cv2.putText(img, info_labels[i], (bx + 6, info_y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

    grid_y_start = info_y + 38 + 10

    # ── Answer grid ──────────────────────────────────────────────
    opts   = OPTION_LBLS[:n_options]
    cell_h = 28
    bubble_r = 9

    if two_columns and n_questions > 10:
        half      = (n_questions + 1) // 2
        col_w     = (SHEET_W - 110) // 2
        col_starts = [GRID_X, GRID_X + col_w + 10]
        splits    = [(1, half), (half + 1, n_questions)]
    else:
        half      = n_questions
        col_w     = SHEET_W - 110
        col_starts = [GRID_X]
        splits    = [(1, n_questions)]

    # cell width within one column
    q_label_w = 40
    opt_cell_w = (col_w - q_label_w) // n_options

    zones = []

    # Compute text-zone rect
    text_zone = {
        'type':  'text',
        'name':  'Student Name',
        'xPct':  round(NAME_BOX_X / SHEET_W, 4),
        'yPct':  round(NAME_BOX_Y / SHEET_H, 4),
        'wPct':  round(NAME_BOX_W / SHEET_W, 4),
        'hPct':  round(NAME_BOX_H / SHEET_H, 4),
    }
    zones.append(text_zone)

    for col_idx, (q_start, q_end) in enumerate(splits):
        cx0 = col_starts[col_idx]
        n_q = q_end - q_start + 1

        # Column option labels
        for ci, lbl in enumerate(opts):
            tx = cx0 + q_label_w + ci * opt_cell_w + opt_cell_w // 2 - 4
            cv2.putText(img, lbl, (tx, grid_y_start - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1)

        # Column separator line
        cv2.line(img,
                 (cx0, grid_y_start - 14),
                 (cx0 + q_label_w + n_options * opt_cell_w, grid_y_start - 14),
                 (150, 150, 150), 1)

        for row, q_num in enumerate(range(q_start, q_end + 1)):
            ry = grid_y_start + row * cell_h
            # Question label
            cv2.putText(img, str(q_num),
                        (cx0 + 4, ry + cell_h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 50, 50), 1)
            # Bubbles
            for ci in range(n_options):
                bx = cx0 + q_label_w + ci * opt_cell_w + opt_cell_w // 2
                by = ry + cell_h // 2
                cv2.circle(img, (bx, by), bubble_r, (150, 150, 150), 1)

            # Thin horizontal rule every 5 rows
            if (q_num) % 5 == 0:
                cv2.line(img,
                         (cx0, ry + cell_h - 2),
                         (cx0 + q_label_w + n_options * opt_cell_w, ry + cell_h - 2),
                         (200, 200, 200), 1)

        # OMR zone relative coords for this column block
        grid_h_px = n_q * cell_h
        omr_zone = {
            'type':   'omr',
            'name':   f'Answers Col {col_idx + 1}',
            'rows':   n_q,
            'cols':   n_options,
            'labels': opts,
            'xPct':   round((cx0 + q_label_w) / SHEET_W, 4),
            'yPct':   round(grid_y_start / SHEET_H, 4),
            'wPct':   round((n_options * opt_cell_w) / SHEET_W, 4),
            'hPct':   round(grid_h_px / SHEET_H, 4),
        }
        zones.append(omr_zone)

    # ── Footer ────────────────────────────────────────────────────
    cv2.putText(img, "MPASAS · Auto-Grading Template · Do not write below this line",
                (55, SHEET_H - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 150, 150), 1)

    # Meta for auto-detect (corner mark positions in px)
    meta = {
        'corner_marks': {
            'TL': (m, m), 'TR': (SHEET_W - m, m),
            'BL': (m, SHEET_H - m), 'BR': (SHEET_W - m, SHEET_H - m),
        },
        'mark_size':  s,
        'mark_thick': t,
        'sheet_w':    SHEET_W,
        'sheet_h':    SHEET_H,
    }

    return img, zones, meta


# ── Auto-detection ─────────────────────────────────────────────────

def auto_detect_zones(
    script_img: np.ndarray,
    zones_template: List[dict],
    meta: Optional[dict] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Detect the 4 corner registration marks in *script_img* and compute
    the homography that maps the fast template onto the script photo.
    Returns the warped (aligned) image and a note string.

    Falls back to the standard ORB aligner on failure.
    """
    gray   = cv2.cvtColor(script_img, cv2.COLOR_BGR2GRAY)
    _, bw  = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = script_img.shape[:2]
    mark_area_min = 60
    mark_area_max = int(w_img * h_img * 0.005)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (mark_area_min < area < mark_area_max):
            continue
        x, y, bw_, bh = cv2.boundingRect(cnt)
        aspect = bw_ / max(bh, 1)
        if 0.5 < aspect < 2.0:
            candidates.append((x + bw_ // 2, y + bh // 2))

    if len(candidates) < 4:
        return None, f"Only {len(candidates)} registration marks found (need 4) – use ORB fallback"

    # Pick the 4 candidates closest to each corner
    cx = w_img / 2; cy = h_img / 2
    TL = min(candidates, key=lambda p: (p[0] - 0)**2     + (p[1] - 0)**2)
    TR = min(candidates, key=lambda p: (p[0] - w_img)**2 + (p[1] - 0)**2)
    BL = min(candidates, key=lambda p: (p[0] - 0)**2     + (p[1] - h_img)**2)
    BR = min(candidates, key=lambda p: (p[0] - w_img)**2 + (p[1] - h_img)**2)

    m = MARK_MARGIN
    dst_pts = np.float32([TL, TR, BR, BL])
    src_pts = np.float32([
        [m, m], [SHEET_W - m, m],
        [SHEET_W - m, SHEET_H - m], [m, SHEET_H - m],
    ])

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, "Homography from registration marks failed"

    aligned = cv2.warpPerspective(script_img, H, (SHEET_W, SHEET_H))
    return aligned, "Auto-aligned via registration marks ✓"
