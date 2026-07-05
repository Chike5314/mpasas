"""
MPASAS – Visualiser Module
Draws green/red/grey bubble overlays on aligned answer sheets
and computes per-student report data.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Colours (BGR) ─────────────────────────────────────────────────
CLR_CORRECT  = (34,  197,  94)   # emerald green
CLR_WRONG    = (239,  68,  68)   # red
CLR_BLANK    = (148, 163, 184)   # slate grey
CLR_KEY      = (245, 158,  11)   # amber  (highlight correct bubble)
ALPHA        = 0.45              # overlay transparency


def _bubble_centers(zone: dict, img_w: int, img_h: int) -> Dict[int, Dict[str, Tuple[int,int]]]:
    """
    Return pixel (cx, cy) for every bubble in an OMR zone.
    Returns {question_num: {label: (cx, cy), ...}, ...}
    """
    x0 = int(zone['xPct'] * img_w)
    y0 = int(zone['yPct'] * img_h)
    w  = int(zone['wPct'] * img_w)
    h  = int(zone['hPct'] * img_h)

    rows   = int(zone.get('rows', 20))
    cols   = int(zone.get('cols', 5))
    labels = zone.get('labels') or [chr(65 + c) for c in range(cols)]

    cell_w = w / cols
    cell_h = h / rows

    centers: Dict[int, Dict[str, Tuple[int,int]]] = {}
    for r in range(rows):
        q = r + 1
        centers[q] = {}
        for c, lbl in enumerate(labels[:cols]):
            cx = int(x0 + c * cell_w + cell_w / 2)
            cy = int(y0 + r * cell_h + cell_h / 2)
            centers[q][lbl] = (cx, cy)
    return centers


def draw_marked_script(
    aligned_img: np.ndarray,
    zones: List[dict],
    student_answers: Dict[str, Optional[str]],
    answer_key: Dict[str, str],
) -> np.ndarray:
    """
    Return a copy of *aligned_img* with coloured overlays on every bubble:
      • Green filled  = student marked this AND it matches the key
      • Red filled    = student marked this AND it is wrong
      • Amber outline = correct answer (always shown, so teacher sees the key)
      • Grey filled   = no mark detected (blank)
    """
    out     = aligned_img.copy()
    overlay = aligned_img.copy()
    h, w    = out.shape[:2]

    q_offset = 0
    for zone in zones:
        if zone.get('type') != 'omr':
            continue

        rows   = int(zone.get('rows', 20))
        cols   = int(zone.get('cols', 5))
        labels = zone.get('labels') or [chr(65 + c) for c in range(cols)]
        bubs   = _bubble_centers(zone, w, h)

        # Estimate bubble radius from cell size
        cell_h = int(zone['hPct'] * h / rows)
        cell_w = int(zone['wPct'] * w / cols)
        radius = max(6, int(min(cell_h, cell_w) * 0.32))

        for local_q, lbl_map in bubs.items():
            global_q   = f"Q{local_q + q_offset}"
            student_ans = (student_answers.get(global_q) or '').upper()
            correct_ans = (answer_key.get(global_q) or '').upper()

            for lbl, (cx, cy) in lbl_map.items():
                l = lbl.upper()

                # Always show the key bubble as amber outline
                if l == correct_ans:
                    cv2.circle(overlay, (cx, cy), radius + 3,
                               CLR_KEY, 2)

                if l == student_ans:
                    if student_ans == correct_ans:
                        clr = CLR_CORRECT
                    else:
                        clr = CLR_WRONG
                    cv2.circle(overlay, (cx, cy), radius, clr, -1)
                    # White letter on top
                    font_scale = max(0.3, radius * 0.045)
                    tw, th = cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX,
                                             font_scale, 1)[0]
                    cv2.putText(overlay, l,
                                (cx - tw // 2, cy + th // 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, (255, 255, 255), 1,
                                cv2.LINE_AA)

        q_offset += rows

    # Blend overlay with original
    cv2.addWeighted(overlay, ALPHA, out, 1 - ALPHA, 0, out)
    # Draw legend
    _draw_legend(out)
    return out


def _draw_legend(img: np.ndarray):
    h, w = img.shape[:2]
    items = [
        (CLR_CORRECT, 'Correct'),
        (CLR_WRONG,   'Wrong'),
        (CLR_KEY,     'Key answer'),
    ]
    x, y = 10, h - 14
    for clr, label in reversed(items):
        text_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        cv2.circle(img, (x + 6, y - 3), 5, clr, -1)
        cv2.putText(img, label, (x + 14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1)
        x += text_w + 30


# ── Per-student report data ────────────────────────────────────────

def build_student_report(
    student_result: dict,
    answer_key: dict,
    session: dict,
) -> dict:
    """
    Return a structured report dict for one student.
    """
    raw = student_result.get('raw_answers', {})
    rows = []
    for q in sorted(answer_key.keys(),
                    key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        correct_ans  = (answer_key.get(q) or '').upper()
        student_ans  = (raw.get(q) or '').upper() or None
        is_correct   = student_ans == correct_ans if student_ans else False
        rows.append({
            'question':    q,
            'key':         correct_ans,
            'student':     student_ans or '—',
            'correct':     is_correct,
            'blank':       student_ans is None,
        })

    correct_count = sum(1 for r in rows if r['correct'])
    total         = len(rows)
    blanks        = sum(1 for r in rows if r['blank'])

    return {
        'student':       student_result,
        'session':       session,
        'rows':          rows,
        'correct_count': correct_count,
        'wrong_count':   total - correct_count - blanks,
        'blank_count':   blanks,
        'total':         total,
        'percentage':    round(student_result['percentage'], 1),
        'passed':        student_result['passed'],
    }
