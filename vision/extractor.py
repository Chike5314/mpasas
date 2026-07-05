"""
vision/extractor.py
===================
Core computer-vision engine for MPASAS.

Pipeline
--------
1. Load template + student images
2. Align student image to template (ORB feature matching → Homography)
3. For every OMR zone: pixel-density analysis per bubble cell
4. For every OCR zone: pytesseract text extraction (name / student ID)
5. Score: compare extracted answers against the answer key
"""

import os, traceback
import numpy as np

# ──────────────────────────────────────────────
# cv2 / PIL – soft import so Flask still starts
# even if packages aren't installed yet.
# ──────────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from PIL import Image as PILImg, ImageFilter, ImageOps
    PIL_OK = True
except ImportError:
    PIL_OK = False


# ──────────────────────────────────────────────────────────────
# 1.  IMAGE  ALIGNMENT
# ──────────────────────────────────────────────────────────────

def align_to_template(student_img, template_img):
    """
    Align *student_img* to *template_img* using ORB feature matching
    and RANSAC homography.  Falls back to plain resize if matching fails.
    Both inputs are BGR numpy arrays.
    Returns an aligned BGR numpy array the same size as template_img.
    """
    h_ref, w_ref = template_img.shape[:2]

    # ── grayscale ──
    g_tpl = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    g_stu = cv2.cvtColor(student_img,  cv2.COLOR_BGR2GRAY)

    # ── ORB detector ──
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)
    kp1, des1 = orb.detectAndCompute(g_tpl, None)
    kp2, des2 = orb.detectAndCompute(g_stu, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return cv2.resize(student_img, (w_ref, h_ref))

    # ── BFMatcher ──
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
    good    = matches[:min(80, len(matches))]

    if len(good) < 4:
        return cv2.resize(student_img, (w_ref, h_ref))

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return cv2.resize(student_img, (w_ref, h_ref))

    return cv2.warpPerspective(student_img, H, (w_ref, h_ref))


# ──────────────────────────────────────────────────────────────
# 2.  OMR  ZONE  EXTRACTION  (pixel-density per bubble cell)
# ──────────────────────────────────────────────────────────────

def extract_omr_zone(image, zone):
    """
    Given a BGR image and a zone descriptor, analyse pixel fill-rate
    per bubble cell and return the most-filled option for each row.

    zone keys
    ---------
    x, y, w, h  – relative coordinates (0-1 of image size)
    rows        – number of questions in this zone
    cols        – number of answer options  (default 4 = A B C D)
    direction   – 'row'  → each row is one question  (standard)
                  'col'  → each column is one question
    """
    ih, iw = image.shape[:2]
    x  = int(zone['x']  * iw);  y  = int(zone['y']  * ih)
    w  = int(zone['w']  * iw);  h  = int(zone['h']  * ih)

    # clamp
    x = max(0, min(x, iw-1));  y = max(0, min(y, ih-1))
    w = min(w, iw-x);           h = min(h, ih-y)

    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return {}

    # ── pre-process ──
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi.copy()
    # adaptive threshold is more robust to uneven lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=8
    )

    rows      = max(1, zone.get('rows', 1))
    cols      = max(1, zone.get('cols', 4))
    options   = ['A','B','C','D','E'][:cols]
    cell_h    = h // rows
    cell_w    = w // cols
    answers   = {}

    for row in range(rows):
        fills = []
        for col in range(cols):
            cy1 = row * cell_h
            cy2 = cy1 + cell_h
            cx1 = col * cell_w
            cx2 = cx1 + cell_w

            # inner margin (removes border lines)
            mg_y = max(2, cell_h // 7)
            mg_x = max(2, cell_w // 7)
            cell = thresh[cy1+mg_y : cy2-mg_y, cx1+mg_x : cx2-mg_x]

            if cell.size == 0:
                fills.append(0.0)
                continue
            fills.append(float(np.sum(cell > 127)) / cell.size)

        # select option: must be clearly the most-filled AND exceed threshold
        MAX_FILL_THRESH = 0.12
        max_f = max(fills) if fills else 0
        if max_f >= MAX_FILL_THRESH:
            best_idx = fills.index(max_f)
            # check it's meaningfully more than second-best (avoid ambiguity)
            sorted_fills = sorted(fills, reverse=True)
            if len(sorted_fills) < 2 or (max_f - sorted_fills[1]) >= 0.05:
                answers[str(row+1)] = options[best_idx] if best_idx < len(options) else ''
            else:
                # ambiguous – mark as multiple / blank
                answers[str(row+1)] = ''
        else:
            answers[str(row+1)] = ''

    return answers


# ──────────────────────────────────────────────────────────────
# 3.  OCR  ZONE  EXTRACTION
# ──────────────────────────────────────────────────────────────

def extract_ocr_zone(image, zone):
    """Extract handwritten / printed text from a zone using pytesseract."""
    try:
        import pytesseract
    except ImportError:
        return ''

    ih, iw = image.shape[:2]
    x  = int(zone['x'] * iw);  y  = int(zone['y'] * ih)
    w  = int(zone['w'] * iw);  h  = int(zone['h'] * ih)
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return ''

    # upscale + denoise for better OCR
    scale = 3
    roi_up = cv2.resize(roi, (roi.shape[1]*scale, roi.shape[0]*scale),
                        interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY) if len(roi_up.shape)==3 else roi_up
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cfg  = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -'
    text = pytesseract.image_to_string(thr, config=cfg).strip()
    return text


# ──────────────────────────────────────────────────────────────
# 4.  MAIN  ENTRY  POINT
# ──────────────────────────────────────────────────────────────

def process_student_script(student_path, template_path, zones,
                            answer_key, points_per_q=1.0, negative_marking=0.0):
    """
    Full pipeline: load → align → extract → score.

    Returns
    -------
    dict with keys:
        name, student_id, answers (dict q_str→option),
        score, correct, wrong, blank
    """
    result = {
        'name': 'Unknown', 'student_id': '',
        'answers': {}, 'score': 0,
        'correct': 0, 'wrong': 0, 'blank': 0,
    }

    if not CV2_OK:
        raise RuntimeError(
            "OpenCV (cv2) is not installed.\n"
            "Run:  pip install opencv-python-headless Pillow pytesseract\n"
            "Then restart the server."
        )

    # ── load images ──
    student_img = cv2.imread(student_path)
    if student_img is None:
        raise ValueError(f"Could not read image: {student_path}")

    template_img = cv2.imread(template_path) if os.path.exists(template_path) else None

    # ── align ──
    if template_img is not None:
        try:
            aligned = align_to_template(student_img, template_img)
        except Exception:
            h, w = template_img.shape[:2]
            aligned = cv2.resize(student_img, (w, h))
    else:
        aligned = student_img

    # ── extract zones ──
    all_answers = {}
    for zone in zones:
        ztype = zone.get('type','omr')
        if ztype == 'omr':
            zone_ans = extract_omr_zone(aligned, zone)
            all_answers.update(zone_ans)
        elif ztype == 'ocr':
            text   = extract_ocr_zone(aligned, zone)
            target = zone.get('target','name')
            if target == 'name' and text:
                result['name'] = text
            elif target == 'id' and text:
                result['student_id'] = text

    result['answers'] = all_answers

    # ── scoring ──
    correct = wrong = blank = 0
    for q_str, stu_ans in all_answers.items():
        corr_ans = answer_key.get(q_str,'')
        if not stu_ans:
            blank += 1
        elif stu_ans == corr_ans:
            correct += 1
        else:
            wrong += 1

    score = max(0.0, correct * points_per_q - wrong * negative_marking)

    result.update({'correct': correct, 'wrong': wrong, 'blank': blank, 'score': score})
    return result


# ──────────────────────────────────────────────────────────────
# 5.  SIMPLE  DEMO  IMAGE  GENERATOR  (for testing without camera)
# ──────────────────────────────────────────────────────────────

def generate_demo_script_image(answer_key, num_questions=20, num_options=4,
                                width=794, height=1123):
    """
    Produce a synthetic OMR image with filled bubbles matching answer_key.
    Useful for unit testing the extraction pipeline without a real camera.
    Returns a BGR numpy array.
    """
    if not CV2_OK:
        raise RuntimeError("cv2 not installed")

    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # margins
    mx, my = int(0.05*width), int(0.15*height)
    zone_w = int(0.90*width)
    zone_h = int(0.78*height)

    cell_h = zone_h // num_questions
    cell_w = zone_w // num_options
    options = ['A','B','C','D','E'][:num_options]

    # column headers
    for ci, opt in enumerate(options):
        hx = mx + ci*cell_w + cell_w//2 - 5
        cv2.putText(img, opt, (hx, my-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 1)

    for qi in range(num_questions):
        q_str    = str(qi+1)
        row_y    = my + qi * cell_h
        # question number
        cv2.putText(img, f'{q_str:>2}.', (5, row_y + cell_h//2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)
        # draw all bubbles (empty circles)
        for ci in range(num_options):
            cx = mx + ci*cell_w + cell_w//2
            cy = row_y + cell_h//2
            r  = min(cell_h, cell_w)//3
            cv2.circle(img, (cx,cy), r, (120,120,120), 1)

        # fill the correct option
        correct_opt = answer_key.get(q_str,'A')
        if correct_opt in options:
            ci = options.index(correct_opt)
            cx = mx + ci*cell_w + cell_w//2
            cy = row_y + cell_h//2
            r  = min(cell_h, cell_w)//3
            cv2.circle(img, (cx,cy), r, (30,30,30), -1)  # filled

    return img
