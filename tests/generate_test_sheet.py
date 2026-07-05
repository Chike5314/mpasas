"""
MPASAS – Test Sheet Generator
Creates a realistic synthetic answer sheet template + 15 student scripts
so you can test the full pipeline without needing a real scanner.

Usage:
    python tests/generate_test_sheet.py
    # → writes to uploads/templates/ and uploads/scripts/
    # → prints the zone JSON and answer key you should paste into the calibrator
"""
import os, sys, json, random
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("OpenCV not installed. Run:  pip install opencv-python")

# ── Configuration ──────────────────────────────────────────────────
NUM_QUESTIONS = 20
NUM_OPTIONS   = 5          # A B C D E
NUM_STUDENTS  = 15
SHEET_W, SHEET_H = 794, 1123   # A4 at 96dpi

ANSWER_KEY = {
    f'Q{i}': random.choice(['A','B','C','D','E'])
    for i in range(1, NUM_QUESTIONS + 1)
}

# Student names for test data
NAMES = [
    "Amara Fonkwe", "Brice Nkemdirim", "Celine Mbah",
    "David Tabi", "Esther Ngum", "Fabrice Djeukam",
    "Grace Fru", "Herman Atanga", "Irene Nso",
    "Jean-Pierre Kamga", "Karine Mbu", "Laurent Tchamba",
    "Marie Ngono", "Noel Takang", "Olivia Mendouga",
]


def draw_template(sheet_w=SHEET_W, sheet_h=SHEET_H,
                  n_questions=NUM_QUESTIONS, n_options=NUM_OPTIONS):
    """Return blank template BGR image + zone definition + pixel coords."""
    img = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 250  # near-white

    # Border
    cv2.rectangle(img, (15, 15), (sheet_w - 15, sheet_h - 15), (180, 180, 180), 1)

    # Header
    cv2.putText(img, "MPASAS DEMO ANSWER SHEET", (40, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.line(img, (40, 65), (sheet_w - 40, 65), (150, 150, 150), 1)

    # ── Name field ─────────────────────────────────────────────────
    name_x, name_y   = 40, 90
    name_w, name_h   = sheet_w - 80, 36
    cv2.rectangle(img, (name_x, name_y), (name_x + name_w, name_y + name_h), (200, 200, 200), 1)
    cv2.putText(img, "Name:", (name_x + 6, name_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    # ── OMR grid ───────────────────────────────────────────────────
    grid_x      = 40
    grid_y      = 150
    cell_w      = 55
    cell_h      = 34
    bubble_r    = 10
    option_lbl  = [chr(65 + i) for i in range(n_options)]

    # Column headers
    for c in range(n_options):
        cx = grid_x + 90 + c * cell_w + cell_w // 2
        cv2.putText(img, option_lbl[c], (cx - 5, grid_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    for row in range(n_questions):
        ry = grid_y + row * cell_h
        # Question label
        cv2.putText(img, f"Q{row+1}", (grid_x + 4, ry + cell_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1)
        # Bubble circles
        for col in range(n_options):
            cx = grid_x + 90 + col * cell_w + cell_w // 2
            cy = ry + cell_h // 2
            cv2.circle(img, (cx, cy), bubble_r, (160, 160, 160), 1)

    # ── Zone definition (relative coords) ─────────────────────────
    grid_w = 90 + n_options * cell_w
    grid_h = n_questions * cell_h

    zones = [
        {
            "type":   "text",
            "name":   "Student Name",
            "xPct":   round(name_x / sheet_w, 4),
            "yPct":   round(name_y / sheet_h, 4),
            "wPct":   round(name_w / sheet_w, 4),
            "hPct":   round(name_h / sheet_h, 4),
        },
        {
            "type":   "omr",
            "name":   "Answers Block",
            "rows":   n_questions,
            "cols":   n_options,
            "labels": option_lbl,
            "xPct":   round(grid_x / sheet_w, 4),
            "yPct":   round(grid_y / sheet_h, 4),
            "wPct":   round(grid_w / sheet_w, 4),
            "hPct":   round(grid_h / sheet_h, 4),
        },
    ]

    # Return bubble pixel centre coords too (for filling student sheets)
    bubble_coords = {}
    for row in range(n_questions):
        ry = grid_y + row * cell_h
        bubble_coords[f'Q{row+1}'] = {}
        for col in range(n_options):
            cx = grid_x + 90 + col * cell_w + cell_w // 2
            cy = ry + cell_h // 2
            bubble_coords[f'Q{row+1}'][option_lbl[col]] = (cx, cy)

    return img, zones, bubble_coords


def fill_student_sheet(template_img, bubble_coords, student_answers,
                       student_name='', noise_level=0.05, warp=True):
    """
    Fill in a copy of the template for one student.
    Optionally adds perspective warp and noise to simulate a phone photo.
    """
    img = template_img.copy()

    # Fill name
    if student_name:
        # Simple text overlay in the name box area
        cv2.putText(img, student_name, (90, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)

    # Fill bubbles
    for q, ans in student_answers.items():
        if ans and q in bubble_coords and ans in bubble_coords[q]:
            cx, cy = bubble_coords[q][ans]
            cv2.circle(img, (cx, cy), 9, (30, 30, 30), -1)

    # Add realistic noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 255,
                                 img.shape).astype(np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Perspective warp (simulate slightly tilted phone photo)
    if warp:
        h, w = img.shape[:2]
        margin = random.randint(8, 22)
        src    = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dx1 = random.randint(-margin, margin)
        dy1 = random.randint(-margin, margin)
        dx2 = random.randint(-margin, margin)
        dy2 = random.randint(-margin, margin)
        dx3 = random.randint(-margin, margin)
        dy3 = random.randint(-margin, margin)
        dx4 = random.randint(-margin, margin)
        dy4 = random.randint(-margin, margin)
        dst = np.float32([
            [dx1, dy1], [w + dx2, dy2],
            [w + dx3, h + dy3], [dx4, h + dy4]
        ])
        M   = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h),
                                  borderValue=(230, 230, 230))

    return img


def generate_student_answers(answer_key, correct_pct=0.65):
    """Generate plausible student answers with some mistakes."""
    options  = ['A', 'B', 'C', 'D', 'E']
    answers  = {}
    for q, correct in answer_key.items():
        if random.random() < correct_pct:
            answers[q] = correct
        elif random.random() < 0.1:
            answers[q] = None   # left blank
        else:
            wrong = [o for o in options if o != correct]
            answers[q] = random.choice(wrong)
    return answers


def run(template_dir: str, scripts_dir: str):
    """Main entry – create all test images."""
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(scripts_dir,  exist_ok=True)

    tmpl_img, zones, bubble_coords = draw_template()

    # Save blank template
    tmpl_path = os.path.join(template_dir, 'demo_template.png')
    cv2.imwrite(tmpl_path, tmpl_img)
    print(f"[MPASAS] Template saved: {tmpl_path}")

    # Generate students
    for i, name in enumerate(NAMES[:NUM_STUDENTS]):
        correct_pct = random.uniform(0.40, 0.95)
        answers     = generate_student_answers(ANSWER_KEY, correct_pct)
        filled      = fill_student_sheet(tmpl_img, bubble_coords, answers,
                                         student_name=name)
        fname       = f"student_{i+1:02d}_{name.split()[0].lower()}.png"
        path        = os.path.join(scripts_dir, fname)
        cv2.imwrite(path, filled)

    print(f"[MPASAS] {NUM_STUDENTS} student scripts saved to: {scripts_dir}")
    print(f"\n[MPASAS] ANSWER KEY:\n{json.dumps(ANSWER_KEY, indent=2)}")
    print(f"\n[MPASAS] ZONES:\n{json.dumps(zones, indent=2)}")
    print(f"\n[MPASAS] Template image: {tmpl_path}")
    print("\n[MPASAS] ✅ All test data generated!")

    return tmpl_path, zones, ANSWER_KEY


if __name__ == '__main__':
    BASE = os.path.join(os.path.dirname(__file__), '..')
    run(
        template_dir=os.path.join(BASE, 'uploads', 'templates'),
        scripts_dir =os.path.join(BASE, 'uploads', 'scripts'),
    )
