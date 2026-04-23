# MPASAS 🎓
### MCQ Paper-based Automatic Scoring and Analytics System

> **"Grade Smarter. Evaluate Better."**
> Born from the experience of manually grading 700+ scripts overnight — MPASAS eliminates that burden and turns every answer sheet into actionable insights.

---

## What it does

MPASAS is a **computer-vision-powered web application** that turns smartphone photos of MCQ answer sheets into structured scores and analytics — no special hardware, no proprietary forms, no internet required after setup.

| Stage | Technology | What happens |
|-------|-----------|-------------|
| **Calibrate** | Interactive canvas editor | You draw zones on your answer sheet template once |
| **Align** | ORB feature matching + Homography | Tilted/dark photos are automatically straightened |
| **Extract** | Pixel density analysis | Filled bubbles are detected per question |
| **Analyse** | Difficulty + Discrimination indices | Charts, pass rates, distractor analysis |

---

## Requirements

- **Python 3.10+** — [python.org/downloads](https://www.python.org/downloads/)
- **pip** (comes with Python)
- No internet needed after first `pip install`

---

## Quick Start (Local / Offline)

```bash
# 1. Clone (or download the zip)
git clone https://github.com/YOUR_USERNAME/mpasas.git
cd mpasas

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Mac / Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies (only needs internet this once)
pip install -r requirements.txt

# 4. Run
python run.py
```

Then open your browser at: **http://localhost:5000**

---

## Generate Demo / Test Data (no real scanner needed)

```bash
python tests/generate_test_sheet.py
```

This creates:
- One blank template image in `uploads/templates/`
- 15 synthetic student scripts in `uploads/scripts/`
- Prints the **zone JSON** and **answer key** to paste into the calibrator

You can then go to the UI, create a template, paste the printed zones/key, and run a full grading session on the generated scripts — all offline.

---

## Run Tests

```bash
python tests/run_tests.py

# For verbose error output:
python tests/run_tests.py --verbose
```

The test suite covers:
- All dependency imports
- Each module (scorer, extractor, aligner, analytics) in isolation
- Test data generation
- Flask app creation, route registration, and database CRUD

---

## How to Use (Step by Step)

### Step 1 — Create a Template
1. Go to **Templates → New Template**
2. Give it a name and upload a **blank, flat scan** of your answer sheet
3. Click **Next: Calibrate Zones**

### Step 2 — Calibrate Zones
1. Click **"Draw OMR Zone"** and drag a box over the bubble grid
2. Enter the number of rows (questions) and columns (options)
3. Click **"Draw Name Zone"** and drag over the student name area
4. Click **"Edit Answer Key"** and set the correct answer for each question
5. Click **"Save & Continue"**

### Step 3 — Grade Scripts
1. Go to **Grade Scripts → New Session**
2. Select your template and set a passing percentage
3. Upload all student script photos (multiple files at once)
4. Click **"Start Grading"**

### Step 4 — Review Analytics
- Score distribution chart
- Per-question difficulty index (p-value)
- Discrimination index (D-value)
- Individual student results table with answer-by-answer breakdown
- Export to CSV or Excel

---

## Project Structure

```
mpasas/
├── run.py                    # Entry point — python run.py
├── app.py                    # Flask routes
├── config.py                 # Configuration
├── requirements.txt
│
├── modules/
│   ├── aligner.py            # ORB + Homography image alignment
│   ├── extractor.py          # Pixel density OMR bubble detection
│   ├── scorer.py             # Answer comparison + scoring
│   ├── analytics.py          # Statistics + chart generation
│   └── models.py             # SQLAlchemy database models
│
├── templates/                # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html            # Dashboard
│   ├── calibrate.html        # Interactive zone editor
│   ├── results.html          # Analytics dashboard
│   └── ...
│
├── static/css/
│   └── mpasas.css            # Full brand design system
│
├── tests/
│   ├── generate_test_sheet.py   # Creates synthetic test data
│   └── run_tests.py             # Full test suite
│
├── uploads/                  # Runtime (auto-created)
│   ├── templates/
│   └── scripts/
├── results/                  # Exported CSV/Excel files
└── instance/
    └── mpasas.db             # SQLite database
```

---

## Putting it on GitHub (so others can try it)

```bash
# 1. Create a repo on github.com (name: mpasas)

# 2. In your project folder:
git init
git add .
git commit -m "Initial release: MPASAS v1.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mpasas.git
git push -u origin main
```

Anyone who clones your repo just runs:
```bash
git clone https://github.com/YOUR_USERNAME/mpasas.git
cd mpasas
pip install -r requirements.txt
python run.py
```

### What to add in `.gitignore`:
```
venv/
__pycache__/
*.pyc
instance/
uploads/
results/
static/charts/
*.db
.env
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| `ModuleNotFoundError: flask` | `pip install flask` |
| Port 5000 already in use | Change `port=5000` in `run.py` to `5001` |
| White/blank canvas in calibrator | Upload the template image first |
| Poor bubble detection | Improve lighting on photos; use the test generator to verify the pipeline |
| Name extraction empty | Pytesseract is optional — name falls back to filename |

---

## Reaching Out for Help

If you run into bugs or want to extend the system, the best ways to get help are:

1. **Open an Issue on GitHub** — describe the error + paste the terminal output
2. **Claude (claude.ai)** — paste the error message and say "this is from the MPASAS project" — the full codebase context will help resolve it quickly
3. For CV tuning (e.g. templates with unusual bubble sizes), share a sample image in the GitHub issue

---

## Brand

**MPASAS** — MCQ Paper-based Automatic Scoring and Analytics System
Tagline: *"Grade Smarter. Evaluate Better."*
Colours: Indigo `#4F46E5` · Purple `#7C3AED` · Amber `#F59E0B` · Emerald `#10B981`

Made in 🇨🇲 Cameroon — inspired by the real problem of teachers spending nights manually grading hundreds of scripts when their time is better spent on evaluation, not arithmetic.

---

## License

MIT — free to use, modify and share.
