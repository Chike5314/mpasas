#!/usr/bin/env python
"""Quick test: verify 50-question sheet extraction works"""
import sys
sys.path.insert(0, '.')
import numpy as np
import cv2
from modules.extractor import extract_all_answers

# Test 50-question sheet extraction
roi_left = np.ones((400, 150, 3), dtype=np.uint8) * 240
roi_right = np.ones((400, 150, 3), dtype=np.uint8) * 240

# Add some filled bubbles to test detection
for i in range(25):
    cv2.circle(roi_left, (30 + (i % 5) * 30, 20 + (i // 5) * 15), 5, (20, 20, 20), -1)
    cv2.circle(roi_right, (30 + (i % 5) * 30, 20 + (i // 5) * 15), 5, (20, 20, 20), -1)

zones = [
    {'type': 'omr', 'rows': 25, 'cols': 5, 'labels': ['A','B','C','D','E'],
     'xPct': 0.0, 'yPct': 0.0, 'wPct': 0.5, 'hPct': 1.0, 'name': 'Left 25'},
    {'type': 'omr', 'rows': 25, 'cols': 5, 'labels': ['A','B','C','D','E'],
     'xPct': 0.5, 'yPct': 0.0, 'wPct': 0.5, 'hPct': 1.0, 'name': 'Right 25'},
]

combined = np.hstack([roi_left, roi_right])
result = extract_all_answers(combined, zones)

print(f"✓ 50-question extraction test PASSED")
q_nums = sorted([int(q[1:]) for q in result['answers'].keys()])
print(f"  Questions extracted: Q{q_nums[0]} to Q{q_nums[-1]}")
print(f"  Total: {len(result['answers'])} answers")
print(f"  Warnings: {len(result['warnings'])}")
if result['warnings']:
    for w in result['warnings'][:3]:
        print(f"    - {w}")
