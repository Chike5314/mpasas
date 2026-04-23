"""
MPASAS – Scorer Module
Compares extracted student answers against the answer key.
"""
from typing import Dict, Optional, Tuple


def score(
    student_answers: Dict[str, Optional[str]],
    answer_key: Dict[str, str],
) -> Tuple[int, int, Dict[str, dict]]:
    """
    Grade a single student's answers.

    Parameters
    ----------
    student_answers : {Q1: 'A', Q2: None (blank), ...}
    answer_key      : {Q1: 'A', Q2: 'C', ...}

    Returns
    -------
    correct   : int
    total     : int
    breakdown : {Q1: {'student': 'A', 'key': 'A', 'correct': True}, ...}
    """
    breakdown: Dict[str, dict] = {}
    correct = 0

    for q, correct_ans in answer_key.items():
        student_ans = student_answers.get(q)
        is_correct  = (student_ans is not None and
                       student_ans.strip().upper() == correct_ans.strip().upper())
        if is_correct:
            correct += 1
        breakdown[q] = {
            'student': student_ans,
            'key':     correct_ans,
            'correct': is_correct,
        }

    total = len(answer_key)
    return correct, total, breakdown


def percentage(correct: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(correct / total * 100, 2)
