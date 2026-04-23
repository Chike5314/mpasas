"""
MPASAS – Analytics Module
Computes MCQ psychometric statistics and generates chart images.
"""
import io, base64
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless / offline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Colour palette (matches brand) ──────────────────────────────────────────
BRAND_INDIGO  = '#4F46E5'
BRAND_PURPLE  = '#7C3AED'
BRAND_AMBER   = '#F59E0B'
BRAND_EMERALD = '#10B981'
BRAND_RED     = '#EF4444'
BRAND_CYAN    = '#06B6D4'
BG            = '#F8FAFC'
TEXT_DARK     = '#1E293B'
TEXT_MID      = '#64748B'


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style_ax(ax, title: str = ''):
    ax.set_facecolor(BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')
    ax.tick_params(colors=TEXT_MID, labelsize=9)
    if title:
        ax.set_title(title, color=TEXT_DARK, fontsize=11, fontweight='bold', pad=10)


# ── Score distribution ────────────────────────────────────────────────────────

def score_distribution_chart(
    percentages: List[float],
    passing_score: float = 50.0,
) -> str:
    """Return base64 PNG of a score-distribution histogram."""
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG)
    _style_ax(ax, 'Score Distribution')

    bins   = list(range(0, 105, 5))
    colors = [BRAND_EMERALD if b >= passing_score else BRAND_RED for b in bins[:-1]]

    n, _, patches = ax.hist(percentages, bins=bins, edgecolor='white', linewidth=0.6)
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)

    ax.axvline(passing_score, color=BRAND_AMBER, linewidth=2, linestyle='--', label=f'Pass mark ({passing_score:.0f}%)')
    ax.set_xlabel('Score (%)', color=TEXT_MID, fontsize=9)
    ax.set_ylabel('Students', color=TEXT_MID, fontsize=9)
    ax.legend(fontsize=8, framealpha=0.8)

    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Per-question difficulty + discrimination ──────────────────────────────────

def per_question_stats(
    results,            # list of StudentResult ORM objects
    answer_key: dict,
) -> List[dict]:
    """
    Returns a list of per-question dicts:
    { question, difficulty_index, discrimination_index, distractors: {A:n, B:n, …} }
    """
    if not results or not answer_key:
        return []

    n = len(results)
    questions = sorted(answer_key.keys(), key=lambda q: int(q[1:]) if q[1:].isdigit() else 0)

    # Sort results by percentage
    sorted_results = sorted(results, key=lambda r: r.percentage)
    cutoff = max(1, int(n * 0.27))
    bottom_group = sorted_results[:cutoff]
    top_group    = sorted_results[n - cutoff:]

    def fraction_correct(group, q):
        correct_ans = answer_key[q].upper()
        import json
        hits = sum(
            1 for r in group
            if json.loads(r.raw_answers_json or '{}').get(q, '').upper() == correct_ans
        )
        return hits / len(group) if group else 0.0

    def distractor_count(q):
        import json
        counts: Dict[str, int] = {}
        for r in results:
            ans = json.loads(r.raw_answers_json or '{}').get(q)
            key = (ans or 'Blank').upper()
            counts[key] = counts.get(key, 0) + 1
        return counts

    stats = []
    import json
    for q in questions:
        correct_ans = answer_key[q].upper()
        total_correct = sum(
            1 for r in results
            if json.loads(r.raw_answers_json or '{}').get(q, '').upper() == correct_ans
        )
        p   = round(total_correct / n, 3) if n else 0.0   # difficulty index
        p_h = fraction_correct(top_group, q)
        p_l = fraction_correct(bottom_group, q)
        d   = round(p_h - p_l, 3)                          # discrimination index

        # Difficulty level label
        if p >= 0.80:
            difficulty_label = 'Easy'
        elif p >= 0.40:
            difficulty_label = 'Moderate'
        else:
            difficulty_label = 'Hard'

        # Discrimination quality label
        if d >= 0.40:
            disc_label = 'Excellent'
        elif d >= 0.30:
            disc_label = 'Good'
        elif d >= 0.20:
            disc_label = 'Fair'
        else:
            disc_label = 'Poor'

        stats.append({
            'question':             q,
            'correct_answer':       correct_ans,
            'total_correct':        total_correct,
            'difficulty_index':     p,
            'difficulty_label':     difficulty_label,
            'discrimination_index': d,
            'discrimination_label': disc_label,
            'distractors':          distractor_count(q),
        })

    return stats


# ── Difficulty bar chart ──────────────────────────────────────────────────────

def difficulty_chart(q_stats: List[dict]) -> str:
    if not q_stats:
        return ''

    labels = [s['question'] for s in q_stats]
    values = [s['difficulty_index'] for s in q_stats]
    colors = [
        BRAND_EMERALD if v >= 0.80 else BRAND_AMBER if v >= 0.40 else BRAND_RED
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.45), 3.5), facecolor=BG)
    _style_ax(ax, 'Question Difficulty Index  (1 = easiest)')

    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(0.80, color=BRAND_EMERALD, linewidth=1, linestyle='--', alpha=0.7)
    ax.axhline(0.40, color=BRAND_AMBER,   linewidth=1, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('p-value', color=TEXT_MID, fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    legend_items = [
        mpatches.Patch(color=BRAND_EMERALD, label='Easy (p ≥ 0.80)'),
        mpatches.Patch(color=BRAND_AMBER,   label='Moderate (0.40–0.79)'),
        mpatches.Patch(color=BRAND_RED,     label='Hard (p < 0.40)'),
    ]
    ax.legend(handles=legend_items, fontsize=8, framealpha=0.8)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Discrimination bar chart ──────────────────────────────────────────────────

def discrimination_chart(q_stats: List[dict]) -> str:
    if not q_stats:
        return ''

    labels = [s['question'] for s in q_stats]
    values = [s['discrimination_index'] for s in q_stats]
    colors = [
        BRAND_INDIGO if v >= 0.40 else BRAND_PURPLE if v >= 0.20 else BRAND_RED
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.45), 3.5), facecolor=BG)
    _style_ax(ax, 'Discrimination Index  (D ≥ 0.40 = Excellent)')

    ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(0.40, color=BRAND_INDIGO, linewidth=1, linestyle='--', alpha=0.7)
    ax.axhline(0.20, color=BRAND_PURPLE, linewidth=1, linestyle='--', alpha=0.7)
    ax.axhline(0.0,  color='#94A3B8',    linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel('D-value', color=TEXT_MID, fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    legend_items = [
        mpatches.Patch(color=BRAND_INDIGO,  label='Excellent (D ≥ 0.40)'),
        mpatches.Patch(color=BRAND_PURPLE,  label='Fair (0.20–0.39)'),
        mpatches.Patch(color=BRAND_RED,     label='Poor / Negative'),
    ]
    ax.legend(handles=legend_items, fontsize=8, framealpha=0.8)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Summary stats ─────────────────────────────────────────────────────────────

def session_summary(results, passing_score: float = 50.0) -> dict:
    if not results:
        return {}

    pcts = [r.percentage for r in results]
    n    = len(pcts)

    return {
        'n':           n,
        'mean':        round(float(np.mean(pcts)),   1),
        'median':      round(float(np.median(pcts)), 1),
        'std':         round(float(np.std(pcts)),    1),
        'min':         round(float(np.min(pcts)),    1),
        'max':         round(float(np.max(pcts)),    1),
        'pass_count':  sum(1 for p in pcts if p >= passing_score),
        'fail_count':  sum(1 for p in pcts if p <  passing_score),
        'pass_rate':   round(sum(1 for p in pcts if p >= passing_score) / n * 100, 1),
    }


# ── Raw-dict versions (no ORM) ────────────────────────────────────

def session_summary_raw(results: list, passing_score: float = 50.0) -> dict:
    """Same as session_summary but accepts plain dicts from sqlite3."""
    if not results:
        return {}

    pcts = [r['percentage'] for r in results]
    n    = len(pcts)

    return {
        'n':          n,
        'mean':       round(float(np.mean(pcts)),   1),
        'median':     round(float(np.median(pcts)), 1),
        'std':        round(float(np.std(pcts)),    1),
        'min':        round(float(np.min(pcts)),    1),
        'max':        round(float(np.max(pcts)),    1),
        'pass_count': sum(1 for p in pcts if p >= passing_score),
        'fail_count': sum(1 for p in pcts if p <  passing_score),
        'pass_rate':  round(sum(1 for p in pcts if p >= passing_score) / n * 100, 1),
    }


def per_question_stats_raw(results: list, answer_key: dict) -> list:
    """Same as per_question_stats but accepts plain dicts from sqlite3."""
    if not results or not answer_key:
        return []

    n = len(results)
    questions = sorted(answer_key.keys(),
                       key=lambda q: int(q[1:]) if q[1:].isdigit() else 0)

    sorted_results = sorted(results, key=lambda r: r['percentage'])
    cutoff       = max(1, int(n * 0.27))
    bottom_group = sorted_results[:cutoff]
    top_group    = sorted_results[n - cutoff:]

    def frac_correct(group, q):
        ca = answer_key[q].upper()
        hits = sum(1 for r in group
                   if (r['raw_answers'].get(q) or '').upper() == ca)
        return hits / len(group) if group else 0.0

    def distractor_count(q):
        counts: Dict[str, int] = {}
        for r in results:
            ans = (r['raw_answers'].get(q) or 'Blank').upper()
            counts[ans] = counts.get(ans, 0) + 1
        return counts

    stats = []
    for q in questions:
        ca = answer_key[q].upper()
        total_correct = sum(
            1 for r in results
            if (r['raw_answers'].get(q) or '').upper() == ca)
        p   = round(total_correct / n, 3) if n else 0.0
        p_h = frac_correct(top_group, q)
        p_l = frac_correct(bottom_group, q)
        d   = round(p_h - p_l, 3)

        difficulty_label    = 'Easy' if p >= 0.80 else 'Moderate' if p >= 0.40 else 'Hard'
        discrimination_label= ('Excellent' if d >= 0.40 else 'Good' if d >= 0.30
                                else 'Fair' if d >= 0.20 else 'Poor')

        stats.append({
            'question':             q,
            'correct_answer':       ca,
            'total_correct':        total_correct,
            'difficulty_index':     p,
            'difficulty_label':     difficulty_label,
            'discrimination_index': d,
            'discrimination_label': discrimination_label,
            'distractors':          distractor_count(q),
        })

    return stats
