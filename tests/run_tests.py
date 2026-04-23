"""
MPASAS – Test Runner
Run all unit + integration tests offline.

    python tests/run_tests.py
    python tests/run_tests.py --verbose
"""
import os, sys, time, traceback, tempfile, json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

# ── Console colours ────────────────────────────────────────────────
G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'; C = '\033[96m'; B = '\033[1m'; X = '\033[0m'
PASS = f"{G}✅ PASS{X}"; FAIL = f"{R}❌ FAIL{X}"; SKIP = f"{Y}⚠️  SKIP{X}"

_results = []

class SkipTest(Exception): pass

def test(name):
    def wrap(fn):
        def runner():
            t0 = time.time()
            try:
                msg = fn() or ''
                _results.append((name, 'pass', msg, time.time()-t0))
                print(f"  {PASS}  {name}  {C}({(time.time()-t0)*1000:.0f}ms){X}")
                if msg: print(f"         → {msg}")
            except SkipTest as e:
                _results.append((name, 'skip', str(e), 0))
                print(f"  {SKIP}  {name}  — {e}")
            except Exception as e:
                _results.append((name, 'fail', str(e), time.time()-t0))
                print(f"  {FAIL}  {name}")
                print(f"         {R}{e}{X}")
                if '--verbose' in sys.argv: traceback.print_exc()
        return runner
    return wrap

# ══════════════════════════════════════════════════════════════════
#   Dependency checks
# ══════════════════════════════════════════════════════════════════

@test("Import: OpenCV")
def t_cv():
    import cv2; return f"v{cv2.__version__}"

@test("Import: NumPy")
def t_np():
    import numpy as np; return f"v{np.__version__}"

@test("Import: Flask")
def t_flask():
    import flask; return f"v{flask.__version__}"

@test("Import: Pandas")
def t_pandas():
    import pandas as pd; return f"v{pd.__version__}"

@test("Import: Matplotlib")
def t_mpl():
    import matplotlib; return f"v{matplotlib.__version__}"

@test("Import: sqlite3 (built-in)")
def t_sqlite():
    import sqlite3; return f"SQLite v{sqlite3.sqlite_version}"

# ══════════════════════════════════════════════════════════════════
#   Module imports
# ══════════════════════════════════════════════════════════════════

@test("Module: aligner")
def t_aligner_import():
    from modules import aligner; return "OK"

@test("Module: extractor")
def t_extractor_import():
    from modules import extractor; return "OK"

@test("Module: scorer")
def t_scorer_import():
    from modules import scorer; return "OK"

@test("Module: analytics")
def t_analytics_import():
    from modules import analytics; return "OK"

@test("Module: models (sqlite3)")
def t_models_import():
    from modules import models; return "OK"

# ══════════════════════════════════════════════════════════════════
#   Scorer
# ══════════════════════════════════════════════════════════════════

@test("Scorer: perfect score")
def t_scorer_perfect():
    from modules.scorer import score, percentage
    key = {f'Q{i}': chr(65+i%5) for i in range(1, 6)}
    correct, total, _ = score(key, key)
    assert correct == 5 == total
    assert percentage(correct, total) == 100.0
    return "5/5 → 100%"

@test("Scorer: zero score")
def t_scorer_zero():
    from modules.scorer import score
    key = {'Q1':'A','Q2':'B','Q3':'C'}
    ans = {'Q1':'B','Q2':'C','Q3':'D'}
    correct, total, _ = score(ans, key)
    assert correct == 0 and total == 3
    return "0/3 → 0%"

@test("Scorer: blanks treated as wrong")
def t_scorer_blanks():
    from modules.scorer import score, percentage
    key = {'Q1':'A','Q2':'B','Q3':'C','Q4':'D'}
    ans = {'Q1':'A','Q2': None,'Q3':'C','Q4':'B'}
    correct, total, _ = score(ans, key)
    assert correct == 2 and total == 4
    assert percentage(correct, total) == 50.0
    return "2/4 → 50%"

@test("Scorer: case-insensitive match")
def t_scorer_case():
    from modules.scorer import score
    key = {'Q1':'A','Q2':'B'}
    ans = {'Q1':'a','Q2':'b'}
    correct, _, _ = score(ans, key)
    assert correct == 2
    return "lowercase answers matched"

# ══════════════════════════════════════════════════════════════════
#   Analytics
# ══════════════════════════════════════════════════════════════════

@test("Analytics: empty results → empty dict")
def t_analytics_empty():
    from modules.analytics import session_summary_raw
    assert session_summary_raw([]) == {}
    return "OK"

@test("Analytics: summary stats correct")
def t_analytics_summary():
    from modules.analytics import session_summary_raw
    fake = [{'percentage': p, 'passed': p >= 50} for p in [40,60,70,80,50]]
    s = session_summary_raw(fake, 50.0)
    assert s['n'] == 5
    assert s['pass_count'] == 4
    assert s['fail_count'] == 1
    return f"mean={s['mean']}%, pass_rate={s['pass_rate']}%"

@test("Analytics: score distribution chart renders")
def t_analytics_chart():
    from modules.analytics import score_distribution_chart
    b64 = score_distribution_chart([30,45,55,60,75,80,90], 50.0)
    assert len(b64) > 500
    return f"base64 PNG ({len(b64)} chars)"

@test("Analytics: difficulty chart renders")
def t_diff_chart():
    from modules.analytics import difficulty_chart
    stats = [{'question':'Q1','difficulty_index':0.7,'discrimination_index':0.3}]
    b64 = difficulty_chart(stats)
    assert len(b64) > 100
    return "OK"

# ══════════════════════════════════════════════════════════════════
#   Extractor
# ══════════════════════════════════════════════════════════════════

@test("Extractor: extract_roi with relative coords")
def t_extractor_roi():
    import numpy as np
    from modules.extractor import extract_roi
    img  = np.zeros((100, 100, 3), dtype=np.uint8)
    zone = {'xPct': 0.1, 'yPct': 0.2, 'wPct': 0.5, 'hPct': 0.3}
    roi  = extract_roi(img, zone)
    assert roi.shape[0] == 30 and roi.shape[1] == 50
    return f"ROI shape {roi.shape[:2]}"

@test("Extractor: analyse_grid detects filled bubble")
def t_extractor_grid():
    import numpy as np, cv2
    from modules.extractor import analyse_grid
    roi = np.ones((40, 200, 3), dtype=np.uint8) * 200
    cv2.circle(roi, (75, 20), 8, (20, 20, 20), -1)   # fill column B
    result = analyse_grid(roi, rows=1, cols=4, labels=['A','B','C','D'])
    assert 1 in result and result[1]['status'] != 'error'
    return f"Q1 → {result[1]['answer']} ({result[1]['status']})"

@test("Extractor: blank sheet gives blank/None answers")
def t_extractor_blank():
    import numpy as np
    from modules.extractor import analyse_grid
    roi = np.ones((80, 200, 3), dtype=np.uint8) * 220   # all light
    result = analyse_grid(roi, rows=2, cols=4)
    assert all(r['status'] == 'blank' for r in result.values())
    return f"{len(result)} blank questions detected"

# ══════════════════════════════════════════════════════════════════
#   Aligner
# ══════════════════════════════════════════════════════════════════

@test("Aligner: identical image aligns with no error")
def t_aligner_identity():
    import numpy as np, cv2
    from modules.aligner import align_to_template
    img = np.ones((300, 250, 3), dtype=np.uint8) * 200
    cv2.rectangle(img, (30, 30), (220, 270), (50, 50, 50), 2)
    cv2.circle(img, (125, 150), 40, (100, 100, 200), -1)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as t1:
        cv2.imwrite(t1.name, img); tp = t1.name
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as t2:
        cv2.imwrite(t2.name, img); sp = t2.name
    aligned, note = align_to_template(tp, sp)
    os.unlink(tp); os.unlink(sp)
    assert aligned is not None
    return f"note: {note[:60]}"

@test("Aligner: missing file returns None gracefully")
def t_aligner_missing():
    from modules.aligner import align_to_template
    result, note = align_to_template('/nonexistent/t.png', '/nonexistent/s.png')
    assert result is None
    return f"note: {note}"

# ══════════════════════════════════════════════════════════════════
#   Database (sqlite3)
# ══════════════════════════════════════════════════════════════════

@test("Database: init creates tables")
def t_db_init():
    import modules.models as m
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        dbp = f.name
    m.init_db(dbp)
    import sqlite3
    con = sqlite3.connect(dbp)
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    con.close(); os.unlink(dbp)
    assert 'exam_template' in tables
    assert 'grading_session' in tables
    assert 'student_result' in tables
    return f"tables: {sorted(tables)}"

@test("Database: template CRUD")
def t_db_template():
    import modules.models as m
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        dbp = f.name
    m.init_db(dbp)
    tid = m.template_create('Test Sheet', 'desc', zones=[{'type':'omr'}],
                             answer_key={'Q1':'A'})
    assert tid > 0
    tmpl = m.template_get(tid)
    assert tmpl['name'] == 'Test Sheet'
    assert tmpl['answer_key'] == {'Q1':'A'}
    m.template_update(tid, name='Updated Sheet')
    assert m.template_get(tid)['name'] == 'Updated Sheet'
    lst = m.template_list()
    assert any(t['id'] == tid for t in lst)
    m.template_delete(tid)
    assert m.template_get(tid) is None
    os.unlink(dbp)
    return "create / get / update / list / delete all OK"

@test("Database: session + result cascade delete")
def t_db_cascade():
    import modules.models as m
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        dbp = f.name
    m.init_db(dbp)
    tid = m.template_create('T', answer_key={'Q1':'A'})
    sid = m.session_create('S1', tid, passing_score=50.0, total_scripts=1)
    m.result_create(sid, 'Alice', '/tmp/a.png', {'Q1':'A'}, {},
                    1, 1, 100.0, True, 'ok')
    assert m.result_count() == 1
    m.session_delete(sid)
    assert m.result_count() == 0   # cascade should remove child results
    m.template_delete(tid)
    os.unlink(dbp)
    return "cascade delete verified"

@test("Database: session get includes results and template")
def t_db_session_get():
    import modules.models as m
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        dbp = f.name
    m.init_db(dbp)
    tid = m.template_create('Tmpl', answer_key={'Q1':'A','Q2':'B'})
    sid = m.session_create('Sess', tid, 50.0, total_scripts=2)
    m.result_create(sid,'Alice','/a.png',{'Q1':'A','Q2':'B'},{},2,2,100.0,True,'ok')
    m.result_create(sid,'Bob',  '/b.png',{'Q1':'B','Q2':'A'},{},0,2,  0.0,False,'ok')
    m.session_update(sid, status='done', processed=2)
    sess = m.session_get(sid)
    assert len(sess['results']) == 2
    assert sess['template']['name'] == 'Tmpl'
    assert sess['pass_count'] == 1
    assert sess['average_score'] == 50.0
    m.session_delete(sid)
    m.template_delete(tid)
    os.unlink(dbp)
    return f"2 results, avg={sess['average_score']}%, pass={sess['pass_count']}"

# ══════════════════════════════════════════════════════════════════
#   Test data generator
# ══════════════════════════════════════════════════════════════════

@test("Generator: creates template + 15 scripts")
def t_generator():
    from tests.generate_test_sheet import run
    with tempfile.TemporaryDirectory() as td:
        tdir = os.path.join(td, 'templates')
        sdir = os.path.join(td, 'scripts')
        tpath, zones, key = run(tdir, sdir)
        assert os.path.exists(tpath)
        scripts = os.listdir(sdir)
        assert len(scripts) == 15
        assert len(key) == 20
        return f"template + {len(scripts)} scripts, {len(key)}-Q key"

# ══════════════════════════════════════════════════════════════════
#   Flask app
# ══════════════════════════════════════════════════════════════════

@test("Flask app: creates successfully")
def t_app_create():
    from app import create_app
    app = create_app()
    assert app is not None
    return "OK"

@test("Flask app: all expected routes registered")
def t_app_routes():
    from app import create_app
    app = create_app()
    routes = {r.endpoint for r in app.url_map.iter_rules()}
    needed = ['index','templates_list','new_template','calibrate',
              'new_session','sessions_list','results',
              'export_csv','export_excel','delete_session']
    missing = [e for e in needed if e not in routes]
    assert not missing, f"Missing: {missing}"
    return f"{len(routes)} routes OK"

@test("Flask app: dashboard renders (test client)")
def t_app_dashboard():
    from app import create_app
    app = create_app()
    with app.test_client() as c:
        r = c.get('/')
        assert r.status_code == 200
        assert b'MPASAS' in r.data
    return "HTTP 200 + MPASAS in HTML"

@test("Flask app: new_template page renders")
def t_app_new_template():
    from app import create_app
    app = create_app()
    with app.test_client() as c:
        r = c.get('/templates/new')
        assert r.status_code == 200
    return "HTTP 200"

@test("Flask app: sessions list renders")
def t_app_sessions():
    from app import create_app
    app = create_app()
    with app.test_client() as c:
        r = c.get('/sessions')
        assert r.status_code == 200
    return "HTTP 200"

# ══════════════════════════════════════════════════════════════════
#   Full integration (generate → grade → analytics)
# ══════════════════════════════════════════════════════════════════

@test("Integration: generate → grade → analytics pipeline")
def t_integration():
    import modules.models as m, cv2
    from tests.generate_test_sheet import run
    from modules import extractor, scorer, analytics

    with tempfile.TemporaryDirectory() as td:
        tdir = os.path.join(td, 'tmpl')
        sdir = os.path.join(td, 'scripts')
        dbp  = os.path.join(td, 'test.db')

        m.init_db(dbp)
        tmpl_path, zones, key = run(tdir, sdir)

        tid = m.template_create('IntegTest', zones=zones, answer_key=key)
        sid = m.session_create('IntegSess', tid, 50.0)

        scripts = [os.path.join(sdir, f) for f in os.listdir(sdir)]
        passed_n = 0

        for sp in scripts:
            img = cv2.imread(sp)
            if img is None: continue
            extr    = extractor.extract_all_answers(img, zones)
            answers = extr['answers']
            correct, total, _ = scorer.score(answers, key)
            pct = scorer.percentage(correct, total)
            passed = pct >= 50.0
            if passed: passed_n += 1
            m.result_create(sid, os.path.basename(sp), sp,
                            answers, {}, correct, total, pct, passed, 'direct')

        m.session_update(sid, status='done', processed=len(scripts))
        sess    = m.session_get(sid)
        summary = analytics.session_summary_raw(sess['results'], 50.0)
        q_stats = analytics.per_question_stats_raw(sess['results'], key)

        assert summary['n'] == len(scripts)
        assert len(q_stats) == len(key)
        assert summary['mean'] >= 0

        return (f"{summary['n']} scripts graded | "
                f"avg={summary['mean']}% | "
                f"pass_rate={summary['pass_rate']}% | "
                f"{len(q_stats)} questions analysed")

# ══════════════════════════════════════════════════════════════════
#   Main
# ══════════════════════════════════════════════════════════════════

ALL = [
    t_cv, t_np, t_flask, t_pandas, t_mpl, t_sqlite,
    t_aligner_import, t_extractor_import, t_scorer_import,
    t_analytics_import, t_models_import,
    t_scorer_perfect, t_scorer_zero, t_scorer_blanks, t_scorer_case,
    t_analytics_empty, t_analytics_summary, t_analytics_chart, t_diff_chart,
    t_extractor_roi, t_extractor_grid, t_extractor_blank,
    t_aligner_identity, t_aligner_missing,
    t_db_init, t_db_template, t_db_cascade, t_db_session_get,
    t_generator,
    t_app_create, t_app_routes, t_app_dashboard,
    t_app_new_template, t_app_sessions,
    t_integration,
]

if __name__ == '__main__':
    print(f"\n{B}{'═'*60}{X}")
    print(f"{B}  MPASAS – Test Suite  ({len(ALL)} tests){X}")
    print(f"{B}{'═'*60}{X}\n")

    for fn in ALL:
        fn()

    passed  = sum(1 for _, s, *_ in _results if s == 'pass')
    failed  = sum(1 for _, s, *_ in _results if s == 'fail')
    skipped = sum(1 for _, s, *_ in _results if s == 'skip')

    print(f"\n{B}{'─'*60}{X}")
    colour = G if not failed else R
    print(f"{B}{colour}  {passed}/{len(ALL)} passed  ·  {failed} failed  ·  {skipped} skipped{X}")
    print(f"{B}{'═'*60}{X}\n")

    if failed:
        print(f"{R}Failed tests:{X}")
        for name, status, msg, _ in _results:
            if status == 'fail':
                print(f"  • {name}: {msg}")
        sys.exit(1)
    else:
        print(f"{G}All tests passed! ✅{X}\n")
