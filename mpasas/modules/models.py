"""
MPASAS – Database Layer
Pure sqlite3 — no ORM dependency needed.
"""
import sqlite3, json, os
from datetime import datetime
from contextlib import contextmanager

DB_PATH: str = ''          # set by init_db()

# ── Bootstrap ──────────────────────────────────────────────────────

def init_db(db_path: str):
    global DB_PATH
    DB_PATH = db_path
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with _conn() as con:
        con.executescript("""
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS exam_template (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT    NOT NULL,
            description     TEXT    DEFAULT '',
            created_at      TEXT    DEFAULT (datetime('now')),
            image_path      TEXT,
            zones_json      TEXT    DEFAULT '[]',
            answer_key_json TEXT    DEFAULT '{}',
            total_questions INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS grading_session (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            template_id   INTEGER NOT NULL REFERENCES exam_template(id) ON DELETE CASCADE,
            created_at    TEXT    DEFAULT (datetime('now')),
            passing_score REAL    DEFAULT 50.0,
            status        TEXT    DEFAULT 'pending',
            total_scripts INTEGER DEFAULT 0,
            processed     INTEGER DEFAULT 0,
            notes         TEXT    DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS student_result (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       INTEGER NOT NULL REFERENCES grading_session(id) ON DELETE CASCADE,
            student_name     TEXT    DEFAULT 'Unknown',
            script_path      TEXT,
            raw_answers_json TEXT    DEFAULT '{}',
            metadata_json    TEXT    DEFAULT '{}',
            score            REAL    DEFAULT 0,
            total            INTEGER DEFAULT 0,
            percentage       REAL    DEFAULT 0,
            passed           INTEGER DEFAULT 0,
            alignment_note   TEXT    DEFAULT '',
            created_at       TEXT    DEFAULT (datetime('now'))
        );
        """)


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Helper ─────────────────────────────────────────────────────────

def _row(con, sql, params=()):
    return con.execute(sql, params).fetchone()

def _rows(con, sql, params=()):
    return con.execute(sql, params).fetchall()

def _now():
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


# ══════════════════════════════════════════════════════════════════
#   ExamTemplate
# ══════════════════════════════════════════════════════════════════

def template_create(name, description='', image_path=None,
                    zones=None, answer_key=None) -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO exam_template
               (name, description, image_path, zones_json, answer_key_json, created_at)
               VALUES (?,?,?,?,?,?)""",
            (name, description, image_path,
             json.dumps(zones or []),
             json.dumps(answer_key or {}),
             _now()))
        return cur.lastrowid


def template_update(template_id, **kw):
    allowed = {'name','description','image_path','zones_json',
                'answer_key_json','total_questions'}
    sets, vals = [], []
    for k, v in kw.items():
        if k in allowed:
            sets.append(f"{k}=?")
            vals.append(v)
    if not sets:
        return
    vals.append(template_id)
    with _conn() as con:
        con.execute(f"UPDATE exam_template SET {','.join(sets)} WHERE id=?", vals)


def template_get(template_id) -> dict | None:
    with _conn() as con:
        row = _row(con, "SELECT * FROM exam_template WHERE id=?", (template_id,))
        return _tmpl_dict(row) if row else None


def template_list() -> list:
    with _conn() as con:
        rows = _rows(con, "SELECT * FROM exam_template ORDER BY created_at DESC")
        result = []
        for r in rows:
            d = _tmpl_dict(r)
            d['session_count'] = _row(con,
                "SELECT COUNT(*) FROM grading_session WHERE template_id=?",
                (r['id'],))[0]
            result.append(d)
        return result


def template_delete(template_id):
    with _conn() as con:
        con.execute("DELETE FROM exam_template WHERE id=?", (template_id,))


def _tmpl_dict(row) -> dict:
    d = dict(row)
    d['zones']      = json.loads(d.get('zones_json') or '[]')
    d['answer_key'] = json.loads(d.get('answer_key_json') or '{}')
    return d


# ══════════════════════════════════════════════════════════════════
#   GradingSession
# ══════════════════════════════════════════════════════════════════

def session_create(name, template_id, passing_score=50.0,
                   total_scripts=0) -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO grading_session
               (name, template_id, passing_score, total_scripts, status, created_at)
               VALUES (?,?,?,?,'processing',?)""",
            (name, template_id, passing_score, total_scripts, _now()))
        return cur.lastrowid


def session_update(session_id, **kw):
    allowed = {'name','status','processed','total_scripts','notes'}
    sets, vals = [], []
    for k, v in kw.items():
        if k in allowed:
            sets.append(f"{k}=?")
            vals.append(v)
    if not sets:
        return
    vals.append(session_id)
    with _conn() as con:
        con.execute(f"UPDATE grading_session SET {','.join(sets)} WHERE id=?", vals)


def session_get(session_id) -> dict | None:
    with _conn() as con:
        row = _row(con, "SELECT * FROM grading_session WHERE id=?", (session_id,))
        if not row:
            return None
        d = _sess_dict(row)
        tmpl = _row(con, "SELECT * FROM exam_template WHERE id=?", (d['template_id'],))
        d['template'] = _tmpl_dict(tmpl) if tmpl else None
        results = _rows(con,
            "SELECT * FROM student_result WHERE session_id=? ORDER BY id",
            (session_id,))
        d['results'] = [_result_dict(r) for r in results]
        # Aggregate stats
        pcts = [r['percentage'] for r in d['results']]
        d['average_score'] = round(sum(pcts)/len(pcts), 1) if pcts else 0.0
        pass_list = [r for r in d['results'] if r['passed']]
        d['pass_count'] = len(pass_list)
        d['pass_rate']  = round(len(pass_list)/len(pcts)*100, 1) if pcts else 0.0
        return d


def session_list(limit=200) -> list:
    with _conn() as con:
        rows = _rows(con,
            "SELECT * FROM grading_session ORDER BY created_at DESC LIMIT ?",
            (limit,))
        result = []
        for row in rows:
            d = _sess_dict(row)
            tmpl = _row(con, "SELECT name FROM exam_template WHERE id=?",
                        (d['template_id'],))
            d['template_name'] = tmpl['name'] if tmpl else '—'
            # Quick aggregate from results
            agg = _row(con,
                """SELECT COUNT(*) as n,
                          AVG(percentage) as avg_pct,
                          SUM(passed) as pass_cnt
                   FROM student_result WHERE session_id=?""",
                (row['id'],))
            d['average_score'] = round(agg['avg_pct'] or 0, 1)
            n = agg['n'] or 1
            d['pass_count'] = agg['pass_cnt'] or 0
            d['pass_rate']  = round((agg['pass_cnt'] or 0) / n * 100, 1)
            result.append(d)
        return result


def session_delete(session_id):
    with _conn() as con:
        con.execute("DELETE FROM grading_session WHERE id=?", (session_id,))


def _sess_dict(row) -> dict:
    return dict(row)


# ══════════════════════════════════════════════════════════════════
#   StudentResult
# ══════════════════════════════════════════════════════════════════

def result_create(session_id, student_name, script_path,
                  raw_answers: dict, metadata: dict,
                  score, total, percentage, passed,
                  alignment_note='') -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO student_result
               (session_id, student_name, script_path,
                raw_answers_json, metadata_json,
                score, total, percentage, passed, alignment_note, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (session_id, student_name, script_path,
             json.dumps(raw_answers), json.dumps(metadata),
             score, total, percentage, int(passed),
             alignment_note, _now()))
        return cur.lastrowid


def result_count() -> int:
    with _conn() as con:
        return _row(con, "SELECT COUNT(*) FROM student_result")[0]


def _result_dict(row) -> dict:
    d = dict(row)
    d['raw_answers'] = json.loads(d.get('raw_answers_json') or '{}')
    d['metadata']    = json.loads(d.get('metadata_json') or '{}')
    d['passed']      = bool(d['passed'])
    return d
