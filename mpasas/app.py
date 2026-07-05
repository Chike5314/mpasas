"""
MPASAS – Flask Application
All routes. Uses pure sqlite3 (no ORM dependency).
"""
import os, json, uuid, io, base64
from datetime import datetime
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, send_file, jsonify)
from werkzeug.utils import secure_filename

from config import Config
import modules.models as db
from modules import aligner, extractor, scorer, analytics, visualiser
from modules import fast_template as ft
import pandas as pd
import cv2
import numpy as np


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    for d in [app.config['TEMPLATE_IMAGES'],
              app.config['SCRIPT_IMAGES'],
              app.config['RESULTS_FOLDER'],
              os.path.join(app.root_path, 'instance'),
              os.path.join(app.root_path, 'static', 'charts')]:
        os.makedirs(d, exist_ok=True)

    db.init_db(os.path.join(app.root_path, 'instance', 'mpasas.db'))

    # ── helpers ───────────────────────────────────────────────────
    def allowed(fn):
        return '.' in fn and fn.rsplit('.', 1)[-1].lower() \
               in app.config['ALLOWED_EXTENSIONS']

    def save_upload(file, folder) -> str:
        ext  = secure_filename(file.filename).rsplit('.', 1)[-1].lower()
        path = os.path.join(folder, f"{uuid.uuid4().hex}.{ext}")
        file.save(path)
        return path

    def img_to_b64(img_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', img_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, 88])
        return base64.b64encode(buf.tobytes()).decode() if ok else ''

    def ts():
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Dashboard ─────────────────────────────────────────────────
    @app.route('/')
    def index():
        return render_template('index.html',
                               templates=db.template_list(),
                               sessions=db.session_list(limit=5),
                               total_graded=db.result_count())

    # ── Templates ─────────────────────────────────────────────────
    @app.route('/templates')
    def templates_list():
        return render_template('templates_page.html', templates=db.template_list())

    @app.route('/templates/new', methods=['GET', 'POST'])
    def new_template():
        if request.method == 'POST':
            name = request.form.get('name', '').strip()
            desc = request.form.get('description', '').strip()
            if not name:
                flash('Template name is required.', 'error')
                return redirect(url_for('new_template'))
            img_path = None
            f = request.files.get('template_image')
            if f and f.filename and allowed(f.filename):
                img_path = save_upload(f, app.config['TEMPLATE_IMAGES'])
            tid = db.template_create(name=name, description=desc, image_path=img_path)
            flash(f'Template "{name}" created – calibrate the zones.', 'success')
            return redirect(url_for('calibrate', template_id=tid))
        return render_template('new_template.html')

    @app.route('/templates/<int:tid>/delete', methods=['POST'])
    def delete_template(tid):
        db.template_delete(tid)
        flash('Template deleted.', 'info')
        return redirect(url_for('templates_list'))

    # ── Calibration ───────────────────────────────────────────────
    @app.route('/templates/<int:tid>/calibrate', methods=['GET', 'POST'])
    def calibrate(tid):
        tmpl = db.template_get(tid)
        if tmpl is None:
            flash('Template not found.', 'error')
            return redirect(url_for('templates_list'))

        if request.method == 'POST':
            updates = {}
            new_img = request.files.get('template_image')
            if new_img and new_img.filename and allowed(new_img.filename):
                updates['image_path'] = save_upload(new_img,
                                                    app.config['TEMPLATE_IMAGES'])
            zones_raw = request.form.get('zones_json', '[]')
            key_raw   = request.form.get('answer_key_json', '{}')
            try:
                zones = json.loads(zones_raw)
                updates['zones_json']      = json.dumps(zones)
                updates['total_questions'] = sum(
                    z.get('rows', 0) for z in zones if z.get('type') == 'omr')
            except json.JSONDecodeError:
                flash('Invalid zone data.', 'error')
                return redirect(url_for('calibrate', tid=tid))
            try:
                updates['answer_key_json'] = json.dumps(json.loads(key_raw))
            except json.JSONDecodeError:
                updates['answer_key_json'] = '{}'
            db.template_update(tid, **updates)
            flash('Template calibrated and saved!', 'success')
            return redirect(url_for('templates_list'))

        img_url = None
        if tmpl.get('image_path') and os.path.exists(tmpl['image_path']):
            img_url = url_for('serve_upload', folder='templates',
                              filename=os.path.basename(tmpl['image_path']))
        return render_template('calibrate.html', template=tmpl,
                               img_url=img_url,
                               zones_json=tmpl.get('zones_json', '[]'),
                               answer_key_json=tmpl.get('answer_key_json', '{}'))

    # ── NEW: Scan answer key from a photo ─────────────────────────
    @app.route('/templates/<int:tid>/scan-key', methods=['GET', 'POST'])
    def scan_answer_key(tid):
        """
        Upload a filled-in marking-guide sheet photo.
        The system extracts answers and shows them for verification/editing
        before saving back to the template.
        """
        tmpl = db.template_get(tid)
        if tmpl is None:
            flash('Template not found.', 'error')
            return redirect(url_for('templates_list'))

        if request.method == 'POST':
            action = request.form.get('action', 'scan')

            if action == 'save':
                # Save the verified/edited key
                key_raw = request.form.get('answer_key_json', '{}')
                try:
                    key = json.loads(key_raw)
                    db.template_update(tid, answer_key_json=json.dumps(key))
                    flash(f'Answer key saved ({len(key)} questions).', 'success')
                except json.JSONDecodeError:
                    flash('Invalid key JSON.', 'error')
                return redirect(url_for('calibrate', tid=tid))

            # action == 'scan' — process uploaded key photo
            f = request.files.get('key_image')
            if not f or not f.filename or not allowed(f.filename):
                flash('Upload a valid image of the filled key sheet.', 'error')
                return redirect(url_for('scan_answer_key', tid=tid))

            key_path = save_upload(f, app.config['SCRIPT_IMAGES'])
            zones    = tmpl.get('zones', [])
            img_path = tmpl.get('image_path')

            # Align the key photo to the template
            if img_path and os.path.exists(img_path):
                aligned, note = aligner.align_to_template(img_path, key_path)
            else:
                aligned = cv2.imread(key_path)
                note    = 'No template image – alignment skipped'

            if aligned is None:
                flash('Could not process image. Check the photo quality.', 'error')
                return redirect(url_for('scan_answer_key', tid=tid))

            # Extract answers from the aligned key photo
            extraction      = extractor.extract_all_answers(aligned, zones)
            scanned_answers = extraction['answers']
            warnings        = extraction['warnings']

            # Build annotated preview (amber = detected answer)
            preview_img = aligned.copy()
            h_img, w_img = preview_img.shape[:2]
            q_offset = 0
            for zone in zones:
                if zone.get('type') != 'omr':
                    continue
                rows   = int(zone.get('rows', 20))
                cols   = int(zone.get('cols', 5))
                labels = zone.get('labels') or [chr(65+c) for c in range(cols)]
                cell_h = int(zone['hPct'] * h_img / rows)
                cell_w = int(zone['wPct'] * w_img / cols)
                radius = max(5, int(min(cell_h, cell_w) * 0.30))
                bubs = visualiser._bubble_centers(zone, w_img, h_img)
                for local_q, lbl_map in bubs.items():
                    gq  = f'Q{local_q + q_offset}'
                    ans = (scanned_answers.get(gq) or '').upper()
                    for lbl, (cx, cy) in lbl_map.items():
                        if lbl.upper() == ans:
                            cv2.circle(preview_img, (cx,cy), radius,
                                       (0, 180, 245), -1)
                q_offset += rows

            preview_b64 = img_to_b64(preview_img)

            return render_template('scan_key.html',
                                   template=tmpl,
                                   scanned_key=scanned_answers,
                                   scanned_key_json=json.dumps(scanned_answers),
                                   warnings=warnings,
                                   align_note=note,
                                   preview_b64=preview_b64)

        return render_template('scan_key.html', template=tmpl,
                               scanned_key=None, scanned_key_json='{}',
                               warnings=[], align_note='', preview_b64='')

    # ── File serving ──────────────────────────────────────────────
    @app.route('/uploads/<folder>/<filename>')
    def serve_upload(folder, filename):
        base = os.path.join(app.config['UPLOAD_FOLDER'],
                            secure_filename(folder))
        return send_file(os.path.join(base, secure_filename(filename)))

    # ── Sessions ──────────────────────────────────────────────────
    @app.route('/sessions')
    def sessions_list():
        return render_template('sessions_list.html', sessions=db.session_list())

    @app.route('/sessions/new', methods=['GET', 'POST'])
    def new_session():
        templates = db.template_list()
        if request.method == 'POST':
            name       = request.form.get('name', '').strip()
            tmpl_id    = request.form.get('template_id', type=int)
            pass_score = request.form.get('passing_score', 50.0, type=float)
            scripts    = request.files.getlist('scripts')

            if not name:
                flash('Session name is required.', 'error')
                return render_template('new_session.html', templates=templates)
            if not tmpl_id:
                flash('Please select a template.', 'error')
                return render_template('new_session.html', templates=templates)
            valid = [f for f in scripts if f and f.filename and allowed(f.filename)]
            if not valid:
                flash('Upload at least one image.', 'error')
                return render_template('new_session.html', templates=templates)

            tmpl = db.template_get(tmpl_id)
            if not tmpl:
                flash('Template not found.', 'error')
                return render_template('new_session.html', templates=templates)

            sid  = db.session_create(name=name, template_id=tmpl_id,
                                     passing_score=pass_score,
                                     total_scripts=len(valid))
            key  = tmpl['answer_key']
            zones= tmpl['zones']
            done = 0

            for f in valid:
                orig = secure_filename(f.filename)
                sp   = save_upload(f, app.config['SCRIPT_IMAGES'])
                try:
                    img_path = tmpl.get('image_path')
                    if img_path and os.path.exists(img_path):
                        aligned, note = aligner.align_to_template(img_path, sp)
                    else:
                        aligned = cv2.imread(sp)
                        note    = 'No template image – alignment skipped'
                    if aligned is None:
                        raise ValueError('Alignment returned None')

                    extr   = extractor.extract_all_answers(aligned, zones)
                    ans    = extr['answers']
                    meta_d = extr['metadata']
                    name_s = extractor.extract_name(aligned, zones) or \
                             os.path.splitext(orig)[0]
                    correct, total, _ = scorer.score(ans, key)
                    pct    = scorer.percentage(correct, total)
                    passed = pct >= pass_score

                    db.result_create(sid, name_s, sp, ans, meta_d,
                                     correct, total, pct, passed, note)
                    done += 1
                except Exception as exc:
                    db.result_create(sid, os.path.splitext(orig)[0], sp,
                                     {}, {}, 0, len(key), 0.0, False,
                                     f'ERROR: {exc}')

            db.session_update(sid, status='done', processed=done)
            flash(f'Graded {done}/{len(valid)} scripts!', 'success')
            return redirect(url_for('results', session_id=sid))

        return render_template('new_session.html', templates=templates)

    # ── Results ───────────────────────────────────────────────────
    @app.route('/sessions/<int:session_id>/results')
    def results(session_id):
        sess = db.session_get(session_id)
        if not sess:
            flash('Session not found.', 'error')
            return redirect(url_for('sessions_list'))
        tmpl    = sess.get('template') or {}
        res     = sess.get('results', [])
        key     = tmpl.get('answer_key', {}) if tmpl else {}
        summary = analytics.session_summary_raw(res, sess['passing_score'])
        q_stats = analytics.per_question_stats_raw(res, key)
        pcts    = [r['percentage'] for r in res]
        return render_template('results.html',
                               session=sess, template=tmpl, results=res,
                               summary=summary, q_stats=q_stats,
                               dist_chart=analytics.score_distribution_chart(pcts, sess['passing_score']) if pcts else '',
                               diff_chart=analytics.difficulty_chart(q_stats) if q_stats else '',
                               disc_chart=analytics.discrimination_chart(q_stats) if q_stats else '')

    # ── NEW: Marked script viewer ─────────────────────────────────
    @app.route('/sessions/<int:session_id>/student/<int:result_id>/marked')
    def view_marked_script(session_id, result_id):
        """
        Annotated photo of the student's answer sheet:
        green = correct, red = wrong, amber outline = key answer.
        """
        sess = db.session_get(session_id)
        if not sess:
            flash('Session not found.', 'error')
            return redirect(url_for('sessions_list'))

        tmpl   = sess.get('template') or {}
        res    = next((r for r in sess.get('results', []) if r['id'] == result_id), None)
        if res is None:
            flash('Result not found.', 'error')
            return redirect(url_for('results', session_id=session_id))

        zones  = tmpl.get('zones', [])
        key    = tmpl.get('answer_key', {})
        img_b64 = ''
        note    = ''

        script_path = res.get('script_path', '')
        if script_path and os.path.exists(script_path):
            tmpl_path = tmpl.get('image_path', '')
            if tmpl_path and os.path.exists(tmpl_path):
                aligned, note = aligner.align_to_template(tmpl_path, script_path)
            else:
                aligned = cv2.imread(script_path)
                note    = 'No template image – using raw script'

            if aligned is not None:
                marked  = visualiser.draw_marked_script(
                    aligned, zones, res['raw_answers'], key)
                img_b64 = img_to_b64(marked)
        else:
            note = 'Script image file not found on disk.'

        return render_template('student_marked.html',
                               session=sess, template=tmpl, result=res,
                               img_b64=img_b64, align_note=note, key=key)

    # ── NEW: Per-student score report ─────────────────────────────
    @app.route('/sessions/<int:session_id>/student/<int:result_id>/report')
    def view_student_report(session_id, result_id):
        sess = db.session_get(session_id)
        if not sess:
            flash('Session not found.', 'error')
            return redirect(url_for('sessions_list'))

        tmpl = sess.get('template') or {}
        res  = next((r for r in sess.get('results', []) if r['id'] == result_id), None)
        if res is None:
            flash('Result not found.', 'error')
            return redirect(url_for('results', session_id=session_id))

        report = visualiser.build_student_report(res, tmpl.get('answer_key', {}), sess)
        return render_template('student_report.html',
                               session=sess, template=tmpl, result=res,
                               report=report)

    # ── NEW: Fast-print template download ─────────────────────────
    @app.route('/fast-template')
    def fast_template_page():
        return render_template('fast_template.html')

    @app.route('/fast-template/generate')
    def generate_fast_template():
        n_q   = request.args.get('questions', 20, type=int)
        n_opt = request.args.get('options',    5, type=int)
        two_c = request.args.get('two_col', 'true') == 'true'
        title = request.args.get('title', 'MPASAS Quick Answer Sheet')

        n_q   = max(5, min(n_q,  100))
        n_opt = max(2, min(n_opt,  6))

        img, zones, meta = ft.generate_fast_template(
            n_questions=n_q, n_options=n_opt,
            two_columns=two_c, sheet_title=title)

        # Save image
        fname    = f"fast_template_{n_q}q_{n_opt}opt_{ts()}.png"
        img_path = os.path.join(app.config['TEMPLATE_IMAGES'], fname)
        cv2.imwrite(img_path, img)

        # Save as a template in the DB
        tid = db.template_create(
            name=f'Fast Template – {n_q}Q {n_opt}-option',
            description=f'Auto-generated printable template ({n_q} questions, {n_opt} options)',
            image_path=img_path,
            zones=zones,
        )
        db.template_update(tid, zones_json=json.dumps(zones),
                           total_questions=n_q)

        flash(f'Fast template created ({n_q} questions). '
              f'Download and print it, then calibrate the answer key.', 'success')
        return redirect(url_for('calibrate', tid=tid))

    @app.route('/fast-template/download')
    def download_fast_template():
        n_q   = request.args.get('questions', 20, type=int)
        n_opt = request.args.get('options',    5, type=int)
        two_c = request.args.get('two_col', 'true') == 'true'
        title = request.args.get('title', 'MPASAS Quick Answer Sheet')
        n_q   = max(5, min(n_q,  100))
        n_opt = max(2, min(n_opt,  6))

        img, _, _ = ft.generate_fast_template(
            n_questions=n_q, n_options=n_opt,
            two_columns=two_c, sheet_title=title)

        _, buf = cv2.imencode('.png', img)
        bio = io.BytesIO(buf.tobytes())
        bio.seek(0)
        return send_file(bio, mimetype='image/png', as_attachment=True,
                         download_name=f'MPASAS_AnswerSheet_{n_q}Q.png')

    # ── Exports ───────────────────────────────────────────────────
    @app.route('/sessions/<int:session_id>/export/csv')
    def export_csv(session_id):
        sess = db.session_get(session_id)
        if not sess:
            flash('Session not found.', 'error')
            return redirect(url_for('sessions_list'))
        rows = _build_rows(sess)
        path = os.path.join(app.config['RESULTS_FOLDER'],
                            f"MPASAS_{session_id}_{ts()}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        return send_file(path, as_attachment=True,
                         download_name=f"MPASAS_{sess['name']}.csv")

    @app.route('/sessions/<int:session_id>/export/excel')
    def export_excel(session_id):
        sess = db.session_get(session_id)
        if not sess:
            flash('Session not found.', 'error')
            return redirect(url_for('sessions_list'))
        rows = _build_rows(sess)
        path = os.path.join(app.config['RESULTS_FOLDER'],
                            f"MPASAS_{session_id}_{ts()}.xlsx")
        pd.DataFrame(rows).to_excel(path, index=False)
        return send_file(path, as_attachment=True,
                         download_name=f"MPASAS_{sess['name']}.xlsx")

    def _build_rows(sess):
        rows = []
        for r in sess.get('results', []):
            row = {'Student': r['student_name'], 'Score': int(r['score']),
                   'Total': r['total'], 'Percentage': round(r['percentage'], 1),
                   'Result': 'PASS' if r['passed'] else 'FAIL'}
            for q, a in r['raw_answers'].items():
                row[q] = a or ''
            rows.append(row)
        return rows

    @app.route('/sessions/<int:session_id>/delete', methods=['POST'])
    def delete_session(session_id):
        db.session_delete(session_id)
        flash('Session deleted.', 'info')
        return redirect(url_for('sessions_list'))

    return app
