import json
import sqlite3
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, g, jsonify, redirect, render_template, request, url_for, flash

BASE = Path(__file__).parent
DB = BASE / "data" / "placement.db"
app = Flask(__name__)
app.secret_key = "change-me-in-production"

FIELDS = [  # name, label, min, max, step
    ("cgpa", "CGPA", 0, 10, 0.01),
    ("backlogs", "Active backlogs", 0, 10, 1),
    ("coding_score", "Coding test score (0-100)", 0, 100, 1),
    ("aptitude_score", "Aptitude score (0-100)", 0, 100, 1),
    ("communication_score", "Communication score (0-100)", 0, 100, 1),
    ("projects", "Projects completed", 0, 20, 1),
    ("internships", "Internships", 0, 10, 1),
    ("certifications", "Certifications", 0, 20, 1),
]

model = joblib.load(BASE / "models" / "best_model.joblib")
METRICS = json.loads((BASE / "models" / "metrics.json").read_text())


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    conn = g.pop("db", None)
    if conn:
        conn.close()


@app.route("/")
def dashboard():
    q = db().execute
    totals = q("""SELECT COUNT(*) AS total, SUM(placed) AS placed,
                         ROUND(100.0 * SUM(placed) / COUNT(*), 1) AS rate,
                         ROUND(AVG(cgpa), 2) AS avg_cgpa FROM students""").fetchone()
    by_branch = q("""SELECT branch, COUNT(*) AS total, SUM(placed) AS placed,
                            ROUND(100.0 * SUM(placed) / COUNT(*), 1) AS rate
                     FROM students GROUP BY branch ORDER BY rate DESC""").fetchall()
    by_cgpa = q("""SELECT CASE WHEN cgpa < 6.5 THEN '< 6.5'
                              WHEN cgpa < 7.5 THEN '6.5 - 7.5'
                              WHEN cgpa < 8.5 THEN '7.5 - 8.5'
                              ELSE '8.5+' END AS band,
                          COUNT(*) AS total,
                          ROUND(100.0 * SUM(placed) / COUNT(*), 1) AS rate
                   FROM students GROUP BY band ORDER BY MIN(cgpa)""").fetchall()
    by_intern = q("""SELECT internships, COUNT(*) AS total,
                            ROUND(100.0 * SUM(placed) / COUNT(*), 1) AS rate
                     FROM students GROUP BY internships ORDER BY internships""").fetchall()
    # Window function: rank students inside their branch by coding score
    top = q("""SELECT * FROM (
                   SELECT name, branch, cgpa, coding_score,
                          RANK() OVER (PARTITION BY branch ORDER BY coding_score DESC, cgpa DESC) AS rnk
                   FROM students) WHERE rnk = 1 ORDER BY coding_score DESC""").fetchall()
    return render_template("dashboard.html", totals=totals, by_branch=by_branch,
                           by_cgpa=by_cgpa, by_intern=by_intern, top=top, m=METRICS)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    values = {}
    if request.method == "POST":
        try:
            values = {f: float(request.form[f]) for f, *_ in FIELDS}
            for f, _, lo, hi, _s in FIELDS:
                if not lo <= values[f] <= hi:
                    raise ValueError(f"{f} must be between {lo} and {hi}")
        except (ValueError, KeyError) as e:
            flash(f"Invalid input: {e}", "error")
        else:
            row = pd.DataFrame([values])[METRICS["features"]]
            prob = float(model.predict_proba(row)[0][1])
            verdict = "Likely to be placed" if prob >= 0.5 else "Needs improvement"
            db().execute("""INSERT INTO predictions (cgpa, backlogs, coding_score, aptitude_score,
                            communication_score, projects, internships, certifications, probability, verdict)
                            VALUES (?,?,?,?,?,?,?,?,?,?)""",
                         (*[values[f] for f in METRICS["features"]], prob, verdict))
            db().commit()
            result = {"prob": round(prob * 100, 1), "verdict": verdict}
    history = db().execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5").fetchall()
    return render_template("predict.html", fields=FIELDS, result=result, values=values, history=history)


@app.route("/students")
def students():
    branch = request.args.get("branch", "")
    placed = request.args.get("placed", "")
    page = max(int(request.args.get("page", 1) or 1), 1)
    where, params = [], []
    if branch:
        where.append("branch = ?"); params.append(branch)
    if placed in ("0", "1"):
        where.append("placed = ?"); params.append(int(placed))
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    total = db().execute(f"SELECT COUNT(*) FROM students {clause}", params).fetchone()[0]
    rows = db().execute(f"SELECT * FROM students {clause} ORDER BY id LIMIT 15 OFFSET ?",
                        (*params, (page - 1) * 15)).fetchall()
    branches = [r[0] for r in db().execute("SELECT DISTINCT branch FROM students ORDER BY branch")]
    return render_template("students.html", rows=rows, branches=branches, branch=branch,
                           placed=placed, page=page, pages=(total + 14) // 15, total=total)


@app.route("/api/stats")
def api_stats():
    rows = db().execute("""SELECT branch, COUNT(*) AS total, SUM(placed) AS placed
                           FROM students GROUP BY branch""").fetchall()
    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    app.run(debug=True)
