"""Trains and compares ML algorithms on the students table; saves the best one."""
import json
import sqlite3
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = Path(__file__).parent
FEATURES = ["cgpa", "backlogs", "coding_score", "aptitude_score", "communication_score",
            "projects", "internships", "certifications"]

with sqlite3.connect(BASE / "data" / "placement.db") as con:
    df = pd.read_sql_query("SELECT * FROM students", con)

X, y = df[FEATURES], df["placed"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results, fitted = [], {}
for name, m in models.items():
    cv = cross_val_score(m, X_tr, y_tr, cv=5, scoring="accuracy").mean()
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    results.append({"model": name, "cv_accuracy": round(cv, 4),
                    "test_accuracy": round(accuracy_score(y_te, pred), 4),
                    "test_f1": round(f1_score(y_te, pred), 4)})
    fitted[name] = m

best = max(results, key=lambda r: (r["cv_accuracy"], r["test_f1"]))
best_model = fitted[best["model"]]
tn, fp, fn, tp = confusion_matrix(y_te, best_model.predict(X_te)).ravel()

importance = {}
if hasattr(best_model, "feature_importances_"):
    importance = dict(zip(FEATURES, best_model.feature_importances_.round(4).tolist()))
else:  # fall back to a Random Forest purely to rank features
    rf = fitted["Random Forest"]
    importance = dict(zip(FEATURES, rf.feature_importances_.round(4).tolist()))

(BASE / "models").mkdir(exist_ok=True)
joblib.dump(best_model, BASE / "models" / "best_model.joblib")
(BASE / "models" / "metrics.json").write_text(json.dumps({
    "features": FEATURES, "results": results, "best": best["model"],
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "importance": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
}, indent=2))

print(pd.DataFrame(results).to_string(index=False))
print(f"\nBest model: {best['model']}  (saved to models/best_model.joblib)")
