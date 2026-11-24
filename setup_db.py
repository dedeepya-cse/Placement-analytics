"""Creates the SQLite database and loads a synthetic student dataset.

NOTE: the data is randomly generated for learning/demo purposes. It is NOT real
student data, so the model's findings should not be treated as real-world facts.
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DB = BASE / "data" / "placement.db"
N = 800
rng = np.random.default_rng(42)

branches = ["CSE", "IT", "ECE", "EEE", "MECH", "AI&DS"]
first = ["Aarav", "Diya", "Ishaan", "Meera", "Rohan", "Sneha", "Karthik", "Anjali", "Vikram", "Priya",
         "Arjun", "Kavya", "Nikhil", "Pooja", "Rahul", "Divya", "Siddharth", "Lakshmi", "Harsha", "Tanvi"]
last = ["Reddy", "Sharma", "Naidu", "Rao", "Kumar", "Iyer", "Patel", "Gupta", "Varma", "Singh"]

df = pd.DataFrame({
    "name": [f"{rng.choice(first)} {rng.choice(last)}" for _ in range(N)],
    "branch": rng.choice(branches, N, p=[.25, .18, .17, .1, .1, .2]),
    "cgpa": np.clip(rng.normal(7.4, 1.0, N), 5.0, 10.0).round(2),
    "backlogs": rng.choice([0, 1, 2, 3], N, p=[.7, .15, .1, .05]),
    "coding_score": np.clip(rng.normal(60, 18, N), 0, 100).astype(int),
    "aptitude_score": np.clip(rng.normal(62, 15, N), 0, 100).astype(int),
    "communication_score": np.clip(rng.normal(65, 14, N), 0, 100).astype(int),
    "projects": rng.integers(0, 5, N),
    "internships": rng.choice([0, 1, 2, 3], N, p=[.4, .35, .2, .05]),
    "certifications": rng.integers(0, 5, N),
})

# Hidden rule used to create labels (with noise so the task is not trivial)
z = (0.9 * (df.cgpa - 7.0) - 0.8 * df.backlogs + 0.04 * (df.coding_score - 60)
     + 0.025 * (df.aptitude_score - 62) + 0.015 * (df.communication_score - 65)
     + 0.35 * df.projects + 0.5 * df.internships + 0.2 * df.certifications - 1.2
     + rng.normal(0, 0.8, N))
df["placed"] = (1 / (1 + np.exp(-z)) > 0.5).astype(int)

DB.parent.mkdir(exist_ok=True)
if DB.exists():
    DB.unlink()
with sqlite3.connect(DB) as con:
    con.executescript((BASE / "schema.sql").read_text())
    df.to_sql("students", con, if_exists="append", index=False)
print(f"Loaded {len(df)} students into {DB}  |  placement rate: {df.placed.mean():.1%}")
