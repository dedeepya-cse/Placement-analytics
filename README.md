# Placement Analytics & Prediction System

A Python Full Stack + Machine Learning project: a Flask web app backed by a SQL database that
analyses student placement data and predicts a student's placement chance with an ML model.

## Features
- **SQL analytics dashboard** – GROUP BY, CASE bands, and a window function (`RANK() OVER (PARTITION BY ...)`)
- **ML model comparison** – Logistic Regression, SVM, Random Forest, Gradient Boosting (5-fold cross-validation + held-out test set); best model is saved and used by the app
- **Prediction form** – enter CGPA, scores, projects etc. and get a placement probability; every prediction is stored in the database
- **Students page** – filtering (branch / status) and pagination using parameterised SQL queries
- JSON endpoint: `/api/stats`

## Tech stack
Python, Flask, SQL (SQLite; schema is MySQL-friendly), scikit-learn, pandas, NumPy, HTML/CSS, Jinja2

## Run it
```bash
pip install -r requirements.txt
python setup_db.py      # creates the DB and loads a synthetic dataset
python train_model.py   # trains the models and saves the best one
python app.py           # open http://127.0.0.1:5000
```

## Project structure
```
app.py            Flask routes + SQL queries
setup_db.py       creates the database and generates the synthetic dataset
train_model.py    trains/compares ML algorithms, saves the best model
schema.sql        table definitions and indexes
templates/        Jinja2 pages (dashboard, predict, students)
static/style.css  styling
```

## Note on the data
The dataset is **randomly generated**, so the patterns it shows are not real-world findings.
Replace it with a real dataset (e.g. your college's placement CSV) by loading it into the `students` table.

## Ideas to extend
Login with roles (student / admin), MySQL instead of SQLite, a REST API, charts with Chart.js, Docker deployment.
