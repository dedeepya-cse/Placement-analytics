-- Written to run on SQLite (used here) and with minor changes on MySQL.
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    branch TEXT NOT NULL,
    cgpa REAL NOT NULL,
    backlogs INTEGER NOT NULL DEFAULT 0,
    coding_score INTEGER NOT NULL,
    aptitude_score INTEGER NOT NULL,
    communication_score INTEGER NOT NULL,
    projects INTEGER NOT NULL DEFAULT 0,
    internships INTEGER NOT NULL DEFAULT 0,
    certifications INTEGER NOT NULL DEFAULT 0,
    placed INTEGER NOT NULL CHECK (placed IN (0, 1))
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    cgpa REAL, backlogs INTEGER, coding_score INTEGER, aptitude_score INTEGER,
    communication_score INTEGER, projects INTEGER, internships INTEGER, certifications INTEGER,
    probability REAL NOT NULL,
    verdict TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_students_branch ON students(branch);
CREATE INDEX IF NOT EXISTS idx_students_placed ON students(placed);
