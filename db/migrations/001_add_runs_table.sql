-- Adds the runs table: one row per training or evaluation run, storing the
-- seed alongside the run record, per the reproducibility section of AGENTS.md.
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  data_source TEXT NOT NULL,      -- 'replay' or 'synthetic'
  instrument TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  final_status TEXT,
  FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);
