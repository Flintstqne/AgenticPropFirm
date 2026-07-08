-- Training progress, written by scripts/train_rl.py to the shared SQLite
-- file so the dashboard can show a live training run the same way it shows
-- live simulation accounts: one row per training run, one row per rollout
-- of logged metrics (reward, episode length, throughput).
CREATE TABLE IF NOT EXISTS training_runs (
  training_run_id INTEGER PRIMARY KEY,
  instrument TEXT NOT NULL,
  algo TEXT NOT NULL,
  seed INTEGER NOT NULL,
  total_timesteps INTEGER NOT NULL,
  status TEXT NOT NULL,        -- running, finished
  started_at TEXT NOT NULL,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS training_metrics (
  metric_id INTEGER PRIMARY KEY,
  training_run_id INTEGER NOT NULL,
  timesteps INTEGER NOT NULL,
  ep_rew_mean REAL,
  ep_len_mean REAL,
  fps REAL,
  recorded_at TEXT NOT NULL,
  FOREIGN KEY (training_run_id) REFERENCES training_runs(training_run_id)
);
