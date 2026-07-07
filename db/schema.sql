CREATE TABLE accounts (
  account_id INTEGER PRIMARY KEY,
  agent_id INTEGER NOT NULL,
  phase TEXT NOT NULL,
  starting_balance REAL NOT NULL,
  current_balance REAL NOT NULL,
  current_equity REAL NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  phase_started_at TEXT NOT NULL
);

CREATE TABLE trades (
  trade_id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL,
  instrument TEXT NOT NULL,
  side TEXT NOT NULL,
  size REAL NOT NULL,
  entry_price REAL NOT NULL,
  exit_price REAL,
  entry_time TEXT NOT NULL,
  exit_time TEXT,
  stop_loss REAL,
  take_profit REAL,
  commission REAL NOT NULL,
  swap REAL NOT NULL,
  realized_pnl REAL,
  FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE TABLE rule_violations (
  violation_id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL,
  rule_name TEXT NOT NULL,
  triggered_at TEXT NOT NULL,
  details TEXT,
  FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE TABLE equity_snapshots (
  snapshot_id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL,
  timestamp TEXT NOT NULL,
  equity REAL NOT NULL,
  balance REAL NOT NULL,
  FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE TABLE agents (
  agent_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  framework_type TEXT NOT NULL,
  notes TEXT
);
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
