"""Account ledger. Every function takes an open sqlite3 connection and commits
its own change. Timestamps are ISO 8601 strings, stored as TEXT."""

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn):
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()


def register_agent(conn, name, framework_type, notes=None):
    cur = conn.execute(
        "INSERT INTO agents (name, framework_type, notes) VALUES (?, ?, ?)",
        (name, framework_type, notes),
    )
    conn.commit()
    return cur.lastrowid


def open_account(conn, agent_id, phase, starting_balance, now):
    cur = conn.execute(
        """INSERT INTO accounts
           (agent_id, phase, starting_balance, current_balance, current_equity,
            status, created_at, phase_started_at)
           VALUES (?, ?, ?, ?, ?, 'active', ?, ?)""",
        (agent_id, phase, starting_balance, starting_balance, starting_balance, now, now),
    )
    conn.commit()
    return cur.lastrowid


def get_account(conn, account_id):
    return conn.execute(
        "SELECT * FROM accounts WHERE account_id = ?", (account_id,)
    ).fetchone()


def set_account_status(conn, account_id, status):
    conn.execute(
        "UPDATE accounts SET status = ? WHERE account_id = ?", (status, account_id)
    )
    conn.commit()


def set_account_phase(conn, account_id, phase, now):
    conn.execute(
        "UPDATE accounts SET phase = ?, phase_started_at = ? WHERE account_id = ?",
        (phase, now, account_id),
    )
    conn.commit()


def record_trade_open(conn, account_id, instrument, side, size, entry_price,
                      entry_time, stop_loss=None, take_profit=None,
                      commission=0.0, swap=0.0):
    cur = conn.execute(
        """INSERT INTO trades
           (account_id, instrument, side, size, entry_price, entry_time,
            stop_loss, take_profit, commission, swap)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (account_id, instrument, side, size, entry_price, entry_time,
         stop_loss, take_profit, commission, swap),
    )
    conn.commit()
    return cur.lastrowid


def record_trade_close(conn, trade_id, exit_price, exit_time, realized_pnl,
                       close_commission=0.0):
    conn.execute(
        """UPDATE trades SET exit_price = ?, exit_time = ?, realized_pnl = ?,
           commission = commission + ? WHERE trade_id = ?""",
        (exit_price, exit_time, realized_pnl, close_commission, trade_id),
    )
    trade = conn.execute(
        "SELECT account_id, commission, swap FROM trades WHERE trade_id = ?",
        (trade_id,),
    ).fetchone()
    net = realized_pnl - trade["commission"] - trade["swap"]
    conn.execute(
        """UPDATE accounts SET current_balance = current_balance + ?
           WHERE account_id = ?""",
        (net, trade["account_id"]),
    )
    conn.commit()


def add_swap(conn, trade_id, amount):
    conn.execute("UPDATE trades SET swap = swap + ? WHERE trade_id = ?",
                 (amount, trade_id))
    conn.commit()


def get_open_trades(conn, account_id):
    return conn.execute(
        "SELECT * FROM trades WHERE account_id = ? AND exit_time IS NULL",
        (account_id,),
    ).fetchall()


def update_equity(conn, account_id, equity):
    conn.execute(
        "UPDATE accounts SET current_equity = ? WHERE account_id = ?",
        (equity, account_id),
    )
    conn.commit()


def snapshot_equity(conn, account_id, timestamp):
    acct = get_account(conn, account_id)
    conn.execute(
        """INSERT INTO equity_snapshots (account_id, timestamp, equity, balance)
           VALUES (?, ?, ?, ?)""",
        (account_id, timestamp, acct["current_equity"], acct["current_balance"]),
    )
    conn.commit()


def record_violation(conn, account_id, rule_name, triggered_at, details=None):
    conn.execute(
        """INSERT INTO rule_violations (account_id, rule_name, triggered_at, details)
           VALUES (?, ?, ?, ?)""",
        (account_id, rule_name, triggered_at, details),
    )
    conn.commit()
