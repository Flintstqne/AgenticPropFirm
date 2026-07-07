// Dashboard backend. Reads the shared SQLite file, serves REST for initial
// loads, polls for new rows on a 500ms interval and forwards them over
// WebSocket. No message queue, no second API layer, per AGENTS.md.

import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import { DatabaseSync } from "node:sqlite"; // ponytail: stdlib sqlite, no native build
import express from "express";
import yaml from "js-yaml";
import { WebSocketServer } from "ws";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const DB_PATH = process.env.PROPFIRM_DB || path.join(ROOT, "db", "propfirm.sqlite");
const PORT = process.env.PORT || 3001;
const POLL_MS = 500;

const db = new DatabaseSync(DB_PATH, { readOnly: true });
const app = express();

app.get("/api/accounts", (_req, res) => {
  res.json(db.prepare(`
    SELECT a.*, g.name AS agent_name, g.framework_type
    FROM accounts a JOIN agents g ON g.agent_id = a.agent_id
    ORDER BY a.account_id`).all());
});

app.get("/api/accounts/:id", (req, res) => {
  const account = db.prepare(`
    SELECT a.*, g.name AS agent_name, g.framework_type
    FROM accounts a JOIN agents g ON g.agent_id = a.agent_id
    WHERE a.account_id = ?`).get(req.params.id);
  if (!account) return res.status(404).json({ error: "no such account" });
  const openPositions = db.prepare(
    "SELECT * FROM trades WHERE account_id = ? AND exit_time IS NULL").all(req.params.id);
  const equity = db.prepare(`
    SELECT timestamp, equity, balance FROM equity_snapshots
    WHERE account_id = ? ORDER BY timestamp`).all(req.params.id);
  res.json({ account, openPositions, equity });
});

app.get("/api/trades", (req, res) => {
  const clauses = ["t.exit_time IS NOT NULL"];
  const params = [];
  if (req.query.agent) { clauses.push("g.agent_id = ?"); params.push(req.query.agent); }
  if (req.query.instrument) { clauses.push("t.instrument = ?"); params.push(req.query.instrument); }
  if (req.query.from) { clauses.push("t.exit_time >= ?"); params.push(req.query.from); }
  if (req.query.to) { clauses.push("t.exit_time <= ?"); params.push(req.query.to); }
  res.json(db.prepare(`
    SELECT t.*, g.name AS agent_name
    FROM trades t
    JOIN accounts a ON a.account_id = t.account_id
    JOIN agents g ON g.agent_id = a.agent_id
    WHERE ${clauses.join(" AND ")}
    ORDER BY t.exit_time DESC LIMIT 500`).all(...params));
});

app.get("/api/violations", (_req, res) => {
  res.json(db.prepare(`
    SELECT v.*, g.name AS agent_name
    FROM rule_violations v
    JOIN accounts a ON a.account_id = v.account_id
    JOIN agents g ON g.agent_id = a.agent_id
    ORDER BY v.triggered_at DESC LIMIT 500`).all());
});

// Leaderboard: Sortino ratio on daily returns from equity snapshots,
// scaled by sqrt(252), ties broken by smaller max drawdown. Formula locked
// in AGENTS.md; do not change once training starts.
app.get("/api/leaderboard", (_req, res) => {
  const accounts = db.prepare(`
    SELECT a.account_id, a.status, a.phase, g.agent_id, g.name AS agent_name, g.framework_type
    FROM accounts a JOIN agents g ON g.agent_id = a.agent_id`).all();
  const rows = accounts.map((acct) => {
    const daily = db.prepare(`
      SELECT MAX(timestamp) AS ts, equity FROM equity_snapshots
      WHERE account_id = ? GROUP BY substr(timestamp, 1, 10) ORDER BY ts`).all(acct.account_id);
    const eq = daily.map((r) => r.equity);
    const returns = eq.slice(1).map((v, i) => (v - eq[i]) / eq[i]);
    let peak = -Infinity, maxDD = 0;
    for (const v of eq) { peak = Math.max(peak, v); maxDD = Math.max(maxDD, (peak - v) / peak); }
    const mean = returns.length ? returns.reduce((a, b) => a + b, 0) / returns.length : 0;
    const downside = returns.filter((r) => r < 0);
    const dd = downside.length
      ? Math.sqrt(downside.reduce((a, b) => a + b * b, 0) / downside.length) : 0;
    const sortino = dd > 0 ? (mean / dd) * Math.sqrt(252) : (mean > 0 ? Infinity : 0);
    return { ...acct, sortino, maxDrawdown: maxDD, days: returns.length };
  });
  rows.sort((a, b) => (b.sortino - a.sortino) || (a.maxDrawdown - b.maxDrawdown));
  res.json(rows.map((r, i) => ({ rank: i + 1, ...r, sortino: Number.isFinite(r.sortino) ? r.sortino : null })));
});

app.get("/api/phases", (_req, res) => {
  res.json(yaml.load(fs.readFileSync(path.join(ROOT, "config", "phases.yaml"), "utf8")));
});

app.get("/api/sessions", (_req, res) => {
  const raw = yaml.load(fs.readFileSync(path.join(ROOT, "config", "sessions.yaml"), "utf8"));
  res.json(Object.fromEntries(Object.entries(raw).filter(([k]) => !k.startsWith("_"))));
});

const server = app.listen(PORT, () => console.log(`dashboard server on :${PORT}, db ${DB_PATH}`));

// --- WebSocket push: poll max rowids, forward new rows ---
const wss = new WebSocketServer({ server, path: "/ws" });
const cursors = {
  equity_snapshots: maxId("snapshot_id", "equity_snapshots"),
  trades: maxId("trade_id", "trades"),
  rule_violations: maxId("violation_id", "rule_violations"),
};

function maxId(col, table) {
  return db.prepare(`SELECT COALESCE(MAX(${col}), 0) AS m FROM ${table}`).get().m;
}

function broadcast(msg) {
  const data = JSON.stringify(msg);
  for (const client of wss.clients) if (client.readyState === 1) client.send(data);
}

setInterval(() => {
  const idCols = { equity_snapshots: "snapshot_id", trades: "trade_id", rule_violations: "violation_id" };
  for (const [table, col] of Object.entries(idCols)) {
    const rows = db.prepare(`SELECT * FROM ${table} WHERE ${col} > ? ORDER BY ${col}`).all(cursors[table]);
    if (rows.length) {
      cursors[table] = rows[rows.length - 1][col];
      broadcast({ type: table, rows });
    }
  }
}, POLL_MS);
