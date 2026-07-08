import React, { useEffect, useMemo, useState } from "react";
import { fmt, get, useLiveFeed, usePoll } from "./api.js";
import SessionRail from "./SessionRail.jsx";
import { EquityCurve, LimitGauge, MetricLine, Sparkline } from "./charts.jsx";

const SCREENS = ["overview", "account", "trades", "leaderboard", "alerts", "training"];

export default function App() {
  const [screen, setScreen] = useState("overview");
  const [selected, setSelected] = useState(null);
  const [accounts, setAccounts] = useState([]);
  const [violations, setViolations] = useState([]);
  const [lastSnap, setLastSnap] = useState(null);
  const [sparks, setSparks] = useState({});   // account_id -> recent equity values
  const phases = usePoll("/api/phases", 60000);

  const loadAccounts = () => get("/api/accounts").then(setAccounts);
  useEffect(() => { loadAccounts(); get("/api/violations").then(setViolations); }, []);

  useLiveFeed({
    equity_snapshots: (rows) => {
      setLastSnap(rows[rows.length - 1]);
      setSparks((prev) => {
        const next = { ...prev };
        for (const r of rows)
          next[r.account_id] = [...(next[r.account_id] || []).slice(-59), r.equity];
        return next;
      });
      loadAccounts();
    },
    rule_violations: (rows) => {
      setViolations((v) => [...rows.slice().reverse(), ...v]);
      loadAccounts();
    },
  });

  const open = (id) => { setSelected(id); setScreen("account"); };

  return (
    <div className="app">
      <SessionRail simTime={lastSnap?.timestamp} mode="REPLAY" />
      <nav className="nav">
        {SCREENS.map((s) => (
          <button key={s} className={screen === s ? "active" : ""} onClick={() => setScreen(s)}>
            {s}
          </button>
        ))}
      </nav>
      <div className="middle">
        <div className="left-rail">
          {accounts.map((a) => (
            <div key={a.account_id} tabIndex={0}
              className={`rail-item ${selected === a.account_id ? "selected" : ""}`}
              onClick={() => open(a.account_id)}>
              <span>{a.agent_name}</span>
              <span className={a.current_equity >= a.starting_balance ? "pos" : "neg"}>
                {fmt(a.current_equity)}
              </span>
            </div>
          ))}
        </div>
        <div className="main">
          {screen === "overview" && <Overview accounts={accounts} sparks={sparks} phases={phases} onOpen={open} />}
          {screen === "account" && <AccountDetail id={selected} phases={phases} />}
          {screen === "trades" && <TradeLedger accounts={accounts} />}
          {screen === "leaderboard" && <Leaderboard />}
          {screen === "alerts" && <AlertLog violations={violations} phases={phases} />}
          {screen === "training" && <Training />}
        </div>
      </div>
      <div className="alert-strip">
        {violations.slice(0, 3).map((v) => (
          <span key={v.violation_id}>
            <span className="neg">{v.rule_name}</span>{" "}
            <span className="caption">{v.agent_name} {v.triggered_at}</span>
          </span>
        ))}
        {violations.length === 0 && <span className="caption">No rule breaches recorded</span>}
      </div>
    </div>
  );
}

function ddUsed(acct, phases) {
  const cfg = phases?.[acct.phase];
  if (!cfg) return 0;
  return (acct.starting_balance - acct.current_equity) /
         (acct.starting_balance * cfg.max_drawdown_pct);
}

function Overview({ accounts, sparks, phases, onOpen }) {
  return (
    <>
      <h1 className="screen-title">Overview</h1>
      <div className="grid" style={{ marginTop: 24 }}>
        {accounts.map((a) => (
          <div key={a.account_id}
            className={`panel account-panel ${a.status === "failed" ? "breached" : ""}`}
            onClick={() => onOpen(a.account_id)}>
            <div className="caption">{a.agent_name} · {a.framework_type}</div>
            <div style={{ margin: "6px 0" }}>
              <span className={`tag ${a.status === "failed" ? "red" : a.phase === "funded" ? "green" : "blue"}`}>
                {a.status === "failed" ? "failed" : a.phase}
              </span>
            </div>
            <div className="hero num">{fmt(a.current_equity)}</div>
            <div style={{ margin: "10px 0" }}>
              <LimitGauge used={ddUsed(a, phases)} label="Drawdown" />
            </div>
            <Sparkline values={sparks[a.account_id]} />
          </div>
        ))}
        {accounts.length === 0 && <div className="empty">No accounts yet</div>}
      </div>
    </>
  );
}

function AccountDetail({ id, phases }) {
  const [data, setData] = useState(null);
  useEffect(() => { if (id) get(`/api/accounts/${id}`).then(setData); }, [id]);
  useLiveFeed({
    equity_snapshots: (rows) => {
      if (!id || !rows.some((r) => r.account_id === id)) return;
      get(`/api/accounts/${id}`).then(setData);
    },
  });
  if (!id) return <div className="empty">Select an account from the left rail</div>;
  if (!data) return null;
  const { account: a, openPositions, equity } = data;
  const cfg = phases?.[a.phase] || {};
  const target = cfg.profit_target_pct != null
    ? a.starting_balance * (1 + cfg.profit_target_pct) : null;
  const floor = cfg.max_drawdown_pct != null
    ? a.starting_balance * (1 - cfg.max_drawdown_pct) : null;
  const dailyUsed = cfg.daily_loss_pct
    ? Math.max(0, (a.current_balance - a.current_equity) / (a.current_balance * cfg.daily_loss_pct)) : 0;
  const tradingDays = new Set(equity.map((e) => e.timestamp.slice(0, 10))).size;
  return (
    <>
      <h1 className="screen-title">{a.agent_name}</h1>
      <div className="panel" style={{ marginTop: 24, padding: 16 }}>
        <EquityCurve points={equity} target={target} floor={floor} />
      </div>
      <div className="blocks">
        <div className="panel block">
          <div className="caption">Balance</div>
          <div className="hero-sm num">{fmt(a.current_balance)}</div>
        </div>
        <div className="panel block">
          <div className="caption">Equity</div>
          <div className="hero-sm num">{fmt(a.current_equity)}</div>
        </div>
        <div className="panel block">
          <div className="caption">Daily loss used</div>
          <div className="hero-sm num">{fmt(dailyUsed * 100, 0)}%</div>
        </div>
        <div className="panel block">
          <div className="caption">Days / minimum</div>
          <div className="hero-sm num">{tradingDays}/{cfg.min_trading_days ?? 0}</div>
        </div>
      </div>
      <h2 className="section-title">Open positions</h2>
      <table style={{ marginTop: 12 }}>
        <thead><tr>
          <th>Instrument</th><th>Side</th><th className="num">Size</th>
          <th className="num">Entry</th><th className="num">Stop</th><th className="num">Target</th>
        </tr></thead>
        <tbody>
          {openPositions.map((p) => (
            <tr key={p.trade_id}>
              <td>{p.instrument}</td><td>{p.side}</td>
              <td className="num">{fmt(p.size, 2)}</td>
              <td className="num">{fmt(p.entry_price, 5)}</td>
              <td className="num">{p.stop_loss ? fmt(p.stop_loss, 5) : "—"}</td>
              <td className="num">{p.take_profit ? fmt(p.take_profit, 5) : "—"}</td>
            </tr>
          ))}
          {openPositions.length === 0 && (
            <tr><td colSpan={6} className="empty">No open positions</td></tr>
          )}
        </tbody>
      </table>
    </>
  );
}

function TradeLedger({ accounts }) {
  const [filters, setFilters] = useState({ agent: "", instrument: "" });
  const [trades, setTrades] = useState([]);
  const query = useMemo(() => {
    const p = new URLSearchParams();
    if (filters.agent) p.set("agent", filters.agent);
    if (filters.instrument) p.set("instrument", filters.instrument);
    return p.toString();
  }, [filters]);
  useEffect(() => { get(`/api/trades?${query}`).then(setTrades); }, [query]);
  useLiveFeed({ trades: () => get(`/api/trades?${query}`).then(setTrades) });
  const instruments = [...new Set(trades.map((t) => t.instrument))];
  return (
    <>
      <h1 className="screen-title">Trade ledger</h1>
      <div style={{ display: "flex", gap: 12, margin: "16px 0" }}>
        <select value={filters.agent} onChange={(e) => setFilters({ ...filters, agent: e.target.value })}>
          <option value="">All agents</option>
          {accounts.map((a) => <option key={a.agent_id} value={a.agent_id}>{a.agent_name}</option>)}
        </select>
        <select value={filters.instrument} onChange={(e) => setFilters({ ...filters, instrument: e.target.value })}>
          <option value="">All instruments</option>
          {instruments.map((i) => <option key={i} value={i}>{i}</option>)}
        </select>
      </div>
      <table>
        <thead><tr>
          <th>Agent</th><th>Instrument</th><th>Side</th><th className="num">Size</th>
          <th className="num">Entry</th><th className="num">Exit</th><th className="num">PnL</th><th>Closed</th>
        </tr></thead>
        <tbody>
          {trades.map((t) => (
            <tr key={t.trade_id}>
              <td>{t.agent_name}</td><td>{t.instrument}</td><td>{t.side}</td>
              <td className="num">{fmt(t.size, 2)}</td>
              <td className="num">{fmt(t.entry_price, 5)}</td>
              <td className="num">{fmt(t.exit_price, 5)}</td>
              <td className={`num ${t.realized_pnl >= 0 ? "pos" : "neg"}`}>{fmt(t.realized_pnl)}</td>
              <td className="num">{t.exit_time}</td>
            </tr>
          ))}
          {trades.length === 0 && <tr><td colSpan={8} className="empty">No closed trades</td></tr>}
        </tbody>
      </table>
    </>
  );
}

function Leaderboard() {
  const rows = usePoll("/api/leaderboard", 5000) || [];
  return (
    <>
      <h1 className="screen-title">Leaderboard</h1>
      <table style={{ marginTop: 16 }}>
        <thead><tr>
          <th className="num">Rank</th><th>Agent</th><th>Type</th><th>Phase</th>
          <th className="num">Sortino</th><th className="num">Max drawdown</th><th className="num">Days</th>
        </tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.account_id}>
              <td className="num hero-sm">{r.rank}</td>
              <td>{r.agent_name}</td><td>{r.framework_type}</td>
              <td><span className={`tag ${r.status === "failed" ? "red" : "blue"}`}>{r.status === "failed" ? "failed" : r.phase}</span></td>
              <td className="num">{r.sortino == null ? "—" : fmt(r.sortino, 2)}</td>
              <td className="num">{fmt(r.maxDrawdown * 100, 1)}%</td>
              <td className="num">{r.days}</td>
            </tr>
          ))}
          {rows.length === 0 && <tr><td colSpan={7} className="empty">No agents ranked yet</td></tr>}
        </tbody>
      </table>
    </>
  );
}

function AlertLog({ violations, phases }) {
  return (
    <>
      <h1 className="screen-title">Alerts and rule violations</h1>
      <table style={{ marginTop: 16 }}>
        <thead><tr><th>When</th><th>Agent</th><th>Rule</th><th>Detail</th></tr></thead>
        <tbody>
          {violations.map((v) => (
            <tr key={v.violation_id}>
              <td className="num">{v.triggered_at}</td>
              <td>{v.agent_name}</td>
              <td><span className="tag red">{v.rule_name}</span></td>
              <td>{describe(v, phases)}</td>
            </tr>
          ))}
          {violations.length === 0 && <tr><td colSpan={4} className="empty">No violations recorded</td></tr>}
        </tbody>
      </table>
    </>
  );
}

function Training() {
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => { get("/api/training-runs").then(setRuns); }, []);
  useEffect(() => {
    if (selected == null) return;
    get(`/api/training-runs/${selected}`).then(setDetail);
  }, [selected]);
  useEffect(() => {
    if (selected == null && runs.length) setSelected(runs[0].training_run_id);
  }, [runs, selected]);

  useLiveFeed({
    training_runs: (rows) => setRuns(rows),
    training_metrics: (rows) => {
      if (selected == null) return;
      const mine = rows.filter((r) => r.training_run_id === selected);
      if (!mine.length) return;
      setDetail((d) => d && { ...d, metrics: [...d.metrics, ...mine] });
    },
  });

  const run = detail?.run;
  const lastMetric = detail?.metrics?.[detail.metrics.length - 1];
  const progressPct = run && lastMetric
    ? Math.min(100, (lastMetric.timesteps / run.total_timesteps) * 100) : 0;

  return (
    <>
      <h1 className="screen-title">Training</h1>
      <div style={{ display: "flex", gap: 8, margin: "16px 0", flexWrap: "wrap" }}>
        {runs.map((r) => (
          <div key={r.training_run_id} tabIndex={0}
            className="tag" style={{ cursor: "pointer", borderLeftColor: r.training_run_id === selected ? "#3E7CFF" : undefined }}
            onClick={() => setSelected(r.training_run_id)}>
            #{r.training_run_id} {r.instrument} {r.algo}
          </div>
        ))}
        {runs.length === 0 && <div className="empty">No training runs yet — run scripts/train_rl.py</div>}
      </div>

      {run && (
        <>
          <div className="blocks">
            <div className="panel block">
              <div className="caption">Status</div>
              <div className="hero-sm">
                <span className={`tag ${run.status === "finished" ? "green" : "blue"}`}>{run.status}</span>
              </div>
            </div>
            <div className="panel block">
              <div className="caption">Timesteps</div>
              <div className="hero-sm num">
                {fmt(lastMetric?.timesteps ?? 0, 0)} / {fmt(run.total_timesteps, 0)}
              </div>
            </div>
            <div className="panel block">
              <div className="caption">Episode reward (mean)</div>
              <div className="hero-sm num">{fmt(lastMetric?.ep_rew_mean, 2)}</div>
            </div>
            <div className="panel block">
              <div className="caption">Throughput</div>
              <div className="hero-sm num">{fmt(lastMetric?.fps, 0)} steps/s</div>
            </div>
          </div>
          <div className="panel" style={{ padding: 16, marginBottom: 8 }}>
            <div className="caption">Progress {progressPct.toFixed(0)}%</div>
            <div className="gauge" style={{ marginTop: 4 }}>
              <div style={{ width: `${progressPct}%`, background: "#3E7CFF" }} />
            </div>
          </div>
          <div className="panel" style={{ padding: 16, marginBottom: 24 }}>
            <MetricLine points={detail.metrics} xKey="timesteps" yKey="ep_rew_mean"
              label="Mean episode reward vs. timesteps" />
          </div>
          <div className="panel" style={{ padding: 16 }}>
            <MetricLine points={detail.metrics} xKey="timesteps" yKey="ep_len_mean"
              label="Mean episode length vs. timesteps" />
          </div>
        </>
      )}
    </>
  );
}

// Plain, direct language rather than an error code, per the design doc.
function describe(v, phases) {
  const names = {
    daily_loss: "exceeded the daily loss limit",
    max_drawdown: "breached the maximum drawdown floor",
    consistency: "took too much profit from a single trading day",
  };
  const base = `${v.agent_name} ${names[v.rule_name] || `failed rule ${v.rule_name}`}`;
  return v.details ? `${base} (${v.details})` : base;
}
