"""Run agents through the evaluation program on the fixed seed set.

Twenty fixed seeds, twenty distinct market condition draws, per AGENTS.md.
Every agent runs the same twenty seeds for a fair leaderboard comparison.
The seed of each run is stored in the runs table.

Baseline policies included so the harness runs end to end without an RL
library. Plug a trained policy in by passing any callable with the same
signature as the baselines: policy(obs, toolbox) -> None.

Usage:
  venv/bin/python scripts/run_training.py momentum EUR_USD 5      # 5 synthetic days per seed
  venv/bin/python scripts/run_training.py random EUR_USD 5
"""

import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm_tools import LLMToolbox               # noqa: E402
from engine import calibration, ledger, synthetic     # noqa: E402
from engine.config import load_contracts              # noqa: E402
from engine.simulator import Simulator                # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "db" / "propfirm.sqlite"

# Locked seed set. Do not reorder or replace values once training starts:
# every agent must face the same twenty market condition draws.
SEEDS = [11, 23, 37, 41, 53, 67, 79, 83, 97, 101,
         113, 127, 139, 149, 163, 173, 191, 199, 211, 223]

# Default generator params when no calibration file exists yet for the
# instrument (calibrate against downloaded history for real evaluations).
DEFAULT_PARAMS = {
    "omega": 1e-9, "alpha": 0.08, "beta": 0.90,
    "p_stay_trend": 0.98, "p_stay_range": 0.98,
    "trend_mu": 3e-5, "revert_kappa": 0.05,
    "jump_prob": 0.001, "jump_scale": 6.0,
    "tick_density": [4] * 24,
}


def momentum_policy(state):
    """Buy when price sits above its short rolling mean, flip when below.
    A baseline to exercise the pipeline, not a strategy."""
    window = state.setdefault("window", [])

    def act(obs, tools):
        inst = state["instrument"]
        price = obs["prices"].get(inst)
        if price is None:
            return
        window.append(price)
        if len(window) < 20:
            return
        ma = sum(window[-20:]) / 20
        have = obs["open_trades"]
        if price > ma and not have:
            tools.place_order(inst, "buy", 0.5, stop_loss=price * 0.997,
                              take_profit=price * 1.006)
        elif price < ma and have:
            for t in have:
                tools.close_position(t["trade_id"])
    return act


def random_policy(state):
    rng = random.Random(state["seed"] + 999)  # own stream, never the market's

    def act(obs, tools):
        inst = state["instrument"]
        r = rng.random()
        if r < 0.02 and not obs["open_trades"]:
            side = "buy" if rng.random() < 0.5 else "sell"
            tools.place_order(inst, side, 0.5)
        elif r > 0.98:
            for t in obs["open_trades"]:
                tools.close_position(t["trade_id"])
    return act


POLICIES = {"momentum": momentum_policy, "random": random_policy}


def run_one(conn, policy_name, instrument, days, seed, data_dir):
    spec = load_contracts()[instrument]
    try:
        params = calibration.load_params(instrument, data_dir=data_dir)
    except FileNotFoundError:
        params = DEFAULT_PARAMS
    out = Path(data_dir) / "calibrated" / f"seed_{seed}" / instrument
    if not out.exists():
        start_ts = datetime(2026, 1, 5, tzinfo=timezone.utc)
        synthetic.generate_day_files(params, spec, _start_price(spec), start_ts,
                                     days, out, seed)

    agent_id = ledger.register_agent(conn, f"{policy_name}-s{seed}", policy_name)
    now = datetime.now(timezone.utc).isoformat()
    acct = ledger.open_account(conn, agent_id, "challenge", 100_000.0, now)
    cur = conn.execute(
        """INSERT INTO runs (account_id, seed, data_source, instrument, started_at)
           VALUES (?, ?, 'synthetic', ?, ?)""", (acct, seed, instrument, now))
    conn.commit()
    run_id = cur.lastrowid

    sim = Simulator(conn, acct, [instrument],
                    source=f"calibrated/seed_{seed}", data_dir=data_dir)
    tools = LLMToolbox(sim)
    state = {"instrument": instrument, "seed": seed}
    act = POLICIES[policy_name](state)
    obs = sim.step()
    while obs is not None:
        act(obs, tools)
        obs = sim.step()

    final = ledger.get_account(conn, acct)
    conn.execute("UPDATE runs SET finished_at = ?, final_status = ? WHERE run_id = ?",
                 (datetime.now(timezone.utc).isoformat(), sim.status, run_id))
    conn.commit()
    print(f"seed {seed:>3}  {policy_name:<10} balance {final['current_balance']:>12.2f}  {sim.status}")
    return sim.status


def _start_price(spec):
    return 1.10 if spec["type"] == "forex" else 5000.0


def main():
    policy_name, instrument, days = sys.argv[1], sys.argv[2], int(sys.argv[3])
    conn = ledger.connect(str(DB))
    conn.executescript((ROOT / "db" / "migrations" / "001_add_runs_table.sql").read_text())
    data_dir = ROOT / "data"
    results = [run_one(conn, policy_name, instrument, days, s, data_dir) for s in SEEDS]
    print(f"\n{results.count('passed')} passed, {results.count('failed')} failed, "
          f"{results.count('active')} still active at data end")


if __name__ == "__main__":
    main()
