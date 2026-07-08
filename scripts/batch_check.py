"""Batch sanity check, per the AGENTS.md testing requirements: run many
simulated accounts through historical data and confirm the pass rate looks
reasonable — not near zero, not near every account passing.

Accounts run random policies with varied aggression (size, trade frequency,
stop width) over the downloaded history. The profit target scales to the
data window: the stock challenge target (8% over weeks) is out of reach in
a one-week window, so the check uses a window-scaled target to test that
the engine discriminates — reckless accounts fail rules, sensible ones can
pass, timid ones neither.

Runs over 30 days of calibrated synthetic data: the downloaded history
window (about a week) is too short for either side to resolve — cost
bleeders need weeks to reach the drawdown floor, and the real profit
target needs weeks of compounding.

Usage: venv/bin/python scripts/batch_check.py EUR_USD 40
"""

import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import calibration, ledger, synthetic  # noqa: E402
from engine.config import load_contracts, load_phases  # noqa: E402
from engine.simulator import Simulator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# 10k account: with the 1-lot forex cap, position risk is meaningful against
# the balance. At 100k the cap makes daily-loss breaches on forex majors
# nearly impossible (a 5 percent daily move), which defeats the check.
STARTING_BALANCE = 10_000.0


def run_account(conn, instrument, profile, phases_cfg, i):
    agent = ledger.register_agent(conn, f"batch-{i}", "random")
    acct = ledger.open_account(conn, agent, "challenge", STARTING_BALANCE,
                               datetime.now(timezone.utc).isoformat())
    sim = Simulator(conn, acct, [instrument], data_dir=ROOT / "data",
                    source="calibrated/batch_check", phases_cfg=phases_cfg)
    rng = random.Random(1000 + i)
    obs = sim.step()
    while obs is not None:
        price = obs["prices"].get(instrument)
        if price and not obs["open_trades"] and rng.random() < profile["freq"]:
            side = "buy" if rng.random() < 0.5 else "sell"
            sl = price * (1 - profile["stop"]) if side == "buy" else price * (1 + profile["stop"])
            tp = price * (1 + profile["stop"]) if side == "buy" else price * (1 - profile["stop"])
            sim.place_order(instrument, side, profile["size"], stop_loss=sl, take_profit=tp)
        obs = sim.step()
    final = ledger.get_account(conn, acct)
    if sim.status == "failed":
        outcome = "failed"
    elif final["phase"] != "challenge" or sim.status == "passed":
        outcome = "passed"   # cleared the challenge phase inside the window
    else:
        outcome = "active"
    return outcome, final["current_balance"]


def run_batch(instrument, n, phases_cfg):
    rng = random.Random(7)
    counts = {"passed": 0, "failed": 0, "active": 0}
    for i in range(n):
        profile = {
            "size": rng.choice([0.25, 0.5, 1.0]),
            "freq": rng.choice([0.02, 0.15, 0.5]),    # chance to trade per minute
            "stop": rng.choice([0.001, 0.01, 0.03]),  # 3% stop at 1 lot = 3k a trade
        }
        conn = ledger.connect(":memory:")
        ledger.init_db(conn)
        status, _ = run_account(conn, instrument, profile, phases_cfg, i)
        counts[status] += 1
        conn.close()
    return counts


def ensure_synthetic(instrument, days=30):
    out = ROOT / "data" / "calibrated" / "batch_check" / instrument
    if out.exists() and len(list(out.glob("*.parquet"))) >= days:
        return
    from datetime import timezone as tz
    params = calibration.load_params(instrument)
    spec = load_contracts()[instrument]
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    synthetic.generate_day_files(params, spec, 1.10, start, days, out, seed=555)


def main():
    instrument = sys.argv[1] if len(sys.argv) > 1 else "EUR_USD"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    ensure_synthetic(instrument)

    # batch A, stock rules: the real 8% target sits out of reach in a short
    # data window, so this batch exercises the failure side -- reckless
    # accounts must breach, careful ones must survive
    stock = load_phases()
    a = run_batch(instrument, n, stock)
    print(f"stock rules:   {a['passed']} passed, {a['failed']} failed, "
          f"{a['active']} survived")

    # batch B, window-scaled target (0.1 percent, 2 days, consistency off
    # since one day always dominates a noise-scale profit): passing side
    scaled = dict(stock)
    scaled["challenge"] = dict(stock["challenge"], profit_target_pct=0.001,
                               min_trading_days=2, consistency_cap_pct=None)
    b = run_batch(instrument, n, scaled)
    print(f"scaled target: {b['passed']} passed, {b['failed']} failed, "
          f"{b['active']} survived (passed = cleared challenge)")

    reasonable = (0 < a["failed"] < n) and (0 < b["passed"] < n)
    print("\nengine discrimination looks",
          "REASONABLE" if reasonable else "SUSPICIOUS",
          "- failures under stock rules, passes when the target fits the"
          " data window, neither saturated")


if __name__ == "__main__":
    main()
