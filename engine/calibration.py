"""Calibrate synthetic generator parameters against downloaded history,
per instrument. A fit for one instrument does not transfer to another.

Pipeline: minute mid returns from tick Parquet (DuckDB resample) ->
GARCH(1,1) via arch -> two-regime Markov switching via statsmodels ->
jump stats from the return tails -> tick density by hour of day.
Writes data/calibrated/<INSTRUMENT>_params.json.
"""

import json
import math
import warnings
from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RET_SCALE = 10_000  # arch fits better on returns scaled up from tiny decimals


def minute_returns(instrument, data_dir=None):
    """Minute-close log returns and per-(hour, minute) tick counts."""
    base = Path(data_dir or DATA_DIR) / "raw" / instrument
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    rows = con.execute(f"""
        SELECT date_trunc('minute', ts) AS minute,
               last((bid + ask) / 2 ORDER BY ts) AS close,
               count(*) AS n_ticks
        FROM read_parquet('{base}/*.parquet')
        GROUP BY 1 ORDER BY 1""").fetchall()
    con.close()
    rets = []
    for (_, prev, _), (_, cur, _) in zip(rows, rows[1:]):
        rets.append(math.log(cur / prev))
    density = [[] for _ in range(24)]
    for minute, _, n in rows:
        density[minute.hour].append(n)
    density = [max(1, round(sum(d) / len(d))) if d else 1 for d in density]
    return rets, density


def fit_garch(rets):
    from arch import arch_model
    scaled = [r * RET_SCALE for r in rets]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(scaled, vol="GARCH", p=1, q=1, mean="Zero").fit(disp="off")
    omega = res.params["omega"] / RET_SCALE ** 2
    return {
        "omega": float(omega),
        "alpha": float(res.params["alpha[1]"]),
        "beta": float(res.params["beta[1]"]),
    }


def fit_regimes(rets):
    """Two-regime Markov switching on returns. The higher-|mean| regime is
    the trending one; transition probabilities feed the generator chain."""
    import numpy as np
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MarkovRegression(np.array(rets) * RET_SCALE, k_regimes=2,
                                 switching_variance=True)
        res = model.fit(search_reps=5)
    p00 = float(res.params[0])      # P(0 -> 0)
    p11 = 1 - float(res.params[1])  # params[1] is P(1 -> 0)
    means = [float(res.params[2]) / RET_SCALE, float(res.params[3]) / RET_SCALE]
    trend_idx = 0 if abs(means[0]) >= abs(means[1]) else 1
    stay = (p00, p11)
    return {
        "p_stay_trend": stay[trend_idx],
        "p_stay_range": stay[1 - trend_idx],
        "trend_mu": abs(means[trend_idx]),
    }


def jump_stats(rets):
    """Returns beyond 4 unconditional sigmas count as jumps."""
    n = len(rets)
    mean = sum(rets) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / n) or 1e-12
    jumps = [r for r in rets if abs(r - mean) > 4 * sd]
    return {
        "jump_prob": len(jumps) / n,
        "jump_scale": (sum(abs(j) for j in jumps) / len(jumps) / sd) if jumps else 6.0,
    }


def calibrate(instrument, data_dir=None, revert_kappa=0.05):
    """Full calibration for one instrument. revert_kappa has no direct
    estimator here; 0.05 per minute is the starting default, tune per
    instrument if synthetic paths range too loosely or too tightly."""
    rets, density = minute_returns(instrument, data_dir)
    params = {"instrument": instrument, "n_minutes_fit": len(rets),
              "revert_kappa": revert_kappa, "tick_density": density}
    params.update(fit_garch(rets))
    params.update(fit_regimes(rets))
    params.update(jump_stats(rets))
    return params


def save_params(params, data_dir=None):
    out = Path(data_dir or DATA_DIR) / "calibrated"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{params['instrument']}_params.json"
    path.write_text(json.dumps(params, indent=2))
    return path


def load_params(instrument, data_dir=None):
    path = Path(data_dir or DATA_DIR) / "calibrated" / f"{instrument}_params.json"
    return json.loads(path.read_text())
