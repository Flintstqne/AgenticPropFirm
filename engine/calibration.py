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
    alpha, beta = float(res.params["alpha[1]"]), float(res.params["beta[1]"])
    # real minute data often fits at alpha+beta ~ 1.0 (integrated GARCH),
    # which makes simulated variance a random walk that can explode; cap
    # persistence just under 1 so the process stays stationary
    if alpha + beta > 0.995:
        scale = 0.995 / (alpha + beta)
        alpha, beta = alpha * scale, beta * scale
    return {"omega": float(omega), "alpha": alpha, "beta": beta}


def fit_regimes(rets):
    """Two-regime Markov switching on returns. The higher-|mean| regime is
    the trending one; transition probabilities feed the generator chain.

    Real minute data carries flat stretches (repeated identical closes) and
    weekend gaps that make the EM step degenerate, so exact-zero returns
    drop out and outliers clip to 20 sigma before fitting. If the fit still
    fails, fall back to persistent default regimes rather than dying."""
    import numpy as np
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    x = np.array([r for r in rets if r != 0.0]) * RET_SCALE
    sd = x.std() or 1.0
    x = np.clip(x, -20 * sd, 20 * sd)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovRegression(x, k_regimes=2, switching_variance=True)
            res = model.fit(search_reps=5)
    except Exception:
        return {"p_stay_trend": 0.98, "p_stay_range": 0.98,
                "trend_mu": float(abs(x.mean()) / RET_SCALE)}
    p00 = float(res.params[0])      # P(0 -> 0)
    p11 = 1 - float(res.params[1])  # params[1] is P(1 -> 0)
    means = [float(res.params[2]) / RET_SCALE, float(res.params[3]) / RET_SCALE]
    trend_idx = 0 if abs(means[0]) >= abs(means[1]) else 1
    # a thin or noisy fit can land near 50/50 transitions, which whipsaws
    # the regime chain every other minute in the generator and shows up as
    # runaway kurtosis in the synthetic path; floor persistence at 0.6 so a
    # weak fit still trends toward "not clearly regime-switching" rather
    # than "switching constantly"
    stay = (max(0.6, p00), max(0.6, p11))
    return {
        "p_stay_trend": stay[trend_idx],
        "p_stay_range": stay[1 - trend_idx],
        "trend_mu": abs(means[trend_idx]),
    }


def fit_kappa(rets):
    """Mean-reversion pull per minute, estimated the same way the generator
    applies it: deviation of log price from a slow EMA anchor (0.005 step),
    kappa = -cov(next return, deviation) / var(deviation), clamped to a sane
    band. Weak or trending data collapses toward zero pull."""
    logp = 0.0
    anchor = 0.0
    devs, nexts = [], []
    for r in rets:
        devs.append(logp - anchor)
        nexts.append(r)
        logp += r
        anchor = 0.995 * anchor + 0.005 * logp
    n = len(devs)
    mean_d = sum(devs) / n
    mean_r = sum(nexts) / n
    var_d = sum((d - mean_d) ** 2 for d in devs) / n
    if var_d == 0:
        return 0.05
    cov = sum((d - mean_d) * (r - mean_r) for d, r in zip(devs, nexts)) / n
    return min(0.5, max(0.0, -cov / var_d))


def jump_stats(rets):
    """Returns beyond 4 unconditional sigmas count as jumps. Clipped at 20
    sigma first: session-boundary gap prints (futures maintenance breaks,
    weekend opens) would otherwise inflate the jump size the generator
    reproduces every simulated day."""
    n = len(rets)
    mean = sum(rets) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / n) or 1e-12
    rets = [max(-20 * sd, min(20 * sd, r)) for r in rets]
    jumps = [r for r in rets if abs(r - mean) > 4 * sd]
    return {
        "jump_prob": len(jumps) / n,
        "jump_scale": (sum(abs(j) for j in jumps) / len(jumps) / sd) if jumps else 6.0,
    }


def calibrate(instrument, data_dir=None):
    """Full calibration for one instrument, every generator parameter
    estimated from that instrument's own history."""
    rets, density = minute_returns(instrument, data_dir)
    clean = [r for r in rets if abs(r) < 0.05]  # weekend gaps out
    var = sum(r * r for r in clean) / len(clean)
    params = {"instrument": instrument, "n_minutes_fit": len(rets),
              "revert_kappa": fit_kappa(rets), "tick_density": density,
              "uncond_var": var}
    params.update(fit_garch(rets))
    # anchor the long-run GARCH level to the sample variance: after the
    # persistence cap, the fitted omega would otherwise set a steady state
    # well above (or below) the real instrument's volatility
    params["omega"] = var * (1 - params["alpha"] - params["beta"])
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
