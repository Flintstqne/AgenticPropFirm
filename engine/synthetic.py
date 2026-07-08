"""Synthetic market data generator. Four components, per AGENTS.md:

1. Volatility clustering: GARCH(1,1) variance recursion using parameters
   calibrated by engine/calibration.py (fit with the arch package).
2. Regime switching: two-state Markov chain, trending vs range bound.
3. Trend persistence in the trending regime, mean-reverting pull in the
   range bound regime.
4. Jump events: small per-minute chance of a move outside normal volatility.

Each stochastic component runs on its own seeded generator stream, so a
change to one component's seed never shifts another component's output.

Output: minute-level mid path expanded to ticks (denser in active hours),
written as the same Parquet tick schema replay reads: ts, bid, ask,
bid_vol, ask_vol.
"""

import math
import random
from datetime import timedelta

import pyarrow as pa
import pyarrow.parquet as pq

TICK_SCHEMA = pa.schema([
    ("ts", pa.timestamp("us", tz="UTC")),
    ("bid", pa.float64()),
    ("ask", pa.float64()),
    ("bid_vol", pa.float64()),
    ("ask_vol", pa.float64()),
])

TRENDING, RANGING = 0, 1


def generate_minutes(params, start_price, n_minutes, seed):
    """Return (prices, regimes): minute-close mid prices and regime labels.

    params keys (all calibrated per instrument, see calibration.py):
      omega, alpha, beta        GARCH(1,1) on minute log returns
      p_stay_trend, p_stay_range  Markov transition probabilities
      trend_mu                  drift magnitude per minute in trend regime
      revert_kappa              pull strength toward the rolling mean, range regime
      jump_prob, jump_scale     per-minute jump probability and size (in sigmas)
    """
    rng_vol = random.Random(seed)
    rng_regime = random.Random(seed + 1)
    rng_jump = random.Random(seed + 2)
    rng_trend = random.Random(seed + 3)

    omega, alpha, beta = params["omega"], params["alpha"], params["beta"]
    # start at the calibrated sample variance when available; the closed-form
    # unconditional variance blows up as alpha+beta approaches 1
    var = params.get("uncond_var") or omega / max(1e-6, 1 - alpha - beta)
    regime = RANGING
    trend_dir = 1.0
    logp = math.log(start_price)
    anchor = logp  # rolling anchor for mean reversion
    prices, regimes = [], []
    last_ret = 0.0

    for _ in range(n_minutes):
        # regime chain
        if regime == TRENDING:
            if rng_regime.random() > params["p_stay_trend"]:
                regime = RANGING
        else:
            if rng_regime.random() > params["p_stay_range"]:
                regime = TRENDING
                trend_dir = 1.0 if rng_trend.random() < 0.5 else -1.0

        # GARCH variance recursion
        var = omega + alpha * last_ret ** 2 + beta * var
        sigma = math.sqrt(var)

        drift = params["trend_mu"] * trend_dir if regime == TRENDING \
            else -params["revert_kappa"] * (logp - anchor)
        shock = rng_vol.gauss(0, sigma)
        jump = 0.0
        if rng_jump.random() < params["jump_prob"]:
            jump = rng_jump.gauss(0, params["jump_scale"] * sigma)

        # only the diffusion shock feeds the GARCH recursion: a jump passing
        # through alpha would persist via beta and inflate variance for many
        # minutes, which real jump-diffusion behavior does not show
        last_ret = shock
        logp += drift + shock + jump
        anchor = 0.995 * anchor + 0.005 * logp  # slow-moving mean
        prices.append(math.exp(logp))
        regimes.append(regime)
    return prices, regimes


def expand_to_ticks(prices, start_ts, spec, density, seed):
    """Turn minute closes into ticks. density: list of 24 ints, mean ticks
    per minute by UTC hour, denser during active sessions. Intraminute path
    is a Brownian bridge between minute closes on its own seeded stream."""
    rng = random.Random(seed)
    half = _half_spread(spec)
    ticks = []
    prev = prices[0]
    for i, close in enumerate(prices):
        minute_ts = start_ts + timedelta(minutes=i)
        n = max(1, int(rng.gauss(density[minute_ts.hour], density[minute_ts.hour] * 0.3)))
        for k in range(n):
            frac = (k + 1) / n
            # bridge: interpolate plus shrinking noise, exact at the close
            mid = prev + (close - prev) * frac \
                + rng.gauss(0, abs(close - prev) * 0.3) * (1 - frac)
            ts = minute_ts + timedelta(seconds=(k * 60.0) / n + rng.uniform(0, 60.0 / n / 2))
            ticks.append((ts, mid - half, mid + half, 1.0, 1.0))
        prev = close
    return ticks


def _half_spread(spec):
    if spec["type"] == "future":
        return spec["spread_ticks"] * spec["tick_size"] / 2
    return spec["spread_pips_peak"] * spec["pip_size"] / 2


def write_ticks(ticks, out_path):
    cols = list(zip(*ticks))
    table = pa.table(
        {n: list(c) for n, c in zip(("ts", "bid", "ask", "bid_vol", "ask_vol"), cols)},
        schema=TICK_SCHEMA)
    pq.write_table(table, out_path)


def generate_day_files(params, spec, start_price, start_ts, days, out_dir, seed):
    """Generate `days` calendar days of ticks, one Parquet file per day,
    matching the layout under data/calibrated/<INSTRUMENT>/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    price = start_price
    for d in range(days):
        day_start = start_ts + timedelta(days=d)
        prices, _ = generate_minutes(params, price, 24 * 60, seed + d * 10)
        ticks = expand_to_ticks(prices, day_start, spec, params["tick_density"], seed + d * 10 + 5)
        write_ticks(ticks, out_dir / f"{day_start.date().isoformat()}.parquet")
        price = prices[-1]
