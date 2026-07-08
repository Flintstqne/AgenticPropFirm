"""Compare synthetic paths against the real history they were calibrated
on: minute-return sigma, volatility clustering (autocorrelation of squared
returns), and excess kurtosis (fat tails). Run after calibration; a PASS
means the generator produces paths statistically in the real instrument's
neighborhood, not that they are indistinguishable.

Usage: venv/bin/python scripts/validate_synthetic.py EUR_USD [more...]
"""

import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import calibration, synthetic  # noqa: E402


def stats(rets):
    sd = statistics.pstdev(rets)
    mean = statistics.fmean(rets)
    r2 = [r * r for r in rets]
    m2 = statistics.fmean(r2)
    ac = (statistics.fmean([(a - m2) * (b - m2) for a, b in zip(r2, r2[1:])])
          / (statistics.pstdev(r2) ** 2 or 1))
    kurt = statistics.fmean([((r - mean) / sd) ** 4 for r in rets]) - 3 if sd else 0
    return sd, ac, kurt


def validate(instrument):
    real, _ = calibration.minute_returns(instrument)
    real = [r for r in real if abs(r) < 0.05]  # drop weekend gaps
    params = calibration.load_params(instrument)
    prices, _ = synthetic.generate_minutes(params, 1.0, len(real), seed=17)
    synth = [math.log(b / a) for a, b in zip(prices, prices[1:])]

    (rs, ra, rk), (ss, sa, sk) = stats(real), stats(synth)
    sigma_ratio = ss / rs if rs else float("inf")
    ok = 0.5 <= sigma_ratio <= 2.0 and sa > 0.01 and sk > 0.5
    print(f"{instrument:>8}  sigma real={rs:.2e} synth={ss:.2e} ratio={sigma_ratio:.2f}  "
          f"vol-cluster real={ra:.3f} synth={sa:.3f}  kurtosis real={rk:.1f} synth={sk:.1f}  "
          f"{'PASS' if ok else 'CHECK'}")
    return ok


if __name__ == "__main__":
    instruments = sys.argv[1:] or ["EUR_USD"]
    results = [validate(i) for i in instruments]
    sys.exit(0 if all(results) else 1)
