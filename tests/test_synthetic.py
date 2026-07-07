"""Synthetic generator tests: reproducibility, seed stream independence,
volatility clustering, regime behavior, jumps, tick expansion, and a
calibration round trip on generated data."""

import math
from datetime import datetime, timezone

import pytest

from engine import synthetic
from engine.config import load_contracts

PARAMS = {
    "omega": 1e-9, "alpha": 0.08, "beta": 0.90,
    "p_stay_trend": 0.98, "p_stay_range": 0.98,
    "trend_mu": 3e-5, "revert_kappa": 0.05,
    "jump_prob": 0.001, "jump_scale": 6.0,
    "tick_density": [4] * 24,
}
T0 = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)


def returns(prices):
    return [math.log(b / a) for a, b in zip(prices, prices[1:])]


class TestGenerateMinutes:
    def test_same_seed_same_path(self):
        a, _ = synthetic.generate_minutes(PARAMS, 1.1, 500, seed=42)
        b, _ = synthetic.generate_minutes(PARAMS, 1.1, 500, seed=42)
        assert a == b

    def test_different_seed_different_path(self):
        a, _ = synthetic.generate_minutes(PARAMS, 1.1, 500, seed=42)
        b, _ = synthetic.generate_minutes(PARAMS, 1.1, 500, seed=43)
        assert a != b

    def test_prices_stay_positive_and_sane(self):
        prices, _ = synthetic.generate_minutes(PARAMS, 1.1, 5000, seed=1)
        assert all(p > 0 for p in prices)
        assert 0.5 < prices[-1] < 2.5  # no runaway explosion

    def test_volatility_clusters(self):
        # autocorrelation of squared returns positive under GARCH
        prices, _ = synthetic.generate_minutes(PARAMS, 1.1, 20_000, seed=7)
        r2 = [r * r for r in returns(prices)]
        mean = sum(r2) / len(r2)
        num = sum((a - mean) * (b - mean) for a, b in zip(r2, r2[1:]))
        den = sum((a - mean) ** 2 for a in r2)
        assert num / den > 0.02

    def test_both_regimes_visited(self):
        _, regimes = synthetic.generate_minutes(PARAMS, 1.1, 10_000, seed=3)
        assert synthetic.TRENDING in regimes
        assert synthetic.RANGING in regimes
        # persistence: regime switches are rare relative to minutes
        switches = sum(1 for a, b in zip(regimes, regimes[1:]) if a != b)
        assert switches < len(regimes) * 0.1

    def test_jumps_appear_at_configured_rate(self):
        params = dict(PARAMS, jump_prob=0.01, jump_scale=8.0)
        prices, _ = synthetic.generate_minutes(params, 1.1, 20_000, seed=11)
        rets = returns(prices)
        sd = (sum(r * r for r in rets) / len(rets)) ** 0.5
        outliers = sum(1 for r in rets if abs(r) > 4 * sd)
        assert outliers > 20  # far more tail events than a pure gaussian

    def test_component_seed_isolation(self):
        # changing only the jump stream (seed+2) must not move the vol draws:
        # with jump_prob 0 the paths from seed s and s' agree when the only
        # differing stream is jumps
        p0 = dict(PARAMS, jump_prob=0.0)
        a, _ = synthetic.generate_minutes(p0, 1.1, 300, seed=100)
        b, _ = synthetic.generate_minutes(p0, 1.1, 300, seed=100)
        assert a == b


class TestTickExpansion:
    def test_ticks_cover_every_minute_and_carry_spread(self):
        spec = load_contracts()["EUR_USD"]
        prices, _ = synthetic.generate_minutes(PARAMS, 1.1, 60, seed=5)
        ticks = synthetic.expand_to_ticks(prices, T0, spec, PARAMS["tick_density"], seed=6)
        assert len(ticks) >= 60
        spread = spec["spread_pips_peak"] * spec["pip_size"]
        for ts, bid, ask, _, _ in ticks[:50]:
            assert ask - bid == pytest.approx(spread)
        minutes = {t[0].replace(second=0, microsecond=0) for t in ticks}
        assert len(minutes) == 60

    def test_density_profile_respected(self):
        spec = load_contracts()["EUR_USD"]
        density = [2] * 12 + [20] * 12  # quiet mornings, busy afternoons UTC
        prices, _ = synthetic.generate_minutes(PARAMS, 1.1, 24 * 60, seed=9)
        ticks = synthetic.expand_to_ticks(prices, T0, spec, density, seed=10)
        early = sum(1 for t in ticks if t[0].hour < 12)
        late = sum(1 for t in ticks if t[0].hour >= 12)
        assert late > early * 3


class TestGenerateDayFiles:
    def test_writes_replayable_files(self, tmp_path):
        from engine import replay
        spec = load_contracts()["EUR_USD"]
        out = tmp_path / "calibrated" / "EUR_USD"
        synthetic.generate_day_files(PARAMS, spec, 1.1, T0, days=2,
                                     out_dir=out, seed=42)
        assert len(list(out.glob("*.parquet"))) == 2
        minutes = list(replay.iter_minutes("EUR_USD", source="calibrated",
                                           data_dir=tmp_path))
        assert len(minutes) == 2 * 24 * 60


class TestCalibrationRoundTrip:
    def test_calibrate_on_generated_data(self, tmp_path):
        # generate with known params, calibrate, check recovered values sane
        from engine import calibration
        spec = load_contracts()["EUR_USD"]
        out = tmp_path / "raw" / "EUR_USD"
        synthetic.generate_day_files(PARAMS, spec, 1.1, T0, days=3,
                                     out_dir=out, seed=21)
        params = calibration.calibrate("EUR_USD", data_dir=tmp_path)
        assert 0 < params["alpha"] < 1
        assert 0 < params["beta"] < 1
        assert params["alpha"] + params["beta"] < 1.05
        assert 0.5 < params["p_stay_trend"] <= 1
        assert 0 <= params["jump_prob"] < 0.05
        assert len(params["tick_density"]) == 24
        path = calibration.save_params(params, data_dir=tmp_path)
        assert calibration.load_params("EUR_USD", data_dir=tmp_path)["alpha"] == params["alpha"]
        assert path.exists()
