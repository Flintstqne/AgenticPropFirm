"""Unit tests for the phase two rule catalog: consistency, minimum trading
days, position size caps, total notional cap."""

from engine import rules
from engine.config import load_contracts, load_phases

CONTRACTS = load_contracts()
LIMITS = load_phases()["position_limits"]


class TestConsistency:
    def test_even_days_pass(self):
        assert rules.consistency([1000, 1000, 1000, 1000, 1000], 0.30) == "pass"

    def test_one_day_dominates_fails(self):
        assert rules.consistency([5000, 100, 100, 100], 0.30) == "fail"

    def test_exactly_at_cap_passes(self):
        # best day 300 of 1000 total = exactly 30%
        assert rules.consistency([300, 300, 250, 150], 0.30) != "fail"

    def test_just_over_cap_fails(self):
        assert rules.consistency([301, 250, 250, 199], 0.30) == "fail"

    def test_no_profit_passes_trivially(self):
        assert rules.consistency([-500, 200, -100], 0.30) == "pass"

    def test_none_cap_disabled(self):
        assert rules.consistency([10_000, 1], None) == "pass"

    def test_losing_days_dont_inflate_share(self):
        # total 1000, best day 900: 90% share even though other days offset
        assert rules.consistency([900, 600, -500], 0.30) == "fail"


class TestMinTradingDays:
    def test_enough_days(self):
        assert rules.min_trading_days(5, 5) == "pass"

    def test_too_few(self):
        assert rules.min_trading_days(4, 5) == "fail"

    def test_zero_required(self):
        assert rules.min_trading_days(0, 0) == "pass"


class TestPositionSize:
    def test_forex_one_lot_cap(self):
        spec = CONTRACTS["EUR_USD"]
        assert rules.position_size(spec, "EUR_USD", 1.0, 0.0, LIMITS) == "pass"
        assert rules.position_size(spec, "EUR_USD", 0.5, 0.6, LIMITS) == "fail"
        assert rules.position_size(spec, "EUR_USD", 0.5, 0.5, LIMITS) == "pass"

    def test_futures_two_contract_cap(self):
        spec = CONTRACTS["ES"]
        assert rules.position_size(spec, "ES", 2, 0, LIMITS) == "pass"
        assert rules.position_size(spec, "ES", 1, 2, LIMITS) == "fail"


class TestTotalNotional:
    def test_under_cap(self):
        # 100k equity, 5x cap = 500k
        assert rules.total_notional(200_000, 200_000, 100_000, LIMITS) == "pass"

    def test_exactly_at_cap_passes(self):
        assert rules.total_notional(400_000, 100_000, 100_000, LIMITS) == "pass"

    def test_over_cap_fails(self):
        assert rules.total_notional(400_000, 100_001, 100_000, LIMITS) == "fail"
