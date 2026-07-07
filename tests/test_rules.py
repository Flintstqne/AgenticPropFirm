"""One unit test per rule in isolation, including a case exactly at the
threshold. Threshold semantics: strictly below the line fails, exactly on
the line warns."""

from engine import rules


class TestDailyLoss:
    # 100k start of day, 5% limit -> line at 95,000
    def test_pass_when_flat(self):
        assert rules.daily_loss(100_000, 100_000, 0.05) == "pass"

    def test_pass_small_loss(self):
        assert rules.daily_loss(97_000, 100_000, 0.05) == "pass"

    def test_warn_at_80pct_of_limit(self):
        assert rules.daily_loss(96_000, 100_000, 0.05) == "warn"

    def test_warn_exactly_at_threshold(self):
        assert rules.daily_loss(95_000, 100_000, 0.05) == "warn"

    def test_fail_below_threshold(self):
        assert rules.daily_loss(94_999.99, 100_000, 0.05) == "fail"

    def test_profit_is_pass(self):
        assert rules.daily_loss(105_000, 100_000, 0.05) == "pass"


class TestMaxDrawdownStatic:
    # 100k starting balance, 10% limit -> line at 90,000
    def test_pass(self):
        assert rules.max_drawdown_static(95_000, 100_000, 0.10) == "pass"

    def test_warn_at_80pct(self):
        assert rules.max_drawdown_static(92_000, 100_000, 0.10) == "warn"

    def test_exactly_at_threshold_warns(self):
        assert rules.max_drawdown_static(90_000, 100_000, 0.10) == "warn"

    def test_fail_below(self):
        assert rules.max_drawdown_static(89_999, 100_000, 0.10) == "fail"


class TestMaxDrawdownTrailing:
    # peak 110k, 10% limit -> line at 99,000
    def test_pass_at_peak(self):
        assert rules.max_drawdown_trailing(110_000, 110_000, 0.10) == "pass"

    def test_fail_below_trailing_line(self):
        assert rules.max_drawdown_trailing(98_999, 110_000, 0.10) == "fail"

    def test_exactly_at_line_warns(self):
        assert rules.max_drawdown_trailing(99_000, 110_000, 0.10) == "warn"

    def test_trailing_stricter_than_static(self):
        # equity 99,500 passes static (line 90k) but warns trailing (line 99k)
        assert rules.max_drawdown_static(99_500, 100_000, 0.10) == "pass"
        assert rules.max_drawdown_trailing(99_500, 110_000, 0.10) == "warn"


class TestMaxDrawdownDispatch:
    def test_static_mode(self):
        assert rules.max_drawdown(95_000, 100_000, 110_000, 0.10, "static") == "pass"

    def test_trailing_mode(self):
        assert rules.max_drawdown(95_000, 100_000, 110_000, 0.10, "trailing") == "fail"
