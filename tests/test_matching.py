import pytest

from engine import matching
from engine.config import load_contracts


@pytest.fixture(scope="module")
def contracts():
    return load_contracts()


class TestFillMarket:
    def test_es_buy_pays_half_spread(self, contracts):
        # ES: 1 tick spread = 0.25, half = 0.125
        assert matching.fill_market(contracts["ES"], "buy", 5000.00) == 5000.125

    def test_es_sell_receives_below_mid(self, contracts):
        assert matching.fill_market(contracts["ES"], "sell", 5000.00) == 4999.875

    def test_eurusd_buy(self, contracts):
        # 0.5 pip spread = 0.00005, half = 0.000025
        assert matching.fill_market(contracts["EUR_USD"], "buy", 1.10000) == pytest.approx(1.100025)

    def test_round_trip_costs_full_spread(self, contracts):
        spec = contracts["EUR_USD"]
        buy = matching.fill_market(spec, "buy", 1.10000)
        sell = matching.fill_market(spec, "sell", 1.10000)
        assert buy - sell == pytest.approx(spec["spread_pips_peak"] * spec["pip_size"])


class TestPipValue:
    def test_usd_counter_pair_fixed(self, contracts):
        assert matching.pip_value_usd(contracts["EUR_USD"], "EUR_USD", 1.10) == 10

    def test_usd_base_pair_recalculated(self, contracts):
        # USD/JPY at 150.00: 0.01 * 100000 / 150 = 6.67 USD per pip
        v = matching.pip_value_usd(contracts["USD_JPY"], "USD_JPY", 150.00)
        assert v == pytest.approx(6.6667, rel=1e-3)


class TestPnl:
    def test_es_long_win(self, contracts):
        # 2 contracts, +10 points, 50 USD/point -> +1000
        pnl = matching.pnl_usd(contracts["ES"], "ES", "buy", 2, 5000.0, 5010.0)
        assert pnl == 1000.0

    def test_es_short_win(self, contracts):
        pnl = matching.pnl_usd(contracts["ES"], "ES", "sell", 1, 5000.0, 4990.0)
        assert pnl == 500.0

    def test_cl_contract_size(self, contracts):
        # 1 contract, +1.00 USD/barrel, 1000 barrels -> +1000
        pnl = matching.pnl_usd(contracts["CL"], "CL", "buy", 1, 70.00, 71.00)
        assert pnl == pytest.approx(1000.0)

    def test_eurusd_long_100_pips(self, contracts):
        # 1 lot, +0.0100 -> 100 pips * 10 USD = +1000
        pnl = matching.pnl_usd(contracts["EUR_USD"], "EUR_USD", "buy", 1, 1.1000, 1.1100)
        assert pnl == pytest.approx(1000.0)

    def test_usdjpy_converts_to_usd(self, contracts):
        # 1 lot long, 150.00 -> 151.00: +100,000 JPY = 662.25 USD at exit
        pnl = matching.pnl_usd(contracts["USD_JPY"], "USD_JPY", "buy", 1, 150.00, 151.00)
        assert pnl == pytest.approx(100_000 / 151.00)

    def test_forex_short_loss(self, contracts):
        pnl = matching.pnl_usd(contracts["EUR_USD"], "EUR_USD", "sell", 0.5, 1.1000, 1.1050)
        assert pnl == pytest.approx(-250.0)


class TestUnrealized:
    def test_long_marks_at_bid(self, contracts):
        spec = contracts["ES"]
        # long from 5000.125 (ask), mid now 5000 -> closes at 4999.875 -> -0.25 pt * 50 = -12.50
        pnl = matching.unrealized_pnl_usd(spec, "ES", "buy", 1, 5000.125, 5000.0)
        assert pnl == pytest.approx(-12.50)
