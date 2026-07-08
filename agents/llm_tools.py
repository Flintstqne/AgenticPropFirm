"""LLM agent tools: plain Python functions an LLM calls as tools. Same
simulation core as the RL wrapper. Every function returns a plain dict a
calling framework can serialize to JSON or render as text.

Time advances only when the agent calls advance_time, one simulated minute
per call, matching the one-observation-one-action cadence."""

from engine import ledger


class LLMToolbox:
    def __init__(self, sim):
        self.sim = sim

    def get_market_state(self):
        """Latest mid price per instrument and current simulated status."""
        return {"prices": dict(self.sim.last_mid), "status": self.sim.status}

    def get_account_state(self):
        """Balance, equity, phase, rule status, and open positions."""
        return self.sim.observation()

    def place_order(self, instrument, side, size, order_type="market",
                    limit_price=None, stop_price=None,
                    stop_loss=None, take_profit=None, trailing_stop=None):
        """Place an order. side: buy|sell. order_type: market|limit|stop|stop_limit.
        trailing_stop is a price distance that follows the best price reached
        since entry, tightening only, never loosening. Fills against ticks in
        the next simulated minute."""
        order_id, err = self.sim.place_order(
            instrument, side, size, order_type,
            limit_price=limit_price, stop_price=stop_price,
            stop_loss=stop_loss, take_profit=take_profit,
            trailing_stop=trailing_stop)
        if err:
            return {"ok": False, "error": err}
        return {"ok": True, "order_id": order_id}

    def cancel_order(self, order_id):
        return {"ok": self.sim.cancel_order(order_id)}

    def close_position(self, trade_id):
        """Close one open trade at the next tick."""
        self.sim.close_position(trade_id)
        return {"ok": True}

    def advance_time(self, minutes=1):
        """Advance the simulation. Returns the latest observation, or the
        terminal state if the account failed or data ran out."""
        obs = None
        for _ in range(minutes):
            nxt = self.sim.step()
            if nxt is None:
                break
            obs = nxt
        return obs or self.sim.observation()

    def get_open_positions(self):
        rows = ledger.get_open_trades(self.sim.conn, self.sim.account_id)
        return {"positions": [dict(r) for r in rows]}
