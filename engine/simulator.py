"""Simulation core shared by both agent interfaces.

Two clocks, per AGENTS.md: the agent acts once per simulated minute; every
tick inside that minute feeds the matching engine. Pending orders (limit,
stop, stop-limit) and stop loss / take profit levels check against every
tick. Daily loss and maximum drawdown check on every tick too, so a breach
fails the account mid-minute, at the moment it happens.

Daily reset at 17:00 US Eastern across all instruments. Phase progression
evaluates at that reset.

Sizes: forex in standard lots, futures in contracts. One trade row per fill;
a close closes one open trade. Equity = balance + unrealized PnL marked at
the closing side of the spread.
"""

from datetime import timedelta
from zoneinfo import ZoneInfo

import statistics
from collections import deque

from engine import ledger, matching, phases, replay, rules
from engine.config import load_contracts, load_phases, load_sessions

ET = ZoneInfo("America/New_York")


def _to_min(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


class Simulator:
    def __init__(self, conn, account_id, instruments, source="raw",
                 data_dir=None, start=None, end=None,
                 contracts=None, phases_cfg=None):
        self.conn = conn
        self.account_id = account_id
        self.contracts = contracts or load_contracts()
        self.phases_cfg = phases_cfg or load_phases()
        self.sessions_cfg = load_sessions()
        self.limits = self.phases_cfg["position_limits"]

        acct = ledger.get_account(conn, account_id)
        self.starting_balance = acct["starting_balance"]
        self.peak_equity = acct["current_equity"]
        self.sod_balance = acct["current_balance"]
        self.equity = acct["current_equity"]

        self.instruments = instruments
        self.streams = {
            i: replay.iter_minutes(i, source=source, data_dir=data_dir,
                                   start=start, end=end)
            for i in instruments
        }
        self.buffers = {}          # instrument -> (minute, ticks) lookahead
        self.last_mid = {}         # instrument -> latest mid price
        self.pending = []          # resting limit/stop/stop-limit orders
        self.current_day_et = None
        self.active_days = set()   # ET dates with at least one fill
        self.daily_pnls = []       # realized+unrealized change per completed day
        self._day_start_equity = self.equity
        self.status = "active"     # active | failed | passed
        self.done = False
        self._next_order_id = 1
        # execution realism state: recent returns for the volatility ratio,
        # last minute's tick count as the liquidity estimate
        self._returns = {i: deque(maxlen=300) for i in instruments}
        self.liquidity = {i: 10.0 for i in instruments}

    # ---- order API (called by agent wrappers) ----

    def place_order(self, instrument, side, size, order_type="market",
                    limit_price=None, stop_price=None,
                    stop_loss=None, take_profit=None):
        """Validate caps, then queue the order for the next minute's ticks.
        Returns (order_id, None) or (None, reason)."""
        spec = self.contracts[instrument]
        open_size = sum(t["size"] for t in ledger.get_open_trades(self.conn, self.account_id)
                        if t["instrument"] == instrument)
        if rules.position_size(spec, instrument, size, open_size, self.limits) == "fail":
            return None, "position size cap"
        mid = self.last_mid.get(instrument)
        if mid is not None:
            open_notional = self._open_notional()
            proposed = matching.notional_usd(spec, instrument, size, mid)
            if rules.total_notional(open_notional, proposed, self.equity, self.limits) == "fail":
                return None, "total notional cap"
            required = matching.margin_required(spec, instrument, size, mid)
            if self._margin_used() + required > self.equity:
                return None, "insufficient margin"
        order = {
            "id": self._next_order_id, "instrument": instrument, "side": side,
            "size": size, "type": order_type, "limit_price": limit_price,
            "stop_price": stop_price, "stop_loss": stop_loss,
            "take_profit": take_profit, "triggered": False,
        }
        self._next_order_id += 1
        self.pending.append(order)
        return order["id"], None

    def cancel_order(self, order_id):
        before = len(self.pending)
        self.pending = [o for o in self.pending if o["id"] != order_id]
        return len(self.pending) < before

    def close_position(self, trade_id):
        """Queue a close: fills at the first tick of the next minute."""
        self.pending.append({"id": self._next_order_id, "type": "close",
                             "trade_id": trade_id})
        self._next_order_id += 1
        return self._next_order_id - 1

    # ---- simulation loop ----

    def step(self):
        """Advance one simulated minute. Returns an observation dict, or None
        when data is exhausted or the account has failed."""
        if self.done:
            return None
        minute = self._pull_next_minute()
        if minute is None:
            self.done = True
            return None
        minute_start, ticks_by_inst = minute

        self._maybe_daily_reset(minute_start)

        # merge ticks across instruments in time order
        merged = sorted(
            ((ts, inst, bid, ask)
             for inst, ticks in ticks_by_inst.items()
             for ts, bid, ask in ticks),
            key=lambda t: t[0])

        for inst, ticks in ticks_by_inst.items():
            self.liquidity[inst] = float(len(ticks))

        for ts, inst, bid, ask in merged:
            mid = (bid + ask) / 2
            prev = self.last_mid.get(inst)
            if prev:
                self._returns[inst].append((mid - prev) / prev)
            self.last_mid[inst] = mid
            self._process_pending(ts, inst, bid, ask)
            self._process_exits(ts, inst, bid, ask)
            self._mark_equity()
            if self._check_hard_rules(ts):
                return None

        ledger.update_equity(self.conn, self.account_id, self.equity)
        ledger.snapshot_equity(self.conn, self.account_id, minute_start.isoformat())
        return self.observation(minute_start)

    def observation(self, now=None):
        acct = ledger.get_account(self.conn, self.account_id)
        phase_cfg = self.phases_cfg[acct["phase"]]
        return {
            "time": now.isoformat() if now else None,
            "prices": dict(self.last_mid),
            "balance": acct["current_balance"],
            "equity": self.equity,
            "phase": acct["phase"],
            "status": self.status,
            "open_trades": [dict(t) for t in ledger.get_open_trades(self.conn, self.account_id)],
            "daily_loss": rules.daily_loss(self.equity, self.sod_balance,
                                           phase_cfg["daily_loss_pct"]),
            "drawdown": rules.max_drawdown(self.equity, self.starting_balance,
                                           self.peak_equity,
                                           phase_cfg["max_drawdown_pct"],
                                           phase_cfg["drawdown_mode"]),
        }

    # ---- internals ----

    def _pull_next_minute(self):
        """Earliest minute across all instrument streams, with every
        instrument's ticks for that minute."""
        for inst in self.instruments:
            if inst not in self.buffers:
                nxt = next(self.streams[inst], None)
                if nxt is not None:
                    self.buffers[inst] = nxt
        if not self.buffers:
            return None
        minute = min(m for m, _ in self.buffers.values())
        out = {}
        for inst in list(self.buffers):
            m, ticks = self.buffers[inst]
            if m == minute:
                out[inst] = ticks
                del self.buffers[inst]
        return minute, out

    def _maybe_daily_reset(self, minute_start):
        # trading day rolls at 17:00 ET: day key = ET date of (ts - 17h)
        et = minute_start.astimezone(ET)
        day = et.date() if et.hour < 17 else et.date() + timedelta(days=1)
        if self.current_day_et is None:
            self.current_day_et = day
            return
        if day != self.current_day_et:
            # overnight swap for every position held through the rollover
            for t in ledger.get_open_trades(self.conn, self.account_id):
                spec = self.contracts[t["instrument"]]
                charge = matching.swap_usd(spec, t["side"], t["size"])
                if charge:
                    ledger.add_swap(self.conn, t["trade_id"], -charge)
            self.daily_pnls.append(self.equity - self._day_start_equity)
            self._evaluate_phase(minute_start)
            acct = ledger.get_account(self.conn, self.account_id)
            self.sod_balance = acct["current_equity"]
            self._day_start_equity = acct["current_equity"]
            self.current_day_et = day

    def _evaluate_phase(self, now):
        acct = ledger.get_account(self.conn, self.account_id)
        cfg = self.phases_cfg[acct["phase"]]
        verdict = phases.evaluate_phase(cfg, self.starting_balance, self.equity,
                                        len(self.active_days), self.daily_pnls)
        if verdict == "advance":
            nxt = phases.advance_account(self.conn, self.account_id,
                                         self.phases_cfg, now.isoformat())
            self.equity = self.starting_balance
            self.peak_equity = self.starting_balance
            self.sod_balance = self.starting_balance
            self.daily_pnls = []
            self.active_days = set()
            if nxt is None:
                self.status = "passed"
                self.done = True
        elif verdict == "fail":
            ledger.record_violation(self.conn, self.account_id, "consistency",
                                    now.isoformat())
            self._fail(now)

    def _process_pending(self, ts, inst, bid, ask):
        still = []
        for o in self.pending:
            if o["type"] == "close":
                trade = self.conn.execute(
                    "SELECT * FROM trades WHERE trade_id = ?", (o["trade_id"],)
                ).fetchone()
                if trade is None or trade["exit_time"] is not None:
                    continue
                if trade["instrument"] == inst:
                    self._close_trade(trade, ts, inst, bid, ask)
                else:
                    still.append(o)
                continue
            if o["instrument"] != inst:
                still.append(o)
                continue
            fill_price = self._try_fill(o, ts, inst, bid, ask)
            if fill_price is None:
                still.append(o)
                continue
            spec = self.contracts[inst]
            ledger.record_trade_open(
                self.conn, self.account_id, inst, o["side"], o["size"],
                fill_price, ts.isoformat(),
                stop_loss=o["stop_loss"], take_profit=o["take_profit"],
                commission=matching.commission_usd(spec, o["size"]))
            self.active_days.add(self.current_day_et)
        self.pending = still

    def _try_fill(self, o, ts, inst, bid, ask):
        """Return a fill price if this tick fills the order, else None.
        Market orders pay the realistic price: widened spread, slippage,
        partial fills, never better than the tick's own bid/ask."""
        side, typ = o["side"], o["type"]
        if typ == "market":
            spec = self.contracts[inst]
            modeled, _ = matching.fill_realistic(
                spec, side, (bid + ask) / 2, o["size"],
                self._active_sessions(inst, ts), self._vol_ratio(inst),
                self.liquidity.get(inst, 10.0))
            return max(ask, modeled) if side == "buy" else min(bid, modeled)
        if typ == "limit":
            if side == "buy" and ask <= o["limit_price"]:
                return ask
            if side == "sell" and bid >= o["limit_price"]:
                return bid
            return None
        if typ == "stop":
            if side == "buy" and ask >= o["stop_price"]:
                return ask
            if side == "sell" and bid <= o["stop_price"]:
                return bid
            return None
        if typ == "stop_limit":
            if not o["triggered"]:
                if (side == "buy" and ask >= o["stop_price"]) or \
                   (side == "sell" and bid <= o["stop_price"]):
                    o["triggered"] = True
            if o["triggered"]:
                if side == "buy" and ask <= o["limit_price"]:
                    return ask
                if side == "sell" and bid >= o["limit_price"]:
                    return bid
            return None
        raise ValueError(f"unknown order type {typ}")

    def _process_exits(self, ts, inst, bid, ask):
        for trade in ledger.get_open_trades(self.conn, self.account_id):
            if trade["instrument"] != inst:
                continue
            long = trade["side"] == "buy"
            close_px = bid if long else ask  # side the position closes at
            sl, tp = trade["stop_loss"], trade["take_profit"]
            hit = (sl is not None and (close_px <= sl if long else close_px >= sl)) or \
                  (tp is not None and (close_px >= tp if long else close_px <= tp))
            if hit:
                self._close_trade(trade, ts, inst, bid, ask)

    def _close_trade(self, trade, ts, inst, bid, ask):
        spec = self.contracts[inst]
        close_px = bid if trade["side"] == "buy" else ask
        pnl = matching.pnl_usd(spec, inst, trade["side"], trade["size"],
                               trade["entry_price"], close_px)
        ledger.record_trade_close(self.conn, trade["trade_id"], close_px,
                                  ts.isoformat(), pnl,
                                  close_commission=matching.commission_usd(spec, trade["size"]))

    def _active_sessions(self, inst, ts):
        """How many configured sessions are open at this tick's ET time."""
        cfg = self.sessions_cfg.get(inst)
        if not cfg:
            return 1
        et = ts.astimezone(ET)
        now = et.hour * 60 + et.minute
        count = 0
        for s in cfg["sessions"]:
            a = _to_min(s["start_et"])
            b = _to_min(s["end_et"])
            inside = a <= now < b if a < b else (now >= a or now < b)
            count += inside
        return count

    def _vol_ratio(self, inst):
        """Recent volatility over its longer baseline, 1.0 = normal."""
        rets = self._returns[inst]
        if len(rets) < 60:
            return 1.0
        recent = statistics.pstdev(list(rets)[-30:])
        base = statistics.pstdev(rets)
        return recent / base if base > 0 else 1.0

    def _margin_used(self):
        total = 0.0
        for t in ledger.get_open_trades(self.conn, self.account_id):
            spec = self.contracts[t["instrument"]]
            mid = self.last_mid.get(t["instrument"], t["entry_price"])
            total += matching.margin_required(spec, t["instrument"], t["size"], mid)
        return total

    def _open_notional(self):
        total = 0.0
        for t in ledger.get_open_trades(self.conn, self.account_id):
            mid = self.last_mid.get(t["instrument"], t["entry_price"])
            total += matching.notional_usd(self.contracts[t["instrument"]],
                                           t["instrument"], t["size"], mid)
        return total

    def _mark_equity(self):
        acct = ledger.get_account(self.conn, self.account_id)
        unreal = 0.0
        for t in ledger.get_open_trades(self.conn, self.account_id):
            mid = self.last_mid.get(t["instrument"], t["entry_price"])
            unreal += matching.unrealized_pnl_usd(
                self.contracts[t["instrument"]], t["instrument"],
                t["side"], t["size"], t["entry_price"], mid)
        self.equity = acct["current_balance"] + unreal
        self.peak_equity = max(self.peak_equity, self.equity)

    def _check_hard_rules(self, ts):
        acct = ledger.get_account(self.conn, self.account_id)
        cfg = self.phases_cfg[acct["phase"]]
        if rules.daily_loss(self.equity, self.sod_balance, cfg["daily_loss_pct"]) == "fail":
            ledger.record_violation(self.conn, self.account_id, "daily_loss",
                                    ts.isoformat(), f"equity {self.equity:.2f}")
            self._fail(ts)
            return True
        if rules.max_drawdown(self.equity, self.starting_balance, self.peak_equity,
                              cfg["max_drawdown_pct"], cfg["drawdown_mode"]) == "fail":
            ledger.record_violation(self.conn, self.account_id, "max_drawdown",
                                    ts.isoformat(), f"equity {self.equity:.2f}")
            self._fail(ts)
            return True
        return False

    def _fail(self, ts):
        # close everything at last known prices, mark failed, stop
        for t in ledger.get_open_trades(self.conn, self.account_id):
            inst = t["instrument"]
            mid = self.last_mid.get(inst, t["entry_price"])
            spec = self.contracts[inst]
            hs = matching.half_spread(spec)
            self._close_trade(t, ts, inst, mid - hs, mid + hs)
        ledger.update_equity(self.conn, self.account_id, self.equity)
        ledger.set_account_status(self.conn, self.account_id, "failed")
        self.status = "failed"
        self.done = True
