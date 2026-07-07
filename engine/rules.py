"""Rules engine. Each rule is an independent function reading account state
and returning "pass", "warn", or "fail".

Threshold semantics, locked here so every rule agrees:
- fail when equity drops strictly below the limit line; sitting exactly on
  the line is a warn, not a fail.
- warn once 80 percent of the distance to the limit is consumed.
"""

WARN_FRACTION = 0.8


def _grade(loss, limit):
    """loss = how far equity has fallen, limit = the dollar distance allowed."""
    if loss > limit:
        return "fail"
    if loss >= limit * WARN_FRACTION:
        return "warn"
    return "pass"


def daily_loss(equity, start_of_day_balance, limit_pct):
    """Fail if equity drops below start-of-day balance minus limit_pct.
    Resets daily at 17:00 Eastern (caller resets start_of_day_balance)."""
    return _grade(start_of_day_balance - equity, start_of_day_balance * limit_pct)


def max_drawdown_static(equity, starting_balance, limit_pct):
    """Drawdown measured against the starting balance."""
    return _grade(starting_balance - equity, starting_balance * limit_pct)


def max_drawdown_trailing(equity, peak_equity, limit_pct):
    """Drawdown measured against the highest equity ever reached."""
    return _grade(peak_equity - equity, peak_equity * limit_pct)


def max_drawdown(equity, starting_balance, peak_equity, limit_pct, mode):
    """Static or trailing, chosen by config per phase, never hardcoded."""
    if mode == "trailing":
        return max_drawdown_trailing(equity, peak_equity, limit_pct)
    return max_drawdown_static(equity, starting_balance, limit_pct)


def consistency(daily_pnls, cap_pct):
    """Cap how much of total profit comes from a single trading day.
    Evaluated at phase completion, not per tick: early in a challenge the
    first profitable day is always 100 percent of profit. Only profitable
    days count toward the share; total profit <= 0 passes trivially."""
    if cap_pct is None:
        return "pass"
    total = sum(daily_pnls)
    if total <= 0:
        return "pass"
    best_day = max(daily_pnls)
    share = best_day / total
    if share > cap_pct:
        return "fail"
    if share >= cap_pct * WARN_FRACTION:
        return "warn"
    return "pass"


def min_trading_days(active_days, required_days):
    """A day counts as active only if the agent placed at least one trade.
    Evaluated at phase completion: fewer days means not yet eligible."""
    return "pass" if active_days >= required_days else "fail"


def position_size(spec, instrument, proposed_size, open_size, limits):
    """Per-instrument cap: 1 standard lot forex, 2 contracts futures.
    Checked at order time against existing open size in that instrument."""
    if spec["type"] == "forex":
        cap = limits["max_forex_lots_per_instrument"]
    else:
        cap = limits["max_futures_contracts_per_instrument"]
    return "pass" if open_size + proposed_size <= cap else "fail"


def total_notional(open_notional_usd, proposed_notional_usd, equity, limits):
    """Cap total open notional across every position at a multiple of equity."""
    cap = equity * limits["max_total_notional_multiple"]
    return "pass" if open_notional_usd + proposed_notional_usd <= cap else "fail"
