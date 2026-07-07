"""Matching engine. Phase one: market orders, fixed spread, instant fill.
Phase five layers execution realism on top: session and volatility aware
spread widening, size-relative slippage, partial fills, commission, swap,
and margin.

Prices passed in are mid prices. Fills happen at mid +/- half spread.
Sizes: forex size is in standard lots, futures size is in contracts.
All PnL is returned in USD.
"""


def half_spread(spec):
    if spec["type"] == "future":
        return spec["spread_ticks"] * spec["tick_size"] / 2
    return spec["spread_pips_peak"] * spec["pip_size"] / 2


def fill_market(spec, side, mid):
    """Return execution price for a market order at the given mid price."""
    hs = half_spread(spec)
    return mid + hs if side == "buy" else mid - hs


def pip_value_usd(spec, instrument, price):
    """Pip value in USD per standard lot. Fixed for USD-counter pairs,
    recalculated against the live rate for USD-base pairs."""
    if spec["pip_value"] is not None:
        return spec["pip_value"]
    return spec["pip_size"] * spec["lot_size"] / price


def pnl_usd(spec, instrument, side, size, entry_price, exit_price):
    """Realized PnL in USD for a closed position."""
    direction = 1 if side == "buy" else -1
    move = (exit_price - entry_price) * direction
    if spec["type"] == "future":
        return move * spec["point_value"] * size
    pnl_counter = move * spec["lot_size"] * size
    if instrument.endswith("_USD"):
        return pnl_counter
    # USD-base pair: counter-currency PnL converts to USD at the exit rate.
    return pnl_counter / exit_price


def notional_usd(spec, instrument, size, price):
    """Open notional exposure in USD, used by the total-exposure cap."""
    if spec["type"] == "future":
        return spec["point_value"] * price * size
    if instrument.startswith("USD_"):
        return spec["lot_size"] * size  # base currency is USD
    return spec["lot_size"] * size * price


def unrealized_pnl_usd(spec, instrument, side, size, entry_price, current_mid):
    """Mark-to-market PnL at the price the position would close at now:
    a long closes at bid, a short closes at ask."""
    close_side = "sell" if side == "buy" else "buy"
    close_price = fill_market(spec, close_side, current_mid)
    return pnl_usd(spec, instrument, side, size, entry_price, close_price)


# ---- phase five: execution realism ----

def spread_multiplier(spec, active_sessions, vol_ratio):
    """How much the base spread widens right now.

    active_sessions: number of sessions open at this moment (from the
    session calendar). Forex spreads tighten during the London/New York
    overlap (2 or more sessions) and widen in quiet single-session hours.
    Futures widen in the low-volume overnight stretch (session count 0 by
    convention for the maintenance-adjacent hours).

    vol_ratio: current volatility over its recent baseline, 1.0 = normal.
    Spreads widen roughly with volatility, capped so a single wild tick
    cannot produce an absurd spread.
    """
    if spec["type"] == "forex":
        session_mult = 1.0 if active_sessions >= 2 else (1.5 if active_sessions == 1 else 3.0)
    else:
        session_mult = 1.0 if active_sessions >= 1 else 2.0
    vol_mult = min(3.0, max(1.0, vol_ratio))
    return session_mult * vol_mult


def slippage(spec, size, liquidity):
    """Extra adverse price movement, in price units, for a market order of
    this size against current liquidity. liquidity: recent ticks per minute,
    a stand-in for depth calibrated from historical tick volume. Small
    orders in deep markets slip nothing; size relative to liquidity scales
    the slip in units of the base half spread."""
    if liquidity <= 0:
        liquidity = 1.0
    return half_spread(spec) * (size / liquidity)


def split_fills(size, liquidity):
    """Partial fills for large orders: an order bigger than what one moment
    absorbs (a tenth of per-minute liquidity) splits into child fills, each
    priced slightly worse than the last by the caller."""
    chunk = max(liquidity / 10.0, 1e-9)
    if size <= chunk:
        return [size]
    fills = []
    left = size
    while left > 1e-12:
        f = min(chunk, left)
        fills.append(f)
        left -= f
    return fills


def fill_realistic(spec, side, mid, size, active_sessions, vol_ratio, liquidity):
    """Volume-weighted fill price with widened spread, slippage, and partial
    fills. Returns (price, n_fills).
    # ponytail: one VWAP trade row per order, not one row per child fill;
    # split rows if per-fill audit detail ever matters
    """
    hs = half_spread(spec) * spread_multiplier(spec, active_sessions, vol_ratio)
    direction = 1 if side == "buy" else -1
    fills = split_fills(size, liquidity)
    total, weight = 0.0, 0.0
    for i, f in enumerate(fills):
        # each child fill slips progressively worse
        px = mid + direction * (hs + slippage(spec, f * (i + 1), liquidity))
        total += px * f
        weight += f
    return total / weight, len(fills)


def commission_usd(spec, size):
    """Commission for one side (open or close) of a trade."""
    return spec.get("commission_per_side", 0.0) * size


def swap_usd(spec, side, size, nights=1):
    """Overnight financing for a position held past the daily rollover.
    Futures carry no swap; financing sits in the futures price itself."""
    if spec["type"] == "future":
        return 0.0
    rate = spec.get("swap_long", 0.0) if side == "buy" else spec.get("swap_short", 0.0)
    return rate * size * nights


def margin_required(spec, instrument, size, price):
    """Margin consumed by a position. Futures: exchange margin per contract.
    Forex: notional / leverage, recalculated against the current rate."""
    if spec["type"] == "future":
        return spec["margin"] * size
    return notional_usd(spec, instrument, size, price) / spec["leverage"]
