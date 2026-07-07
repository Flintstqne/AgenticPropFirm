"""Phase progression: challenge -> verification -> funded.

evaluate_phase runs at the daily reset (17:00 Eastern) and decides whether an
account advances, keeps trading, or has hit a phase-completion gate failure.
Hard rule breaches (daily loss, drawdown) fail the account the moment they
happen, on tick, outside this module.

On advance, the account balance and equity reset to starting_balance,
modeling a real prop firm issuing a fresh account for each phase.
"""

from engine import ledger, rules


def evaluate_phase(phase_cfg, starting_balance, current_equity,
                   active_days, daily_pnls):
    """Return "advance", "continue", or "fail".

    "fail" here only comes from the consistency gate: profit target reached
    but one day carries too much of the profit. Real firms treat this as a
    soft failure; version one fails the account and records the violation.
    """
    target_pct = phase_cfg["profit_target_pct"]
    if target_pct is None:
        return "continue"  # funded phase: no target, runs until a rule failure
    if current_equity < starting_balance * (1 + target_pct):
        return "continue"
    if rules.min_trading_days(active_days, phase_cfg["min_trading_days"]) != "pass":
        return "continue"  # target hit but not enough active days yet
    if rules.consistency(daily_pnls, phase_cfg["consistency_cap_pct"]) == "fail":
        return "fail"
    return "advance"


def advance_account(conn, account_id, phases_cfg, now):
    """Move the account to its next phase and reset balance and equity."""
    acct = ledger.get_account(conn, account_id)
    next_phase = phases_cfg[acct["phase"]]["next"]
    if next_phase is None:
        return None
    ledger.set_account_phase(conn, account_id, next_phase, now)
    conn.execute(
        """UPDATE accounts SET current_balance = starting_balance,
           current_equity = starting_balance WHERE account_id = ?""",
        (account_id,),
    )
    conn.commit()
    return next_phase
