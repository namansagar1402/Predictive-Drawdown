"""
engine/market_calendar.py
--------------------------
NSE trading-session and options-expiry conventions.

IMPORTANT: these conventions are set by SEBI/NSE circulars and have
changed before (most recently 1 September 2025, when NIFTY 50's weekly
expiry moved from Thursday to Tuesday, and Bank Nifty / Fin Nifty /
Midcap Nifty were cut back to monthly-only). Rules encoded here reflect
that regime as of mid-2026. Always cross-check against NSE's current
circular before relying on this for anything beyond a prototype/demo --
do not hardcode this into a live trading decision without verifying.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

IST_OPEN = dt.time(9, 15)
IST_CLOSE = dt.time(15, 30)

# Indices with weekly expiry vs. monthly-only, per the Sept-2025 SEBI rules.
WEEKLY_EXPIRY_INDICES = {"NIFTY"}          # NSE's one weekly-expiry benchmark
MONTHLY_ONLY_INDICES = {"BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
NSE_WEEKLY_EXPIRY_WEEKDAY = 1   # Monday=0 ... Tuesday=1 (NSE, since 1 Sep 2025)

# Approximate strike intervals (verify against the current NSE contract
# specification page for the exact instrument -- these can and do change).
STRIKE_INTERVALS = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
}
DEFAULT_STOCK_STRIKE_INTERVAL_RULE = [
    (500, 5), (1000, 10), (2500, 20), (5000, 50), (float("inf"), 100),
]


def is_market_open(now: dt.datetime | None = None) -> bool:
    """Very approximate: IST trading window on a weekday. Does NOT account
    for NSE trading holidays (Diwali, Republic Day, etc.) -- plug in the
    official NSE holiday list for anything beyond a demo."""
    now = now or dt.datetime.now()
    if now.weekday() >= 5:  # Sat/Sun
        return False
    return IST_OPEN <= now.time() <= IST_CLOSE


def next_weekday(d: dt.date, weekday: int) -> dt.date:
    """Next date on/after `d` falling on the given weekday (Mon=0..Sun=6)."""
    days_ahead = (weekday - d.weekday()) % 7
    return d + dt.timedelta(days=days_ahead)


def last_weekday_of_month(year: int, month: int, weekday: int) -> dt.date:
    if month == 12:
        first_of_next = dt.date(year + 1, 1, 1)
    else:
        first_of_next = dt.date(year, month + 1, 1)
    last_day = first_of_next - dt.timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - dt.timedelta(days=offset)


def next_weekly_expiry(symbol: str = "NIFTY", from_date: dt.date | None = None) -> dt.date:
    """Next weekly expiry date for a weekly-expiry index (NIFTY only, per
    current rules). Raises for indices that no longer have weekly expiry."""
    if symbol.upper() not in WEEKLY_EXPIRY_INDICES:
        raise ValueError(
            f"{symbol} does not have a weekly expiry under current NSE rules "
            f"(only {WEEKLY_EXPIRY_INDICES} do). Use next_monthly_expiry instead."
        )
    from_date = from_date or dt.date.today()
    candidate = next_weekday(from_date, NSE_WEEKLY_EXPIRY_WEEKDAY)
    if candidate == from_date and dt.datetime.now().time() > IST_CLOSE:
        candidate = next_weekday(from_date + dt.timedelta(days=1), NSE_WEEKLY_EXPIRY_WEEKDAY)
    return candidate


def next_monthly_expiry(symbol: str = "NIFTY", from_date: dt.date | None = None) -> dt.date:
    """Next monthly expiry: last Tuesday of the current month, rolling to
    next month if that date has already passed."""
    from_date = from_date or dt.date.today()
    candidate = last_weekday_of_month(from_date.year, from_date.month, NSE_WEEKLY_EXPIRY_WEEKDAY)
    if candidate < from_date:
        y, m = (from_date.year, from_date.month + 1) if from_date.month < 12 else (from_date.year + 1, 1)
        candidate = last_weekday_of_month(y, m, NSE_WEEKLY_EXPIRY_WEEKDAY)
    return candidate


def list_upcoming_expiries(symbol: str, n: int = 6, from_date: dt.date | None = None) -> list[dt.date]:
    """Convenience list of the next `n` relevant expiries for a symbol."""
    from_date = from_date or dt.date.today()
    symbol = symbol.upper()
    if symbol in WEEKLY_EXPIRY_INDICES:
        expiries = []
        cursor = from_date
        for _ in range(n):
            e = next_weekly_expiry(symbol, cursor)
            expiries.append(e)
            cursor = e + dt.timedelta(days=1)
        return expiries
    else:
        expiries = []
        cursor = from_date
        for _ in range(n):
            e = next_monthly_expiry(symbol if symbol in MONTHLY_ONLY_INDICES else "NIFTY", cursor)
            expiries.append(e)
            cursor = e + dt.timedelta(days=1)
        return expiries


def strike_interval_for(symbol: str, spot: float | None = None) -> int:
    symbol = symbol.upper()
    if symbol in STRIKE_INTERVALS:
        return STRIKE_INTERVALS[symbol]
    spot = spot or 1000
    for threshold, step in DEFAULT_STOCK_STRIKE_INTERVAL_RULE:
        if spot < threshold:
            return step
    return 100


def atm_strike(spot: float, interval: int) -> float:
    return round(spot / interval) * interval
