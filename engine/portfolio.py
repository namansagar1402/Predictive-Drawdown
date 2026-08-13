"""
engine/portfolio.py
---------------------
Multi-asset Indian equity & options portfolio tracking.

A `Position` is user-entered (this module does not place or receive
orders -- it is a read-only tracker/analyzer). Equity/index positions are
marked to market directly from a live underlying price; option positions
are marked to market via Black-Scholes using the live underlying price
plus a user-supplied (or default) implied volatility, since no live
option-premium feed is wired in (see engine.options_chain docstring).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import greeks as gk

ASSET_TYPES = ["Equity", "Index", "Call Option", "Put Option"]


@dataclass
class Position:
    label: str                      # display name, e.g. "NIFTY 25000 CE"
    asset_type: str                 # one of ASSET_TYPES
    yf_symbol: str                  # yfinance ticker for the underlying, e.g. "RELIANCE.NS", "^NSEI"
    quantity: float                 # positive = long, negative = short (units, not lots)
    entry_price: float              # per-unit entry price (premium for options)
    strike: float | None = None     # option positions only
    expiry: dt.date | None = None   # option positions only
    iv: float | None = None         # option positions only (annualised, e.g. 0.14)


def mark_to_market(
    positions: list[Position],
    live_prices: dict[str, float],
    r: float = 0.065,
    default_iv: float = 0.15,
) -> pd.DataFrame:
    """
    Compute a per-position mark-to-market snapshot: current value, P&L,
    and (for options) Delta/Gamma, given a dict of live underlying prices
    keyed by yf_symbol.
    """
    rows = []
    for p in positions:
        S = live_prices.get(p.yf_symbol, np.nan)
        if p.asset_type in ("Equity", "Index"):
            value_per_unit = S
            delta = 1.0
            gamma = 0.0
        else:
            if p.strike is None or p.expiry is None:
                raise ValueError(f"Position '{p.label}' is an option but missing strike/expiry.")
            T = max((p.expiry - dt.date.today()).days, 0) / 365.0
            T = max(T, 1e-4)
            iv = p.iv if p.iv else default_iv
            if p.asset_type == "Call Option":
                value_per_unit = float(gk.bs_call_price(S, p.strike, T, r, iv))
                delta = float(gk.bs_delta_call(S, p.strike, T, r, iv))
            else:  # Put Option, via put-call parity
                call_val = float(gk.bs_call_price(S, p.strike, T, r, iv))
                value_per_unit = call_val - S + p.strike * np.exp(-r * T)
                value_per_unit = max(value_per_unit, 0.0)
                delta = float(gk.bs_delta_call(S, p.strike, T, r, iv)) - 1.0
            gamma = float(gk.bs_gamma(S, p.strike, T, r, iv))

        position_value = value_per_unit * p.quantity if not np.isnan(value_per_unit) else np.nan
        entry_value = p.entry_price * p.quantity
        pnl = position_value - entry_value if not np.isnan(position_value) else np.nan

        rows.append({
            "Label": p.label,
            "Type": p.asset_type,
            "Underlying": p.yf_symbol,
            "Underlying Price": S,
            "Qty": p.quantity,
            "Entry Price": p.entry_price,
            "Current Price": value_per_unit,
            "Position Value": position_value,
            "P&L": pnl,
            "Delta (per unit)": delta,
            "Position Delta": delta * p.quantity,
            "Position Gamma": gamma * p.quantity,
        })
    return pd.DataFrame(rows)


def portfolio_summary(mtm_df: pd.DataFrame) -> dict:
    """Aggregate portfolio-level statistics from a mark_to_market DataFrame."""
    return {
        "total_value": float(np.nansum(mtm_df["Position Value"])),
        "total_pnl": float(np.nansum(mtm_df["P&L"])),
        "net_delta": float(np.nansum(mtm_df["Position Delta"])),
        "net_gamma": float(np.nansum(mtm_df["Position Gamma"])),
        "n_positions": len(mtm_df),
    }
