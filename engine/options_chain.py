"""
engine/options_chain.py
-------------------------
Model-derived options chain for NSE-style indices/stocks.

No free, reliable, ToS-safe live NSE options-chain feed exists for a
prototype like this (yfinance does not cover Indian option chains, and
scraping nseindia.com's internal API is fragile and against its terms of
use). Instead, this module builds a *synthetic* chain: real strikes and
expiries following NSE conventions (engine.market_calendar), priced with
Black-Scholes off a live (or user-supplied) spot price and a
volatility-smile approximation, clearly labelled as model output.

If you have access to a licensed data vendor or broker API (Kite Connect,
Breeze, etc.) that provides real option-chain LTP/OI/IV, swap
`generate_chain`'s pricing step for a real data pull -- the rest of the
app (Greeks, aggregation) is written against the same DataFrame shape
either way.
"""

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd

from . import market_calendar as mcal
from . import greeks as gk


def _smile_iv(moneyness: np.ndarray, atm_iv: float, skew: float = 0.06, curvature: float = 0.15) -> np.ndarray:
    """
    Simple, illustrative volatility-smile approximation: IV rises for deep
    ITM/OTM strikes, with a negative skew (puts richer than calls) typical
    of index options. `moneyness` = log(K/S). This is a stylised shape, not
    a market-calibrated smile -- flag this clearly to anyone reading chain
    IVs off this module.
    """
    return atm_iv + skew * (-moneyness) + curvature * moneyness ** 2


def generate_chain(
    spot: float,
    symbol: str = "NIFTY",
    expiry: dt.date | None = None,
    atm_iv: float = 0.13,
    r: float = 0.065,
    n_strikes_each_side: int = 10,
) -> pd.DataFrame:
    """
    Build a synthetic call/put chain around `spot` for the given (or next
    default) expiry, using NSE strike-interval conventions.
    """
    symbol_u = symbol.upper()
    interval = mcal.strike_interval_for(symbol_u, spot)
    atm = mcal.atm_strike(spot, interval)

    if expiry is None:
        expiry = (
            mcal.next_weekly_expiry(symbol_u) if symbol_u in mcal.WEEKLY_EXPIRY_INDICES
            else mcal.next_monthly_expiry(symbol_u)
        )

    T_years = max((expiry - dt.date.today()).days, 0) / 365.0
    T_years = max(T_years, 1e-4)

    strikes = np.array([atm + i * interval for i in range(-n_strikes_each_side, n_strikes_each_side + 1)])
    moneyness = np.log(strikes / spot)
    ivs = np.clip(_smile_iv(moneyness, atm_iv), 0.03, 2.0)

    call_price = gk.bs_call_price(spot, strikes, T_years, r, ivs)
    call_delta = gk.bs_delta_call(spot, strikes, T_years, r, ivs)
    call_gamma = gk.bs_gamma(spot, strikes, T_years, r, ivs)

    # Put via put-call parity: P = C - S + K e^{-rT}
    put_price = call_price - spot + strikes * np.exp(-r * T_years)
    put_delta = call_delta - 1.0
    put_gamma = call_gamma  # gamma identical for calls/puts at same strike/expiry

    df = pd.DataFrame({
        "strike": strikes,
        "iv": ivs,
        "call_ltp": np.round(call_price, 2),
        "call_delta": np.round(call_delta, 4),
        "call_gamma": np.round(call_gamma, 6),
        "put_ltp": np.round(np.maximum(put_price, 0.0), 2),
        "put_delta": np.round(put_delta, 4),
        "put_gamma": np.round(put_gamma, 6),
        "moneyness": np.round(moneyness, 4),
    })
    df.attrs["spot"] = spot
    df.attrs["expiry"] = expiry
    df.attrs["symbol"] = symbol_u
    df.attrs["T_years"] = T_years
    df.attrs["atm_strike"] = atm
    df.attrs["is_synthetic"] = True
    return df
