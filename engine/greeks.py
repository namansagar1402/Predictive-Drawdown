"""
engine/greeks.py
-----------------
Phase II (Major Project), Section 8.6: Dynamic Greek-Based Threshold Monitoring.

Computes Black-Scholes Delta/Gamma for a long-call position as the
underlying evolves and time-to-expiry decays, and tracks the *velocity*
(rate of change) of these Greeks as an early-warning signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * np.maximum(T, 0)) * norm.cdf(d2)


def bs_delta_call(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)


def bs_gamma(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    T = np.maximum(T, 1e-8)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def option_greek_series(
    S_path: np.ndarray,
    K: float,
    T_start: float,
    T_end: float,
    r: float,
    sigma: float,
) -> pd.DataFrame:
    """
    Compute the option value, Delta, and Gamma along a single price path,
    with time-to-expiry decaying linearly from T_start to T_end (both in
    years). T_start/T_end are the option's remaining life at the start and
    end of the *session* being simulated -- NOT necessarily 0, since a
    single intraday session is normally a small slice of a multi-week
    option's life. Passing T_end=0 reproduces "expiry-day" behaviour.
    """
    n = len(S_path)
    time_grid = np.linspace(T_start, max(T_end, 0.0), n)
    price = bs_call_price(S_path, K, time_grid, r, sigma)
    delta = bs_delta_call(S_path, K, time_grid, r, sigma)
    gamma = bs_gamma(S_path, K, time_grid, r, sigma)
    return pd.DataFrame({
        "S": S_path,
        "T_minus_t": time_grid,
        "option_value": price,
        "delta": delta,
        "gamma": gamma,
    })


def greek_velocity(series: np.ndarray, dt: float, smooth_window: int = 5) -> np.ndarray:
    """
    Finite-difference velocity (rate of change) of a Greek series, smoothed
    with a short rolling mean to reduce tick-noise before it feeds the
    composite trigger.
    """
    vel = np.gradient(series, dt)
    smoothed = pd.Series(vel).rolling(smooth_window, min_periods=1).mean().to_numpy()
    return smoothed


def normalized_greek_velocity(delta_series, gamma_series, dt, smooth_window=5) -> np.ndarray:
    """
    Combine Delta-velocity and Gamma-velocity into a single normalised
    ([0, 1]-ish) early-warning signal via a rolling z-score of their
    absolute combined magnitude, passed through a logistic squashing
    function so it is comparable in scale to the other two composite-score
    inputs (breach probability, normalised CVaR).
    """
    v_delta = greek_velocity(delta_series, dt, smooth_window)
    v_gamma = greek_velocity(gamma_series, dt, smooth_window)
    combined = np.abs(v_delta) + np.abs(v_gamma)

    roll = pd.Series(combined).rolling(30, min_periods=5)
    z = (combined - roll.mean().to_numpy()) / (roll.std().to_numpy() + 1e-9)
    z = np.nan_to_num(z, nan=0.0)
    return 1.0 / (1.0 + np.exp(-z))
