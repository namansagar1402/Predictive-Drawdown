"""
engine/risk.py
--------------
Phase II (Major Project), Section 8.7: Micro-Interval Conditional VaR (CVaR).

Rolling, historical-simulation CVaR (Expected Shortfall) computed on a
short window of recent returns, vectorised with a sliding-window view for
tick-level performance.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def rolling_var_cvar(returns: np.ndarray, window: int, alpha: float = 0.05):
    """
    Historical-simulation rolling VaR and CVaR (Expected Shortfall) at
    confidence level (1 - alpha).

    Returns
    -------
    var : ndarray, same length as `returns`, NaN before the first full window.
    cvar : ndarray, same length as `returns`, NaN before the first full window.
    """
    n = len(returns)
    var = np.full(n, np.nan)
    cvar = np.full(n, np.nan)
    if n < window:
        return var, cvar

    windows = sliding_window_view(returns, window)          # (n - window + 1, window)
    var_thresh = np.quantile(windows, alpha, axis=1)         # VaR per window (a return level)
    mask = windows <= var_thresh[:, None]
    counts = mask.sum(axis=1)
    counts_safe = np.where(counts == 0, 1, counts)
    sums = np.where(mask, windows, 0.0).sum(axis=1)
    cvar_vals = sums / counts_safe

    var[window - 1:] = var_thresh
    cvar[window - 1:] = cvar_vals
    return var, cvar


def normalize_cvar(cvar: np.ndarray, max_acceptable_loss: float) -> np.ndarray:
    """
    Map (typically negative) CVaR figures onto [0, 1], where 1 means the
    expected shortfall already equals or exceeds the maximum acceptable
    loss for the position. NaNs (warm-up period) map to 0 (no signal yet).
    """
    loss = np.nan_to_num(-cvar, nan=0.0)  # flip sign: loss is positive
    normalized = np.clip(loss / max(max_acceptable_loss, 1e-9), 0.0, 1.0)
    return normalized
