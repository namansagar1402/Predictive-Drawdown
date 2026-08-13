"""
engine/simulation.py
--------------------
Phase I (Minor Project): stochastic price-process modelling.

Implements:
  - Vectorised Geometric Brownian Motion (GBM) path simulation.
  - Merton jump-diffusion path simulation (GBM + compound Poisson jumps).
  - Parameter calibration for both models from an observed log-return series,
    using the maximum-likelihood / threshold-based jump-filtering approach
    described in the project synopsis (Section 8.5).

All simulation is on a *tick* grid: `n_steps` ticks over a session of length
`T` (in years, e.g. one trading day = 1/252). This stands in for the
Level-2 tick data described in Section 8.2; where real historical tick data
is unavailable, this module IS the "calibrated synthetic order-flow
generator" fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ProcessParams:
    """Container for a (possibly jump-augmented) GBM parameter set."""
    mu: float
    sigma: float
    lam: float = 0.0          # jump intensity (jumps per year)
    jump_mean: float = 0.0    # mean log-jump size
    jump_std: float = 0.0     # std of log-jump size

    def as_dict(self) -> dict:
        return dict(mu=self.mu, sigma=self.sigma, lam=self.lam,
                    jump_mean=self.jump_mean, jump_std=self.jump_std)


def simulate_paths(
    S0: float,
    params: ProcessParams,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate `n_paths` price paths of length `n_steps + 1` (including S0)
    under GBM (+ optional Merton jump-diffusion if params.lam > 0).

    Returns
    -------
    S : ndarray, shape (n_paths, n_steps + 1)
        Simulated price paths.
    jump_flags : ndarray, shape (n_paths, n_steps), int
        Number of jumps realised in each tick interval (0 in almost all
        GBM-only ticks; used for diagnostics / labelling).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    Z = rng.standard_normal((n_paths, n_steps))
    diffusion = (params.mu - 0.5 * params.sigma ** 2) * dt + params.sigma * np.sqrt(dt) * Z

    jump_counts = np.zeros((n_paths, n_steps), dtype=int)
    jump_component = np.zeros((n_paths, n_steps))

    if params.lam > 0:
        jump_counts = rng.poisson(params.lam * dt, size=(n_paths, n_steps))
        max_jumps = int(jump_counts.max())
        for k in range(1, max_jumps + 1):
            mask = jump_counts >= k
            n_draw = int(mask.sum())
            if n_draw == 0:
                continue
            draws = rng.normal(params.jump_mean, max(params.jump_std, 1e-12), size=n_draw)
            jump_component[mask] += draws

    log_returns = diffusion + jump_component
    log_price = np.log(S0) + np.cumsum(log_returns, axis=1)
    S = np.exp(np.hstack([np.full((n_paths, 1), np.log(S0)), log_price]))
    return S, jump_counts


def calibrate_gbm(prices: np.ndarray, dt: float) -> ProcessParams:
    """
    Maximum-likelihood estimation of (mu, sigma) for a pure-diffusion GBM
    from an observed price series (Section 8.5, first paragraph).
    """
    log_ret = np.diff(np.log(prices))
    sigma_hat = float(np.std(log_ret, ddof=1)) / np.sqrt(dt)
    mean_hat = float(np.mean(log_ret))
    mu_hat = mean_hat / dt + 0.5 * sigma_hat ** 2
    return ProcessParams(mu=mu_hat, sigma=sigma_hat)


def calibrate_jump_diffusion(
    prices: np.ndarray,
    dt: float,
    k_threshold: float = 4.0,
    rolling_window: int = 20,
) -> tuple[ProcessParams, np.ndarray]:
    """
    Threshold-based jump-filtering calibration (Section 8.5, second
    paragraph): log-returns whose absolute value exceeds `k_threshold`
    times a *robust* local volatility estimate are classified as jumps and
    removed before re-estimating the continuous (mu, sigma) on the
    residual series; the removed observations calibrate (lam, jump_mean,
    jump_std) separately.

    A rolling median-absolute-deviation (MAD, scaled by 1.4826 to be
    consistent with the standard deviation under normality) is used
    instead of a plain rolling standard deviation for the local volatility
    estimate, because a plain rolling std is itself inflated by the jumps
    it is trying to detect -- a well-known weakness of naive threshold
    jump-detection methods. MAD is far more robust to the outliers it is
    screening for.

    Returns
    -------
    params : ProcessParams
    is_jump : ndarray[bool]
        Boolean mask (length = len(prices) - 1) flagging detected jump ticks,
        useful for overlaying detected jumps on a chart.
    """
    log_ret = np.diff(np.log(prices))
    s = pd.Series(log_ret)
    roll_median = s.rolling(rolling_window, min_periods=max(5, rolling_window // 4)).median()
    abs_dev = (s - roll_median).abs()
    mad = (
        abs_dev.rolling(rolling_window, min_periods=max(5, rolling_window // 4))
        .median()
        .bfill()
        .ffill()
        .to_numpy()
    ) * 1.4826
    mad = np.where(mad <= 0, np.std(log_ret) + 1e-12, mad)

    is_jump = np.abs(log_ret) > (k_threshold * mad)
    diffusion_rets = log_ret[~is_jump]
    jump_rets = log_ret[is_jump]

    if len(diffusion_rets) < 2:
        diffusion_rets = log_ret  # degenerate fallback

    sigma_hat = float(np.std(diffusion_rets, ddof=1)) / np.sqrt(dt)
    mu_hat = float(np.mean(diffusion_rets)) / dt + 0.5 * sigma_hat ** 2

    n_obs = len(log_ret)
    lam_hat = float(is_jump.sum()) / (n_obs * dt) if n_obs > 0 else 0.0
    jump_mean_hat = float(np.mean(jump_rets)) if len(jump_rets) > 0 else 0.0
    jump_std_hat = float(np.std(jump_rets, ddof=1)) if len(jump_rets) > 1 else 0.0

    params = ProcessParams(
        mu=mu_hat, sigma=sigma_hat, lam=lam_hat,
        jump_mean=jump_mean_hat, jump_std=jump_std_hat,
    )
    return params, is_jump


def atr_proxy(S: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Synthetic Average-True-Range proxy for a single price series (no OHLC
    bars are available from a tick simulation, so we use the rolling mean
    absolute tick-to-tick price change as a stand-in, scaled to be roughly
    comparable to a conventional ATR). Used for the ATR trailing-stop
    baseline (Objective 6.1).
    """
    diffs = np.abs(np.diff(S, prepend=S[0]))
    atr = pd.Series(diffs).rolling(window, min_periods=1).mean().to_numpy()
    return atr
