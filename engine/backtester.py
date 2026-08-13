"""
engine/backtester.py
---------------------
Ties Phase I (simulation.py) and Phase II (greeks.py, risk.py, trigger.py)
together into a single per-session pipeline, then runs an event-driven,
queue-aware execution simulation comparing three exit strategies:

  1. static   - exit when option-value drawdown from its running peak
                exceeds a fixed percentage threshold.
  2. atr      - exit when the underlying crosses a volatility-adjusted
                (ATR-based) trailing stop.
  3. predictive - exit when the composite trigger score (Section 8.8)
                crosses its ROC-calibrated threshold.

Execution/latency model
------------------------
No real Level-2 order-book data is available (Section 8.2), so a
stylised, explicitly-documented synthetic queue model is used: static and
ATR triggers fire at common, "crowded" round-number-style levels that many
participants act on simultaneously, so they are assigned a *longer*
simulated queue delay; the predictive trigger fires idiosyncratically
ahead of the crowd, so it is assigned a *shorter* delay. During the delay,
the (already-simulated) forward price path continues to move -- so if a
jump/flash-crash is in progress, waiting in the queue costs real,
measurable slippage. This is a deliberate, clearly-labelled simplification
(see Section 7, "Model Limitations"), not a claim about real exchange
microstructure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from . import simulation, greeks, risk, trigger as trig


@dataclass
class SessionConfig:
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.06
    sigma: float = 0.25
    lam: float = 8.0            # jumps / year
    jump_mean: float = -0.01
    jump_std: float = 0.02
    mu: float = 0.08
    T_days: int = 1             # session length in trading days (path simulation horizon)
    option_days_to_expiry: float = 30.0   # option's remaining life at session start
    n_steps: int = 375          # ticks per session (~1-min bars in a 375-min NSE day)
    static_threshold: float = 0.20    # 20% drawdown in option value
    atr_multiplier: float = 2.5
    cvar_window: int = 30
    cvar_alpha: float = 0.05
    max_acceptable_loss: float = 0.30   # for CVaR normalisation
    breach_horizon_ticks: int = 10
    fpr_tolerance: float = 0.05
    seed: int | None = 42


@dataclass
class SessionResult:
    df: pd.DataFrame
    trigger_ticks: dict
    fills: dict
    composite: "trig.CompositeTrigger"
    jump_flags: np.ndarray


TRADING_DAYS_PER_YEAR = 252


def _latency_ticks(strategy: str, local_vol_percentile: float, rng: np.random.Generator) -> int:
    """Synthetic queue-latency model (see module docstring)."""
    if strategy == "predictive":
        base = 1.0
        crowd = 1.0
    else:
        base = 3.0
        crowd = 1.5 + local_vol_percentile  # crowded, volatility-dependent queue
    return int(rng.poisson(base * crowd) + 1)


def run_single_session(
    cfg: SessionConfig,
    S_external: np.ndarray | None = None,
    dt_override: float | None = None,
) -> SessionResult:
    """
    Run the full Phase I + Phase II pipeline for one session.

    If `S_external` is provided (e.g. a slice of real NIFTY 50 prices from
    `engine.data_loader`), that price series is used directly instead of
    simulating a new one, so the exact same Greek/CVaR/trigger/execution
    machinery can be evaluated on real historical data. `dt_override` lets
    the caller specify the real tick spacing (in years) for that series;
    if omitted, a daily (1/252) spacing is assumed.
    """
    rng = np.random.default_rng(cfg.seed)

    if S_external is not None:
        S = np.asarray(S_external, dtype=float)
        n_steps_actual = len(S) - 1
        dt = dt_override if dt_override else (1 / TRADING_DAYS_PER_YEAR)
        T = dt * n_steps_actual
        # Jumps are not "known" for real data -- detect them post-hoc with
        # the same robust threshold filter used for calibration, purely for
        # display / tail-risk inflation purposes (Section 8.4/8.5).
        try:
            calib_params, is_jump_mask = simulation.calibrate_jump_diffusion(S, dt)
            jump_flags = is_jump_mask.astype(int)
            lam_for_pjump = calib_params.lam
        except Exception:
            jump_flags = np.zeros(n_steps_actual, dtype=int)
            lam_for_pjump = cfg.lam
    else:
        T = cfg.T_days / TRADING_DAYS_PER_YEAR
        n_steps_actual = cfg.n_steps
        dt = T / n_steps_actual
        params = simulation.ProcessParams(cfg.mu, cfg.sigma, cfg.lam, cfg.jump_mean, cfg.jump_std)
        S, jump_counts = simulation.simulate_paths(cfg.S0, params, T, n_steps_actual, 1, seed=cfg.seed)
        S = S[0]
        jump_flags = jump_counts[0]
        lam_for_pjump = params.lam

    g = greeks.option_greek_series(
        S, cfg.K,
        T_start=cfg.option_days_to_expiry / TRADING_DAYS_PER_YEAR,
        T_end=max(cfg.option_days_to_expiry / TRADING_DAYS_PER_YEAR - T, 0.0),
        r=cfg.r, sigma=cfg.sigma,
    )
    running_peak = g["option_value"].cummax()
    drawdown = (running_peak - g["option_value"]) / running_peak.replace(0, np.nan)
    drawdown = drawdown.fillna(0.0).to_numpy()

    atr = simulation.atr_proxy(S, window=14)
    running_peak_S = pd.Series(S).cummax().to_numpy()
    atr_stop_level = running_peak_S - cfg.atr_multiplier * atr

    v_greek = greeks.normalized_greek_velocity(g["delta"].to_numpy(), g["gamma"].to_numpy(), dt)

    option_returns = np.diff(g["option_value"].to_numpy(), prepend=g["option_value"].iloc[0])
    _, cvar = risk.rolling_var_cvar(option_returns, cfg.cvar_window, cfg.cvar_alpha)
    cvar_norm = risk.normalize_cvar(cvar, cfg.max_acceptable_loss * g["option_value"].iloc[0])

    local_vol = pd.Series(option_returns).rolling(20, min_periods=5).std().bfill().to_numpy()
    distance_to_breach = np.maximum(cfg.static_threshold - drawdown, 0.0) * running_peak.to_numpy()
    p_jump = trig.breach_probability_proxy(
        distance_to_breach, np.maximum(local_vol, 1e-9), cfg.breach_horizon_ticks, dt, lam_for_pjump
    )

    labels = trig.make_breach_labels(drawdown, cfg.static_threshold, cfg.breach_horizon_ticks)
    composite = trig.CompositeTrigger()
    X = np.column_stack([p_jump, v_greek, cvar_norm])
    if labels.sum() >= 2 and (len(labels) - labels.sum()) >= 2:
        composite.fit(X, labels)
        composite.calibrate_threshold(composite.score(p_jump, v_greek, cvar_norm), labels, cfg.fpr_tolerance)
    else:
        composite.threshold = 0.5
    score = composite.score(p_jump, v_greek, cvar_norm)

    df = g.copy()
    df["drawdown"] = drawdown
    df["atr"] = atr
    df["atr_stop_level"] = atr_stop_level
    df["v_greek"] = v_greek
    df["cvar"] = cvar
    df["cvar_norm"] = cvar_norm
    df["p_jump"] = p_jump
    df["composite_score"] = score
    df["jump_flag"] = np.concatenate([[0], jump_flags])

    n = len(df)
    static_hits = np.where(drawdown >= cfg.static_threshold)[0]
    atr_hits = np.where(S <= atr_stop_level)[0]
    atr_hits = atr_hits[atr_hits > 5]  # ignore warm-up noise
    predictive_hits = np.where(score >= composite.threshold)[0]

    trigger_ticks = {
        "static": int(static_hits[0]) if len(static_hits) else None,
        "atr": int(atr_hits[0]) if len(atr_hits) else None,
        "predictive": int(predictive_hits[0]) if len(predictive_hits) else None,
    }

    vol_pctile = pd.Series(local_vol).rank(pct=True).fillna(0.5).to_numpy()

    fills = {}
    for name, tick in trigger_ticks.items():
        if tick is None:
            fills[name] = None
            continue
        lat = _latency_ticks(name, float(vol_pctile[tick]), rng)
        fill_tick = min(tick + lat, n - 1)
        trigger_value = df["option_value"].iloc[tick]
        fill_value = df["option_value"].iloc[fill_tick]
        slippage = trigger_value - fill_value  # positive = cost (sold for less than intended)
        fills[name] = {
            "trigger_tick": tick,
            "fill_tick": fill_tick,
            "latency_ticks": lat,
            "trigger_value": float(trigger_value),
            "fill_value": float(fill_value),
            "slippage": float(slippage),
            "pnl": float(fill_value - df["option_value"].iloc[0]),
        }

    return SessionResult(df=df, trigger_ticks=trigger_ticks, fills=fills, composite=composite, jump_flags=jump_flags)


def run_backtest(cfg: SessionConfig, n_paths: int, base_seed: int = 0) -> pd.DataFrame:
    """
    Monte Carlo backtest: repeat `run_single_session` across `n_paths`
    independent *simulated* sessions and collect the per-strategy fill
    outcomes into a tidy DataFrame for aggregation (Section 12).
    """
    rows = []
    for i in range(n_paths):
        path_cfg = SessionConfig(**{**cfg.__dict__, "seed": base_seed + i})
        result = run_single_session(path_cfg)
        for strategy, fill in result.fills.items():
            if fill is None:
                rows.append({
                    "path_id": i, "strategy": strategy, "triggered": False,
                    "latency_ticks": np.nan, "slippage": 0.0, "pnl": np.nan,
                })
            else:
                rows.append({
                    "path_id": i, "strategy": strategy, "triggered": True,
                    "latency_ticks": fill["latency_ticks"], "slippage": fill["slippage"],
                    "pnl": fill["pnl"],
                })
    return pd.DataFrame(rows)


def run_historical_backtest(
    cfg: SessionConfig,
    price_series: np.ndarray,
    session_len: int,
    n_sessions: int,
    stride: int | None = None,
    dt_override: float | None = None,
) -> pd.DataFrame:
    """
    Real-data counterpart to `run_backtest`: slices a real historical price
    series (e.g. NIFTY 50 closes from `engine.data_loader`) into multiple
    overlapping windows and runs the identical pipeline on each, instead of
    Monte Carlo simulating new paths. Lets the thesis defence show "how
    would this have performed on real recent NIFTY history" directly.
    """
    from . import data_loader

    windows = data_loader.slice_sessions(price_series, session_len, n_sessions, stride)
    rows = []
    for i, window in enumerate(windows):
        result = run_single_session(cfg, S_external=window, dt_override=dt_override)
        for strategy, fill in result.fills.items():
            if fill is None:
                rows.append({
                    "path_id": i, "strategy": strategy, "triggered": False,
                    "latency_ticks": np.nan, "slippage": 0.0, "pnl": np.nan,
                })
            else:
                rows.append({
                    "path_id": i, "strategy": strategy, "triggered": True,
                    "latency_ticks": fill["latency_ticks"], "slippage": fill["slippage"],
                    "pnl": fill["pnl"],
                })
    return pd.DataFrame(rows)
