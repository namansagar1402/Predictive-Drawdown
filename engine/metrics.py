"""
engine/metrics.py
------------------
Section 12: Evaluation Metrics and Success Criteria.

Standard risk-adjusted performance measures plus the statistical-
significance machinery (paired Wilcoxon signed-rank test + bootstrap
confidence intervals) used to compare the predictive mechanism against
the static/ATR baselines, rather than relying on a single point estimate.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def max_drawdown(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / np.where(running_max == 0, 1, running_max)
    return float(np.max(drawdowns))


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - rf / periods_per_year
    if np.std(excess, ddof=1) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray, periods_per_year: int = 252) -> float:
    annual_return = float(np.mean(returns) * periods_per_year)
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    return annual_return / mdd


def paired_significance_test(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Paired, non-parametric significance test (Wilcoxon signed-rank) for the
    difference (a - b) between two strategies' matched-path metrics (e.g.
    predictive vs. static latency on the same simulated paths), used
    instead of a plain mean comparison since latency/slippage distributions
    are typically right-skewed (Section 12).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b
    if len(diff) < 2 or np.all(diff == 0):
        return {"n": len(diff), "median_diff": float(np.median(diff)) if len(diff) else np.nan,
                "p_value": np.nan, "statistic": np.nan}
    try:
        stat, p = stats.wilcoxon(a, b)
    except ValueError:
        stat, p = np.nan, np.nan
    return {"n": len(diff), "median_diff": float(np.median(diff)), "p_value": float(p), "statistic": float(stat)}


def bootstrap_ci(data: np.ndarray, statistic=np.mean, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap confidence interval for an arbitrary statistic."""
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return {"point_estimate": np.nan, "lower": np.nan, "upper": np.nan}
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    n = len(data)
    for i in range(n_boot):
        sample = data[rng.integers(0, n, size=n)]
        boot_stats[i] = statistic(sample)
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(boot_stats, [alpha, 1 - alpha])
    return {"point_estimate": float(statistic(data)), "lower": float(lower), "upper": float(upper)}
