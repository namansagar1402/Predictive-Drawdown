"""
engine/trigger.py
------------------
Phase II (Major Project), Section 8.8: Composite Predictive Trigger.

  Score_t = sigmoid( w1 * P_jump,t + w2 * V_Greek,t + w3 * C_VaR,t + b )

  - P_jump,t : short-horizon breach probability, estimated via a Brownian-
    motion first-passage-time approximation (reflection principle) inflated
    by the calibrated jump intensity for tail risk (this is the
    "jump-adjusted breach probability" from Section 8.4).
  - V_Greek,t : normalised Delta/Gamma velocity (engine.greeks).
  - C_VaR,t   : normalised rolling CVaR (engine.risk).

Weights (w1, w2, w3, b) are fit by logistic regression against labelled
historical breach / no-breach outcomes (scipy.optimize, no sklearn
dependency), and the firing threshold is chosen via ROC analysis to hold
the false-trigger rate below an explicit tolerance.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def breach_probability_proxy(
    distance_to_breach: np.ndarray,
    local_vol: np.ndarray,
    horizon_ticks: int,
    dt: float,
    lam: float = 0.0,
) -> np.ndarray:
    """
    Fast analytic proxy for "probability of breaching the drawdown
    threshold within the next `horizon_ticks` ticks", using the reflection-
    principle first-passage-time approximation for a driftless Brownian
    motion:

        P(hit within h) ~= 2 * (1 - Phi(distance / (vol * sqrt(h * dt))))

    then inflated by an extra tail-risk term proportional to the
    calibrated jump intensity (lam), reflecting that jump-diffusion paths
    breach more often than pure diffusion would predict (Section 8.4).

    Parameters
    ----------
    distance_to_breach : current (positive) distance from breach, in the
        same units as local_vol (e.g. option-value drawdown units).
    local_vol : recent local volatility estimate of that same quantity.
    horizon_ticks, dt : the probability horizon, h * dt.
    lam : calibrated jump intensity (jumps / year); higher lam raises the
        estimated breach probability via a simple exponential inflation.
    """
    vol_h = np.maximum(local_vol, 1e-9) * np.sqrt(horizon_ticks * dt)
    diffusion_p = 2 * (1 - norm.cdf(np.maximum(distance_to_breach, 0) / vol_h))
    jump_inflation = 1 - np.exp(-lam * horizon_ticks * dt)
    p = diffusion_p + (1 - diffusion_p) * jump_inflation
    return np.clip(p, 0.0, 1.0)


class CompositeTrigger:
    """Weighted logistic combination of the three composite-score inputs."""

    def __init__(self, weights: np.ndarray | None = None, bias: float = 0.0):
        self.w = np.asarray(weights) if weights is not None else np.array([2.0, 1.0, 2.0])
        self.b = bias
        self.threshold = 0.5
        self.fit_diagnostics: dict = {}

    def score(self, p_jump, v_greek, c_var) -> np.ndarray:
        X = np.column_stack([p_jump, v_greek, c_var])
        z = X @ self.w + self.b
        return sigmoid(z)

    def fit(self, X: np.ndarray, y: np.ndarray, l2: float = 1e-3):
        """Fit (w, b) by regularised logistic regression via L-BFGS-B."""
        n, d = X.shape
        y = y.astype(float)

        def neg_log_likelihood(params):
            w, b = params[:d], params[d]
            z = X @ w + b
            p = sigmoid(z)
            eps = 1e-9
            ll = y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
            return -np.mean(ll) + l2 * np.sum(w ** 2)

        x0 = np.zeros(d + 1)
        res = minimize(neg_log_likelihood, x0, method="L-BFGS-B")
        self.w, self.b = res.x[:d], res.x[d]
        self.fit_diagnostics = {"converged": res.success, "final_loss": float(res.fun)}
        return self

    def calibrate_threshold(self, scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.05):
        """
        ROC-based threshold selection (Section 8.8): among thresholds that
        keep the false-positive rate at or below `target_fpr`, pick the one
        with the highest true-positive rate (detection rate).
        """
        thr, tpr, fpr = roc_threshold(scores, labels, target_fpr)
        self.threshold = thr
        self.fit_diagnostics.update({"threshold": thr, "tpr_at_threshold": tpr, "fpr_at_threshold": fpr})
        return thr


def roc_curve_manual(scores: np.ndarray, labels: np.ndarray):
    """Manually computed ROC curve (no sklearn dependency)."""
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    P = labels.sum()
    N = len(labels) - P
    P = max(P, 1)
    N = max(N, 1)

    tps = np.cumsum(labels_sorted == 1)
    fps = np.cumsum(labels_sorted == 0)
    tpr = tps / P
    fpr = fps / N
    thresholds = scores_sorted
    return thresholds, tpr, fpr


def roc_threshold(scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.05):
    """Pick the threshold with FPR <= target_fpr maximising TPR."""
    thresholds, tpr, fpr = roc_curve_manual(scores, labels)
    eligible = np.where(fpr <= target_fpr)[0]
    if len(eligible) == 0:
        # Cannot meet the target FPR; fall back to the point closest to it.
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
    else:
        idx = eligible[np.argmax(tpr[eligible])]
    return float(thresholds[idx]), float(tpr[idx]), float(fpr[idx])


def make_breach_labels(drawdown: np.ndarray, breach_level: float, horizon_ticks: int) -> np.ndarray:
    """
    Label tick t as 1 if drawdown crosses `breach_level` at any point within
    the next `horizon_ticks` ticks (used to train/calibrate the trigger).
    """
    n = len(drawdown)
    breached = (drawdown >= breach_level).astype(int)
    cs = np.concatenate([[0], np.cumsum(breached)])  # cs[i] = sum(breached[:i])
    idx = np.arange(n)
    end_idx = np.minimum(idx + horizon_ticks, n - 1)
    # breaches strictly after t, up to and including end_idx
    window_hits = cs[end_idx + 1] - cs[idx + 1]
    labels = (window_hits > 0).astype(int)
    return labels
