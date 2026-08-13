"""
app.py
------
Predictive, Latency-Optimized Exit Mechanism for Dynamic Drawdown Containment
Interactive Streamlit prototype -- Minor Project (Phase I) + Major Project (Phase II)

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from engine import simulation, greeks, backtester, metrics, data_loader

st.set_page_config(
    page_title="Strategy Engine — Predictive Drawdown Containment",
    page_icon="🔬",
    layout="wide",
)

# --------------------------------------------------------------------------
# Sidebar -- data source (synthetic vs. real NIFTY 50)
# --------------------------------------------------------------------------
st.sidebar.title("📡 Data Source")
data_source = st.sidebar.radio(
    "Where should the underlying price series come from?",
    ["Synthetic (Monte Carlo)", "NIFTY 50 — Yahoo Finance", "Upload CSV"],
    help="Synthetic mode uses the GBM/jump-diffusion simulator directly. "
         "The other two modes calibrate the same simulator from real NIFTY 50 "
         "data and/or let you replay real historical sessions in Tab 5.",
)

if "nifty_series" not in st.session_state:
    st.session_state["nifty_series"] = None
    st.session_state["nifty_dt"] = None
    st.session_state["nifty_calib"] = None

if data_source == "NIFTY 50 — Yahoo Finance":
    with st.sidebar.expander("Fetch settings", expanded=True):
        period = st.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=4)
        interval = st.selectbox("Bar interval", ["1d", "1h", "30m", "15m", "5m"], index=0,
                                help="Intraday intervals are only available for a recent window "
                                     "regardless of the period chosen -- a Yahoo Finance limit, not ours.")
        if st.button("⬇️ Fetch NIFTY 50 data"):
            with st.spinner("Fetching NIFTY 50 (^NSEI) from Yahoo Finance..."):
                try:
                    series = data_loader.fetch_nifty_yfinance(period=period, interval=interval)
                    st.session_state["nifty_series"] = series
                    st.session_state["nifty_dt"] = data_loader.infer_dt_years(series.index)
                    st.success(f"Fetched {len(series)} bars, {series.index[0].date()} → {series.index[-1].date()}.")
                except data_loader.DataFetchError as e:
                    st.error(str(e))

elif data_source == "Upload CSV":
    with st.sidebar.expander("CSV upload", expanded=True):
        st.caption("Expected columns: a Date/Datetime column and a Close/LTP price column.")
        uploaded = st.file_uploader("Upload NIFTY 50 (or any) price history CSV", type=["csv"])
        if uploaded is not None:
            try:
                series = data_loader.load_csv_upload(uploaded)
                st.session_state["nifty_series"] = series
                st.session_state["nifty_dt"] = data_loader.infer_dt_years(series.index)
                st.success(f"Parsed {len(series)} rows, {series.index[0].date()} → {series.index[-1].date()}.")
            except data_loader.DataFetchError as e:
                st.error(str(e))

if st.session_state["nifty_series"] is not None:
    _s = st.session_state["nifty_series"]
    st.sidebar.caption(f"Loaded: {len(_s)} bars ending {_s.index[-1].date()}, last close {_s.iloc[-1]:.2f}")
    if st.sidebar.button("🧮 Calibrate GBM/jump-diffusion from this data"):
        calib, is_jump = simulation.calibrate_jump_diffusion(_s.to_numpy(), st.session_state["nifty_dt"])
        st.session_state["nifty_calib"] = calib
        st.sidebar.success(
            f"Calibrated: μ={calib.mu:.3f}, σ={calib.sigma:.3f}, λ={calib.lam:.2f}/yr "
            f"({int(is_jump.sum())} jumps detected)"
        )

st.sidebar.divider()

# --------------------------------------------------------------------------
# Sidebar -- simulation & strategy parameters
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Simulation Parameters")

_calib = st.session_state.get("nifty_calib")
_default_S0 = float(st.session_state["nifty_series"].iloc[-1]) if st.session_state["nifty_series"] is not None else 100.0

with st.sidebar.expander("Underlying / Option", expanded=True):
    S0 = st.number_input("Spot price S₀", value=round(_default_S0, 2), step=1.0)
    K = st.number_input("Strike K", value=float(round(_default_S0 / 50) * 50 if _default_S0 > 1000 else round(_default_S0, 2)), step=1.0)
    r = st.number_input("Risk-free rate r", value=0.06, step=0.01, format="%.2f")
    sigma = st.slider("Volatility σ (annualised)", 0.05, 1.0, float(_calib.sigma) if _calib else 0.25, 0.01)
    option_days_to_expiry = st.slider("Option days-to-expiry at session start", 1, 90, 30)

with st.sidebar.expander("GBM + Jump-Diffusion (Section 8.3 / 8.4)", expanded=True):
    mu = st.slider("Drift μ (annualised)", -1.0, 1.0, float(np.clip(_calib.mu, -1, 1)) if _calib else 0.08, 0.01)
    lam = st.slider("Jump intensity λ (jumps/year)", 0.0, 40.0, float(_calib.lam) if _calib else 8.0, 0.5)
    jump_mean = st.slider("Mean jump size (log-return)", -0.15, 0.05, float(np.clip(_calib.jump_mean, -0.15, 0.05)) if _calib else -0.03, 0.005)
    jump_std = st.slider("Jump size std-dev", 0.0, 0.15, float(np.clip(_calib.jump_std, 0, 0.15)) if _calib else 0.05, 0.005)

with st.sidebar.expander("Session / Exit Strategies", expanded=True):
    T_days = st.slider("Session length (trading days)", 1, 5, 1)
    n_steps = st.slider("Ticks per session", 100, 1000, 375, 25)
    static_threshold = st.slider("Static exit: option-value drawdown %", 0.05, 0.5, 0.20, 0.01)
    atr_multiplier = st.slider("ATR trailing-stop multiplier", 0.5, 5.0, 2.5, 0.1)
    fpr_tolerance = st.slider("Predictive trigger: max tolerated false-trigger rate", 0.01, 0.30, 0.05, 0.01)

with st.sidebar.expander("Advanced (Risk / Labelling)", expanded=False):
    cvar_window = st.slider("CVaR rolling window (ticks)", 10, 100, 30, 5)
    cvar_alpha = st.slider("CVaR tail probability α", 0.01, 0.20, 0.05, 0.01)
    max_acceptable_loss = st.slider("Max acceptable loss (fraction of option value)", 0.05, 1.0, 0.30, 0.05)
    breach_horizon_ticks = st.slider("Breach-probability horizon (ticks)", 2, 50, 10, 1)

seed = st.sidebar.number_input("Random seed", value=42, step=1)

cfg = backtester.SessionConfig(
    S0=S0, K=K, r=r, sigma=sigma, lam=lam, jump_mean=jump_mean, jump_std=jump_std, mu=mu,
    T_days=T_days, option_days_to_expiry=option_days_to_expiry, n_steps=n_steps,
    static_threshold=static_threshold, atr_multiplier=atr_multiplier,
    cvar_window=cvar_window, cvar_alpha=cvar_alpha, max_acceptable_loss=max_acceptable_loss,
    breach_horizon_ticks=breach_horizon_ticks, fpr_tolerance=fpr_tolerance, seed=int(seed),
)

st.title("📉 Predictive, Latency-Optimized Exit Mechanism")
st.caption(
    "Interactive prototype for Dynamic Drawdown Containment in derivatives trading — "
    "Minor Project (Phase I: stochastic modelling) + Major Project (Phase II: composite trigger, "
    "execution simulation, dashboard). **Simulation on synthetic/historical data only — "
    "not connected to any live trading account.**"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Live Risk Dashboard", "🚨 Execution & Alert Log", "📊 Backtest Comparison",
    "🧮 Phase I Calibration", "🇮🇳 NIFTY 50 Real Data",
])

# --------------------------------------------------------------------------
# Shared: run a single session (cached on the config values)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_single_session(cfg_dict):
    cfg_local = backtester.SessionConfig(**cfg_dict)
    result = backtester.run_single_session(cfg_local)
    return result.df, result.trigger_ticks, result.fills, result.composite.w, result.composite.b, result.composite.threshold


df, trigger_ticks, fills, w, b, threshold = _cached_single_session(cfg.__dict__)

STRATEGY_COLORS = {"static": "#E07A5F", "atr": "#81B29A", "predictive": "#3D5A80"}

# --------------------------------------------------------------------------
# TAB 1 -- Live Risk Dashboard
# --------------------------------------------------------------------------
with tab1:
    st.subheader("Live Risk Dashboard")
    ticks = np.arange(len(df))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ticks, y=df["S"], name="Underlying S", line=dict(color="#3D5A80", width=2)))
    fig.add_trace(
        go.Scatter(x=ticks, y=df["composite_score"], name="Composite breach score", line=dict(color="#E07A5F", dash="dot")),
        secondary_y=True,
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="#E07A5F", secondary_y=True,
                  annotation_text="Predictive firing threshold")

    for name, tick in trigger_ticks.items():
        if tick is not None:
            fig.add_vline(x=tick, line_color=STRATEGY_COLORS[name], line_width=2,
                          annotation_text=f"{name} trigger", annotation_position="top")

    jump_ticks = np.where(df["jump_flag"] > 0)[0]
    if len(jump_ticks):
        fig.add_trace(go.Scatter(
            x=jump_ticks, y=df["S"].iloc[jump_ticks], mode="markers", name="Jump event",
            marker=dict(color="black", size=9, symbol="x"),
        ))

    fig.update_layout(height=460, title="Underlying price with breach-probability overlay and trigger events",
                      legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="Underlying price", secondary_y=False)
    fig.update_yaxes(title_text="Composite score (0-1)", range=[0, 1], secondary_y=True)
    st.plotly_chart(fig, width='stretch')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=float(df["cvar_norm"].iloc[-1]) * 100,
            title={"text": "Normalised CVaR (%)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E07A5F"}},
        ))
        gauge.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(gauge, width='stretch')
    with c2:
        st.metric("Delta / Gamma velocity (last tick)", f"{df['v_greek'].iloc[-1]:.3f}")
        st.metric("Jump-adjusted breach probability", f"{df['p_jump'].iloc[-1]:.3f}")
    with c3:
        pred_fill = fills.get("predictive")
        st.metric("Predictive decision latency", f"{pred_fill['latency_ticks']} ticks" if pred_fill else "not triggered")
        static_fill = fills.get("static")
        st.metric("Static decision latency", f"{static_fill['latency_ticks']} ticks" if static_fill else "not triggered")
    with c4:
        if pred_fill and static_fill:
            st.metric("Latency saved vs. static", f"{static_fill['latency_ticks'] - pred_fill['latency_ticks']} ticks")
            st.metric("Slippage saved vs. static", f"{static_fill['slippage'] - pred_fill['slippage']:.4f}")
        else:
            st.info("Both static and predictive must trigger in this session to compare directly.")

    with st.expander("Raw session data"):
        st.dataframe(df, width='stretch', height=300)

# --------------------------------------------------------------------------
# TAB 2 -- Execution & Alert Log
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Execution & Alert Log")

    log_rows = []
    for name, tick in trigger_ticks.items():
        if tick is None:
            log_rows.append({"Strategy": name, "Tick": np.nan, "Event": "No trigger this session", "Latency (ticks)": np.nan, "Fill value": np.nan})
            continue
        fill = fills[name]
        log_rows.append({"Strategy": name, "Tick": tick, "Event": "Trigger fired", "Latency (ticks)": np.nan, "Fill value": np.nan})
        log_rows.append({
            "Strategy": name, "Tick": fill["fill_tick"], "Event": "Order filled (simulated queue)",
            "Latency (ticks)": float(fill["latency_ticks"]), "Fill value": round(float(fill["fill_value"]), 4),
        })
    log_df = pd.DataFrame(log_rows)
    log_df["Tick"] = pd.to_numeric(log_df["Tick"], errors="coerce")
    log_df = log_df.sort_values("Tick").reset_index(drop=True)
    st.dataframe(log_df, width='stretch')

    st.markdown("##### Synthetic order-book depth ladder (illustrative snapshot near the predictive trigger)")
    st.caption(
        "No real Level-2 feed is used (Section 8.2); this ladder is a stylised, randomly generated "
        "snapshot for illustrating the queue-aware execution concept, not derived from live market data."
    )
    snap_tick = trigger_ticks.get("predictive") or trigger_ticks.get("static") or 0
    mid = float(df["S"].iloc[snap_tick])
    rng_ladder = np.random.default_rng(snap_tick + 1)
    levels = np.round(mid + np.arange(-5, 6) * 0.05, 2)
    volumes = rng_ladder.integers(20, 400, size=len(levels))
    side = ["Ask" if lv > mid else ("Bid" if lv < mid else "Mid") for lv in levels]
    ladder_df = pd.DataFrame({"Level": levels, "Side": side, "Volume": volumes})
    ladder_fig = go.Figure()
    for s_type, color in [("Bid", "#81B29A"), ("Ask", "#E07A5F")]:
        sub = ladder_df[ladder_df.Side == s_type]
        ladder_fig.add_trace(go.Bar(x=sub.Level, y=sub.Volume, name=s_type, marker_color=color))
    ladder_fig.update_layout(height=320, xaxis_title="Price level", yaxis_title="Depth (synthetic)")
    st.plotly_chart(ladder_fig, width='stretch')

# --------------------------------------------------------------------------
# TAB 3 -- Backtest Comparison
# --------------------------------------------------------------------------
with tab3:
    st.subheader("Backtest Comparison — Static vs. ATR vs. Predictive")
    n_paths = st.slider("Number of Monte Carlo paths", 20, 1000, 200, 20)
    run_bt = st.button("▶ Run backtest", type="primary")

    if "bt_results" not in st.session_state:
        st.session_state["bt_results"] = None

    if run_bt:
        with st.spinner(f"Simulating {n_paths} sessions..."):
            st.session_state["bt_results"] = backtester.run_backtest(cfg, n_paths=n_paths, base_seed=1000)

    bt = st.session_state["bt_results"]
    if bt is None:
        st.info("Click **Run backtest** to simulate multiple sessions and compare the three exit strategies.")
    else:
        summary = bt.groupby("strategy").agg(
            trigger_rate=("triggered", "mean"),
            mean_latency=("latency_ticks", "mean"),
            mean_slippage=("slippage", "mean"),
            mean_pnl=("pnl", "mean"),
        ).round(4)
        st.markdown("##### Summary statistics")
        st.dataframe(summary, width='stretch')

        fig_box = go.Figure()
        for strat, color in STRATEGY_COLORS.items():
            sub = bt[(bt.strategy == strat) & (bt.triggered)]
            fig_box.add_trace(go.Box(y=sub["latency_ticks"], name=strat, marker_color=color))
        fig_box.update_layout(height=380, title="Decision-to-fill latency distribution (ticks)")
        st.plotly_chart(fig_box, width='stretch')

        fig_slip = go.Figure()
        for strat, color in STRATEGY_COLORS.items():
            sub = bt[(bt.strategy == strat) & (bt.triggered)]
            fig_slip.add_trace(go.Box(y=sub["slippage"], name=strat, marker_color=color))
        fig_slip.update_layout(height=380, title="Realised slippage distribution")
        st.plotly_chart(fig_slip, width='stretch')

        st.markdown("##### Statistical significance (Section 12): predictive vs. each baseline")
        piv_lat = bt.pivot(index="path_id", columns="strategy", values="latency_ticks")
        piv_slip = bt.pivot(index="path_id", columns="strategy", values="slippage")

        rows = []
        for baseline in ["static", "atr"]:
            for metric_name, piv in [("latency_ticks", piv_lat), ("slippage", piv_slip)]:
                sig = metrics.paired_significance_test(piv["predictive"], piv[baseline])
                ci = metrics.bootstrap_ci((piv["predictive"] - piv[baseline]).dropna().to_numpy())
                rows.append({
                    "Comparison": f"predictive vs. {baseline}", "Metric": metric_name,
                    "n paired paths": sig["n"], "Median difference": round(sig["median_diff"], 4),
                    "Wilcoxon p-value": round(sig["p_value"], 4) if not np.isnan(sig["p_value"]) else "n/a",
                    "Bootstrap 95% CI": f"[{ci['lower']:.4f}, {ci['upper']:.4f}]",
                })
        st.dataframe(pd.DataFrame(rows), width='stretch')
        st.caption(
            "Negative 'median difference' for latency/slippage means the predictive mechanism is faster / "
            "cheaper than the baseline on the matched paths. A small p-value (< 0.05) indicates the "
            "difference is unlikely to be due to chance given this many simulated paths."
        )

# --------------------------------------------------------------------------
# TAB 4 -- Phase I Calibration
# --------------------------------------------------------------------------
with tab4:
    st.subheader("Phase I — Parameter Calibration Correctness Check")
    st.caption(
        "Simulates a 2-year daily 'historical' series from known true parameters, then re-estimates "
        "those parameters from the simulated data alone, to verify the calibration procedure in "
        "Section 8.5 recovers its own inputs before it is trusted on real data."
    )

    colA, colB = st.columns(2)
    with colA:
        true_mu = st.slider("True μ", -0.5, 0.5, 0.08, 0.01, key="cal_mu")
        true_sigma = st.slider("True σ", 0.05, 0.8, 0.25, 0.01, key="cal_sigma")
    with colB:
        true_lam = st.slider("True λ (jumps/yr)", 0.0, 30.0, 8.0, 0.5, key="cal_lam")
        true_jump_std = st.slider("True jump std", 0.0, 0.15, 0.05, 0.005, key="cal_jstd")
    k_threshold = st.slider("Jump-detection threshold (× robust local σ)", 1.0, 8.0, 4.0, 0.5)

    if st.button("▶ Simulate & calibrate"):
        true_p = simulation.ProcessParams(true_mu, true_sigma, true_lam, -abs(true_jump_std), true_jump_std)
        S_hist, jump_counts = simulation.simulate_paths(100.0, true_p, T=2.0, n_steps=504, n_paths=1, seed=7)
        dt_daily = 1 / 252
        calib_gbm = simulation.calibrate_gbm(S_hist[0], dt_daily)
        calib_jd, is_jump = simulation.calibrate_jump_diffusion(S_hist[0], dt_daily, k_threshold=k_threshold)

        st.markdown("##### Recovered vs. true parameters")
        comp_df = pd.DataFrame({
            "Parameter": ["mu (drift)", "sigma (volatility)", "lambda (jump intensity)", "jump_std"],
            "True value": [true_p.mu, true_p.sigma, true_p.lam, true_p.jump_std],
            "GBM-only estimate": [calib_gbm.mu, calib_gbm.sigma, np.nan, np.nan],
            "Jump-diffusion estimate": [calib_jd.mu, calib_jd.sigma, calib_jd.lam, calib_jd.jump_std],
        }).round(4)
        st.dataframe(comp_df, width='stretch')
        st.info(
            f"Detected {int(is_jump.sum())} jump ticks out of {int(jump_counts.sum())} true simulated jump events "
            "(over 504 trading days). Note that sigma typically recovers well, while mu (drift) remains noisy "
            "even over 2 years of daily data — its estimation error scales with 1/√T regardless of sampling "
            "frequency, a standard and worth-documenting limitation of drift estimation."
        )

        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(y=S_hist[0], mode="lines", name="Simulated 'historical' price"))
        jump_idx = np.where(is_jump)[0] + 1
        if len(jump_idx):
            price_fig.add_trace(go.Scatter(x=jump_idx, y=S_hist[0][jump_idx], mode="markers",
                                           name="Detected jump", marker=dict(color="red", size=7, symbol="x")))
        price_fig.update_layout(height=380, title="2-year simulated daily series with detected jumps")
        st.plotly_chart(price_fig, width='stretch')

# --------------------------------------------------------------------------
# TAB 5 -- NIFTY 50 Real Data (replay + historical backtest)
# --------------------------------------------------------------------------
with tab5:
    st.subheader("NIFTY 50 — Real Data Replay & Historical Backtest")
    nifty_series = st.session_state.get("nifty_series")

    if nifty_series is None:
        st.info(
            "No real data loaded yet. Use the **📡 Data Source** panel in the sidebar to fetch "
            "NIFTY 50 from Yahoo Finance, or upload a CSV, then come back to this tab."
        )
    else:
        dt_real = st.session_state["nifty_dt"]
        st.caption(
            f"Loaded {len(nifty_series)} bars from {nifty_series.index[0].date()} to "
            f"{nifty_series.index[-1].date()} (inferred step ≈ {dt_real*252*375:.1f} trading minutes)."
        )

        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=nifty_series.index, y=nifty_series.values, name="NIFTY 50 Close"))
        price_fig.update_layout(height=350, title="Loaded NIFTY 50 history")
        st.plotly_chart(price_fig, width='stretch')

        st.markdown("##### Replay a single historical window through the strategy engine")
        max_len = len(nifty_series)
        session_len = st.slider("Session length (bars)", 20, min(max_len - 1, 375), min(100, max_len - 1))
        offset = st.slider(
            "Start offset from the most recent bar (0 = most recent window)",
            0, max(0, max_len - session_len - 1), 0,
        )
        end_idx = max_len - offset
        start_idx = max(0, end_idx - session_len)
        window = nifty_series.to_numpy()[start_idx:end_idx]
        window_dates = nifty_series.index[start_idx:end_idx]

        if st.button("▶ Replay this window"):
            replay_cfg = backtester.SessionConfig(
                K=K, r=r, sigma=sigma, mu=mu, lam=lam, jump_mean=jump_mean, jump_std=jump_std,
                option_days_to_expiry=option_days_to_expiry, static_threshold=static_threshold,
                atr_multiplier=atr_multiplier, cvar_window=min(cvar_window, max(5, session_len // 3)),
                cvar_alpha=cvar_alpha, max_acceptable_loss=max_acceptable_loss,
                breach_horizon_ticks=min(breach_horizon_ticks, max(2, session_len // 5)),
                fpr_tolerance=fpr_tolerance, seed=int(seed),
            )
            result = backtester.run_single_session(replay_cfg, S_external=window, dt_override=dt_real)

            replay_fig = make_subplots(specs=[[{"secondary_y": True}]])
            replay_fig.add_trace(go.Scatter(x=window_dates, y=window, name="NIFTY 50 (replayed)", line=dict(color="#3D5A80")))
            replay_fig.add_trace(
                go.Scatter(x=window_dates, y=result.df["composite_score"], name="Composite breach score",
                          line=dict(color="#E07A5F", dash="dot")),
                secondary_y=True,
            )
            for name, tick in result.trigger_ticks.items():
                if tick is not None:
                    replay_fig.add_vline(x=window_dates[tick], line_color=STRATEGY_COLORS[name],
                                        annotation_text=f"{name} trigger")
            replay_fig.update_layout(height=440, title="Real NIFTY 50 window — triggers overlaid")
            st.plotly_chart(replay_fig, width='stretch')

            fills_df = pd.DataFrame([
                {"Strategy": k, **({kk: vv for kk, vv in v.items()} if v else {"note": "not triggered"})}
                for k, v in result.fills.items()
            ])
            st.dataframe(fills_df, width='stretch')

        st.divider()
        st.markdown("##### Historical backtest across many real overlapping windows")
        n_sessions = st.slider("Number of historical windows", 5, 100, 30, 5)
        hist_session_len = st.slider("Window length (bars)", 20, min(max_len - 1, 375), min(60, max_len - 1), key="hist_len")

        if st.button("▶ Run historical backtest"):
            with st.spinner(f"Replaying {n_sessions} real NIFTY 50 windows..."):
                bt_cfg = backtester.SessionConfig(
                    K=K, r=r, sigma=sigma, mu=mu, lam=lam, jump_mean=jump_mean, jump_std=jump_std,
                    option_days_to_expiry=option_days_to_expiry, static_threshold=static_threshold,
                    atr_multiplier=atr_multiplier, cvar_window=min(cvar_window, max(5, hist_session_len // 3)),
                    cvar_alpha=cvar_alpha, max_acceptable_loss=max_acceptable_loss,
                    breach_horizon_ticks=min(breach_horizon_ticks, max(2, hist_session_len // 5)),
                    fpr_tolerance=fpr_tolerance, seed=int(seed),
                )
                hist_bt = backtester.run_historical_backtest(
                    bt_cfg, nifty_series.to_numpy(), session_len=hist_session_len,
                    n_sessions=n_sessions, dt_override=dt_real,
                )
            st.session_state["hist_bt_results"] = hist_bt

        hist_bt = st.session_state.get("hist_bt_results")
        if hist_bt is not None:
            summary = hist_bt.groupby("strategy").agg(
                trigger_rate=("triggered", "mean"),
                mean_latency=("latency_ticks", "mean"),
                mean_slippage=("slippage", "mean"),
                mean_pnl=("pnl", "mean"),
            ).round(4)
            st.dataframe(summary, width='stretch')

            piv_lat = hist_bt.pivot(index="path_id", columns="strategy", values="latency_ticks")
            piv_slip = hist_bt.pivot(index="path_id", columns="strategy", values="slippage")
            rows = []
            for baseline in ["static", "atr"]:
                for metric_name, piv in [("latency_ticks", piv_lat), ("slippage", piv_slip)]:
                    sig = metrics.paired_significance_test(piv["predictive"], piv[baseline])
                    rows.append({
                        "Comparison": f"predictive vs. {baseline}", "Metric": metric_name,
                        "n paired windows": sig["n"], "Median difference": round(sig["median_diff"], 4),
                        "Wilcoxon p-value": round(sig["p_value"], 4) if not np.isnan(sig["p_value"]) else "n/a",
                    })
            st.dataframe(pd.DataFrame(rows), width='stretch')
            st.caption(
                "This is real historical NIFTY 50 price action replayed through the identical strategy "
                "engine used in the synthetic Backtest Comparison tab — the headline result your thesis "
                "defence should actually quote, alongside the synthetic Monte Carlo numbers."
            )
