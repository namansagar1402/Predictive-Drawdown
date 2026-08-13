import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Your original imports follow below
from engine import data_loader as dl, market_calendar as mcal, portfolio as pf

"""
app.py
------
Home page of the Indian Equity & Options Portfolio Analyzer.
Run with:  streamlit run app.py

This is a read-only market-data / analytics / research tool. It does not
place orders and is not connected to any brokerage account.
"""

import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import data_loader as dl, market_calendar as mcal, portfolio as pf

st.set_page_config(
    page_title="Indian Equity & Options Portfolio Analyzer",
    page_icon="🏠",
    layout="wide",
)

for key, default in [
    ("nifty_series", None), ("nifty_dt", None), ("nifty_calib", None),
    ("portfolio_positions", []), ("live_prices", {}), ("mock_feed", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🏠 Indian Equity & Options Portfolio Analyzer")
st.caption(
    "Read-only market analytics and research tool — no orders are placed and no brokerage "
    "account is connected. Live data via Yahoo Finance (`yfinance`), which is polling-based and "
    "typically delayed by a few minutes for NSE symbols, not a true tick-by-tick feed. "
    "Use the sidebar page navigator (**Portfolio**, **Options Chain**, **Strategy Engine**) to explore further."
)

market_open = mcal.is_market_open()
status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    st.metric("Market status (approx., IST hours only)", "🟢 Open" if market_open else "🔴 Closed")
with status_col2:
    st.metric("Next NIFTY weekly expiry", str(mcal.next_weekly_expiry("NIFTY")))
with status_col3:
    st.metric("Next NIFTY monthly expiry", str(mcal.next_monthly_expiry("NIFTY")))

st.divider()
st.subheader("📈 NIFTY 50 — Macro Reference")

col_fetch, col_period = st.columns([1, 2])
with col_period:
    period = st.select_slider("History window", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], value="6mo")
with col_fetch:
    st.write("")
    fetch_clicked = st.button("⬇️ Refresh NIFTY 50 data", type="primary")

if fetch_clicked or st.session_state["nifty_series"] is None:
    with st.spinner("Fetching NIFTY 50 (^NSEI)..."):
        try:
            series = dl.fetch_nifty_yfinance(period=period, interval="1d")
            st.session_state["nifty_series"] = series
            st.session_state["nifty_dt"] = dl.infer_dt_years(series.index)
        except dl.DataFetchError as e:
            st.error(
                f"{e}\n\nThis sandbox/network may not have access to Yahoo Finance. "
                "On your own machine with normal internet access this should work; "
                "otherwise use the CSV upload option on the Strategy Engine page."
            )

nifty_series = st.session_state["nifty_series"]
if nifty_series is not None and len(nifty_series) > 1:
    last = float(nifty_series.iloc[-1])
    prev = float(nifty_series.iloc[-2])
    chg = last - prev
    chg_pct = chg / prev * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("NIFTY 50 Last Close", f"{last:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
    m2.metric("Period High", f"{nifty_series.max():,.2f}")
    m3.metric("Period Low", f"{nifty_series.min():,.2f}")
    realised_vol = float(np.diff(np.log(nifty_series.to_numpy())).std() * np.sqrt(252) * 100)
    m4.metric("Realised Volatility (annualised)", f"{realised_vol:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nifty_series.index, y=nifty_series.values, name="NIFTY 50",
                             line=dict(color="#3D5A80", width=2)))
    fig.update_layout(height=420, title=f"NIFTY 50 (^NSEI) — {period} history", margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Click **Refresh NIFTY 50 data** above to load the macro reference chart.")

st.divider()
st.subheader("💼 Portfolio Snapshot")

positions = st.session_state["portfolio_positions"]
if not positions:
    st.info("No positions yet. Add some on the **Portfolio** page (see sidebar navigation).")
else:
    symbols = sorted({p.yf_symbol for p in positions})
    with st.spinner("Fetching live prices for your positions..."):
        try:
            live_prices = dl.fetch_live_quotes(symbols)
        except dl.DataFetchError:
            live_prices = {}
    if not live_prices and nifty_series is not None:
        # Fall back to last known NIFTY close for ^NSEI only, so the page
        # still shows something useful when live quotes are unreachable.
        live_prices = {"^NSEI": float(nifty_series.iloc[-1])}
    st.session_state["live_prices"] = live_prices

    mtm = pf.mark_to_market(positions, live_prices)
    summary = pf.portfolio_summary(mtm)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positions", summary["n_positions"])
    c2.metric("Portfolio Value", f"₹{summary['total_value']:,.0f}")
    c3.metric("Unrealised P&L", f"₹{summary['total_pnl']:,.0f}")
    c4.metric("Net Delta Exposure", f"{summary['net_delta']:,.1f}")

    missing = [s for s in symbols if s not in live_prices]
    if missing:
        st.warning(f"Could not fetch a live price for: {', '.join(missing)} — shown as unavailable below.")
    st.dataframe(mtm, width="stretch")
