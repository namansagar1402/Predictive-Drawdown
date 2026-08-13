"""
pages/1_Portfolio.py
----------------------
Multi-asset Indian equity & options portfolio tracker. Positions are
entered manually here (this is a read-only analyzer, not an order-entry
system) and marked to market from live/delayed Yahoo Finance prices.
"""

import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import data_loader as dl, market_calendar as mcal, portfolio as pf

st.set_page_config(page_title="Portfolio — Indian Equity & Options Analyzer", page_icon="💼", layout="wide")

for key, default in [("portfolio_positions", []), ("live_prices", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("💼 Multi-Asset Portfolio")
st.caption(
    "Enter positions manually below — nothing here places real orders. Equity/Index positions "
    "mark to market from a live quote; option positions mark to market via Black-Scholes using "
    "the live underlying price and the IV you specify (no live option-premium feed is wired in — "
    "see engine/options_chain.py docstring)."
)

st.subheader("➕ Add a position")
with st.form("add_position", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        label = st.text_input("Label", placeholder="e.g. NIFTY 25000 CE")
        asset_type = st.selectbox("Type", pf.ASSET_TYPES)
    with c2:
        watch_choice = st.selectbox("Underlying (from watchlist)", list(dl.DEFAULT_WATCHLIST.keys()))
        custom_symbol = st.text_input("...or custom yfinance ticker (overrides above)", placeholder="e.g. WIPRO.NS")
        yf_symbol = custom_symbol.strip() if custom_symbol.strip() else dl.DEFAULT_WATCHLIST[watch_choice]
    with c3:
        quantity = st.number_input("Quantity (+long / -short, units)", value=1.0, step=1.0)
        entry_price = st.number_input("Entry price (per unit)", value=100.0, step=0.5)

    strike = expiry = iv = None
    if asset_type in ("Call Option", "Put Option"):
        oc1, oc2, oc3 = st.columns(3)
        underlying_key = next((k for k, v in dl.DEFAULT_WATCHLIST.items() if v == yf_symbol), None)
        default_interval = mcal.strike_interval_for(underlying_key or yf_symbol.replace(".NS", ""))
        with oc1:
            strike = st.number_input("Strike", value=float(default_interval * 100), step=float(default_interval))
        with oc2:
            expiry = st.date_input("Expiry", value=mcal.next_weekly_expiry("NIFTY") if underlying_key == "NIFTY 50"
                                    else mcal.next_monthly_expiry(underlying_key or "NIFTY"))
        with oc3:
            iv = st.number_input("IV (annualised, e.g. 0.14)", value=0.15, step=0.01, format="%.2f")

    submitted = st.form_submit_button("Add position", type="primary")
    if submitted:
        if not label:
            st.warning("Please give the position a label.")
        else:
            pos = pf.Position(
                label=label, asset_type=asset_type, yf_symbol=yf_symbol,
                quantity=quantity, entry_price=entry_price,
                strike=strike, expiry=expiry, iv=iv,
            )
            st.session_state["portfolio_positions"].append(pos)
            st.success(f"Added '{label}'.")

st.divider()
st.subheader("📋 Current Positions")

positions = st.session_state["portfolio_positions"]
if not positions:
    st.info("No positions yet — add one above.")
else:
    col_a, col_b = st.columns([3, 1])
    with col_b:
        idx_to_remove = st.selectbox(
            "Remove a position", options=list(range(len(positions))),
            format_func=lambda i: positions[i].label,
        )
        if st.button("🗑️ Remove selected"):
            positions.pop(idx_to_remove)
            st.rerun()

    symbols = sorted({p.yf_symbol for p in positions})
    with st.spinner("Fetching live prices..."):
        try:
            live_prices = dl.fetch_live_quotes(symbols)
        except dl.DataFetchError as e:
            st.error(str(e))
            live_prices = {}
    st.session_state["live_prices"] = live_prices

    missing = [s for s in symbols if s not in live_prices]
    if missing:
        st.warning(
            f"No live price available for: {', '.join(missing)} (market closed, network blocked, or "
            "unrecognised ticker). Those rows will show as unavailable rather than a guessed number."
        )

    mtm = pf.mark_to_market(positions, live_prices)
    summary = pf.portfolio_summary(mtm)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Positions", summary["n_positions"])
    m2.metric("Portfolio Value", f"₹{summary['total_value']:,.0f}")
    m3.metric("Unrealised P&L", f"₹{summary['total_pnl']:,.0f}")
    m4.metric("Net Delta", f"{summary['net_delta']:,.1f}")
    m5.metric("Net Gamma", f"{summary['net_gamma']:,.4f}")

    st.dataframe(mtm, width="stretch")

    pnl_fig = go.Figure(go.Bar(
        x=mtm["Label"], y=mtm["P&L"],
        marker_color=np.where(mtm["P&L"] >= 0, "#81B29A", "#E07A5F"),
    ))
    pnl_fig.update_layout(height=350, title="P&L by position")
    st.plotly_chart(pnl_fig, width="stretch")

    st.caption(
        "Net Delta/Gamma are simple sums of per-position Delta/Gamma × quantity across different "
        "underlyings — treat this as a rough risk indicator, not a properly beta-adjusted "
        "cross-asset exposure figure (that would need per-stock betas to the index, which this "
        "prototype does not compute)."
    )
