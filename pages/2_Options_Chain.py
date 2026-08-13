"""
pages/2_Options_Chain.py
--------------------------
Model-derived (synthetic) options chain viewer for NIFTY / Bank Nifty /
individual NSE stocks. See engine/options_chain.py docstring: this is
Black-Scholes output off a live spot price, not a live NSE option-chain
feed (none is reliably available for free/ToS-safe use).
"""

import datetime as dt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from engine import data_loader as dl, market_calendar as mcal, options_chain as oc

st.set_page_config(page_title="Options Chain — Indian Equity & Options Analyzer", page_icon="⛓️", layout="wide")

st.title("⛓️ Options Chain (Model-Derived)")
st.warning(
    "**This chain is synthetic**, not live NSE data: strikes and expiries follow real NSE "
    "conventions, but LTP/Greeks are computed with Black-Scholes off a live/last-known spot price "
    "and the IV you set below (with a simple illustrative smile). No free, reliable, ToS-safe live "
    "NSE option-chain feed exists for a prototype like this — plug in a licensed vendor "
    "(Kite Connect, Breeze, etc.) in `engine/options_chain.py` if you have one.",
    icon="⚠️",
)

c1, c2, c3 = st.columns(3)
with c1:
    underlying_name = st.selectbox("Underlying", list(dl.DEFAULT_WATCHLIST.keys()), index=0)
    yf_symbol = dl.DEFAULT_WATCHLIST[underlying_name]
    symbol_key = "NIFTY" if underlying_name == "NIFTY 50" else (
        "BANKNIFTY" if underlying_name == "BANK NIFTY" else underlying_name.replace(" ", "")
    )
with c2:
    fetch_spot = st.button("⬇️ Fetch live spot", type="primary")
with c3:
    atm_iv = st.slider("ATM IV (annualised)", 0.05, 1.0, 0.13, 0.01)

if "chain_spot" not in st.session_state:
    st.session_state["chain_spot"] = 24988.0 if symbol_key == "NIFTY" else 100.0

if fetch_spot:
    with st.spinner(f"Fetching {underlying_name}..."):
        try:
            quotes = dl.fetch_live_quotes([yf_symbol])
            if yf_symbol in quotes:
                st.session_state["chain_spot"] = quotes[yf_symbol]
                st.success(f"Spot fetched: {quotes[yf_symbol]:,.2f}")
            else:
                st.error("Could not fetch a live price for this symbol (market closed / network blocked).")
        except dl.DataFetchError as e:
            st.error(str(e))

spot = st.number_input("Spot price (editable — auto-filled by Fetch)", value=float(st.session_state["chain_spot"]), step=1.0)

expiries = mcal.list_upcoming_expiries(symbol_key, n=6)
expiry = st.selectbox("Expiry", expiries, format_func=lambda d: d.strftime("%d %b %Y (%a)"))
n_strikes = st.slider("Strikes each side of ATM", 3, 20, 10)

chain = oc.generate_chain(spot=spot, symbol=symbol_key, expiry=expiry, atm_iv=atm_iv, n_strikes_each_side=n_strikes)

st.caption(
    f"ATM strike: **{chain.attrs['atm_strike']:,.0f}** | Days to expiry: "
    f"**{(expiry - dt.date.today()).days}** | Strike interval: "
    f"**{mcal.strike_interval_for(symbol_key, spot)}**"
)

display_cols = ["call_delta", "call_gamma", "call_ltp", "iv", "strike", "put_ltp", "put_gamma", "put_delta"]
styled = chain[display_cols].copy()
styled.attrs = {}
styled = styled.rename(columns={
    "call_delta": "Call Δ", "call_gamma": "Call Γ", "call_ltp": "Call LTP",
    "iv": "IV", "strike": "Strike",
    "put_ltp": "Put LTP", "put_gamma": "Put Γ", "put_delta": "Put Δ",
})


def _highlight_atm(row):
    is_atm = row["Strike"] == chain.attrs["atm_strike"]
    return ["background-color: #FFF3CD" if is_atm else "" for _ in row]


st.dataframe(styled.style.apply(_highlight_atm, axis=1), width="stretch", height=460)

col_chart1, col_chart2 = st.columns(2)
with col_chart1:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=chain["strike"], y=chain["call_ltp"], name="Call LTP", line=dict(color="#3D5A80")))
    fig_price.add_trace(go.Scatter(x=chain["strike"], y=chain["put_ltp"], name="Put LTP", line=dict(color="#E07A5F")))
    fig_price.add_vline(x=spot, line_dash="dash", annotation_text="Spot")
    fig_price.update_layout(height=380, title="Option premium vs. strike")
    st.plotly_chart(fig_price, width="stretch")
with col_chart2:
    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(x=chain["strike"], y=chain["iv"] * 100, name="IV smile", line=dict(color="#81B29A")))
    fig_smile.add_vline(x=spot, line_dash="dash", annotation_text="Spot")
    fig_smile.update_layout(height=380, title="Illustrative volatility smile (%)")
    st.plotly_chart(fig_smile, width="stretch")
