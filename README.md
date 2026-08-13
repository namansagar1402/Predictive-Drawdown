# Predictive-Drawdown

# Indian Equity & Options Portfolio Analyzer (+ Predictive Exit Strategy Engine)

Multi-page Streamlit app: a live/delayed NIFTY 50 macro dashboard,
multi-asset portfolio tracker, and synthetic NSE options chain, built on
top of the original B.Tech Minor Project (Phase I) / Major Project (Phase
II) predictive-exit synopsis engine, which is preserved as the
**Strategy Engine** page.

**This is a read-only market-data / analytics / research tool. No page
places orders or connects to a brokerage account.**

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).
Streamlit auto-discovers the `pages/` folder and shows **Home**,
**Portfolio**, **Options Chain**, and **Strategy Engine** in the sidebar.

## Project structure

```
predictive_exit_app/
├── app.py                       # Home page: NIFTY 50 macro dashboard + portfolio snapshot
├── pages/
│   ├── 1_Portfolio.py            # Multi-asset position entry + live mark-to-market
│   ├── 2_Options_Chain.py        # Synthetic NSE-style options chain viewer
│   └── 3_Strategy_Engine.py      # The original predictive-exit simulator (5 tabs, unchanged)
├── requirements.txt
├── test_engine.py                # standalone smoke test for the engine (no UI)
└── engine/
    ├── simulation.py             # Phase I — GBM + Merton jump-diffusion, calibration (§8.2–8.5)
    ├── greeks.py                 # Phase II — Black-Scholes Greeks + velocity monitor (§8.6)
    ├── risk.py                   # Phase II — rolling micro-interval CVaR (§8.7)
    ├── trigger.py                # Phase II — composite logistic trigger + ROC threshold (§8.8)
    ├── backtester.py             # Event-driven, queue-aware execution simulation (synthetic AND real data)
    ├── data_loader.py            # Live/historical NIFTY 50 + multi-symbol quotes, CSV fallback, mock feed
    ├── market_calendar.py        # NSE trading hours + expiry-date conventions (NEW)
    ├── options_chain.py          # Synthetic NSE-style options chain (NEW)
    ├── portfolio.py              # Multi-asset position tracking + mark-to-market (NEW)
    └── metrics.py                # Evaluation metrics + statistical significance (§12)
```

Section numbers refer to the accompanying project synopsis document.

## What this is (and isn't)

Three honesty points to keep in mind, especially for a thesis defence:

1. **"Live" data is delayed polling, not a real-time feed.** `yfinance`
   wraps Yahoo Finance, which typically lags NSE by a few minutes. The
   `MockLiveFeed` class in `data_loader.py` is an explicit, clearly-labelled
   simulated tick generator (Gaussian jitter around the last real price) —
   useful for demoing a "live-looking" UI when the market is closed or
   between polls, but it is not real market data and the UI should never
   present it as such.
2. **The options chain is model output, not a live NSE feed.** No free,
   reliable, ToS-safe source of real NSE option-chain data exists for a
   prototype like this. `engine/options_chain.py` builds a Black-Scholes
   chain off the live spot with real NSE strike/expiry conventions, and
   says so directly in the UI. Swap in a licensed vendor (Kite Connect,
   Breeze, Sensibull's API, etc.) there if you get access to one.
3. **NSE expiry-day conventions changed on 1 September 2025** (NIFTY
   weekly expiry moved Thursday → Tuesday; Bank Nifty/Fin Nifty/Midcap
   Nifty lost their weekly expiries entirely, per a Nov-2024 SEBI
   circular). `engine/market_calendar.py` encodes the *current* (mid-2026)
   rules with an explicit comment that this has changed before and should
   be re-verified against NSE's live circular before being trusted beyond
   a prototype.

## NIFTY 50 integration (Strategy Engine page)

The sidebar's **📡 Data Source** panel offers three modes:

1. **Synthetic (Monte Carlo)** — the original mode: you set μ, σ, λ, jump
   parameters by hand and everything is simulated.
2. **NIFTY 50 — Yahoo Finance** — fetches real `^NSEI` history via
   `yfinance` (needs internet access on your machine; blocked in some
   sandboxed/offline environments). Pick a period + bar interval, click
   **Fetch**, then **🧮 Calibrate GBM/jump-diffusion from this data** in
   the sidebar to auto-populate μ, σ, λ, jump parameters, and S₀ from the
   real series (using the Section 8.5 calibration procedure).
3. **Upload CSV** — same calibration flow, but from a CSV you provide
   (columns: a Date/Datetime column + a Close/LTP/Price column). Use this
   if Yahoo Finance is unreachable, rate-limited, or you have your own
   NSE/broker export.

Once real data is loaded, the **NIFTY 50 Real Data** tab lets you:
- **Replay a single historical window** (e.g. a specific volatile week)
  through the exact same Greek/CVaR/composite-trigger/execution pipeline
  used in the synthetic tabs, and see where each strategy would have
  fired.
- **Run a historical backtest** across many overlapping real windows
  (`engine.data_loader.slice_sessions`), producing the same latency/
  slippage/statistical-significance table as the synthetic Backtest
  Comparison tab — but on real NIFTY 50 price action. **This is the
  result to actually quote in your thesis defence**, with the synthetic
  Monte Carlo numbers as a supporting robustness check, not the other
  way around.

If `yfinance` can't reach Yahoo Finance (no internet, corporate firewall,
etc.), the app shows a clear on-screen error and falls back gracefully —
it will not crash. Use the CSV upload option in that case.

## What each page/tab shows

- **Home** — NIFTY 50 macro reference chart + realised vol, approximate
  market-open status, next weekly/monthly expiry, and a portfolio snapshot
  if you've added positions.
- **Portfolio** — add/remove multi-asset positions (equity, index, call/put
  options), see live mark-to-market, P&L by position, and aggregate
  Delta/Gamma.
- **Options Chain** — pick an underlying + expiry, view the synthetic
  chain with Greeks, an IV-smile chart, and a premium-vs-strike chart.
- **Strategy Engine** (the original prototype, unchanged):
  1. *Live Risk Dashboard* — one simulated session with breach-probability
     overlay, CVaR gauge, Greek-velocity readout, trigger points.
  2. *Execution & Alert Log* — timestamped trigger→fill event log +
     illustrative synthetic order-book depth ladder.
  3. *Backtest Comparison* — Monte Carlo comparison of static/ATR/predictive
     exits with the paired Wilcoxon test + bootstrap CI (§12).
  4. *Phase I Calibration* — simulate-then-recalibrate correctness check.
  5. *NIFTY 50 Real Data* — replay/backtest on real historical NIFTY 50.

## Key modelling decisions and limitations (read before citing results)

- **No real Level-2 data is used.** Per synopsis §8.2/§7, all data is
  simulated. The execution/queue-latency model in `backtester.py` is a
  deliberately simple, clearly-labelled stand-in: static/ATR triggers are
  assumed to fire at "crowded" levels (longer simulated queue), the
  predictive trigger is assumed to fire idiosyncratically ahead of the
  crowd (shorter queue). Treat absolute latency numbers as illustrative;
  only the *relative* comparison between strategies is the point.
- **Drift (μ) is very hard to estimate**, even from 2 years of daily data
  — its standard error scales with 1/√T regardless of sampling frequency.
  Volatility (σ) calibrates much better. This is a real, well-known
  property of GBM, not a bug — see the Phase I Calibration tab.
- **Jump detection is imperfect by design.** The threshold-based filter
  uses a robust (MAD-based) local volatility estimate rather than a plain
  rolling standard deviation (which would be inflated by the jumps it's
  trying to find), but will still miss jumps that are close in size to
  normal daily noise. Use the "Jump-detection threshold" slider on the
  Calibration tab to see this trade-off directly.
- **Constant volatility within each calibration window** is assumed
  (no GARCH / stochastic volatility) — noted in synopsis §7 as a
  Model Limitation and a natural extension for future work.
- **Composite trigger weights are fit per-session** on a synthetic
  breach/no-breach labelling of that same session (since no persistent
  historical label set exists yet). For a real thesis defense, fit the
  weights once on a large held-out calibration set and freeze them before
  evaluating on a separate backtest set, to avoid the look-ahead bias the
  synopsis explicitly warns against (§8.8).

## Suggested next steps for the thesis

- Replace the synthetic tick generator with real historical Level-2 or
  1-minute OHLC data (e.g. NSE) once available, and re-run calibration.
- Freeze composite-trigger weights on a dedicated calibration window
  (see limitation above) rather than re-fitting per session.
- Extend `simulation.py` with a GARCH(1,1) volatility layer for the
  "Future Work" item noted in the synopsis.
- Add unit tests (`pytest`) around `engine/` for the Major Project
  software-engineering deliverable.
- If you get access to a licensed data vendor or broker API (Kite Connect,
  Breeze, etc.), swap it into `data_loader.fetch_live_quotes` and
  `options_chain.generate_chain` — both are already structured so the rest
  of the app doesn't need to change shape.
- `market_calendar.py`'s holiday handling is approximate (weekday-only,
  no NSE holiday list). Wire in NSE's published annual trading-holiday
  calendar for exact expiry-shift behaviour if this ever needs to be
  more than a prototype.
- Add basic rate-limiting/backoff around `yfinance` calls before pointing
  this at a public deployment — Yahoo Finance will throttle aggressive
  polling.
