"""
engine/data_loader.py
----------------------
Real-data integration: NIFTY 50 (Section 8.2 fallback -> now a *primary*
data path when internet access is available).

Two ways in:
  1. Live/historical fetch via yfinance (ticker "^NSEI").
  2. Manual CSV upload (Date + Close columns), for offline use or if
     Yahoo Finance is unreachable / rate-limited.

Both paths return a plain pandas Series of closing prices indexed by
timestamp, plus an inferred `dt` (time-step size in years) so the rest of
the engine (calibration, simulation, backtester) doesn't need to know or
care where the data came from.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd

NIFTY_TICKER = "^NSEI"
BANKNIFTY_TICKER = "^NSEBANK"
TRADING_DAYS_PER_YEAR = 252
TRADING_MINUTES_PER_DAY = 375

# A small default watchlist of common NSE symbols -> Yahoo Finance tickers.
# yfinance convention: NSE-listed stocks take a ".NS" suffix; indices use
# the "^" prefix names Yahoo assigns them (these are fixed, well-known
# mappings, not something that changes with SEBI rules).
DEFAULT_WATCHLIST = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "INFOSYS": "INFY.NS",
    "ICICI BANK": "ICICIBANK.NS",
}


class DataFetchError(RuntimeError):
    """Raised when live data cannot be fetched, with a user-facing message."""


def fetch_live_quotes(yf_symbols: list[str]) -> dict[str, float]:
    """
    Fetch the latest available price for multiple symbols in one batch.
    Uses `Ticker.fast_info` (a lightweight endpoint) per symbol; falls back
    to a short `.history()` pull if fast_info is unavailable for a symbol.
    Missing/failed symbols are simply absent from the returned dict rather
    than raising, so one bad ticker doesn't take down the whole portfolio
    view -- the caller should treat absence as "price unavailable".
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise DataFetchError("The 'yfinance' package is not installed. Run: pip install yfinance") from e

    prices: dict[str, float] = {}
    for sym in yf_symbols:
        try:
            t = yf.Ticker(sym)
            price = None
            try:
                price = t.fast_info.get("lastPrice") or t.fast_info.get("last_price")
            except Exception:
                price = None
            if price is None:
                hist = t.history(period="2d", interval="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            if price is not None:
                prices[sym] = float(price)
        except Exception:
            continue  # leave this symbol out; caller shows "unavailable"
    return prices


class MockLiveFeed:
    """
    Explicit "mock websocket" wrapper: simulates intra-poll tick updates by
    jittering around the last known real price with small Gaussian noise,
    purely so a dashboard has something to animate between real yfinance
    polls (which only refresh every few seconds/minutes) or when the
    market is closed. This is clearly-labelled simulated data, not a real
    market feed -- do not present its output as live prices without that
    label in the UI.
    """

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self._last_prices: dict[str, float] = {}

    def seed_from_real(self, prices: dict[str, float]):
        self._last_prices.update(prices)

    def next_tick(self, symbol: str, vol_bps: float = 3.0) -> float:
        """One simulated tick for `symbol`, jittered from its last price."""
        base = self._last_prices.get(symbol)
        if base is None or not np.isfinite(base):
            return float("nan")
        shock = self._rng.normal(0, vol_bps / 10_000.0)
        new_price = base * (1 + shock)
        self._last_prices[symbol] = new_price
        return new_price


def fetch_nifty_yfinance(period: str = "2y", interval: str = "1d") -> pd.Series:
    """
    Fetch NIFTY 50 (^NSEI) closing prices from Yahoo Finance via yfinance.

    Parameters
    ----------
    period : yfinance period string, e.g. "5d", "1mo", "6mo", "1y", "2y", "5y".
        Note: Yahoo Finance restricts intraday history depth (e.g. 1m bars
        are typically only available for the last ~7-30 days regardless of
        the period requested).
    interval : yfinance interval string, e.g. "1d", "1h", "30m", "5m".

    Raises
    ------
    DataFetchError if yfinance is not installed, there is no network
    access, or Yahoo Finance returns no data for the requested ticker.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise DataFetchError(
            "The 'yfinance' package is not installed. Run: pip install yfinance"
        ) from e

    try:
        ticker = yf.Ticker(NIFTY_TICKER)
        hist = ticker.history(period=period, interval=interval, auto_adjust=True)
    except Exception as e:
        raise DataFetchError(
            f"Could not fetch NIFTY 50 data from Yahoo Finance ({e}). "
            "Check your internet connection, or use the CSV upload option instead."
        ) from e

    if hist is None or hist.empty:
        raise DataFetchError(
            "Yahoo Finance returned no data for NIFTY 50 (^NSEI) with the "
            "requested period/interval. Try a shorter period (intraday history "
            "is limited to the last few weeks) or use the CSV upload option."
        )

    close = hist["Close"].dropna()
    close.name = "Close"
    return close


def load_csv_upload(file_obj) -> pd.Series:
    """
    Parse an uploaded CSV into a closing-price Series. Accepts common NSE /
    broker export column-name variants (case-insensitive): a date-like
    column ('Date', 'Datetime', 'Timestamp') and a price column ('Close',
    'close', 'Adj Close', 'LTP').
    """
    raw = file_obj.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.strip() for c in df.columns]

    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")), None)
    close_col = next((c for c in df.columns if c.lower() in ("close", "adj close", "ltp", "price")), None)

    if date_col is None or close_col is None:
        raise DataFetchError(
            f"Could not find recognisable Date/Close columns in the uploaded CSV "
            f"(found columns: {list(df.columns)}). Expected something like "
            "'Date' and 'Close'."
        )

    # Try both ISO (YYYY-MM-DD) and day-first (DD-MM-YYYY, common in NSE
    # exports) parsing, and keep whichever loses fewer rows to NaT --
    # forcing one or the other silently drops valid rows in the other format.
    parsed_iso = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
    parsed_dayfirst = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    if parsed_iso.notna().sum() >= parsed_dayfirst.notna().sum():
        df[date_col] = parsed_iso
    else:
        df[date_col] = parsed_dayfirst

    df = df.dropna(subset=[date_col, close_col]).sort_values(date_col)
    series = pd.Series(df[close_col].astype(float).to_numpy(), index=df[date_col], name="Close")
    if len(series) < 10:
        raise DataFetchError("Uploaded CSV has fewer than 10 valid rows after parsing.")
    return series


def infer_dt_years(index: pd.DatetimeIndex) -> float:
    """
    Infer the average time-step size in years from a price series' index,
    used to convert calibrated per-tick parameters into annualised units.
    Falls back to a daily assumption if the index has fewer than 2 points
    or an unparseable frequency.
    """
    if len(index) < 2:
        return 1 / TRADING_DAYS_PER_YEAR
    diffs = pd.Series(index).diff().dropna().dt.total_seconds()
    median_seconds = float(diffs.median())
    if median_seconds >= 12 * 3600:  # roughly daily or coarser
        return 1 / TRADING_DAYS_PER_YEAR
    # Intraday: express the step as a fraction of a 375-minute trading day.
    minutes = median_seconds / 60.0
    return (minutes / TRADING_MINUTES_PER_DAY) / TRADING_DAYS_PER_YEAR


def slice_sessions(prices: np.ndarray, session_len: int, n_sessions: int, stride: int | None = None):
    """
    Slice a long real price series into `n_sessions` overlapping (or
    strided) windows of length `session_len`, most-recent-first, for use
    as historical "sessions" in the backtester (in place of Monte Carlo
    simulated paths).
    """
    stride = stride or max(1, session_len // 4)
    n = len(prices)
    windows = []
    start = n - session_len
    while start >= 0 and len(windows) < n_sessions:
        windows.append(prices[start:start + session_len])
        start -= stride
    return windows
