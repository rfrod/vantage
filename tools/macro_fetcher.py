"""
MacroFetcher — collects macro and sector context for a given ticker.

Data layers (in order of dependency):
  1. Ticker metadata       : sector, industry, country via yfinance
  2. Sector ETF trend      : price performance and relative strength vs SPY
  3. Macro indicators      : ^TNX, ^VIX, DX-Y.NYB, GC=F, CL=F, ^GSPC, ^NDX, ^IRX
  4. Macro & sector news   : reuses NewsFetcher on sector ETF + macro keywords
  5. FRED economic data    : CPI, Fed Funds Rate, unemployment, GDP (optional — requires FRED_API_KEY)

FRED is silently skipped if FRED_API_KEY env var is absent or invalid.
A configurable rate limiter (default 120 calls/min) is applied to all FRED requests.
"""

import os
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List

# NewsFetcher is reused for macro/sector news
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.news_fetcher import NewsFetcher


# ── Sector ETF map ────────────────────────────────────────────────────────────
SECTOR_ETF_MAP: Dict[str, str] = {
    "Technology":               "XLK",
    "Financial Services":       "XLF",
    "Financials":               "XLF",
    "Healthcare":               "XLV",
    "Health Care":              "XLV",
    "Energy":                   "XLE",
    "Consumer Cyclical":        "XLY",
    "Consumer Discretionary":   "XLY",
    "Consumer Defensive":       "XLP",
    "Consumer Staples":         "XLP",
    "Industrials":              "XLI",
    "Basic Materials":          "XLB",
    "Materials":                "XLB",
    "Utilities":                "XLU",
    "Real Estate":              "XLRE",
    "Communication Services":   "XLC",
}

# ── FRED series to fetch ──────────────────────────────────────────────────────
FRED_SERIES: Dict[str, str] = {
    "CPI YoY (%)":              "CPIAUCSL",      # Consumer Price Index (monthly)
    "Core CPI YoY (%)":        "CPILFESL",      # Core CPI ex food & energy
    "Fed Funds Rate (%)":      "FEDFUNDS",      # Effective Fed Funds Rate
    "Unemployment Rate (%)":   "UNRATE",        # US Unemployment Rate
    "GDP Growth QoQ (%)":      "A191RL1Q225SBEA", # Real GDP growth (quarterly)
    "10Y-2Y Spread (pp)":      None,            # Computed from TNX and IRX
    "PCE Inflation YoY (%)":   "PCEPI",         # PCE Price Index
}


class _FredRateLimiter:
    """Token-bucket rate limiter for FRED API calls."""

    def __init__(self, calls_per_min: int = 120):
        self._min_interval = 60.0 / max(calls_per_min, 1)
        self._last_call    = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last_call
        sleep_for = self._min_interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_call = time.monotonic()


class MacroFetcher:
    """
    Fetches macro and sector context for a given ticker.

    Configurable parameters:
    - fred_calls_per_min (int)  : Max FRED API calls per minute. Default 120.
    - news_max_age_days (int)   : Max age of macro/sector news articles. Default 7.
    - news_max_articles (int)   : Max articles to fetch per news query. Default 10.
    - history_period (str)      : yfinance period for ETF/macro price history. Default '3mo'.
    - fred_api_key (str|None)   : FRED API key. Falls back to FRED_API_KEY env var.
                                  If absent, FRED section is silently skipped.
    """

    def __init__(
        self,
        fred_calls_per_min: int = 120,
        news_max_age_days: int = 7,
        news_max_articles: int = 10,
        history_period: str = "3mo",
        fred_api_key: Optional[str] = None,
    ):
        self.fred_calls_per_min = fred_calls_per_min
        self.news_max_age_days  = news_max_age_days
        self.news_max_articles  = news_max_articles
        self.history_period     = history_period

        # Resolve FRED key
        self._fred_key = fred_api_key or os.environ.get("FRED_API_KEY", "")
        self._fred_limiter = _FredRateLimiter(fred_calls_per_min)
        self._fred_client  = None
        if self._fred_key:
            try:
                from fredapi import Fred
                self._fred_client = Fred(api_key=self._fred_key)
            except Exception:
                self._fred_client = None

        self._news_fetcher = NewsFetcher(
            max_age_days=news_max_age_days,
        )
        self._news_max_articles = news_max_articles

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _f(v, fmt=".2f", fallback="N/A") -> str:
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _pct_change(series: pd.Series, periods: int) -> Optional[float]:
        try:
            if len(series) < periods + 1:
                return None
            return float((series.iloc[-1] / series.iloc[-periods - 1] - 1) * 100)
        except Exception:
            return None

    def _price_summary(self, symbol: str, label: str) -> Dict:
        """Fetch price history for a symbol and compute key stats."""
        try:
            hist = yf.Ticker(symbol).history(period=self.history_period)
            if hist.empty:
                return {"error": f"No data for {symbol}"}
            close = hist["Close"]
            last  = float(close.iloc[-1])
            return {
                "symbol":   symbol,
                "label":    label,
                "last":     last,
                "1d_chg":   self._pct_change(close, 1),
                "5d_chg":   self._pct_change(close, 5),
                "1mo_chg":  self._pct_change(close, 21),
                "3mo_chg":  self._pct_change(close, 63),
                "52w_high": float(hist["High"].max()),
                "52w_low":  float(hist["Low"].min()),
                "pos_52w":  (last - float(hist["Low"].min())) /
                            max(float(hist["High"].max()) - float(hist["Low"].min()), 0.01) * 100,
            }
        except Exception as e:
            return {"error": str(e), "symbol": symbol}

    # ── Section builders ──────────────────────────────────────────────────────

    def _build_ticker_meta(self, ticker_symbol: str) -> tuple:
        """Return (meta_block_str, sector, sector_etf)."""
        lines = ["--- TICKER MACRO CONTEXT ---"]
        try:
            info   = yf.Ticker(ticker_symbol).info
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")
            country  = info.get("country", "Unknown")
            exchange = info.get("exchange", "Unknown")
            etf      = SECTOR_ETF_MAP.get(sector, "SPY")
            lines.append(f"  Ticker    : {ticker_symbol.upper()}")
            lines.append(f"  Sector    : {sector}  (ETF proxy: {etf})")
            lines.append(f"  Industry  : {industry}")
            lines.append(f"  Country   : {country}  |  Exchange: {exchange}")
            return "\n".join(lines), sector, etf
        except Exception as e:
            return f"  Error fetching ticker meta: {e}", "Unknown", "SPY"

    def _build_sector_trend(self, sector_etf: str, sector: str) -> str:
        """Sector ETF performance vs SPY."""
        lines = [f"--- SECTOR TREND ({sector_etf} vs SPY) ---"]
        etf_data = self._price_summary(sector_etf, f"{sector} ETF")
        spy_data = self._price_summary("SPY", "S&P 500")

        for d in [etf_data, spy_data]:
            if "error" in d:
                lines.append(f"  {d.get('symbol','?')}: {d['error']}")
                continue
            lines.append(f"\n  [{d['label']} — {d['symbol']}]")
            lines.append(f"    Last Price    : ${self._f(d['last'])}")
            lines.append(f"    1D Change     : {self._f(d['1d_chg'], '+.2f')}%")
            lines.append(f"    5D Change     : {self._f(d['5d_chg'], '+.2f')}%")
            lines.append(f"    1M Change     : {self._f(d['1mo_chg'], '+.2f')}%")
            lines.append(f"    3M Change     : {self._f(d['3mo_chg'], '+.2f')}%")
            lines.append(f"    52W High      : ${self._f(d['52w_high'])}")
            lines.append(f"    52W Low       : ${self._f(d['52w_low'])}")
            lines.append(f"    52W Position  : {self._f(d['pos_52w'], '.1f')}%")

        # Relative strength: ETF vs SPY (1M and 3M)
        if "error" not in etf_data and "error" not in spy_data:
            rs_1m = (etf_data.get("1mo_chg") or 0) - (spy_data.get("1mo_chg") or 0)
            rs_3m = (etf_data.get("3mo_chg") or 0) - (spy_data.get("3mo_chg") or 0)
            rs_signal = (
                "OUTPERFORMING" if rs_1m > 1 else
                "UNDERPERFORMING" if rs_1m < -1 else
                "IN LINE"
            )
            lines.append(f"\n  Relative Strength vs SPY:")
            lines.append(f"    1M RS         : {self._f(rs_1m, '+.2f')}pp  [{rs_signal}]")
            lines.append(f"    3M RS         : {self._f(rs_3m, '+.2f')}pp")

        return "\n".join(lines)

    def _build_macro_indicators(self) -> str:
        """Key macro market indicators from yfinance."""
        lines = ["--- MACRO MARKET INDICATORS ---"]

        indicators = [
            ("^GSPC",    "S&P 500"),
            ("^NDX",     "NASDAQ 100"),
            ("^VIX",     "VIX (Fear Gauge)"),
            ("^TNX",     "10Y Treasury Yield (%)"),
            ("^IRX",     "3M Treasury Yield (%)"),
            ("DX-Y.NYB", "US Dollar Index"),
            ("GC=F",     "Gold (USD/oz)"),
            ("CL=F",     "WTI Crude Oil (USD/bbl)"),
        ]

        tnx_last = None
        irx_last = None

        for symbol, label in indicators:
            d = self._price_summary(symbol, label)
            if "error" in d:
                lines.append(f"  {label:<28}: N/A  ({d['error']})")
                continue
            last = d["last"]
            chg_1d = d["1d_chg"]
            chg_1m = d["1mo_chg"]
            lines.append(
                f"  {label:<28}: {self._f(last, '.2f')}  "
                f"1D: {self._f(chg_1d, '+.2f')}%  "
                f"1M: {self._f(chg_1m, '+.2f')}%"
            )
            if symbol == "^TNX":
                tnx_last = last
            if symbol == "^IRX":
                irx_last = last / 100 * 10 if last else None  # IRX is in annualised %

        # Yield curve (10Y - 3M spread)
        if tnx_last is not None and irx_last is not None:
            spread = tnx_last - irx_last
            curve_signal = (
                "INVERTED (recession signal)" if spread < 0 else
                "FLAT (caution)" if spread < 0.5 else
                "NORMAL (positive slope)"
            )
            lines.append(f"\n  Yield Curve (10Y - 3M)  : {self._f(spread, '+.2f')}pp  [{curve_signal}]")

        # VIX regime
        vix_data = self._price_summary("^VIX", "VIX")
        if "error" not in vix_data:
            vix = vix_data["last"]
            vix_regime = (
                "EXTREME FEAR (>40)" if vix > 40 else
                "HIGH FEAR (30–40)" if vix > 30 else
                "ELEVATED (20–30)" if vix > 20 else
                "NORMAL (<20)"
            )
            lines.append(f"  VIX Regime              : {vix_regime}")

        return "\n".join(lines)

    def _build_fred_data(self) -> str:
        """Optional FRED economic indicators. Silently skipped if no key."""
        if not self._fred_client:
            return (
                "--- FRED ECONOMIC DATA ---\n"
                "  [Skipped — FRED_API_KEY not set. "
                "Register free at https://fred.stlouisfed.org/docs/api/fred/ "
                "and set FRED_API_KEY in your environment to enable this section.]"
            )

        lines = ["--- FRED ECONOMIC DATA ---"]
        series_to_fetch = {
            "CPI YoY (%)":          "CPIAUCSL",
            "Core CPI YoY (%)":     "CPILFESL",
            "Fed Funds Rate (%)":   "FEDFUNDS",
            "Unemployment Rate (%)":"UNRATE",
            "GDP Growth QoQ (%)":   "A191RL1Q225SBEA",
            "PCE Inflation YoY (%)":"PCEPI",
        }

        for label, series_id in series_to_fetch.items():
            try:
                self._fred_limiter.wait()
                data = self._fred_client.get_series(series_id, observation_start="2020-01-01")
                data = data.dropna()
                if data.empty:
                    lines.append(f"  {label:<28}: N/A")
                    continue
                latest_val   = float(data.iloc[-1])
                latest_date  = data.index[-1].strftime("%Y-%m")
                prev_val     = float(data.iloc[-2]) if len(data) > 1 else None
                yoy_val      = float(data.iloc[-13]) if len(data) > 13 else None

                # For CPI and PCE, compute YoY % change
                if series_id in ("CPIAUCSL", "CPILFESL", "PCEPI") and yoy_val:
                    yoy_chg = (latest_val / yoy_val - 1) * 100
                    lines.append(
                        f"  {label:<28}: {self._f(yoy_chg, '.2f')}%  "
                        f"(index: {self._f(latest_val, '.1f')}, as of {latest_date})"
                    )
                else:
                    mom_chg = (latest_val - prev_val) if prev_val is not None else None
                    lines.append(
                        f"  {label:<28}: {self._f(latest_val, '.2f')}%  "
                        f"(MoM: {self._f(mom_chg, '+.2f')}pp, as of {latest_date})"
                    )
            except Exception as e:
                lines.append(f"  {label:<28}: Error — {e}")

        # Fed Funds Rate context
        try:
            self._fred_limiter.wait()
            ff = self._fred_client.get_series("FEDFUNDS", observation_start="2022-01-01").dropna()
            if len(ff) >= 2:
                current = float(ff.iloc[-1])
                peak    = float(ff.max())
                trough  = float(ff.min())
                trend   = "CUTTING" if float(ff.iloc[-1]) < float(ff.iloc[-6]) else \
                          "HIKING"  if float(ff.iloc[-1]) > float(ff.iloc[-6]) else "HOLDING"
                lines.append(f"\n  Fed Policy Cycle:")
                lines.append(f"    Current Rate  : {self._f(current)}%")
                lines.append(f"    Cycle Peak    : {self._f(peak)}%")
                lines.append(f"    Cycle Trough  : {self._f(trough)}%")
                lines.append(f"    Trend (6M)    : {trend}")
        except Exception:
            pass

        return "\n".join(lines)

    def _build_macro_news(self, sector_etf: str, ticker_symbol: str) -> str:
        """Macro and sector news using NewsFetcher on ETF and macro keywords."""
        lines = ["--- MACRO & SECTOR NEWS ---"]

        # Fetch news for the sector ETF
        articles = self._news_fetcher.get_consolidated_news(
            sector_etf, max_articles=self._news_max_articles
        )
        lines.append(f"\n  [Sector ETF News — {sector_etf}]  ({len(articles)} articles)")
        if not articles:
            lines.append("  No recent articles found.")
        else:
            for a in articles:
                lines.append(
                    f"  [{a.get('age_label','?')}] [{a.get('source','?')}] "
                    f"{a.get('title','')}"
                )
                summary = a.get('summary', '')
                if summary:
                    lines.append(f"    {summary[:200]}")

        return "\n".join(lines)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_formatted_context(self, ticker_symbol: str) -> str:
        """
        Fetch and compute all macro/sector context for a ticker.
        Returns a single pre-formatted string ready for LLM injection.
        """
        meta_block, sector, sector_etf = self._build_ticker_meta(ticker_symbol)

        fred_status = (
            "FRED: enabled" if self._fred_client else
            "FRED: disabled (FRED_API_KEY not set)"
        )
        header = (
            f"=== MACRO / SECTOR DATA: {ticker_symbol.upper()} ===\n"
            f"Sector: {sector} | ETF Proxy: {sector_etf} | "
            f"{fred_status} | "
            f"Data as of: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        sections = [
            header,
            "",
            meta_block,
            "",
            self._build_sector_trend(sector_etf, sector),
            "",
            self._build_macro_indicators(),
            "",
            self._build_fred_data(),
            "",
            self._build_macro_news(sector_etf, ticker_symbol),
        ]
        return "\n".join(sections)
