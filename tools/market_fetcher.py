import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timezone
from typing import Optional


class MarketDataFetcher:
    """
    Fetches real OHLCV price history and computes technical indicators
    for a given ticker using yfinance and the `ta` library.

    Configurable parameters:
    - period (str)          : yfinance period string for price history.
                              Default '1y'. Supports '6mo', '1y', '2y'.
    - interval (str)        : yfinance interval. Default '1d' (daily).
    - recent_sessions (int) : Number of most-recent OHLCV rows to include
                              in the recent price action table. Default 10.
    - gap_threshold (float) : Minimum overnight gap size (as fraction of
                              prior close) to flag as a notable gap.
                              Default 0.02 (2%).
    - volume_spike_mult (float): Volume multiple vs 20d avg to flag as
                              an abnormal volume session. Default 2.0.
    """

    def __init__(
        self,
        period: str = "1y",
        interval: str = "1d",
        recent_sessions: int = 10,
        gap_threshold: float = 0.02,
        volume_spike_mult: float = 2.0,
    ):
        self.period = period
        self.interval = interval
        self.recent_sessions = recent_sessions
        self.gap_threshold = gap_threshold
        self.volume_spike_mult = volume_spike_mult

    # ── Formatting helpers ────────────────────────────────────────────────

    @staticmethod
    def _f(value, fmt=".2f", fallback="N/A") -> str:
        try:
            return format(float(value), fmt)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _fp(value, fallback="N/A") -> str:
        """Format as percentage."""
        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _fv(value, fallback="N/A") -> str:
        """Format volume as human-readable."""
        try:
            v = float(value)
            if v >= 1e9:
                return f"{v/1e9:.2f}B"
            if v >= 1e6:
                return f"{v/1e6:.2f}M"
            if v >= 1e3:
                return f"{v/1e3:.1f}K"
            return str(int(v))
        except (TypeError, ValueError):
            return fallback

    # ── Section builders ──────────────────────────────────────────────────

    def _build_price_snapshot(self, info: dict, hist: pd.DataFrame) -> str:
        close = hist["Close"]
        high  = hist["High"]
        low   = hist["Low"]

        current_price = close.iloc[-1]
        prev_close    = close.iloc[-2] if len(close) > 1 else None
        day_change    = (current_price / prev_close - 1) if prev_close else None

        week_high  = high.tail(5).max()
        week_low   = low.tail(5).min()
        month_high = high.tail(21).max()
        month_low  = low.tail(21).min()

        w52_high = info.get("fiftyTwoWeekHigh") or high.max()
        w52_low  = info.get("fiftyTwoWeekLow")  or low.min()
        w52_pos  = (current_price - w52_low) / (w52_high - w52_low) if (w52_high - w52_low) > 0 else None
        w52_chg  = info.get("52WeekChange")

        lines = ["--- PRICE SNAPSHOT ---"]
        lines.append(f"  Current Price      : ${self._f(current_price)}")
        lines.append(f"  Day Change         : {self._fp(day_change)}")
        lines.append(f"  5-Session High/Low : ${self._f(week_high)} / ${self._f(week_low)}")
        lines.append(f"  21-Session High/Low: ${self._f(month_high)} / ${self._f(month_low)}")
        lines.append(f"  52-Week High       : ${self._f(w52_high)}")
        lines.append(f"  52-Week Low        : ${self._f(w52_low)}")
        lines.append(f"  52-Week Position   : {self._fp(w52_pos)} of range (0%=low, 100%=high)")
        lines.append(f"  52-Week Change     : {self._fp(w52_chg)}")
        return "\n".join(lines)

    def _build_moving_averages(self, close: pd.Series, info: dict) -> str:
        current = close.iloc[-1]

        sma20  = close.rolling(20).mean().iloc[-1]
        sma50  = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]
        ema9   = close.ewm(span=9, adjust=False).mean().iloc[-1]
        ema20  = close.ewm(span=20, adjust=False).mean().iloc[-1]

        # Golden / Death cross detection (SMA50 vs SMA200)
        sma50_series  = close.rolling(50).mean()
        sma200_series = close.rolling(200).mean()
        cross_signal = "N/A"
        if len(sma50_series.dropna()) >= 2 and len(sma200_series.dropna()) >= 2:
            if sma50_series.iloc[-2] <= sma200_series.iloc[-2] and sma50_series.iloc[-1] > sma200_series.iloc[-1]:
                cross_signal = "GOLDEN CROSS (bullish)"
            elif sma50_series.iloc[-2] >= sma200_series.iloc[-2] and sma50_series.iloc[-1] < sma200_series.iloc[-1]:
                cross_signal = "DEATH CROSS (bearish)"
            elif sma50_series.iloc[-1] > sma200_series.iloc[-1]:
                cross_signal = "SMA50 above SMA200 (bullish regime)"
            else:
                cross_signal = "SMA50 below SMA200 (bearish regime)"

        lines = ["--- MOVING AVERAGES ---"]
        lines.append(f"  EMA9               : ${self._f(ema9)}  ({'+' if current >= ema9 else ''}{self._f((current/ema9-1)*100)}% vs price)")
        lines.append(f"  EMA20              : ${self._f(ema20)}  ({'+' if current >= ema20 else ''}{self._f((current/ema20-1)*100)}% vs price)")
        lines.append(f"  SMA20              : ${self._f(sma20)}  ({'+' if current >= sma20 else ''}{self._f((current/sma20-1)*100)}% vs price)")
        lines.append(f"  SMA50              : ${self._f(sma50)}  ({'+' if current >= sma50 else ''}{self._f((current/sma50-1)*100)}% vs price)")
        lines.append(f"  SMA200             : ${self._f(sma200)}  ({'+' if current >= sma200 else ''}{self._f((current/sma200-1)*100)}% vs price)")
        lines.append(f"  MA Cross Signal    : {cross_signal}")
        return "\n".join(lines)

    def _build_momentum(self, close: pd.Series, high: pd.Series, low: pd.Series) -> str:
        # RSI
        rsi14 = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        rsi_zone = (
            "OVERBOUGHT (>70)" if rsi14 > 70
            else "OVERSOLD (<30)" if rsi14 < 30
            else "NEUTRAL (30-70)"
        )

        # MACD
        macd_ind  = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd_val  = macd_ind.macd().iloc[-1]
        macd_sig  = macd_ind.macd_signal().iloc[-1]
        macd_hist = macd_ind.macd_diff().iloc[-1]
        macd_cross = (
            "Bullish crossover" if macd_val > macd_sig and macd_ind.macd().iloc[-2] <= macd_ind.macd_signal().iloc[-2]
            else "Bearish crossover" if macd_val < macd_sig and macd_ind.macd().iloc[-2] >= macd_ind.macd_signal().iloc[-2]
            else ("Above signal (bullish)" if macd_val > macd_sig else "Below signal (bearish)")
        )

        # Stochastic
        stoch     = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        stoch_k   = stoch.stoch().iloc[-1]
        stoch_d   = stoch.stoch_signal().iloc[-1]
        stoch_zone = (
            "OVERBOUGHT (>80)" if stoch_k > 80
            else "OVERSOLD (<20)" if stoch_k < 20
            else "NEUTRAL"
        )

        # ADX
        adx_ind    = ta.trend.ADXIndicator(high, low, close, window=14)
        adx_val    = adx_ind.adx().iloc[-1]
        adx_plus   = adx_ind.adx_pos().iloc[-1]
        adx_minus  = adx_ind.adx_neg().iloc[-1]
        trend_str  = (
            "STRONG TREND (>25)" if adx_val > 25
            else "WEAK/NO TREND (<25)"
        )
        trend_dir  = "Bullish (+DI > -DI)" if adx_plus > adx_minus else "Bearish (-DI > +DI)"

        lines = ["--- MOMENTUM INDICATORS ---"]
        lines.append(f"  RSI(14)            : {self._f(rsi14)}  [{rsi_zone}]")
        lines.append(f"  MACD               : {self._f(macd_val, '.4f')}  Signal: {self._f(macd_sig, '.4f')}  Hist: {self._f(macd_hist, '.4f')}")
        lines.append(f"  MACD Status        : {macd_cross}")
        lines.append(f"  Stoch %K/%D        : {self._f(stoch_k)} / {self._f(stoch_d)}  [{stoch_zone}]")
        lines.append(f"  ADX(14)            : {self._f(adx_val)}  [{trend_str}]  Direction: {trend_dir}")
        return "\n".join(lines)

    def _build_volatility(self, close: pd.Series, high: pd.Series, low: pd.Series) -> str:
        # Bollinger Bands
        bb        = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper  = bb.bollinger_hband().iloc[-1]
        bb_mid    = bb.bollinger_mavg().iloc[-1]
        bb_lower  = bb.bollinger_lband().iloc[-1]
        bb_pband  = bb.bollinger_pband().iloc[-1]
        bb_wband  = bb.bollinger_wband().iloc[-1]
        bb_pos    = (
            "Near upper band (extended)" if bb_pband > 0.8
            else "Near lower band (compressed)" if bb_pband < 0.2
            else "Mid-band (neutral)"
        )

        # ATR
        atr14 = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        atr_pct = atr14 / close.iloc[-1] * 100

        # Historical volatility (20-day annualised)
        log_returns = close.pct_change().dropna()
        hv20 = log_returns.tail(20).std() * (252 ** 0.5) * 100

        lines = ["--- VOLATILITY ---"]
        lines.append(f"  Bollinger Upper    : ${self._f(bb_upper)}")
        lines.append(f"  Bollinger Mid      : ${self._f(bb_mid)}")
        lines.append(f"  Bollinger Lower    : ${self._f(bb_lower)}")
        lines.append(f"  BB %B              : {self._f(bb_pband)}  [{bb_pos}]")
        lines.append(f"  BB Width           : {self._f(bb_wband)}")
        lines.append(f"  ATR(14)            : ${self._f(atr14)}  ({self._f(atr_pct)}% of price)")
        lines.append(f"  HV20 (annualised)  : {self._f(hv20)}%")
        return "\n".join(lines)

    def _build_volume(self, close: pd.Series, volume: pd.Series) -> str:
        vol_sma20  = volume.rolling(20).mean()
        vol_sma50  = volume.rolling(50).mean()
        latest_vol = volume.iloc[-1]
        vs_20d     = latest_vol / vol_sma20.iloc[-1] if vol_sma20.iloc[-1] > 0 else None
        vs_50d     = latest_vol / vol_sma50.iloc[-1] if vol_sma50.iloc[-1] > 0 else None

        # Flag abnormal volume sessions in the last 20 days
        recent_vol  = volume.tail(20)
        spikes      = recent_vol[recent_vol > vol_sma20.tail(20) * self.volume_spike_mult]
        spike_dates = [str(d)[:10] for d in spikes.index.tolist()]

        # OBV trend (simple: last 5 vs 20 sessions)
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        obv_trend = "Rising (accumulation)" if obv.iloc[-1] > obv.iloc[-5] else "Falling (distribution)"

        lines = ["--- VOLUME ANALYSIS ---"]
        lines.append(f"  Latest Volume      : {self._fv(latest_vol)}")
        lines.append(f"  20d Avg Volume     : {self._fv(vol_sma20.iloc[-1])}")
        lines.append(f"  50d Avg Volume     : {self._fv(vol_sma50.iloc[-1])}")
        lines.append(f"  Vol vs 20d Avg     : {self._f(vs_20d)}x")
        lines.append(f"  Vol vs 50d Avg     : {self._f(vs_50d)}x")
        lines.append(f"  OBV Trend (5d)     : {obv_trend}")
        if spike_dates:
            lines.append(f"  Volume Spikes (>{self.volume_spike_mult}x 20d avg, last 20 sessions):")
            for d in spike_dates[-5:]:  # show up to 5 most recent
                lines.append(f"    - {d}")
        else:
            lines.append(f"  Volume Spikes      : None in last 20 sessions")
        return "\n".join(lines)

    def _build_support_resistance(self, high: pd.Series, low: pd.Series, close: pd.Series) -> str:
        """Derive support/resistance levels from rolling highs/lows."""
        r1_20  = high.tail(20).max()
        r2_50  = high.tail(50).max()
        r3_252 = high.max()
        s1_20  = low.tail(20).min()
        s2_50  = low.tail(50).min()
        s3_252 = low.min()
        current = close.iloc[-1]

        lines = ["--- SUPPORT / RESISTANCE (Rolling High/Low Levels) ---"]
        lines.append(f"  Resistance R1 (20d): ${self._f(r1_20)}  ({'+' if current < r1_20 else ''}{self._f((r1_20/current-1)*100)}% from price)")
        lines.append(f"  Resistance R2 (50d): ${self._f(r2_50)}  ({'+' if current < r2_50 else ''}{self._f((r2_50/current-1)*100)}% from price)")
        lines.append(f"  Resistance R3 (52w): ${self._f(r3_252)}  ({'+' if current < r3_252 else ''}{self._f((r3_252/current-1)*100)}% from price)")
        lines.append(f"  Support   S1 (20d) : ${self._f(s1_20)}  ({self._f((current/s1_20-1)*100)}% above)")
        lines.append(f"  Support   S2 (50d) : ${self._f(s2_50)}  ({self._f((current/s2_50-1)*100)}% above)")
        lines.append(f"  Support   S3 (52w) : ${self._f(s3_252)}  ({self._f((current/s3_252-1)*100)}% above)")
        return "\n".join(lines)

    def _build_gap_analysis(self, hist: pd.DataFrame) -> str:
        """Detect notable overnight gaps in recent history."""
        df = hist.copy()
        df["prev_close"] = df["Close"].shift(1)
        df["gap_pct"]    = (df["Open"] - df["prev_close"]) / df["prev_close"]
        gaps = df[df["gap_pct"].abs() >= self.gap_threshold].tail(10)

        lines = ["--- NOTABLE GAPS (last 252 sessions, threshold ≥"
                 f" {self.gap_threshold*100:.0f}%) ---"]
        if gaps.empty:
            lines.append(f"  No gaps ≥ {self.gap_threshold*100:.0f}% detected.")
        else:
            for dt, row in gaps.iterrows():
                direction = "GAP UP  " if row["gap_pct"] > 0 else "GAP DOWN"
                lines.append(
                    f"  {str(dt)[:10]}  {direction}  "
                    f"{'+' if row['gap_pct'] > 0 else ''}{row['gap_pct']*100:.2f}%  "
                    f"(Open: ${row['Open']:.2f}, Prev Close: ${row['prev_close']:.2f})"
                )
        return "\n".join(lines)

    def _build_recent_sessions(self, hist: pd.DataFrame) -> str:
        """Show the most recent N sessions as an OHLCV table."""
        df = hist[["Open", "High", "Low", "Close", "Volume"]].tail(self.recent_sessions).copy()
        df = df.sort_index(ascending=False)

        lines = [f"--- RECENT {self.recent_sessions} SESSIONS (newest first) ---"]
        lines.append(f"  {'Date':<12}  {'Open':>8}  {'High':>8}  {'Low':>8}  {'Close':>8}  {'Volume':>10}  {'Change':>8}")
        lines.append("  " + "-" * 72)

        prev_close = None
        for dt, row in df.iterrows():
            chg = ""
            if prev_close is not None:
                chg = f"{(row['Close']/prev_close - 1)*100:+.2f}%"
            prev_close = row["Close"]
            lines.append(
                f"  {str(dt)[:10]:<12}  "
                f"${row['Open']:>7.2f}  "
                f"${row['High']:>7.2f}  "
                f"${row['Low']:>7.2f}  "
                f"${row['Close']:>7.2f}  "
                f"{self._fv(row['Volume']):>10}  "
                f"{chg:>8}"
            )
        return "\n".join(lines)

    # ── Public API ────────────────────────────────────────────────────────

    def get_market_data(self, ticker: str) -> dict:
        """
        Fetch and compute all market/technical data for a ticker.
        Returns a dict of pre-formatted string sections.
        """
        t    = yf.Ticker(ticker)
        info = t.info or {}
        hist = t.history(period=self.period, interval=self.interval)

        if hist.empty:
            return {"error": f"No price history available for {ticker}."}

        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        volume = hist["Volume"]

        return {
            "price_snapshot":    self._build_price_snapshot(info, hist),
            "moving_averages":   self._build_moving_averages(close, info),
            "momentum":          self._build_momentum(close, high, low),
            "volatility":        self._build_volatility(close, high, low),
            "volume":            self._build_volume(close, volume),
            "support_resistance":self._build_support_resistance(high, low, close),
            "gap_analysis":      self._build_gap_analysis(hist),
            "recent_sessions":   self._build_recent_sessions(hist),
        }

    def get_formatted_context(self, ticker: str) -> str:
        """
        Return a single pre-formatted string with all market/technical data,
        ready to be injected directly into an LLM prompt context.
        """
        data = self.get_market_data(ticker)

        if "error" in data:
            return f"=== MARKET DATA: {ticker.upper()} ===\nERROR: {data['error']}"

        meta = (
            f"Ticker: {ticker.upper()} | "
            f"Period: {self.period} | "
            f"Interval: {self.interval} | "
            f"Data as of: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        sections = [
            f"=== MARKET & TECHNICAL DATA: {ticker.upper()} ===",
            meta,
            "",
            data["price_snapshot"],
            "",
            data["moving_averages"],
            "",
            data["momentum"],
            "",
            data["volatility"],
            "",
            data["volume"],
            "",
            data["support_resistance"],
            "",
            data["gap_analysis"],
            "",
            data["recent_sessions"],
        ]
        return "\n".join(sections)
