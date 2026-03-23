import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, date, timedelta
from typing import Optional, List, Tuple, Dict, Any


class OptionsFetcher:
    """
    Fetches and computes options analytics for a given ticker using yfinance.

    Configurable parameters:
    - expirations_for_pcr (int)  : Number of nearest expirations to aggregate
                                   for Put/Call ratio. Default 8.
    - oi_table_rows (int)        : Number of strikes to show in the OI
                                   distribution table. Default 10.
    - top_walls (int)            : Number of top call/put OI walls to report.
                                   Default 5.
    - hv_period (str)            : yfinance period for historical volatility
                                   used in IV Rank estimation. Default '1y'.
    - medium_dte_target (int)    : Target DTE for the "medium-term" expiry
                                   (used for ATM IV and skew). Default 30.
    """

    def __init__(
        self,
        expirations_for_pcr: int = 8,
        oi_table_rows: int = 10,
        top_walls: int = 5,
        hv_period: str = "1y",
        medium_dte_target: int = 30,
    ):
        self.expirations_for_pcr = expirations_for_pcr
        self.oi_table_rows       = oi_table_rows
        self.top_walls           = top_walls
        self.hv_period           = hv_period
        self.medium_dte_target   = medium_dte_target

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _f(v, fmt=".2f", fallback="N/A") -> str:
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _fv(v, fallback="N/A") -> str:
        try:
            v = float(v)
            if v >= 1e6:
                return f"{v/1e6:.2f}M"
            if v >= 1e3:
                return f"{v/1e3:.1f}K"
            return str(int(v))
        except (TypeError, ValueError):
            return fallback

    def _get_spot(self, ticker: yf.Ticker) -> float:
        try:
            price = ticker.info.get("regularMarketPrice")
            if price:
                return float(price)
        except Exception:
            pass
        hist = ticker.history(period="2d")
        return float(hist["Close"].iloc[-1])

    def _dte(self, exp_str: str) -> int:
        """Days to expiration from today."""
        exp_date = date.fromisoformat(exp_str)
        return max((exp_date - date.today()).days, 0)

    def _nearest_exp(self, exps: tuple) -> str:
        return exps[0] if exps else None

    def _medium_exp(self, exps: tuple) -> Optional[str]:
        """Return the expiration closest to medium_dte_target DTE."""
        if not exps:
            return None
        best, best_diff = exps[0], abs(self._dte(exps[0]) - self.medium_dte_target)
        for e in exps[1:]:
            diff = abs(self._dte(e) - self.medium_dte_target)
            if diff < best_diff:
                best, best_diff = e, diff
        return best

    def _atm_iv(self, chain_calls: pd.DataFrame, chain_puts: pd.DataFrame, spot: float) -> Dict:
        """Extract ATM IV for calls and puts."""
        result = {"call_iv": None, "put_iv": None, "strike": None}
        try:
            c = chain_calls.copy()
            c["dist"] = (c["strike"] - spot).abs()
            atm_call = c.sort_values("dist").iloc[0]

            p = chain_puts.copy()
            p["dist"] = (p["strike"] - spot).abs()
            atm_put = p.sort_values("dist").iloc[0]

            result["strike"]  = float(atm_call["strike"])
            result["call_iv"] = float(atm_call["impliedVolatility"])
            result["put_iv"]  = float(atm_put["impliedVolatility"])
        except Exception:
            pass
        return result

    def _max_pain(self, calls: pd.DataFrame, puts: pd.DataFrame) -> Optional[float]:
        """
        Compute max pain: the strike at which total option holder losses
        (= market maker gains) are maximised.
        """
        try:
            all_strikes = sorted(
                set(calls["strike"].dropna().tolist()) |
                set(puts["strike"].dropna().tolist())
            )
            min_loss_strike, min_loss = None, float("inf")
            for s in all_strikes:
                call_loss = calls.apply(
                    lambda r: max(0, s - r["strike"]) * (r["openInterest"] or 0), axis=1
                ).sum()
                put_loss = puts.apply(
                    lambda r: max(0, r["strike"] - s) * (r["openInterest"] or 0), axis=1
                ).sum()
                total = call_loss + put_loss
                if total < min_loss:
                    min_loss, min_loss_strike = total, s
            return min_loss_strike
        except Exception:
            return None

    def _hv_annualised(self, ticker: yf.Ticker) -> Optional[float]:
        """20-day annualised historical volatility from price history."""
        try:
            hist = ticker.history(period=self.hv_period)
            log_ret = hist["Close"].pct_change().dropna()
            return float(log_ret.tail(20).std() * (252 ** 0.5) * 100)
        except Exception:
            return None

    # ── Section builders ──────────────────────────────────────────────────

    def _build_overview(self, spot: float, exps: tuple) -> str:
        lines = ["--- OPTIONS OVERVIEW ---"]
        lines.append(f"  Spot Price         : ${self._f(spot)}")
        lines.append(f"  Expirations Listed : {len(exps)}")
        if exps:
            lines.append(f"  Nearest Expiry     : {exps[0]}  (DTE: {self._dte(exps[0])})")
            lines.append(f"  Farthest Expiry    : {exps[-1]}  (DTE: {self._dte(exps[-1])})")
            lines.append(f"  Next 6 Expirations : {', '.join(exps[:6])}")
        return "\n".join(lines)

    def _build_pcr(self, ticker: yf.Ticker, exps: tuple, spot: float) -> str:
        """Aggregate Put/Call ratio across the first N expirations."""
        lines = [f"--- PUT/CALL RATIO (first {min(self.expirations_for_pcr, len(exps))} expirations) ---"]
        total_call_oi = total_put_oi = total_call_vol = total_put_vol = 0.0
        rows = []
        for exp in exps[:self.expirations_for_pcr]:
            try:
                chain = ticker.option_chain(exp)
                c_oi  = float(chain.calls["openInterest"].fillna(0).sum())
                p_oi  = float(chain.puts["openInterest"].fillna(0).sum())
                c_vol = float(chain.calls["volume"].fillna(0).sum())
                p_vol = float(chain.puts["volume"].fillna(0).sum())
                total_call_oi  += c_oi
                total_put_oi   += p_oi
                total_call_vol += c_vol
                total_put_vol  += p_vol
                rows.append((exp, self._dte(exp), c_oi, p_oi, c_vol, p_vol))
            except Exception:
                continue

        lines.append(f"  {'Expiry':<12}  {'DTE':>4}  {'Call OI':>10}  {'Put OI':>10}  {'Call Vol':>10}  {'Put Vol':>10}")
        lines.append("  " + "-" * 64)
        for exp, dte, c_oi, p_oi, c_vol, p_vol in rows:
            lines.append(
                f"  {exp:<12}  {dte:>4}  "
                f"{self._fv(c_oi):>10}  {self._fv(p_oi):>10}  "
                f"{self._fv(c_vol):>10}  {self._fv(p_vol):>10}"
            )
        lines.append("  " + "-" * 64)
        lines.append(
            f"  {'TOTAL':<12}  {'':>4}  "
            f"{self._fv(total_call_oi):>10}  {self._fv(total_put_oi):>10}  "
            f"{self._fv(total_call_vol):>10}  {self._fv(total_put_vol):>10}"
        )

        pcr_oi  = total_put_oi  / total_call_oi  if total_call_oi  > 0 else None
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else None

        pcr_oi_signal = (
            "BEARISH (>1.0)" if pcr_oi and pcr_oi > 1.0
            else "BULLISH (<0.7)" if pcr_oi and pcr_oi < 0.7
            else "NEUTRAL (0.7–1.0)" if pcr_oi else "N/A"
        )
        pcr_vol_signal = (
            "BEARISH (>1.0)" if pcr_vol and pcr_vol > 1.0
            else "BULLISH (<0.7)" if pcr_vol and pcr_vol < 0.7
            else "NEUTRAL (0.7–1.0)" if pcr_vol else "N/A"
        )

        lines.append(f"\n  Put/Call OI Ratio  : {self._f(pcr_oi, '.3f')}  [{pcr_oi_signal}]")
        lines.append(f"  Put/Call Vol Ratio : {self._f(pcr_vol, '.3f')}  [{pcr_vol_signal}]")
        return "\n".join(lines)

    def _build_iv(self, ticker: yf.Ticker, exps: tuple, spot: float, hv20: Optional[float]) -> str:
        """ATM IV, IV skew, and IV Rank estimate."""
        lines = ["--- IMPLIED VOLATILITY ---"]

        nearest_exp = self._nearest_exp(exps)
        medium_exp  = self._medium_exp(exps)

        for label, exp in [("Nearest", nearest_exp), (f"~{self.medium_dte_target}DTE", medium_exp)]:
            if not exp:
                continue
            try:
                chain = ticker.option_chain(exp)
                atm   = self._atm_iv(chain.calls, chain.puts, spot)
                dte   = self._dte(exp)
                call_iv_pct = atm["call_iv"] * 100 if atm["call_iv"] else None
                put_iv_pct  = atm["put_iv"]  * 100 if atm["put_iv"]  else None
                skew = (atm["put_iv"] - atm["call_iv"]) * 100 if (atm["put_iv"] and atm["call_iv"]) else None
                skew_signal = (
                    "Elevated put premium (downside fear)" if skew and skew > 2
                    else "Elevated call premium (upside demand)" if skew and skew < -2
                    else "Balanced skew" if skew is not None else "N/A"
                )
                lines.append(f"\n  [{label} — {exp}, DTE={dte}]")
                lines.append(f"    ATM Strike       : ${self._f(atm['strike'])}")
                lines.append(f"    ATM Call IV      : {self._f(call_iv_pct)}%")
                lines.append(f"    ATM Put  IV      : {self._f(put_iv_pct)}%")
                lines.append(f"    IV Skew (P-C)    : {self._f(skew, '+.1f')}pp  [{skew_signal}]")
            except Exception as e:
                lines.append(f"\n  [{label} — {exp}]: Error fetching chain — {e}")

        # IV Rank estimate (ATM IV vs HV20)
        lines.append("\n  [IV Rank Estimate]")
        if hv20:
            try:
                chain0 = ticker.option_chain(nearest_exp)
                atm0   = self._atm_iv(chain0.calls, chain0.puts, spot)
                atm_iv_pct = (atm0["call_iv"] or 0) * 100
                iv_rank = (atm_iv_pct - hv20 * 0.5) / (hv20 * 1.5 - hv20 * 0.5) * 100
                iv_rank = max(0, min(100, iv_rank))
                iv_env  = (
                    "HIGH IV (>60 — options expensive, favour selling)" if iv_rank > 60
                    else "LOW IV (<40 — options cheap, favour buying)" if iv_rank < 40
                    else "NORMAL IV (40–60)"
                )
                lines.append(f"    HV20 (annualised): {self._f(hv20)}%")
                lines.append(f"    ATM IV (nearest) : {self._f(atm_iv_pct)}%")
                lines.append(f"    IV Rank Est.     : {self._f(iv_rank, '.0f')}  [{iv_env}]")
                lines.append(f"    NOTE: IV Rank is estimated using HV20 as a proxy for the IV range.")
            except Exception:
                lines.append("    IV Rank: could not compute.")
        else:
            lines.append("    IV Rank: HV20 unavailable.")

        return "\n".join(lines)

    def _build_walls_and_pain(self, ticker: yf.Ticker, exps: tuple, spot: float) -> str:
        """Max pain and top OI walls for the most active expiration."""
        lines = ["--- MAX PAIN & GAMMA WALLS ---"]

        # Use the nearest expiry with meaningful OI
        target_exp = None
        for exp in exps[:6]:
            try:
                chain = ticker.option_chain(exp)
                total_oi = (
                    chain.calls["openInterest"].fillna(0).sum() +
                    chain.puts["openInterest"].fillna(0).sum()
                )
                if total_oi > 1000:
                    target_exp = exp
                    break
            except Exception:
                continue

        if not target_exp:
            return "\n".join(lines + ["  Insufficient OI data for wall analysis."])

        try:
            chain = ticker.option_chain(target_exp)
            calls = chain.calls.copy()
            puts  = chain.puts.copy()

            max_pain_strike = self._max_pain(calls, puts)
            mp_dist = ((max_pain_strike - spot) / spot * 100) if max_pain_strike else None

            lines.append(f"\n  Analysis expiry    : {target_exp}  (DTE: {self._dte(target_exp)})")
            lines.append(f"  Max Pain Strike    : ${self._f(max_pain_strike)}  "
                         f"({self._f(mp_dist, '+.2f')}% from spot)")
            lines.append(f"  Spot               : ${self._f(spot)}")

            # Top call OI walls
            top_calls = calls.nlargest(self.top_walls, "openInterest")[
                ["strike", "openInterest", "impliedVolatility", "volume"]
            ]
            lines.append(f"\n  Top {self.top_walls} Call OI Walls (resistance):")
            lines.append(f"    {'Strike':>8}  {'OI':>8}  {'IV':>7}  {'Vol':>8}  {'vs Spot':>8}")
            for _, r in top_calls.iterrows():
                dist = (r["strike"] - spot) / spot * 100
                lines.append(
                    f"    ${r['strike']:>7.1f}  "
                    f"{self._fv(r['openInterest']):>8}  "
                    f"{self._f(r['impliedVolatility']*100, '.1f'):>6}%  "
                    f"{self._fv(r['volume']):>8}  "
                    f"{dist:>+7.2f}%"
                )

            # Top put OI walls
            top_puts = puts.nlargest(self.top_walls, "openInterest")[
                ["strike", "openInterest", "impliedVolatility", "volume"]
            ]
            lines.append(f"\n  Top {self.top_walls} Put OI Walls (support):")
            lines.append(f"    {'Strike':>8}  {'OI':>8}  {'IV':>7}  {'Vol':>8}  {'vs Spot':>8}")
            for _, r in top_puts.iterrows():
                dist = (r["strike"] - spot) / spot * 100
                lines.append(
                    f"    ${r['strike']:>7.1f}  "
                    f"{self._fv(r['openInterest']):>8}  "
                    f"{self._f(r['impliedVolatility']*100, '.1f'):>6}%  "
                    f"{self._fv(r['volume']):>8}  "
                    f"{dist:>+7.2f}%"
                )
        except Exception as e:
            lines.append(f"  Error computing walls: {e}")

        return "\n".join(lines)

    def _build_oi_distribution(self, ticker: yf.Ticker, exps: tuple, spot: float) -> str:
        """OI distribution table around ATM for the nearest active expiry."""
        lines = ["--- OI DISTRIBUTION (ATM ± strikes) ---"]

        target_exp = None
        for exp in exps[:6]:
            try:
                chain = ticker.option_chain(exp)
                total_oi = (
                    chain.calls["openInterest"].fillna(0).sum() +
                    chain.puts["openInterest"].fillna(0).sum()
                )
                if total_oi > 500:
                    target_exp = exp
                    break
            except Exception:
                continue

        if not target_exp:
            return "\n".join(lines + ["  Insufficient OI data."])

        try:
            chain = ticker.option_chain(target_exp)
            calls = chain.calls[["strike", "openInterest", "volume", "impliedVolatility"]].copy()
            puts  = chain.puts[["strike", "openInterest", "volume", "impliedVolatility"]].copy()
            calls.columns = ["strike", "call_oi", "call_vol", "call_iv"]
            puts.columns  = ["strike", "put_oi",  "put_vol",  "put_iv"]

            merged = pd.merge(calls, puts, on="strike", how="outer").fillna(0)
            merged = merged.sort_values("strike")

            # Filter to rows nearest ATM
            merged["dist"] = (merged["strike"] - spot).abs()
            merged = merged.sort_values("dist").head(self.oi_table_rows).sort_values("strike")

            lines.append(f"  Expiry: {target_exp}  |  Spot: ${self._f(spot)}")
            lines.append(f"  {'Strike':>8}  {'Call OI':>9}  {'Call Vol':>9}  {'Call IV':>8}  "
                         f"{'Put OI':>9}  {'Put Vol':>9}  {'Put IV':>8}  {'P/C OI':>7}")
            lines.append("  " + "-" * 82)
            for _, r in merged.iterrows():
                atm_marker = " <-- ATM" if abs(r["strike"] - spot) < 2.5 else ""
                pcr = r["put_oi"] / r["call_oi"] if r["call_oi"] > 0 else float("inf")
                lines.append(
                    f"  ${r['strike']:>7.1f}  "
                    f"{self._fv(r['call_oi']):>9}  "
                    f"{self._fv(r['call_vol']):>9}  "
                    f"{self._f(r['call_iv']*100, '.1f'):>7}%  "
                    f"{self._fv(r['put_oi']):>9}  "
                    f"{self._fv(r['put_vol']):>9}  "
                    f"{self._f(r['put_iv']*100, '.1f'):>7}%  "
                    f"{self._f(pcr, '.2f'):>7}"
                    f"{atm_marker}"
                )
        except Exception as e:
            lines.append(f"  Error building OI table: {e}")

        return "\n".join(lines)

    # ── Public API ────────────────────────────────────────────────────────

    def get_formatted_context(self, ticker_symbol: str) -> str:
        """
        Fetch and compute all options analytics for a ticker.
        Returns a single pre-formatted string ready for LLM injection.
        """
        t    = yf.Ticker(ticker_symbol)
        exps = t.options  # tuple of expiry strings

        if not exps:
            return (
                f"=== OPTIONS DATA: {ticker_symbol.upper()} ===\n"
                f"ERROR: No options data available for {ticker_symbol}."
            )

        spot  = self._get_spot(t)
        hv20  = self._hv_annualised(t)

        meta = (
            f"Ticker: {ticker_symbol.upper()} | "
            f"Spot: ${self._f(spot)} | "
            f"Expirations: {len(exps)} | "
            f"Data as of: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        sections = [
            f"=== OPTIONS DATA: {ticker_symbol.upper()} ===",
            meta,
            "",
            self._build_overview(spot, exps),
            "",
            self._build_pcr(t, exps, spot),
            "",
            self._build_iv(t, exps, spot, hv20),
            "",
            self._build_walls_and_pain(t, exps, spot),
            "",
            self._build_oi_distribution(t, exps, spot),
        ]
        return "\n".join(sections)
