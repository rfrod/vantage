import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Optional


class FundamentalsFetcher:
    """
    Fetches comprehensive fundamental data for a given ticker using yfinance.

    Configurable parameters:
    - quarters (int): Number of quarterly periods to pull for income statement,
      balance sheet, and cash flow trends. Default is 12 (3 years).
    - earnings_quarters (int): Number of quarters of EPS actual vs estimate
      history to include. Default is 12.

    All data is sourced from Yahoo Finance via yfinance — no API key required.
    """

    # Snapshot fields pulled from yf.Ticker.info
    _VALUATION_KEYS = [
        "trailingPE", "forwardPE", "priceToBook",
        "priceToSalesTrailing12Months", "enterpriseToEbitda",
        "enterpriseToRevenue",
    ]
    _PROFITABILITY_KEYS = [
        "grossMargins", "operatingMargins", "profitMargins",
        "ebitdaMargins", "returnOnEquity", "returnOnAssets",
    ]
    _GROWTH_KEYS = [
        "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    ]
    _BALANCE_SHEET_KEYS = [
        "totalCash", "totalDebt", "debtToEquity",
        "currentRatio", "quickRatio",
        "freeCashflow", "operatingCashflow",
    ]
    _INCOME_KEYS = [
        "totalRevenue", "ebitda", "netIncomeToCommon",
        "trailingEps", "forwardEps", "revenuePerShare",
    ]
    _MARKET_KEYS = [
        "marketCap", "enterpriseValue", "sharesOutstanding",
        "beta", "dividendYield", "payoutRatio",
        "shortRatio", "shortPercentOfFloat",
        "currentPrice",
    ]
    _ANALYST_KEYS = [
        "recommendationKey", "numberOfAnalystOpinions",
        "targetMeanPrice",
    ]

    def __init__(self, quarters: int = 12, earnings_quarters: int = 12):
        self.quarters = quarters
        self.earnings_quarters = earnings_quarters

    # ── Formatting helpers ────────────────────────────────────────────────

    @staticmethod
    def _fmt_large(value: Optional[float]) -> str:
        """Format large numbers as $B / $M for readability."""
        if value is None:
            return "N/A"
        if abs(value) >= 1e12:
            return f"${value / 1e12:.2f}T"
        if abs(value) >= 1e9:
            return f"${value / 1e9:.2f}B"
        if abs(value) >= 1e6:
            return f"${value / 1e6:.2f}M"
        return f"${value:,.0f}"

    @staticmethod
    def _fmt_pct(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{value * 100:.2f}%"

    @staticmethod
    def _fmt_ratio(value: Optional[float], decimals: int = 2) -> str:
        if value is None:
            return "N/A"
        return f"{value:.{decimals}f}x"

    @staticmethod
    def _fmt_val(value: Optional[float], prefix: str = "", suffix: str = "", decimals: int = 2) -> str:
        if value is None:
            return "N/A"
        return f"{prefix}{value:.{decimals}f}{suffix}"

    def _safe_info(self, info: dict, key: str) -> Any:
        return info.get(key)

    # ── Section builders ──────────────────────────────────────────────────

    def _build_snapshot(self, info: dict) -> str:
        """Build the valuation / profitability / market snapshot block."""
        lines = []

        lines.append("--- VALUATION SNAPSHOT ---")
        lines.append(f"  Current Price      : {self._fmt_val(self._safe_info(info, 'currentPrice'), prefix='$')}")
        lines.append(f"  Market Cap         : {self._fmt_large(self._safe_info(info, 'marketCap'))}")
        lines.append(f"  Enterprise Value   : {self._fmt_large(self._safe_info(info, 'enterpriseValue'))}")
        lines.append(f"  Trailing P/E       : {self._fmt_ratio(self._safe_info(info, 'trailingPE'))}")
        lines.append(f"  Forward P/E        : {self._fmt_ratio(self._safe_info(info, 'forwardPE'))}")
        lines.append(f"  Price/Book         : {self._fmt_ratio(self._safe_info(info, 'priceToBook'))}")
        lines.append(f"  Price/Sales (TTM)  : {self._fmt_ratio(self._safe_info(info, 'priceToSalesTrailing12Months'))}")
        lines.append(f"  EV/EBITDA          : {self._fmt_ratio(self._safe_info(info, 'enterpriseToEbitda'))}")
        lines.append(f"  EV/Revenue         : {self._fmt_ratio(self._safe_info(info, 'enterpriseToRevenue'))}")

        lines.append("\n--- PROFITABILITY ---")
        lines.append(f"  Gross Margin       : {self._fmt_pct(self._safe_info(info, 'grossMargins'))}")
        lines.append(f"  Operating Margin   : {self._fmt_pct(self._safe_info(info, 'operatingMargins'))}")
        lines.append(f"  Net Margin         : {self._fmt_pct(self._safe_info(info, 'profitMargins'))}")
        lines.append(f"  EBITDA Margin      : {self._fmt_pct(self._safe_info(info, 'ebitdaMargins'))}")
        lines.append(f"  Return on Equity   : {self._fmt_pct(self._safe_info(info, 'returnOnEquity'))}")
        lines.append(f"  Return on Assets   : {self._fmt_pct(self._safe_info(info, 'returnOnAssets'))}")

        lines.append("\n--- GROWTH (YoY) ---")
        lines.append(f"  Revenue Growth     : {self._fmt_pct(self._safe_info(info, 'revenueGrowth'))}")
        lines.append(f"  Earnings Growth    : {self._fmt_pct(self._safe_info(info, 'earningsGrowth'))}")
        lines.append(f"  EPS Growth (QoQ)   : {self._fmt_pct(self._safe_info(info, 'earningsQuarterlyGrowth'))}")

        lines.append("\n--- BALANCE SHEET & CASH FLOW ---")
        lines.append(f"  Total Cash         : {self._fmt_large(self._safe_info(info, 'totalCash'))}")
        lines.append(f"  Total Debt         : {self._fmt_large(self._safe_info(info, 'totalDebt'))}")
        lines.append(f"  Debt/Equity        : {self._fmt_val(self._safe_info(info, 'debtToEquity'), suffix='%')}")
        lines.append(f"  Current Ratio      : {self._fmt_ratio(self._safe_info(info, 'currentRatio'))}")
        lines.append(f"  Quick Ratio        : {self._fmt_ratio(self._safe_info(info, 'quickRatio'))}")
        lines.append(f"  Free Cash Flow     : {self._fmt_large(self._safe_info(info, 'freeCashflow'))}")
        lines.append(f"  Operating Cash Flow: {self._fmt_large(self._safe_info(info, 'operatingCashflow'))}")

        lines.append("\n--- INCOME (TTM) ---")
        lines.append(f"  Total Revenue      : {self._fmt_large(self._safe_info(info, 'totalRevenue'))}")
        lines.append(f"  EBITDA             : {self._fmt_large(self._safe_info(info, 'ebitda'))}")
        lines.append(f"  Net Income         : {self._fmt_large(self._safe_info(info, 'netIncomeToCommon'))}")
        lines.append(f"  Trailing EPS       : {self._fmt_val(self._safe_info(info, 'trailingEps'), prefix='$')}")
        lines.append(f"  Forward EPS        : {self._fmt_val(self._safe_info(info, 'forwardEps'), prefix='$')}")

        lines.append("\n--- MARKET & RISK ---")
        lines.append(f"  Beta               : {self._fmt_val(self._safe_info(info, 'beta'))}")
        lines.append(f"  Dividend Yield     : {self._fmt_pct(self._safe_info(info, 'dividendYield'))}")
        lines.append(f"  Payout Ratio       : {self._fmt_pct(self._safe_info(info, 'payoutRatio'))}")
        lines.append(f"  Short Ratio        : {self._fmt_val(self._safe_info(info, 'shortRatio'), suffix='d')}")
        lines.append(f"  Short % of Float   : {self._fmt_pct(self._safe_info(info, 'shortPercentOfFloat'))}")
        lines.append(f"  Shares Outstanding : {self._fmt_large(self._safe_info(info, 'sharesOutstanding')).replace('$', '')}")

        lines.append("\n--- ANALYST CONSENSUS ---")
        lines.append(f"  Recommendation     : {(self._safe_info(info, 'recommendationKey') or 'N/A').upper()}")
        lines.append(f"  # Analyst Opinions : {self._safe_info(info, 'numberOfAnalystOpinions') or 'N/A'}")
        lines.append(f"  Mean Price Target  : {self._fmt_val(self._safe_info(info, 'targetMeanPrice'), prefix='$')}")
        analyst_targets = info.get("_analyst_targets", {})
        if analyst_targets:
            lines.append(f"  High Target        : {self._fmt_val(analyst_targets.get('high'), prefix='$')}")
            lines.append(f"  Low Target         : {self._fmt_val(analyst_targets.get('low'), prefix='$')}")
            lines.append(f"  Median Target      : {self._fmt_val(analyst_targets.get('median'), prefix='$')}")

        return "\n".join(lines)

    def _build_quarterly_income(self, ticker: yf.Ticker) -> str:
        """Build a quarterly income statement trend table."""
        try:
            df = ticker.quarterly_income_stmt
            if df is None or df.empty:
                return "Quarterly income statement: not available."

            rows_wanted = [
                "Total Revenue", "Gross Profit", "Operating Income",
                "Net Income", "Diluted EPS", "EBITDA",
            ]
            # Select available rows and limit to self.quarters columns
            available = [r for r in rows_wanted if r in df.index]
            df_sub = df.loc[available, :].iloc[:, : self.quarters]

            # Format columns as Q labels
            col_labels = [c.strftime("%Y-Q%q") if hasattr(c, "strftime") else str(c)[:7]
                          for c in df_sub.columns]

            lines = [f"--- QUARTERLY INCOME STATEMENT (last {len(df_sub.columns)} quarters) ---"]
            header = f"  {'Metric':<28}" + "".join(f"  {lbl:>12}" for lbl in col_labels)
            lines.append(header)
            lines.append("  " + "-" * (28 + 14 * len(col_labels)))

            for row in available:
                values = df_sub.loc[row]
                if row == "Diluted EPS":
                    formatted = [self._fmt_val(v, prefix="$") if pd.notna(v) else "N/A" for v in values]
                else:
                    formatted = [self._fmt_large(v) if pd.notna(v) else "N/A" for v in values]
                lines.append(f"  {row:<28}" + "".join(f"  {v:>12}" for v in formatted))

            return "\n".join(lines)
        except Exception as e:
            return f"Quarterly income statement: error ({e})"

    def _build_quarterly_cashflow(self, ticker: yf.Ticker) -> str:
        """Build a quarterly cash flow trend table."""
        try:
            df = ticker.quarterly_cashflow
            if df is None or df.empty:
                return "Quarterly cash flow: not available."

            rows_wanted = [
                "Operating Cash Flow", "Free Cash Flow",
                "Capital Expenditure", "Repurchase Of Capital Stock",
            ]
            available = [r for r in rows_wanted if r in df.index]
            df_sub = df.loc[available, :].iloc[:, : self.quarters]

            col_labels = [c.strftime("%Y-Q%q") if hasattr(c, "strftime") else str(c)[:7]
                          for c in df_sub.columns]

            lines = [f"--- QUARTERLY CASH FLOW (last {len(df_sub.columns)} quarters) ---"]
            header = f"  {'Metric':<32}" + "".join(f"  {lbl:>12}" for lbl in col_labels)
            lines.append(header)
            lines.append("  " + "-" * (32 + 14 * len(col_labels)))

            for row in available:
                values = df_sub.loc[row]
                formatted = [self._fmt_large(v) if pd.notna(v) else "N/A" for v in values]
                lines.append(f"  {row:<32}" + "".join(f"  {v:>12}" for v in formatted))

            return "\n".join(lines)
        except Exception as e:
            return f"Quarterly cash flow: error ({e})"

    def _build_earnings_history(self, ticker: yf.Ticker) -> str:
        """Build EPS actual vs estimate surprise history."""
        try:
            df = ticker.earnings_history
            if df is None or df.empty:
                return "Earnings history: not available."

            df = df.tail(self.earnings_quarters).copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=False)

            lines = [f"--- EPS SURPRISE HISTORY (last {len(df)} quarters) ---"]
            lines.append(f"  {'Quarter':<14}  {'Actual EPS':>12}  {'Est. EPS':>12}  {'Surprise':>10}  {'Surprise %':>12}")
            lines.append("  " + "-" * 66)

            for qtr, row in df.iterrows():
                actual = self._fmt_val(row.get("epsActual"), prefix="$")
                est = self._fmt_val(row.get("epsEstimate"), prefix="$")
                diff = row.get("epsDifference")
                surprise_pct = row.get("surprisePercent")
                diff_str = (f"+${diff:.2f}" if diff and diff >= 0 else f"-${abs(diff):.2f}") if diff is not None else "N/A"
                surp_str = self._fmt_pct(surprise_pct) if surprise_pct is not None else "N/A"
                lines.append(f"  {str(qtr)[:10]:<14}  {actual:>12}  {est:>12}  {diff_str:>10}  {surp_str:>12}")

            return "\n".join(lines)
        except Exception as e:
            return f"Earnings history: error ({e})"

    def _build_balance_sheet_trend(self, ticker: yf.Ticker) -> str:
        """Build a quarterly balance sheet trend for key health metrics."""
        try:
            df = ticker.quarterly_balance_sheet
            if df is None or df.empty:
                return "Quarterly balance sheet: not available."

            rows_wanted = [
                "Total Debt", "Total Assets", "Stockholders Equity",
                "Cash And Cash Equivalents", "Net Debt",
            ]
            available = [r for r in rows_wanted if r in df.index]
            df_sub = df.loc[available, :].iloc[:, : self.quarters]

            col_labels = [c.strftime("%Y-Q%q") if hasattr(c, "strftime") else str(c)[:7]
                          for c in df_sub.columns]

            lines = [f"--- QUARTERLY BALANCE SHEET (last {len(df_sub.columns)} quarters) ---"]
            header = f"  {'Metric':<30}" + "".join(f"  {lbl:>12}" for lbl in col_labels)
            lines.append(header)
            lines.append("  " + "-" * (30 + 14 * len(col_labels)))

            for row in available:
                values = df_sub.loc[row]
                formatted = [self._fmt_large(v) if pd.notna(v) else "N/A" for v in values]
                lines.append(f"  {row:<30}" + "".join(f"  {v:>12}" for v in formatted))

            return "\n".join(lines)
        except Exception as e:
            return f"Quarterly balance sheet: error ({e})"

    # ── Public API ────────────────────────────────────────────────────────

    def get_fundamentals(self, ticker: str) -> Dict[str, str]:
        """
        Fetch and format all fundamental data for a ticker.

        Returns a dict with keys:
            snapshot, quarterly_income, quarterly_cashflow,
            quarterly_balance_sheet, earnings_history, meta
        Each value is a pre-formatted string ready to be injected into an LLM prompt.
        """
        t = yf.Ticker(ticker)
        info = t.info or {}

        # Attach analyst price targets into info for snapshot builder
        try:
            targets = t.analyst_price_targets
            if targets:
                info["_analyst_targets"] = targets
        except Exception:
            pass

        snapshot = self._build_snapshot(info)
        quarterly_income = self._build_quarterly_income(t)
        quarterly_cashflow = self._build_quarterly_cashflow(t)
        quarterly_balance_sheet = self._build_balance_sheet_trend(t)
        earnings_history = self._build_earnings_history(t)

        meta = (
            f"Ticker: {ticker.upper()} | "
            f"Sector: {info.get('sector', 'N/A')} | "
            f"Industry: {info.get('industry', 'N/A')} | "
            f"Exchange: {info.get('exchange', 'N/A')} | "
            f"Data as of: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
            f"Quarters requested: {self.quarters}"
        )

        return {
            "meta": meta,
            "snapshot": snapshot,
            "quarterly_income": quarterly_income,
            "quarterly_cashflow": quarterly_cashflow,
            "quarterly_balance_sheet": quarterly_balance_sheet,
            "earnings_history": earnings_history,
        }

    def get_formatted_context(self, ticker: str) -> str:
        """
        Return a single pre-formatted string containing all fundamental data,
        ready to be injected directly into an LLM prompt context.
        """
        data = self.get_fundamentals(ticker)
        sections = [
            f"=== FUNDAMENTAL DATA: {ticker.upper()} ===",
            data["meta"],
            "",
            data["snapshot"],
            "",
            data["quarterly_income"],
            "",
            data["quarterly_cashflow"],
            "",
            data["quarterly_balance_sheet"],
            "",
            data["earnings_history"],
        ]
        return "\n".join(sections)
