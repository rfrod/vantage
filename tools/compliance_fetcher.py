"""
ComplianceFetcher — collects compliance and governance signals for a given ticker.

Six data blocks:
  1. Short Interest          : short float %, short ratio, shares short (yfinance)
  2. Insider Transactions    : last N insider buy/sell transactions, net 90-day sentiment (yfinance)
  3. Institutional Ownership : top holders, % institutional, recent change signal (yfinance)
  4. Earnings Calendar       : next earnings date, blackout window flag, last analyst rating changes (yfinance)
  5. SEC EDGAR Filings       : last N 8-K material event filings (EDGAR free API, no key required)
  6. Compliance-Relevant News: ticker news filtered by legal/regulatory/governance keywords (NewsFetcher)

No API key required for any source.
"""

import os
import sys
import json
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.news_fetcher import NewsFetcher

# ── Compliance keyword groups ─────────────────────────────────────────────────
COMPLIANCE_KEYWORDS: Dict[str, List[str]] = {
    "Legal / Regulatory": [
        "lawsuit", "litigation", "SEC investigation", "DOJ", "antitrust",
        "class action", "subpoena", "indictment", "settlement", "fine",
        "penalty", "regulatory action", "CFTC", "FTC", "FINRA",
    ],
    "Financial Integrity": [
        "restatement", "accounting fraud", "audit", "material weakness",
        "going concern", "earnings manipulation", "revenue recognition",
        "write-down", "impairment", "goodwill write-off",
    ],
    "Governance": [
        "insider trading", "executive departure", "CEO resign", "CFO resign",
        "board shake-up", "proxy fight", "activist investor",
        "whistleblower", "corporate governance",
    ],
    "Sector-Specific": [
        "FDA", "recall", "safety warning", "clinical hold",
        "sanctions", "export ban", "trade restriction", "blacklist",
        "data breach", "cybersecurity incident", "GDPR",
    ],
}

ALL_COMPLIANCE_TERMS = [
    term for terms in COMPLIANCE_KEYWORDS.values() for term in terms
]


class ComplianceFetcher:
    """
    Fetches compliance and governance signals for a given ticker.

    Configurable parameters:
    - insider_lookback_days (int)  : Days of insider transactions to include. Default 180.
    - top_holders (int)            : Number of top institutional holders to show. Default 10.
    - analyst_changes (int)        : Number of recent analyst rating changes to show. Default 5.
    - edgar_filings (int)          : Number of recent 8-K filings to retrieve. Default 5.
    - news_max_age_days (int)      : Max age of compliance news articles. Default 30.
    - news_max_articles (int)      : Max compliance news articles to return. Default 15.
    - blackout_window_days (int)   : Days before earnings to flag as blackout window. Default 14.
    """

    def __init__(
        self,
        insider_lookback_days: int = 180,
        top_holders: int = 10,
        analyst_changes: int = 5,
        edgar_filings: int = 5,
        news_max_age_days: int = 30,
        news_max_articles: int = 15,
        blackout_window_days: int = 14,
    ):
        self.insider_lookback_days  = insider_lookback_days
        self.top_holders            = top_holders
        self.analyst_changes        = analyst_changes
        self.edgar_filings          = edgar_filings
        self.news_max_age_days      = news_max_age_days
        self.news_max_articles      = news_max_articles
        self.blackout_window_days   = blackout_window_days

        self._news_fetcher = NewsFetcher(max_age_days=news_max_age_days)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _f(v, fmt=".2f", fallback="N/A") -> str:
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _fmt_date(dt) -> str:
        try:
            if isinstance(dt, str):
                return dt[:10]
            return pd.Timestamp(dt).strftime("%Y-%m-%d")
        except Exception:
            return "?"

    # ── Block 1: Short Interest ───────────────────────────────────────────────

    def _build_short_interest(self, info: dict) -> str:
        lines = ["--- SHORT INTEREST ---"]
        short_pct   = info.get("shortPercentOfFloat")
        short_ratio = info.get("shortRatio")
        shares_short = info.get("sharesShort")
        shares_out   = info.get("sharesOutstanding")

        if short_pct is not None:
            pct_val = float(short_pct) * 100
            flag = ""
            if pct_val > 30:
                flag = "  ⚑ VERY HIGH — potential short squeeze or strong bearish conviction"
            elif pct_val > 20:
                flag = "  ⚑ HIGH — elevated short interest, monitor closely"
            elif pct_val > 10:
                flag = "  ⚑ MODERATE — above-average short interest"
            lines.append(f"  Short % of Float   : {self._f(pct_val, '.2f')}%{flag}")
        else:
            lines.append("  Short % of Float   : N/A")

        lines.append(f"  Short Ratio (DTC)  : {self._f(short_ratio, '.1f')} days")
        if shares_short:
            lines.append(f"  Shares Short       : {int(shares_short):,}")
        if shares_out:
            lines.append(f"  Shares Outstanding : {int(shares_out):,}")

        return "\n".join(lines)

    # ── Block 2: Insider Transactions ─────────────────────────────────────────

    def _build_insider_transactions(self, ticker_obj: yf.Ticker) -> str:
        lines = ["--- INSIDER TRANSACTIONS ---"]
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.insider_lookback_days)

        try:
            txns = ticker_obj.insider_transactions
            if txns is None or txns.empty:
                lines.append("  No insider transaction data available.")
                return "\n".join(lines)

            # Normalise column names
            txns = txns.copy()
            txns.columns = [c.lower().replace(" ", "_") for c in txns.columns]

            # Filter by lookback window
            date_col = next((c for c in txns.columns if "date" in c), None)
            if date_col:
                txns[date_col] = pd.to_datetime(txns[date_col], utc=True, errors="coerce")
                txns = txns[txns[date_col] >= cutoff]

            if txns.empty:
                lines.append(f"  No insider transactions in the last {self.insider_lookback_days} days.")
                return "\n".join(lines)

            # Compute net sentiment
            shares_col   = next((c for c in txns.columns if "shares" in c), None)
            text_col     = next((c for c in txns.columns if "text" in c or "transaction" in c), None)
            insider_col  = next((c for c in txns.columns if "insider" in c or "name" in c), None)
            title_col    = next((c for c in txns.columns if "title" in c or "position" in c), None)

            buys = sells = 0
            buy_shares = sell_shares = 0

            for _, row in txns.iterrows():
                text = str(row.get(text_col, "")).lower() if text_col else ""
                shares = abs(float(row.get(shares_col, 0) or 0)) if shares_col else 0
                if "sale" in text or "sell" in text or "sold" in text:
                    sells += 1
                    sell_shares += shares
                elif "purchase" in text or "buy" in text or "bought" in text or "acquisition" in text:
                    buys += 1
                    buy_shares += shares

            net_signal = (
                "NET BUYER (bullish signal)" if buys > sells else
                "NET SELLER (bearish signal)" if sells > buys else
                "MIXED"
            )
            lines.append(f"  Lookback            : {self.insider_lookback_days} days")
            lines.append(f"  Insider Buys        : {buys} transactions ({int(buy_shares):,} shares)")
            lines.append(f"  Insider Sells       : {sells} transactions ({int(sell_shares):,} shares)")
            lines.append(f"  Net Sentiment       : {net_signal}")
            lines.append("")
            lines.append("  Recent transactions:")

            for _, row in txns.head(8).iterrows():
                date_str  = self._fmt_date(row.get(date_col)) if date_col else "?"
                insider   = str(row.get(insider_col, "Unknown"))[:30] if insider_col else "?"
                title     = str(row.get(title_col, ""))[:25] if title_col else ""
                text      = str(row.get(text_col, ""))[:30] if text_col else ""
                shares    = int(abs(float(row.get(shares_col, 0) or 0))) if shares_col else 0
                lines.append(
                    f"    {date_str}  {insider:<30}  {title:<25}  "
                    f"{text:<30}  {shares:>10,} shares"
                )

        except Exception as e:
            lines.append(f"  Error fetching insider data: {e}")

        return "\n".join(lines)

    # ── Block 3: Institutional Ownership ─────────────────────────────────────

    def _build_institutional_ownership(self, ticker_obj: yf.Ticker, info: dict) -> str:
        lines = ["--- INSTITUTIONAL OWNERSHIP ---"]

        inst_pct = info.get("heldPercentInstitutions")
        insider_pct = info.get("heldPercentInsiders")

        if inst_pct is not None:
            lines.append(f"  Institutional Ownership : {self._f(float(inst_pct)*100, '.1f')}%")
        if insider_pct is not None:
            lines.append(f"  Insider Ownership       : {self._f(float(insider_pct)*100, '.1f')}%")

        try:
            holders = ticker_obj.institutional_holders
            if holders is not None and not holders.empty:
                holders = holders.head(self.top_holders)
                lines.append(f"\n  Top {self.top_holders} Institutional Holders:")
                lines.append(f"  {'Holder':<40} {'Shares':>15} {'% Out':>8} {'Date Reported'}")
                lines.append("  " + "-" * 80)
                for _, row in holders.iterrows():
                    name   = str(row.get("Holder", "?"))[:38]
                    shares = int(row.get("Shares", 0) or 0)
                    pct    = float(row.get("% Out", 0) or 0) * 100
                    date   = self._fmt_date(row.get("Date Reported", "?"))
                    lines.append(f"  {name:<40} {shares:>15,} {pct:>7.2f}% {date}")
            else:
                lines.append("  No institutional holder data available.")
        except Exception as e:
            lines.append(f"  Error fetching institutional holders: {e}")

        return "\n".join(lines)

    # ── Block 4: Earnings Calendar & Analyst Ratings ─────────────────────────

    def _build_earnings_and_ratings(self, ticker_obj: yf.Ticker) -> str:
        lines = ["--- EARNINGS CALENDAR & ANALYST RATINGS ---"]

        # Earnings date — yfinance returns a dict in newer versions
        try:
            cal = ticker_obj.calendar
            earnings_date = None
            if isinstance(cal, dict):
                # New yfinance API: dict with 'Earnings Date' key
                ed = cal.get("Earnings Date")
                if ed:
                    earnings_date = pd.Timestamp(ed[0]) if isinstance(ed, list) else pd.Timestamp(ed)
            elif cal is not None and hasattr(cal, "index") and "Earnings Date" in cal.index:
                # Old yfinance API: DataFrame
                dates = cal.loc["Earnings Date"]
                earnings_date = pd.Timestamp(dates.iloc[0]) if hasattr(dates, "iloc") else pd.Timestamp(dates)

            if earnings_date is not None:
                next_date_utc = earnings_date.tz_localize("UTC") if earnings_date.tzinfo is None else earnings_date.tz_convert("UTC")
                days_to_earnings = (next_date_utc - datetime.now(timezone.utc)).days
                blackout_flag = ""
                if 0 <= days_to_earnings <= self.blackout_window_days:
                    blackout_flag = f"  ⚑ BLACKOUT WINDOW — earnings in {days_to_earnings} days, trading restrictions may apply"
                lines.append(f"  Next Earnings Date  : {earnings_date.strftime('%Y-%m-%d')}  ({days_to_earnings} days away){blackout_flag}")
            else:
                lines.append("  Next Earnings Date  : N/A")
        except Exception as e:
            lines.append(f"  Earnings calendar error: {e}")

        # Analyst rating changes
        try:
            upgrades = ticker_obj.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                upgrades = upgrades.sort_index(ascending=False).head(self.analyst_changes)
                lines.append(f"\n  Last {self.analyst_changes} Analyst Rating Changes:")
                lines.append(f"  {'Date':<12} {'Firm':<30} {'To Grade':<20} {'From Grade':<20} {'Action'}")
                lines.append("  " + "-" * 95)
                for date_idx, row in upgrades.iterrows():
                    date_str   = self._fmt_date(date_idx)
                    firm       = str(row.get("Firm", "?"))[:28]
                    to_grade   = str(row.get("ToGrade", "?"))[:18]
                    from_grade = str(row.get("FromGrade", "?"))[:18]
                    action     = str(row.get("Action", "?"))
                    lines.append(f"  {date_str:<12} {firm:<30} {to_grade:<20} {from_grade:<20} {action}")
            else:
                lines.append("\n  No analyst rating change data available.")
        except Exception as e:
            lines.append(f"\n  Analyst ratings error: {e}")

        return "\n".join(lines)

    # ── Block 5: SEC EDGAR Filings ────────────────────────────────────────────

    def _build_edgar_filings(self, ticker_symbol: str) -> str:
        lines = ["--- SEC EDGAR RECENT FILINGS (8-K Material Events) ---"]
        try:
            # Use EDGAR full-text search API (no key required)
            url = (
                f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker_symbol}%22"
                f"&dateRange=custom&startdt={(datetime.now()-timedelta(days=180)).strftime('%Y-%m-%d')}"
                f"&forms=8-K&hits.hits._source=period_of_report,display_names,file_date,form_type,entity_name"
                f"&hits.hits.total.value=true&hits.hits._source.period_of_report=true"
            )
            # Try the simpler EDGAR company search endpoint
            cik_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker_symbol}&type=8-K&dateb=&owner=include&count={self.edgar_filings}&search_text=&output=atom"
            headers = {"User-Agent": "VantageBot research@vantage.ai"}
            resp = requests.get(cik_url, headers=headers, timeout=10)

            if resp.status_code == 200:
                # Parse the Atom feed
                import xml.etree.ElementTree as ET
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(resp.text)
                entries = root.findall("atom:entry", ns)
                if not entries:
                    lines.append("  No recent 8-K filings found.")
                else:
                    lines.append(f"  Last {min(self.edgar_filings, len(entries))} 8-K filings:")
                    for entry in entries[:self.edgar_filings]:
                        title   = entry.findtext("atom:title", default="?", namespaces=ns)
                        updated = entry.findtext("atom:updated", default="?", namespaces=ns)[:10]
                        link_el = entry.find("atom:link", ns)
                        link    = link_el.get("href", "") if link_el is not None else ""
                        lines.append(f"    [{updated}] {title}")
                        if link:
                            lines.append(f"             {link}")
            else:
                lines.append(f"  EDGAR API returned status {resp.status_code}. Skipping.")

        except Exception as e:
            lines.append(f"  Error fetching EDGAR filings: {e}")

        return "\n".join(lines)

    # ── Block 6: Compliance-Relevant News ────────────────────────────────────

    def _build_compliance_news(self, ticker_symbol: str) -> str:
        lines = ["--- COMPLIANCE-RELEVANT NEWS ---"]

        try:
            all_articles = self._news_fetcher.get_consolidated_news(
                ticker_symbol, max_articles=50
            )

            # Filter articles that contain any compliance keyword
            flagged: List[Dict[str, Any]] = []
            for article in all_articles:
                text = (
                    (article.get("title") or "") + " " +
                    (article.get("summary") or "")
                ).lower()
                matched_groups = []
                for group, terms in COMPLIANCE_KEYWORDS.items():
                    matched_terms = [t for t in terms if t.lower() in text]
                    if matched_terms:
                        matched_groups.append(f"{group}: {', '.join(matched_terms[:3])}")
                if matched_groups:
                    article["_compliance_flags"] = matched_groups
                    flagged.append(article)

            flagged = flagged[:self.news_max_articles]

            if not flagged:
                lines.append(
                    f"  No compliance-relevant news found in the last {self.news_max_age_days} days. "
                    "This is a positive signal."
                )
            else:
                lines.append(f"  {len(flagged)} compliance-relevant article(s) found:")
                for a in flagged:
                    lines.append(
                        f"\n  [{a.get('age_label','?')}] [{a.get('source','?')}] "
                        f"{a.get('title','')}"
                    )
                    flags = a.get("_compliance_flags", [])
                    if flags:
                        lines.append(f"    Flags: {' | '.join(flags)}")
                    summary = a.get("summary", "")
                    if summary:
                        lines.append(f"    {summary[:250]}")

        except Exception as e:
            lines.append(f"  Error fetching compliance news: {e}")

        return "\n".join(lines)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_formatted_context(self, ticker_symbol: str) -> str:
        """
        Fetch all six compliance data blocks for a ticker.
        Returns a single pre-formatted string ready for LLM injection.
        """
        ticker_obj = yf.Ticker(ticker_symbol)
        info = ticker_obj.info or {}

        company_name = info.get("longName", ticker_symbol)
        sector       = info.get("sector", "Unknown")
        exchange     = info.get("exchange", "Unknown")

        header = (
            f"=== COMPLIANCE DATA: {ticker_symbol.upper()} ===\n"
            f"Company: {company_name} | Sector: {sector} | Exchange: {exchange} | "
            f"Data as of: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        sections = [
            header,
            "",
            self._build_short_interest(info),
            "",
            self._build_insider_transactions(ticker_obj),
            "",
            self._build_institutional_ownership(ticker_obj, info),
            "",
            self._build_earnings_and_ratings(ticker_obj),
            "",
            self._build_edgar_filings(ticker_symbol),
            "",
            self._build_compliance_news(ticker_symbol),
        ]
        return "\n".join(sections)
