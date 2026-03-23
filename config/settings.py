"""
Centralised configuration settings for the Vantage multi-agent system.

All values are loaded from environment variables when present, falling back
to the defaults defined here. Set overrides in the .env file at the project root.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Market Analyst ────────────────────────────────────────────────────────────
MARKET_PERIOD           = os.getenv("MARKET_PERIOD", "1y")
MARKET_INTERVAL         = os.getenv("MARKET_INTERVAL", "1d")
MARKET_RECENT_SESSIONS  = int(os.getenv("MARKET_RECENT_SESSIONS", "10"))
MARKET_GAP_THRESHOLD    = float(os.getenv("MARKET_GAP_THRESHOLD", "0.02"))
MARKET_VOLUME_SPIKE_MULT = float(os.getenv("MARKET_VOLUME_SPIKE_MULT", "2.0"))

# ── Fundamentals Analyst ──────────────────────────────────────────────────────
FUNDAMENTALS_QUARTERS          = int(os.getenv("FUNDAMENTALS_QUARTERS", "12"))
FUNDAMENTALS_EARNINGS_QUARTERS = int(os.getenv("FUNDAMENTALS_EARNINGS_QUARTERS", "12"))

# ── News Analyst ──────────────────────────────────────────────────────────────
NEWS_FINNHUB_CALLS_PER_MIN = int(os.getenv("NEWS_FINNHUB_CALLS_PER_MIN", "60"))
NEWS_MAX_ARTICLES          = int(os.getenv("NEWS_MAX_ARTICLES", "15"))
NEWS_MAX_AGE_DAYS          = int(os.getenv("NEWS_MAX_AGE_DAYS", "7"))

# ── Sentiment Analyst ─────────────────────────────────────────────────────────
SENTIMENT_STOCKTWITS_LIMIT     = int(os.getenv("SENTIMENT_STOCKTWITS_LIMIT", "30"))
_reddit_subs_env               = os.getenv("SENTIMENT_REDDIT_SUBREDDITS", "")
SENTIMENT_REDDIT_SUBREDDITS    = (
    [s.strip() for s in _reddit_subs_env.split(",") if s.strip()]
    if _reddit_subs_env else None
)
SENTIMENT_REDDIT_POSTS_PER_SUB = int(os.getenv("SENTIMENT_REDDIT_POSTS_PER_SUB", "10"))
SENTIMENT_REDDIT_TIME_FILTER   = os.getenv("SENTIMENT_REDDIT_TIME_FILTER", "week")
SENTIMENT_MAX_AGE_HOURS        = int(os.getenv("SENTIMENT_MAX_AGE_HOURS", "48"))

# ── Options Strategist ────────────────────────────────────────────────────────
OPTIONS_EXPIRATIONS_FOR_PCR = int(os.getenv("OPTIONS_EXPIRATIONS_FOR_PCR", "8"))
OPTIONS_OI_TABLE_ROWS       = int(os.getenv("OPTIONS_OI_TABLE_ROWS", "10"))
OPTIONS_TOP_WALLS           = int(os.getenv("OPTIONS_TOP_WALLS", "5"))
OPTIONS_MEDIUM_DTE_TARGET   = int(os.getenv("OPTIONS_MEDIUM_DTE_TARGET", "30"))

# ── Macro / Sector Analyst ────────────────────────────────────────────────────
MACRO_FRED_CALLS_PER_MIN = int(os.getenv("MACRO_FRED_CALLS_PER_MIN", "120"))
MACRO_NEWS_MAX_AGE_DAYS  = int(os.getenv("MACRO_NEWS_MAX_AGE_DAYS", "7"))
MACRO_NEWS_MAX_ARTICLES  = int(os.getenv("MACRO_NEWS_MAX_ARTICLES", "10"))
MACRO_HISTORY_PERIOD     = os.getenv("MACRO_HISTORY_PERIOD", "3mo")
MACRO_FRED_API_KEY       = os.getenv("FRED_API_KEY") or None

# ── Compliance Agent ──────────────────────────────────────────────────────────
COMPLIANCE_INSIDER_LOOKBACK_DAYS = int(os.getenv("COMPLIANCE_INSIDER_LOOKBACK_DAYS", "180"))
COMPLIANCE_TOP_HOLDERS           = int(os.getenv("COMPLIANCE_TOP_HOLDERS", "10"))
COMPLIANCE_ANALYST_CHANGES       = int(os.getenv("COMPLIANCE_ANALYST_CHANGES", "5"))
COMPLIANCE_EDGAR_FILINGS         = int(os.getenv("COMPLIANCE_EDGAR_FILINGS", "5"))
COMPLIANCE_NEWS_MAX_AGE_DAYS     = int(os.getenv("COMPLIANCE_NEWS_MAX_AGE_DAYS", "30"))
COMPLIANCE_NEWS_MAX_ARTICLES     = int(os.getenv("COMPLIANCE_NEWS_MAX_ARTICLES", "15"))
COMPLIANCE_BLACKOUT_WINDOW_DAYS  = int(os.getenv("COMPLIANCE_BLACKOUT_WINDOW_DAYS", "14"))
