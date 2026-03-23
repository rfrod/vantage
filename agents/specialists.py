import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional
from agents.base_specialist import BaseSpecialistAgent
from schemas.state import OutlierTicker, SpecialistReport
from tools.news_fetcher import NewsFetcher
from tools.fundamentals_fetcher import FundamentalsFetcher
from tools.market_fetcher import MarketDataFetcher
from tools.sentiment_fetcher import SentimentFetcher
from tools.options_fetcher import OptionsFetcher
from tools.macro_fetcher import MacroFetcher
from tools.compliance_fetcher import ComplianceFetcher

from config.settings import (
    MARKET_PERIOD, MARKET_INTERVAL, MARKET_RECENT_SESSIONS,
    MARKET_GAP_THRESHOLD, MARKET_VOLUME_SPIKE_MULT,
    FUNDAMENTALS_QUARTERS, FUNDAMENTALS_EARNINGS_QUARTERS,
    NEWS_FINNHUB_CALLS_PER_MIN, NEWS_MAX_ARTICLES, NEWS_MAX_AGE_DAYS,
    SENTIMENT_STOCKTWITS_LIMIT, SENTIMENT_REDDIT_SUBREDDITS,
    SENTIMENT_REDDIT_POSTS_PER_SUB, SENTIMENT_REDDIT_TIME_FILTER,
    SENTIMENT_MAX_AGE_HOURS,
    OPTIONS_EXPIRATIONS_FOR_PCR, OPTIONS_OI_TABLE_ROWS,
    OPTIONS_TOP_WALLS, OPTIONS_MEDIUM_DTE_TARGET,
    MACRO_FRED_CALLS_PER_MIN, MACRO_NEWS_MAX_AGE_DAYS,
    MACRO_NEWS_MAX_ARTICLES, MACRO_HISTORY_PERIOD, MACRO_FRED_API_KEY,
    COMPLIANCE_INSIDER_LOOKBACK_DAYS, COMPLIANCE_TOP_HOLDERS,
    COMPLIANCE_ANALYST_CHANGES, COMPLIANCE_EDGAR_FILINGS,
    COMPLIANCE_NEWS_MAX_AGE_DAYS, COMPLIANCE_NEWS_MAX_ARTICLES,
    COMPLIANCE_BLACKOUT_WINDOW_DAYS,
)

from prompts.specialists import (
    MARKET_ANALYST_PROMPT,
    FUNDAMENTALS_ANALYST_PROMPT,
    NEWS_ANALYST_PROMPT,
    SENTIMENT_ANALYST_PROMPT,
    OPTIONS_STRATEGIST_PROMPT,
    MACRO_SECTOR_ANALYST_PROMPT,
    COMPLIANCE_AGENT_PROMPT,
    DATA_QUALITY_AGENT_PROMPT,
)


class MarketAnalyst(BaseSpecialistAgent):
    def __init__(
        self,
        period: str = MARKET_PERIOD,
        interval: str = MARKET_INTERVAL,
        recent_sessions: int = MARKET_RECENT_SESSIONS,
        gap_threshold: float = MARKET_GAP_THRESHOLD,
        volume_spike_mult: float = MARKET_VOLUME_SPIKE_MULT,
    ):
        super().__init__(
            role_name="Market Analyst Agent",
            role_description=MARKET_ANALYST_PROMPT,
        )
        self._fetcher = MarketDataFetcher(
            period=period,
            interval=interval,
            recent_sessions=recent_sessions,
            gap_threshold=gap_threshold,
            volume_spike_mult=volume_spike_mult,
        )
        self._period = period
        self._interval = interval

    def analyze(self, ticker_data: OutlierTicker, additional_context: dict = None) -> SpecialistReport:
        market_context = self._fetcher.get_formatted_context(ticker_data.ticker)
        enriched_context = {
            **(additional_context or {}),
            "market_data": market_context,
            "period": self._period,
            "interval": self._interval,
        }
        return super().analyze(ticker_data, enriched_context)


class FundamentalsAnalyst(BaseSpecialistAgent):
    def __init__(
        self,
        quarters: int = FUNDAMENTALS_QUARTERS,
        earnings_quarters: int = FUNDAMENTALS_EARNINGS_QUARTERS,
    ):
        super().__init__(
            role_name="Fundamentals Analyst Agent",
            role_description=FUNDAMENTALS_ANALYST_PROMPT,
        )
        self._fetcher = FundamentalsFetcher(
            quarters=quarters,
            earnings_quarters=earnings_quarters,
        )
        self._quarters = quarters
        self._earnings_quarters = earnings_quarters

    def analyze(self, ticker_data: OutlierTicker, additional_context: dict = None) -> SpecialistReport:
        fundamentals_context = self._fetcher.get_formatted_context(ticker_data.ticker)
        enriched_context = {
            **(additional_context or {}),
            "fundamentals_data": fundamentals_context,
            "quarters_pulled": self._quarters,
            "earnings_quarters_pulled": self._earnings_quarters,
        }
        return super().analyze(ticker_data, enriched_context)


class NewsAnalyst(BaseSpecialistAgent):
    def __init__(
        self,
        finnhub_calls_per_min: int = NEWS_FINNHUB_CALLS_PER_MIN,
        max_articles: int = NEWS_MAX_ARTICLES,
        max_age_days: int = NEWS_MAX_AGE_DAYS,
    ):
        super().__init__(
            role_name="News Analyst Agent",
            role_description=NEWS_ANALYST_PROMPT,
        )
        self._fetcher = NewsFetcher(
            finnhub_calls_per_min=finnhub_calls_per_min,
            max_age_days=max_age_days,
        )
        self._max_articles = max_articles
        self._max_age_days = max_age_days

    def analyze(self, ticker_data: OutlierTicker, additional_context: dict = None) -> SpecialistReport:
        articles = self._fetcher.get_consolidated_news(
            ticker_data.ticker, max_articles=self._max_articles
        )

        if articles:
            news_lines = []
            for i, a in enumerate(articles, 1):
                stale_marker = "  *** STALE — treat as background only ***" if "STALE" in a["age_label"] else ""
                news_lines.append(
                    f"{i}. [{a['age_label']}] [{a['source']}] {a['published_at']}\n"
                    f"   Title  : {a['title']}\n"
                    f"   Summary: {a['summary'][:300] if a['summary'] else 'N/A'}\n"
                    f"   URL    : {a['url']}"
                    + (f"\n{stale_marker}" if stale_marker else "")
                )
            news_context = "\n\n".join(news_lines)
            freshest = articles[0]["age_label"] if articles else "N/A"
            stale_count = sum(1 for a in articles if "STALE" in a["age_label"])
            news_meta = (
                f"Articles fetched: {len(articles)} "
                f"(max age filter: {self._max_age_days}d, "
                f"freshest: {freshest}, "
                f"stale articles: {stale_count})"
            )
        else:
            news_context = "No news articles could be retrieved from any source."
            news_meta = f"Articles fetched: 0 (max age filter: {self._max_age_days}d)"

        enriched_context = {
            **(additional_context or {}),
            "news_meta": news_meta,
            "news_articles": news_context,
            "sources_used": list({a["source"].split(" ")[0] for a in articles}),
        }
        return super().analyze(ticker_data, enriched_context)


class SentimentAnalyst(BaseSpecialistAgent):
    def __init__(
        self,
        stocktwits_limit: int = SENTIMENT_STOCKTWITS_LIMIT,
        reddit_subreddits: Optional[List[str]] = SENTIMENT_REDDIT_SUBREDDITS,
        reddit_posts_per_sub: int = SENTIMENT_REDDIT_POSTS_PER_SUB,
        reddit_time_filter: str = SENTIMENT_REDDIT_TIME_FILTER,
        max_age_hours: int = SENTIMENT_MAX_AGE_HOURS,
    ):
        super().__init__(
            role_name="Sentiment Analyst Agent",
            role_description=SENTIMENT_ANALYST_PROMPT,
        )
        self._fetcher = SentimentFetcher(
            stocktwits_limit=stocktwits_limit,
            reddit_subreddits=reddit_subreddits,
            reddit_posts_per_sub=reddit_posts_per_sub,
            reddit_time_filter=reddit_time_filter,
            max_age_hours=max_age_hours,
        )

    def analyze(self, ticker_data: OutlierTicker, additional_context: dict = None) -> SpecialistReport:
        sentiment_context = self._fetcher.get_formatted_context(ticker_data.ticker)
        enriched_context = {
            **(additional_context or {}),
            "sentiment_data": sentiment_context,
        }
        return super().analyze(ticker_data, enriched_context)


class OptionsStrategist(BaseSpecialistAgent):
    def __init__(
        self,
        expirations_for_pcr: int = OPTIONS_EXPIRATIONS_FOR_PCR,
        oi_table_rows: int = OPTIONS_OI_TABLE_ROWS,
        top_walls: int = OPTIONS_TOP_WALLS,
        medium_dte_target: int = OPTIONS_MEDIUM_DTE_TARGET,
    ):
        super().__init__(
            role_name="Options Strategist Agent",
            role_description=OPTIONS_STRATEGIST_PROMPT,
        )
        self._fetcher = OptionsFetcher(
            expirations_for_pcr=expirations_for_pcr,
            oi_table_rows=oi_table_rows,
            top_walls=top_walls,
            medium_dte_target=medium_dte_target,
        )

    def analyze(self, ticker_data: OutlierTicker, additional_context: dict = None) -> SpecialistReport:
        options_context = self._fetcher.get_formatted_context(ticker_data.ticker)
        enriched_context = {
            **(additional_context or {}),
            "options_data": options_context,
        }
        return super().analyze(ticker_data, enriched_context)


class MacroSectorAnalyst(BaseSpecialistAgent):
    def __init__(
        self,
        fred_calls_per_min: int = MACRO_FRED_CALLS_PER_MIN,
        news_max_age_days: int = MACRO_NEWS_MAX_AGE_DAYS,
        news_max_articles: int = MACRO_NEWS_MAX_ARTICLES,
        history_period: str = MACRO_HISTORY_PERIOD,
        fred_api_key: Optional[str] = MACRO_FRED_API_KEY,
    ):
        super().__init__(
            role_name="Macro / Sector Analyst Agent",
            role_description=MACRO_SECTOR_ANALYST_PROMPT,
        )
        self._fetcher = MacroFetcher(
            fred_calls_per_min=fred_calls_per_min,
            news_max_age_days=news_max_age_days,
            news_max_articles=news_max_articles,
            history_period=history_period,
            fred_api_key=fred_api_key,
        )

    def analyze(self, ticker_data: OutlierTicker, additional_context: dict = None) -> SpecialistReport:
        macro_context = self._fetcher.get_formatted_context(ticker_data.ticker)
        enriched_context = {
            **(additional_context or {}),
            "macro_data": macro_context,
        }
        return super().analyze(ticker_data, enriched_context)


class ComplianceAgent(BaseSpecialistAgent):
    def __init__(
        self,
        insider_lookback_days: int = COMPLIANCE_INSIDER_LOOKBACK_DAYS,
        top_holders: int = COMPLIANCE_TOP_HOLDERS,
        analyst_changes: int = COMPLIANCE_ANALYST_CHANGES,
        edgar_filings: int = COMPLIANCE_EDGAR_FILINGS,
        news_max_age_days: int = COMPLIANCE_NEWS_MAX_AGE_DAYS,
        news_max_articles: int = COMPLIANCE_NEWS_MAX_ARTICLES,
        blackout_window_days: int = COMPLIANCE_BLACKOUT_WINDOW_DAYS,
    ):
        super().__init__(
            role_name="Compliance / Policy Agent",
            role_description=COMPLIANCE_AGENT_PROMPT,
        )
        self._fetcher = ComplianceFetcher(
            insider_lookback_days=insider_lookback_days,
            top_holders=top_holders,
            analyst_changes=analyst_changes,
            edgar_filings=edgar_filings,
            news_max_age_days=news_max_age_days,
            news_max_articles=news_max_articles,
            blackout_window_days=blackout_window_days,
        )

    def analyze(self, ticker_data: OutlierTicker, additional_context: dict = None) -> SpecialistReport:
        compliance_context = self._fetcher.get_formatted_context(ticker_data.ticker)
        enriched_context = {
            **(additional_context or {}),
            "compliance_data": compliance_context,
        }
        return super().analyze(ticker_data, enriched_context)


class DataQualityAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            role_name="Data Quality / Verification Agent",
            role_description=DATA_QUALITY_AGENT_PROMPT,
        )
