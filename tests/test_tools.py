"""
Unit tests for all fetcher tools.
All external API calls are mocked via fixtures in conftest.py.
"""
import sys
import os
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.market_fetcher import MarketDataFetcher
from tools.fundamentals_fetcher import FundamentalsFetcher
from tools.news_fetcher import NewsFetcher
from tools.sentiment_fetcher import SentimentFetcher
from tools.options_fetcher import OptionsFetcher
from tools.macro_fetcher import MacroFetcher
from tools.compliance_fetcher import ComplianceFetcher


# ── MarketDataFetcher ─────────────────────────────────────────────────────────

class TestMarketDataFetcher:
    def test_returns_all_sections(self, mock_yf_ticker):
        context = MarketDataFetcher().get_formatted_context("AAPL")
        for section in [
            "--- PRICE SNAPSHOT ---",
            "--- MOVING AVERAGES ---",
            "--- MOMENTUM INDICATORS ---",
            "--- VOLATILITY ---",
            "--- VOLUME ANALYSIS ---",
            "--- SUPPORT / RESISTANCE",
            "--- RECENT",
        ]:
            assert section in context, f"Missing section: {section}"

    def test_ticker_appears_in_header(self, mock_yf_ticker):
        context = MarketDataFetcher().get_formatted_context("AAPL")
        assert "AAPL" in context

    def test_configurable_period(self, mock_yf_ticker):
        fetcher = MarketDataFetcher(period="6mo", recent_sessions=5)
        context = fetcher.get_formatted_context("AAPL")
        assert "AAPL" in context
        assert "--- RECENT 5 SESSIONS" in context


# ── FundamentalsFetcher ───────────────────────────────────────────────────────

class TestFundamentalsFetcher:
    def test_returns_all_sections(self, mock_yf_ticker):
        context = FundamentalsFetcher().get_formatted_context("AAPL")
        for section in [
            "--- VALUATION SNAPSHOT ---",
            "--- QUARTERLY INCOME STATEMENT",
            "--- QUARTERLY CASH FLOW",
            "--- EPS SURPRISE HISTORY",
            "--- QUARTERLY BALANCE SHEET",
        ]:
            assert section in context, f"Missing section: {section}"

    def test_ticker_in_header(self, mock_yf_ticker):
        context = FundamentalsFetcher().get_formatted_context("AAPL")
        assert "AAPL" in context

    def test_configurable_quarters(self, mock_yf_ticker):
        fetcher = FundamentalsFetcher(quarters=4, earnings_quarters=4)
        context = fetcher.get_formatted_context("AAPL")
        assert "AAPL" in context


# ── NewsFetcher ───────────────────────────────────────────────────────────────

class TestNewsFetcher:
    def test_returns_articles_from_yfinance(self, mock_yf_ticker):
        if "FINNHUB_API_KEY" in os.environ:
            del os.environ["FINNHUB_API_KEY"]
        news = NewsFetcher().get_consolidated_news("AAPL")
        assert len(news) > 0
        assert any("Test News 1" in n["title"] for n in news)

    def test_returns_articles_from_finnhub(self, mock_yf_ticker, mock_finnhub):
        os.environ["FINNHUB_API_KEY"] = "test_key"
        news = NewsFetcher().get_consolidated_news("AAPL")
        assert any("Finnhub News 1" in n["title"] for n in news)
        del os.environ["FINNHUB_API_KEY"]

    def test_articles_have_age_label(self, mock_yf_ticker):
        if "FINNHUB_API_KEY" in os.environ:
            del os.environ["FINNHUB_API_KEY"]
        news = NewsFetcher().get_consolidated_news("AAPL")
        for article in news:
            assert "age_label" in article

    def test_max_age_days_filters_old_articles(self, mock_yf_ticker):
        if "FINNHUB_API_KEY" in os.environ:
            del os.environ["FINNHUB_API_KEY"]
        # max_age_days=0 should filter out all articles
        news = NewsFetcher(max_age_days=0).get_consolidated_news("AAPL")
        assert len(news) == 0


# ── SentimentFetcher ──────────────────────────────────────────────────────────

class TestSentimentFetcher:
    def test_returns_all_sections(self, mock_subprocess, mock_requests, mock_time_sleep):
        context = SentimentFetcher().get_formatted_context("AAPL")
        for section in [
            "--- STOCKTWITS ($AAPL) ---",
            "--- REDDIT ($AAPL) ---",
            "--- AGGREGATED SIGNAL ---",
        ]:
            assert section in context, f"Missing section: {section}"

    def test_stocktwits_message_appears(self, mock_subprocess, mock_requests, mock_time_sleep):
        context = SentimentFetcher().get_formatted_context("AAPL")
        assert "StockTwits message 1" in context

    def test_reddit_post_appears(self, mock_subprocess, mock_requests, mock_time_sleep):
        context = SentimentFetcher().get_formatted_context("AAPL")
        assert "Reddit Post 1" in context


# ── OptionsFetcher ────────────────────────────────────────────────────────────

class TestOptionsFetcher:
    def test_returns_all_sections(self, mock_yf_ticker):
        context = OptionsFetcher().get_formatted_context("AAPL")
        for section in [
            "--- OPTIONS OVERVIEW ---",
            "--- PUT/CALL RATIO",
            "--- IMPLIED VOLATILITY ---",
            "--- MAX PAIN & GAMMA WALLS ---",
            "--- OI DISTRIBUTION",
        ]:
            assert section in context, f"Missing section: {section}"

    def test_ticker_in_header(self, mock_yf_ticker):
        context = OptionsFetcher().get_formatted_context("AAPL")
        assert "AAPL" in context


# ── MacroFetcher ──────────────────────────────────────────────────────────────

class TestMacroFetcher:
    def test_returns_all_sections_with_fred(self, mock_yf_ticker, mock_yf_download, mock_fred, mock_time_sleep):
        os.environ["FRED_API_KEY"] = "test_key"
        context = MacroFetcher().get_formatted_context("AAPL")
        for section in [
            "--- SECTOR TREND",
            "--- MACRO MARKET INDICATORS ---",
            "--- FRED ECONOMIC DATA ---",
            "--- MACRO & SECTOR NEWS ---",
        ]:
            assert section in context, f"Missing section: {section}"
        assert "FRED: enabled" in context
        del os.environ["FRED_API_KEY"]

    def test_fred_skipped_without_key(self, mock_yf_ticker, mock_yf_download, mock_time_sleep):
        if "FRED_API_KEY" in os.environ:
            del os.environ["FRED_API_KEY"]
        context = MacroFetcher().get_formatted_context("AAPL")
        assert "--- FRED ECONOMIC DATA ---" in context
        assert "FRED_API_KEY not set" in context


# ── ComplianceFetcher ─────────────────────────────────────────────────────────

class TestComplianceFetcher:
    def test_returns_all_sections(self, mock_yf_ticker, mock_requests):
        context = ComplianceFetcher().get_formatted_context("AAPL")
        for section in [
            "--- SHORT INTEREST ---",
            "--- INSIDER TRANSACTIONS ---",
            "--- INSTITUTIONAL OWNERSHIP ---",
            "--- EARNINGS CALENDAR & ANALYST RATINGS ---",
            "--- SEC EDGAR RECENT FILINGS",
            "--- COMPLIANCE-RELEVANT NEWS ---",
        ]:
            assert section in context, f"Missing section: {section}"

    def test_insider_name_appears(self, mock_yf_ticker, mock_requests):
        context = ComplianceFetcher().get_formatted_context("AAPL")
        assert "Tim Cook" in context

    def test_institutional_holder_appears(self, mock_yf_ticker, mock_requests):
        context = ComplianceFetcher().get_formatted_context("AAPL")
        assert "Vanguard" in context

    def test_edgar_filing_appears(self, mock_yf_ticker, mock_requests):
        context = ComplianceFetcher().get_formatted_context("AAPL")
        assert "8-K" in context
