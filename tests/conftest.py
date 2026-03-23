"""
Shared pytest fixtures for the Vantage test suite.

Key mock strategy for LangChain agents:
  - Agents use the pattern: chain = prompt | self.llm.with_structured_output(Schema)
  - The | operator creates a RunnableSequence (LangChain internal)
  - To intercept the final .invoke() call, with_structured_output must return
    a real RunnableLambda (not a MagicMock), so the pipe operator works correctly.
  - We use a schema-dispatch lambda that returns the correct Pydantic object per call.

Key mock strategy for external APIs:
  - Patch at the module level where the name is looked up (not where it is defined)
  - fredapi is imported lazily inside a function in macro_fetcher.py, so we patch
    'fredapi.Fred' at the top-level module, not 'tools.macro_fetcher.fredapi.Fred'
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import datetime
from langchain_core.runnables import RunnableLambda


# ── Pydantic model factory ────────────────────────────────────────────────────

def _make_pydantic_objects():
    from schemas.state import (
        SpecialistReport, SynthesisReport, DebateTurn, DebateSummary,
        RiskRecommendation, RiskSummary, FinalDecision,
    )

    spec_report = SpecialistReport(
        agent_name="Mock Agent",
        findings=["Finding 1", "Finding 2"],
        flags=["Flag 1"],
    )
    synthesis = SynthesisReport(
        ticker="AAPL",
        outlier_classification="++",
        what_happened=["Price went up"],
        key_technical_findings=["Bullish trend"],
        key_fundamental_findings=["Strong growth"],
        key_news_findings=["Positive earnings"],
        key_sentiment_findings=["Bullish retail"],
        key_options_findings=["Call buying"],
        major_risks=["Valuation"],
        initial_directional_bias="Bullish",
        candidate_trade_expressions=["Buy stock"],
    )
    debate_turn = DebateTurn(agent="Bull", arguments=["Strong fundamentals"])
    debate_summary = DebateSummary(
        strongest_bull_arguments=["Growth"],
        strongest_bear_arguments=["Valuation"],
        unresolved_uncertainties=["Macro"],
        current_score="Bullish edge",
    )
    risk_rec = RiskRecommendation(
        persona="Neutral",
        whether_to_trade=True,
        trade_type="Buy Stock",
        suggested_position_size_pct=5.0,
        max_loss_framing="2% of portfolio",
        stop_review_conditions="Close below 50d MA",
        invalidation_thesis="Earnings miss",
    )
    risk_summary = RiskSummary(
        conservative_rec=RiskRecommendation(
            persona="Conservative", whether_to_trade=True, trade_type="Buy Stock",
            suggested_position_size_pct=2.0, max_loss_framing="1%",
            stop_review_conditions="Below 50d MA", invalidation_thesis="Miss"),
        neutral_rec=risk_rec,
        aggressive_rec=RiskRecommendation(
            persona="Aggressive", whether_to_trade=True, trade_type="Buy Stock",
            suggested_position_size_pct=10.0, max_loss_framing="5%",
            stop_review_conditions="Below 200d MA", invalidation_thesis="Miss"),
        committee_consensus="Proceed with caution",
    )
    final_decision = FinalDecision(
        ticker="AAPL",
        final_action="BUY STOCK",
        confidence_score=8,
        top_3_reasons=["A", "B", "C"],
        top_3_risks=["X", "Y", "Z"],
        recommended_position_size="5%",
        recommended_strike_expiration="N/A",
        what_to_monitor_next=["Earnings"],
        audit_trail=["Market", "Fundamentals"],
    )

    return {
        "SpecialistReport": spec_report,
        "SynthesisReport": synthesis,
        "DebateTurn": debate_turn,
        "DebateSummary": debate_summary,
        "RiskRecommendation": risk_rec,
        "RiskSummary": risk_summary,
        "FinalDecision": final_decision,
    }


# ── yfinance fixtures ─────────────────────────────────────────────────────────

def _make_mock_ticker():
    """Build a fully-populated yfinance Ticker mock."""
    mock_ticker = MagicMock()

    mock_ticker.info = {
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "country": "United States",
        "exchange": "NMS",
        "regularMarketPrice": 150.0,
        "previousClose": 148.0,
        "fiftyTwoWeekHigh": 180.0,
        "fiftyTwoWeekLow": 120.0,
        "trailingPE": 25.5,
        "forwardPE": 22.1,
        "priceToBook": 10.5,
        "profitMargins": 0.25,
        "grossMargins": 0.43,
        "operatingMargins": 0.30,
        "returnOnEquity": 1.50,
        "returnOnAssets": 0.28,
        "revenueGrowth": 0.08,
        "earningsGrowth": 0.12,
        "totalRevenue": 400_000_000_000,
        "ebitda": 130_000_000_000,
        "netIncomeToCommon": 100_000_000_000,
        "trailingEps": 6.50,
        "forwardEps": 7.20,
        "totalCash": 60_000_000_000,
        "totalDebt": 110_000_000_000,
        "debtToEquity": 180.0,
        "currentRatio": 0.95,
        "freeCashflow": 90_000_000_000,
        "beta": 1.2,
        "dividendYield": 0.005,
        "shortPercentOfFloat": 0.05,
        "shortRatio": 2.5,
        "sharesShort": 75_000_000,
        "sharesOutstanding": 15_500_000_000,
        "heldPercentInstitutions": 0.60,
        "heldPercentInsiders": 0.05,
        "recommendationKey": "buy",
        "numberOfAnalystOpinions": 40,
        "targetMeanPrice": 200.0,
        "targetHighPrice": 250.0,
        "targetLowPrice": 160.0,
    }

    # .history()
    dates = pd.date_range(end=datetime.date.today(), periods=100, freq='B')
    hist_df = pd.DataFrame({
        "Open":   np.linspace(140, 150, 100),
        "High":   np.linspace(142, 152, 100),
        "Low":    np.linspace(138, 148, 100),
        "Close":  np.linspace(141, 151, 100),
        "Volume": np.random.randint(1_000_000, 5_000_000, 100).astype(float),
    }, index=dates)
    mock_ticker.history.return_value = hist_df

    # .news
    mock_ticker.news = [{
        "title": "Test News 1",
        "publisher": "Yahoo Finance",
        "link": "http://example.com/1",
        "providerPublishTime": int(datetime.datetime.now().timestamp()),
    }]

    # .options and .option_chain()
    mock_ticker.options = ("2025-04-17", "2025-05-15")
    chain_mock = MagicMock()
    chain_mock.calls = pd.DataFrame({
        "strike":            [140.0, 150.0, 160.0],
        "lastPrice":         [10.0,   5.0,   1.0],
        "bid":               [9.9,    4.9,   0.9],
        "ask":               [10.1,   5.1,   1.1],
        "volume":            [100,    500,   200],
        "openInterest":      [1000,   5000,  2000],
        "impliedVolatility": [0.20,   0.25,  0.30],
        "inTheMoney":        [True,   False, False],
    })
    chain_mock.puts = pd.DataFrame({
        "strike":            [140.0, 150.0, 160.0],
        "lastPrice":         [1.0,   5.0,  10.0],
        "bid":               [0.9,   4.9,   9.9],
        "ask":               [1.1,   5.1,  10.1],
        "volume":            [200,   500,   100],
        "openInterest":      [2000,  5000,  1000],
        "impliedVolatility": [0.30,  0.25,  0.20],
        "inTheMoney":        [False, False,  True],
    })
    mock_ticker.option_chain.return_value = chain_mock

    # Fundamentals DataFrames
    q_idx = pd.to_datetime(["2024-03-31", "2023-12-31", "2023-09-30", "2023-06-30"])
    mock_ticker.quarterly_income_stmt = pd.DataFrame(
        {d: {"Total Revenue": 100_000, "Gross Profit": 43_000,
             "Operating Income": 30_000, "Net Income": 25_000,
             "Diluted EPS": 1.5, "EBITDA": 35_000} for d in q_idx}
    )
    mock_ticker.quarterly_cashflow = pd.DataFrame(
        {d: {"Operating Cash Flow": 30_000, "Free Cash Flow": 25_000,
             "Capital Expenditure": -5_000, "Repurchase Of Capital Stock": -10_000}
         for d in q_idx}
    )
    mock_ticker.quarterly_balance_sheet = pd.DataFrame(
        {d: {"Total Assets": 500_000, "Total Debt": 100_000,
             "Stockholders Equity": 50_000, "Cash And Cash Equivalents": 60_000}
         for d in q_idx}
    )
    mock_ticker.earnings_history = pd.DataFrame(
        {"epsActual": [1.1, 1.2], "epsEstimate": [1.0, 1.1], "surprisePercent": [0.10, 0.09]},
        index=pd.to_datetime(["2023-12-31", "2024-03-31"]),
    )
    mock_ticker.analyst_price_targets = {
        "mean": 200.0, "high": 250.0, "low": 160.0, "median": 195.0,
        "numberOfAnalysts": 40,
    }

    # Compliance data
    mock_ticker.insider_transactions = pd.DataFrame({
        "Start Date": [datetime.date.today() - datetime.timedelta(days=10)],
        "Insider":    ["Tim Cook"],
        "Position":   ["CEO"],
        "Transaction":["Sale"],
        "Shares":     [10_000],
        "Value":      [1_500_000],
    })
    mock_ticker.institutional_holders = pd.DataFrame({
        "Holder":        ["Vanguard", "BlackRock"],
        "Shares":        [100_000_000, 80_000_000],
        "Date Reported": [datetime.date.today(), datetime.date.today()],
        "% Out":         [0.08, 0.06],
        "Value":         [15_000_000_000, 12_000_000_000],
    })
    mock_ticker.calendar = {
        "Earnings Date": [datetime.date.today() + datetime.timedelta(days=20)]
    }
    mock_ticker.upgrades_downgrades = pd.DataFrame(
        {"Firm": ["Goldman Sachs"], "To Grade": ["Buy"],
         "From Grade": ["Neutral"], "Action": ["up"]},
        index=pd.to_datetime([datetime.date.today() - datetime.timedelta(days=5)]),
    )

    return mock_ticker


@pytest.fixture
def mock_yf_ticker():
    """Patch yfinance.Ticker in all tool modules."""
    mock_ticker = _make_mock_ticker()
    targets = [
        "tools.market_fetcher.yf.Ticker",
        "tools.fundamentals_fetcher.yf.Ticker",
        "tools.options_fetcher.yf.Ticker",
        "tools.macro_fetcher.yf.Ticker",
        "tools.compliance_fetcher.yf.Ticker",
        "tools.news_fetcher.yf.Ticker",
    ]
    patchers = [patch(t, return_value=mock_ticker) for t in targets]
    mocks = [p.start() for p in patchers]
    yield mocks[0]
    for p in patchers:
        p.stop()


@pytest.fixture
def mock_yf_download():
    """
    Patch yfinance.download in all modules that use it.

    The QuantScreener calls yf.download twice per ticker:
      1st call: monthly data over N years  → used to compute mean/std and last_close
      2nd call: last 2 days               → used to get stock_hj (today's price)

    To reliably trigger a '++' classification:
      - Historical data: stable prices with small std (e.g. linspace 100→105)
        → last_close = 105, std ≈ 0.002, mean ≈ 0.001
        → ddd = 105 * (1 + 0.001 + 2*0.002) = 105 * 1.005 ≈ 105.5
      - Current price: 200 → well above ddd → classified as '++'
    """
    # Historical monthly data — stable, small variance
    hist_dates = pd.date_range(end=datetime.date.today() - datetime.timedelta(days=30),
                               periods=60, freq='ME')
    hist_prices = np.linspace(100, 105, 60)
    hist_df = pd.DataFrame({"Close": hist_prices}, index=hist_dates)

    # Current 2-day data — big price jump to trigger ++ outlier
    today = datetime.date.today()
    curr_dates = pd.date_range(end=today, periods=2, freq='B')
    curr_df = pd.DataFrame({"Close": [105.0, 200.0]}, index=curr_dates)

    # Generic df for other tools (macro fetcher etc.)
    generic_dates = pd.date_range(end=datetime.date.today(), periods=100, freq='B')
    generic_df = pd.DataFrame({"Close": np.linspace(100, 200, 100)}, index=generic_dates)

    call_count = {"n": 0}

    def _side_effect(*args, **kwargs):
        call_count["n"] += 1
        period = kwargs.get("period", "")
        interval = kwargs.get("interval", "1d")
        if period == "2d" or interval == "1d":
            return curr_df
        if interval == "1mo":
            return hist_df
        return generic_df

    targets = [
        "agents.quant_screener.yf.download",
        "tools.macro_fetcher.yf.download",
    ]
    patchers = [patch(t, side_effect=_side_effect) for t in targets]
    mocks = [p.start() for p in patchers]
    yield mocks[0]
    for p in patchers:
        p.stop()


# ── LLM fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_openai():
    """
    Patch ChatOpenAI in all agent modules.

    Uses RunnableLambda so that the LangChain pipe operator (prompt | llm.with_structured_output(...))
    produces a real RunnableSequence whose .invoke() returns the correct Pydantic object.
    """
    objects = _make_pydantic_objects()

    def make_runnable(schema, **kwargs):
        obj = objects.get(schema.__name__, objects["SpecialistReport"])
        return RunnableLambda(lambda _: obj)

    mock_llm = MagicMock()
    mock_llm.with_structured_output.side_effect = make_runnable

    targets = [
        "agents.base_specialist.ChatOpenAI",
        "agents.manager_debate.ChatOpenAI",
        "agents.risk_pm.ChatOpenAI",
    ]
    patchers = [patch(t, return_value=mock_llm) for t in targets]
    mocks = [p.start() for p in patchers]
    yield mocks[0]
    for p in patchers:
        p.stop()


# ── External API fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def mock_finnhub():
    """Patch finnhub.Client in the news fetcher."""
    with patch("tools.news_fetcher.finnhub.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.company_news.return_value = [{
            "headline": "Finnhub News 1",
            "summary": "Summary 1",
            "source": "CNBC",
            "url": "http://example.com/finnhub1",
            "datetime": int(datetime.datetime.now().timestamp()),
        }]
        mock_client_class.return_value = mock_client
        yield mock_client_class


@pytest.fixture
def mock_requests():
    """Patch requests.get in all tool modules."""
    def _side_effect(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "sec.gov" in url:
            resp.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>8-K - Current report</title>
    <link href="http://example.com/8k"/>
    <updated>2024-01-01T12:00:00Z</updated>
  </entry>
</feed>"""
        else:
            resp.json.return_value = {
                "data": {
                    "children": [{
                        "data": {
                            "title": "Reddit Post 1",
                            "selftext": "Bullish text",
                            "score": 100,
                            "upvote_ratio": 0.95,
                            "num_comments": 50,
                            "created_utc": int(datetime.datetime.now().timestamp()),
                            "permalink": "/r/stocks/comments/1",
                        }
                    }]
                }
            }
        return resp

    targets = [
        "tools.sentiment_fetcher.requests.get",
        "tools.compliance_fetcher.requests.get",
    ]
    patchers = [patch(t, side_effect=_side_effect) for t in targets]
    mocks = [p.start() for p in patchers]
    yield mocks[0]
    for p in patchers:
        p.stop()


@pytest.fixture
def mock_subprocess():
    """Patch subprocess.run for StockTwits curl calls."""
    with patch("tools.sentiment_fetcher.subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """{
            "symbol": {"watchlist_count": 10000},
            "messages": [
                {
                    "body": "StockTwits message 1",
                    "created_at": "%s",
                    "entities": {"sentiment": {"basic": "Bullish"}}
                }
            ]
        }""" % datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        mock_run.return_value = mock_result
        yield mock_run


@pytest.fixture
def mock_fred():
    """Patch fredapi.Fred — patched at the top-level since it is imported lazily."""
    with patch("fredapi.Fred") as mock_fred_class:
        mock_client = MagicMock()
        dates = pd.date_range(end=datetime.date.today(), periods=12, freq='ME')
        series = pd.Series(np.linspace(2.0, 3.0, 12), index=dates)
        mock_client.get_series.return_value = series
        mock_fred_class.return_value = mock_client
        yield mock_fred_class


@pytest.fixture
def mock_time_sleep():
    """Patch time.sleep everywhere to avoid real delays."""
    targets = [
        "tools.sentiment_fetcher.time.sleep",
        "tools.macro_fetcher.time.sleep",
        "tools.news_fetcher.time.sleep",
    ]
    patchers = [patch(t) for t in targets]
    mocks = [p.start() for p in patchers]
    yield mocks[0]
    for p in patchers:
        p.stop()
