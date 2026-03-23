"""
Unit tests for all Vantage agents.
All LLM and external API calls are mocked via fixtures in conftest.py.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.state import (
    OutlierTicker, SpecialistReport, SynthesisReport,
    DebateTurn, DebateSummary, RiskRecommendation, RiskSummary, FinalDecision
)
from agents.specialists import (
    MarketAnalyst, FundamentalsAnalyst, NewsAnalyst, SentimentAnalyst,
    OptionsStrategist, MacroSectorAnalyst, ComplianceAgent, DataQualityAgent
)
from agents.manager_debate import ResearchManager, DebateAgent, DebateModerator
from agents.risk_pm import RiskAnalyst, RiskCommitteeChair, PortfolioManager
from agents.quant_screener import QuantScreener


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_ticker():
    return OutlierTicker(ticker="AAPL", classification="++")


@pytest.fixture
def sample_reports():
    return [
        SpecialistReport(agent_name="Market Analyst", findings=["Up"], flags=[]),
        SpecialistReport(agent_name="News Analyst", findings=["Good news"], flags=[]),
    ]


@pytest.fixture
def sample_synthesis():
    return SynthesisReport(
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


@pytest.fixture
def sample_debate_summary():
    return DebateSummary(
        strongest_bull_arguments=["Growth"],
        strongest_bear_arguments=["Valuation"],
        unresolved_uncertainties=["Macro"],
        current_score="Bullish edge",
    )


@pytest.fixture
def sample_risk_recs():
    def _make(persona, pct):
        return RiskRecommendation(
            persona=persona, whether_to_trade=True, trade_type="Buy Stock",
            suggested_position_size_pct=pct, max_loss_framing=f"{pct/5:.0f}%",
            stop_review_conditions="Below 50d MA", invalidation_thesis="Earnings miss",
        )
    return [_make("Conservative", 2.0), _make("Neutral", 5.0), _make("Aggressive", 10.0)]


@pytest.fixture
def sample_risk_summary(sample_risk_recs):
    return RiskSummary(
        conservative_rec=sample_risk_recs[0],
        neutral_rec=sample_risk_recs[1],
        aggressive_rec=sample_risk_recs[2],
        committee_consensus="Proceed with caution",
    )


# ── QuantScreener ─────────────────────────────────────────────────────────────

class TestQuantScreener:
    def test_detects_outlier(self, mock_yf_download):
        screener = QuantScreener(verbose=False)
        outliers = screener.screen_tickers(["AAPL"])
        assert len(outliers) > 0
        assert outliers[0].ticker == "AAPL"
        assert outliers[0].classification in ["+", "++", "-", "--"]

    def test_returns_empty_for_no_data(self, mock_yf_download):
        # Override with empty DataFrame
        with pytest.MonkeyPatch().context() as mp:
            import agents.quant_screener as qs_mod
            import yfinance as yf
            mp.setattr(qs_mod.yf, "download", lambda *a, **kw: pd.DataFrame())
            screener = QuantScreener(verbose=False)
            outliers = screener.screen_tickers(["EMPTY"])
            assert outliers == []

    def test_screen_multiple_tickers(self, mock_yf_download):
        screener = QuantScreener(verbose=False)
        outliers = screener.screen_tickers(["AAPL", "MSFT"])
        assert all(o.ticker in ["AAPL", "MSFT"] for o in outliers)


# ── Specialist Agents ─────────────────────────────────────────────────────────

class TestSpecialistAgents:
    def test_market_analyst(self, mock_openai, mock_yf_ticker, mock_yf_download, mock_time_sleep, sample_ticker):
        report = MarketAnalyst().analyze(sample_ticker)
        assert isinstance(report, SpecialistReport)
        assert report.agent_name == "Mock Agent"
        assert len(report.findings) > 0

    def test_fundamentals_analyst(self, mock_openai, mock_yf_ticker, sample_ticker):
        report = FundamentalsAnalyst().analyze(sample_ticker)
        assert isinstance(report, SpecialistReport)
        assert len(report.findings) > 0

    def test_news_analyst(self, mock_openai, mock_yf_ticker, sample_ticker):
        if "FINNHUB_API_KEY" in os.environ:
            del os.environ["FINNHUB_API_KEY"]
        report = NewsAnalyst().analyze(sample_ticker)
        assert isinstance(report, SpecialistReport)

    def test_sentiment_analyst(self, mock_openai, mock_subprocess, mock_requests, mock_time_sleep, sample_ticker):
        report = SentimentAnalyst().analyze(sample_ticker)
        assert isinstance(report, SpecialistReport)

    def test_options_strategist(self, mock_openai, mock_yf_ticker, sample_ticker):
        report = OptionsStrategist().analyze(sample_ticker)
        assert isinstance(report, SpecialistReport)

    def test_macro_sector_analyst(self, mock_openai, mock_yf_ticker, mock_yf_download, mock_time_sleep, sample_ticker):
        if "FRED_API_KEY" in os.environ:
            del os.environ["FRED_API_KEY"]
        report = MacroSectorAnalyst().analyze(sample_ticker)
        assert isinstance(report, SpecialistReport)

    def test_compliance_agent(self, mock_openai, mock_yf_ticker, mock_requests, sample_ticker):
        report = ComplianceAgent().analyze(sample_ticker)
        assert isinstance(report, SpecialistReport)

    def test_data_quality_agent(self, mock_openai, mock_yf_ticker, sample_ticker):
        report = DataQualityAgent().analyze(sample_ticker)
        assert isinstance(report, SpecialistReport)

    def test_all_8_agents_instantiate(self):
        """All 8 specialist agents must instantiate without error."""
        agents = [
            MarketAnalyst(), FundamentalsAnalyst(), NewsAnalyst(),
            SentimentAnalyst(), OptionsStrategist(), MacroSectorAnalyst(),
            ComplianceAgent(), DataQualityAgent(),
        ]
        assert len(agents) == 8


# ── ResearchManager ───────────────────────────────────────────────────────────

class TestResearchManager:
    def test_synthesize_returns_synthesis_report(self, mock_openai, sample_ticker, sample_reports):
        synthesis = ResearchManager().synthesize(sample_ticker, sample_reports)
        assert isinstance(synthesis, SynthesisReport)
        assert synthesis.ticker == "AAPL"

    def test_synthesis_has_required_fields(self, mock_openai, sample_ticker, sample_reports):
        synthesis = ResearchManager().synthesize(sample_ticker, sample_reports)
        assert synthesis.initial_directional_bias in ["Bullish", "Bearish", "Neutral"]
        assert len(synthesis.candidate_trade_expressions) > 0
        assert len(synthesis.what_happened) > 0


# ── Debate Agents ─────────────────────────────────────────────────────────────

class TestDebateAgents:
    def test_bull_generates_turn(self, mock_openai, sample_ticker, sample_synthesis):
        turn = DebateAgent("Bull").generate_arguments(sample_ticker, sample_synthesis, [])
        assert isinstance(turn, DebateTurn)
        assert len(turn.arguments) > 0

    def test_bear_generates_turn(self, mock_openai, sample_ticker, sample_synthesis):
        turn = DebateAgent("Bear").generate_arguments(sample_ticker, sample_synthesis, [])
        assert isinstance(turn, DebateTurn)
        assert len(turn.arguments) > 0

    def test_debate_with_history(self, mock_openai, sample_ticker, sample_synthesis):
        history = [DebateTurn(agent="Bull", arguments=["Up"])]
        turn = DebateAgent("Bear").generate_arguments(sample_ticker, sample_synthesis, history)
        assert isinstance(turn, DebateTurn)

    def test_moderator_summarizes(self, mock_openai, sample_ticker):
        history = [
            DebateTurn(agent="Bull", arguments=["Up"]),
            DebateTurn(agent="Bear", arguments=["Down"]),
        ]
        summary = DebateModerator().summarize(sample_ticker, history)
        assert isinstance(summary, DebateSummary)
        assert len(summary.strongest_bull_arguments) > 0
        assert len(summary.strongest_bear_arguments) > 0


# ── Risk Agents ───────────────────────────────────────────────────────────────

class TestRiskAgents:
    @pytest.mark.parametrize("persona", ["Conservative", "Neutral", "Aggressive"])
    def test_risk_analyst_personas(self, mock_openai, sample_ticker, sample_synthesis, sample_debate_summary, persona):
        rec = RiskAnalyst(persona).assess_risk(sample_ticker, sample_synthesis, sample_debate_summary)
        assert isinstance(rec, RiskRecommendation)
        assert rec.whether_to_trade in [True, False]
        assert rec.suggested_position_size_pct >= 0

    def test_risk_committee_chair(self, mock_openai, sample_ticker, sample_risk_recs):
        summary = RiskCommitteeChair().summarize(sample_ticker, sample_risk_recs)
        assert isinstance(summary, RiskSummary)
        assert summary.committee_consensus != ""


# ── PortfolioManager ──────────────────────────────────────────────────────────

class TestPortfolioManager:
    def test_make_decision_returns_final_decision(
        self, mock_openai, sample_ticker, sample_synthesis, sample_debate_summary, sample_risk_summary
    ):
        decision = PortfolioManager().make_decision(
            sample_ticker, sample_synthesis, sample_debate_summary, sample_risk_summary
        )
        assert isinstance(decision, FinalDecision)
        assert decision.final_action in [
            "BUY STOCK", "SELL CASH-SECURED PUT", "WATCHLIST", "REJECT", "BUY STOCK"
        ]
        assert 1 <= decision.confidence_score <= 10

    def test_decision_has_audit_trail(
        self, mock_openai, sample_ticker, sample_synthesis, sample_debate_summary, sample_risk_summary
    ):
        decision = PortfolioManager().make_decision(
            sample_ticker, sample_synthesis, sample_debate_summary, sample_risk_summary
        )
        assert len(decision.audit_trail) > 0
        assert len(decision.top_3_reasons) == 3
        assert len(decision.top_3_risks) == 3
        assert len(decision.what_to_monitor_next) > 0
