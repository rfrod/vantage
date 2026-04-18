"""
Integration tests for the full LangGraph workflow.
All external API calls are mocked via fixtures in conftest.py.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.full_graph import create_full_workflow
from schemas.state import OutlierTicker


def _base_state(tickers):
    return {
        "tickers": tickers,
        "outliers": [],
        "current_ticker": None,
        "specialist_reports": [],
        "synthesis": None,
        "debate_history": [],
        "debate_summary": None,
        "risk_recommendations": [],
        "risk_summary": None,
        "final_decisions": [],
        "errors": [],
    }


class TestFullWorkflow:
    def test_happy_path(
        self,
        mock_openai,
        mock_yf_ticker,
        mock_yf_download,
        mock_subprocess,
        mock_requests,
        mock_time_sleep,
        tmp_path,
        monkeypatch,
    ):
        """Full pipeline runs end-to-end and produces a FinalDecision for a single ticker."""
        monkeypatch.chdir(tmp_path)

        workflow = create_full_workflow(debug=False)
        final_state = workflow.invoke(_base_state(["AAPL"]))

        # Specialist analysis — 8 agents
        assert len(final_state["specialist_reports"]) == 8

        # Synthesis
        assert final_state["synthesis"] is not None
        assert final_state["synthesis"].ticker == "AAPL"

        # Debate — 2 turns (Bull + Bear)
        assert len(final_state["debate_history"]) == 2
        assert final_state["debate_summary"] is not None

        # Risk — 3 personas
        assert len(final_state["risk_recommendations"]) == 3
        assert final_state["risk_summary"] is not None

        # Final decision
        assert len(final_state["final_decisions"]) == 1
        assert final_state["final_decisions"][0].ticker == "AAPL"

    def test_no_outliers_exits_early(
        self,
        tmp_path,
        monkeypatch,
    ):
        """When no outlier is detected, the pipeline terminates after quant screening."""
        monkeypatch.chdir(tmp_path)

        with patch("workflows.full_graph.QuantScreener") as mock_screener_cls:
            mock_screener = mock_screener_cls.return_value
            mock_screener.screen_tickers.return_value = []

            workflow = create_full_workflow(debug=False)
            final_state = workflow.invoke(_base_state(["FLAT"]))

        assert len(final_state["outliers"]) == 0
        assert final_state["current_ticker"] is None
        assert len(final_state["specialist_reports"]) == 0
        assert final_state["synthesis"] is None
        assert len(final_state["final_decisions"]) == 0

    def test_debug_mode_does_not_crash(
        self,
        mock_openai,
        mock_yf_ticker,
        mock_yf_download,
        mock_subprocess,
        mock_requests,
        mock_time_sleep,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        """Running with debug=True should produce extra output but not raise."""
        monkeypatch.chdir(tmp_path)

        workflow = create_full_workflow(debug=True)
        final_state = workflow.invoke(_base_state(["AAPL"]))

        captured = capsys.readouterr()
        assert "[STATE]" in captured.out
        assert "[DEBUG]" in captured.out
        assert len(final_state["final_decisions"]) == 1

    def test_multiple_tickers_all_processed(
        self,
        mock_openai,
        mock_yf_ticker,
        mock_yf_download,
        mock_subprocess,
        mock_requests,
        mock_time_sleep,
        tmp_path,
        monkeypatch,
    ):
        """
        When the QuantScreener returns multiple outliers, the graph loops and
        produces a FinalDecision for EACH outlier.
        Note: the mock LLM always returns ticker='AAPL' in FinalDecision regardless
        of the current ticker — we verify count and empty queue, not ticker names.
        """
        monkeypatch.chdir(tmp_path)

        # Inject two outliers directly so the loop runs twice
        with patch("workflows.full_graph.QuantScreener") as mock_screener_cls:
            mock_screener = mock_screener_cls.return_value
            mock_screener.screen_tickers.return_value = [
                OutlierTicker(ticker="AAPL", classification="++"),
                OutlierTicker(ticker="MSFT", classification="--"),
            ]

            workflow = create_full_workflow(debug=False)
            final_state = workflow.invoke(_base_state(["AAPL", "MSFT"]))

        # Both tickers must have been processed — 2 decisions produced
        assert len(final_state["final_decisions"]) == 2

        # Outliers queue should be empty after the loop
        assert len(final_state["outliers"]) == 0

        # Each run should have produced exactly 8 specialist reports (not accumulated)
        assert len(final_state["specialist_reports"]) == 8

    def test_min_sigma_2_filters_single_sigma(
        self,
        mock_openai,
        mock_yf_ticker,
        mock_yf_download,
        mock_subprocess,
        mock_requests,
        mock_time_sleep,
        tmp_path,
        monkeypatch,
    ):
        """
        With min_sigma=2, single-sigma outliers (+/-) are filtered out.
        Only ++ and -- outliers proceed to the full pipeline.
        """
        monkeypatch.chdir(tmp_path)

        with patch("workflows.full_graph.QuantScreener") as mock_screener_cls:
            mock_screener = mock_screener_cls.return_value
            # Return one strong and one weak outlier
            mock_screener.screen_tickers.return_value = [
                OutlierTicker(ticker="NVDA", classification="++"),  # strong — should pass
                OutlierTicker(ticker="TSLA", classification="+"),   # weak — should be filtered
            ]

            workflow = create_full_workflow(debug=False, min_sigma=2)
            final_state = workflow.invoke(_base_state(["NVDA", "TSLA"]))

        # Only the strong outlier (NVDA) should have been processed — 1 decision
        assert len(final_state["final_decisions"]) == 1
        # The outliers queue should be empty (TSLA was filtered, NVDA was processed)
        assert len(final_state["outliers"]) == 0

    def test_min_sigma_1_passes_all_outliers(
        self,
        mock_openai,
        mock_yf_ticker,
        mock_yf_download,
        mock_subprocess,
        mock_requests,
        mock_time_sleep,
        tmp_path,
        monkeypatch,
    ):
        """
        With min_sigma=1 (default), all outliers (+, ++, -, --) are processed.
        """
        monkeypatch.chdir(tmp_path)

        with patch("workflows.full_graph.QuantScreener") as mock_screener_cls:
            mock_screener = mock_screener_cls.return_value
            mock_screener.screen_tickers.return_value = [
                OutlierTicker(ticker="AAPL", classification="+"),
                OutlierTicker(ticker="TSLA", classification="-"),
            ]

            workflow = create_full_workflow(debug=False, min_sigma=1)
            final_state = workflow.invoke(_base_state(["AAPL", "TSLA"]))

        # Both single-sigma outliers should have been processed
        assert len(final_state["final_decisions"]) == 2
        assert len(final_state["outliers"]) == 0
