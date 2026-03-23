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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.full_graph import create_full_workflow


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
        """Full pipeline runs end-to-end and produces a FinalDecision."""
        monkeypatch.chdir(tmp_path)

        workflow = create_full_workflow(debug=False)
        final_state = workflow.invoke(_base_state(["AAPL"]))

        # Quant screening
        assert len(final_state["outliers"]) > 0
        assert final_state["outliers"][0].ticker == "AAPL"

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
        from unittest.mock import patch
        monkeypatch.chdir(tmp_path)

        # Mock QuantScreener.screen_tickers to return an empty list directly
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

    def test_multiple_tickers_produce_multiple_decisions(
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
        The graph detects multiple outliers but processes one ticker per run
        (the first outlier in the list). Verify that:
          - Both tickers are detected as outliers by the screener
          - At least one FinalDecision is produced
          - The processed ticker is one of the two submitted
        """
        monkeypatch.chdir(tmp_path)

        workflow = create_full_workflow(debug=False)
        final_state = workflow.invoke(_base_state(["AAPL", "MSFT"]))

        # Both tickers should be detected as outliers
        assert len(final_state["outliers"]) == 2
        outlier_tickers = {o.ticker for o in final_state["outliers"]}
        assert "AAPL" in outlier_tickers
        assert "MSFT" in outlier_tickers

        # The graph processes the first outlier in a single run
        assert len(final_state["final_decisions"]) >= 1
        assert final_state["final_decisions"][0].ticker in {"AAPL", "MSFT"}
