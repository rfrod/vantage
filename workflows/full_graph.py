import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from schemas.state import GraphState, OutlierTicker, SpecialistReport
from agents.quant_screener import QuantScreener
from agents.specialists import (
    MarketAnalyst, FundamentalsAnalyst, NewsAnalyst, SentimentAnalyst,
    OptionsStrategist, MacroSectorAnalyst, ComplianceAgent, DataQualityAgent
)
from agents.manager_debate import ResearchManager, DebateAgent, DebateModerator
from agents.risk_pm import RiskAnalyst, RiskCommitteeChair, PortfolioManager


# ── Debug helpers ─────────────────────────────────────────────────────────────

def _dbg(label: str, content, debug: bool):
    """Print a structured debug block for a single piece of content."""
    if not debug:
        return
    width = 60
    print(f"\n{'─'*width}")
    print(f"  [DEBUG] {label}")
    print(f"{'─'*width}")
    if isinstance(content, list):
        for item in content:
            print(f"  • {item}")
    elif hasattr(content, "model_dump"):
        print(json.dumps(content.model_dump(), indent=2))
    else:
        print(f"  {content}")
    print(f"{'─'*width}")


def _serialize_state(state: GraphState) -> dict:
    """
    Convert a GraphState TypedDict into a JSON-serialisable dict.
    Pydantic models are dumped via model_dump(); everything else is
    left as-is (strings, ints, lists of primitives).
    """
    def _convert(value):
        if value is None:
            return None
        if isinstance(value, list):
            return [_convert(v) for v in value]
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return value

    return {k: _convert(v) for k, v in state.items()}


def _print_state(node_name: str, state: GraphState, debug: bool):
    """Print the full state snapshot after a node completes."""
    if not debug:
        return
    width = 60
    print(f"\n{'═'*width}")
    print(f"  [STATE] After node: {node_name}")
    print(f"{'═'*width}")
    print(json.dumps(_serialize_state(state), indent=2, default=str))
    print(f"{'═'*width}")


# ── Workflow factory ──────────────────────────────────────────────────────────

def create_full_workflow(debug: bool = False):
    workflow = StateGraph(GraphState)

    # ── 1. Quant Screening ────────────────────────────────────────────────
    def quant_screening_node(state: GraphState):
        tickers = state.get("tickers", [])
        screener = QuantScreener(verbose=debug)
        outliers = screener.screen_tickers(tickers)
        print(f"[Quant Screening] {len(outliers)} outlier(s) found out of {len(tickers)} ticker(s).")
        if debug:
            for o in outliers:
                print(f"  → {o.ticker}  [{o.classification}]")
        new_state = {**state, "outliers": outliers}
        _print_state("quant_screening", new_state, debug)
        return {"outliers": outliers}

    workflow.add_node("quant_screening", quant_screening_node)

    def check_outliers(state: GraphState):
        return "end" if not state.get("outliers") else "set_current_ticker"

    # ── 2. Set Current Ticker ─────────────────────────────────────────────
    def set_current_ticker_node(state: GraphState):
        outliers = state.get("outliers", [])
        if not outliers:
            new_state = {**state, "current_ticker": None}
            _print_state("set_current_ticker", new_state, debug)
            return {"current_ticker": None}
        current = outliers[0]
        print(f"\n[Pipeline] Processing ticker: {current.ticker}  [{current.classification}]")
        new_state = {**state, "current_ticker": current, "specialist_reports": []}
        _print_state("set_current_ticker", new_state, debug)
        return {"current_ticker": current, "specialist_reports": []}

    workflow.add_node("set_current_ticker", set_current_ticker_node)

    # ── 3. Parallel Specialist Analysis ───────────────────────────────────
    def specialist_analysis_node(state: GraphState):
        ticker = state.get("current_ticker")
        if not ticker:
            return {}

        print(f"[Specialist Analysis] Running 8 specialist agents for {ticker.ticker}...")
        agents = [
            MarketAnalyst(),
            FundamentalsAnalyst(),
            NewsAnalyst(),
            SentimentAnalyst(),
            OptionsStrategist(),
            MacroSectorAnalyst(),
            ComplianceAgent(),
            DataQualityAgent()
        ]

        reports = []
        for agent in agents:
            try:
                report = agent.analyze(ticker, {"mock_data": "True"})
                reports.append(report)
                if debug:
                    _dbg(f"{agent.role_name} — Findings", report.findings, debug=True)
                    if report.flags:
                        _dbg(f"{agent.role_name} — Flags", report.flags, debug=True)
                else:
                    print(f"  ✓ {agent.role_name}")
            except Exception as e:
                print(f"  ✗ Error in {agent.role_name}: {e}")

        new_state = {**state, "specialist_reports": reports}
        _print_state("specialist_analysis", new_state, debug)
        return {"specialist_reports": reports}

    workflow.add_node("specialist_analysis", specialist_analysis_node)

    # ── 4. Research Manager Synthesis ─────────────────────────────────────
    def synthesis_node(state: GraphState):
        ticker = state.get("current_ticker")
        reports = state.get("specialist_reports", [])
        if not ticker or not reports:
            return {}

        print(f"[Research Manager] Synthesizing {len(reports)} specialist report(s)...")
        manager = ResearchManager()
        synthesis = manager.synthesize(ticker, reports)
        _dbg("Research Manager — Synthesis", synthesis, debug=debug)
        new_state = {**state, "synthesis": synthesis, "debate_history": []}
        _print_state("synthesis", new_state, debug)
        return {"synthesis": synthesis, "debate_history": []}

    workflow.add_node("synthesis", synthesis_node)

    # ── 5. Bull / Bear Debate ─────────────────────────────────────────────
    def debate_bull_node(state: GraphState):
        ticker = state.get("current_ticker")
        synthesis = state.get("synthesis")
        history = state.get("debate_history", [])

        print("[Debate] Bull agent presenting case...")
        bull = DebateAgent("Bull")
        turn = bull.generate_arguments(ticker, synthesis, history)
        _dbg("Bull Agent — Arguments", turn.arguments, debug=debug)
        new_state = {**state, "debate_history": history + [turn]}
        _print_state("debate_bull", new_state, debug)
        return {"debate_history": [turn]}

    workflow.add_node("debate_bull", debate_bull_node)

    def debate_bear_node(state: GraphState):
        ticker = state.get("current_ticker")
        synthesis = state.get("synthesis")
        history = state.get("debate_history", [])

        print("[Debate] Bear agent presenting case...")
        bear = DebateAgent("Bear")
        turn = bear.generate_arguments(ticker, synthesis, history)
        _dbg("Bear Agent — Arguments", turn.arguments, debug=debug)
        new_state = {**state, "debate_history": history + [turn]}
        _print_state("debate_bear", new_state, debug)
        return {"debate_history": [turn]}

    workflow.add_node("debate_bear", debate_bear_node)

    def debate_moderator_node(state: GraphState):
        ticker = state.get("current_ticker")
        history = state.get("debate_history", [])

        print("[Debate] Moderator summarizing debate...")
        moderator = DebateModerator()
        summary = moderator.summarize(ticker, history)
        _dbg("Debate Moderator — Summary", summary, debug=debug)
        new_state = {**state, "debate_summary": summary, "risk_recommendations": []}
        _print_state("debate_moderator", new_state, debug)
        return {"debate_summary": summary, "risk_recommendations": []}

    workflow.add_node("debate_moderator", debate_moderator_node)

    # ── 6. Risk Sizing ────────────────────────────────────────────────────
    def risk_analysis_node(state: GraphState):
        ticker = state.get("current_ticker")
        synthesis = state.get("synthesis")
        debate_summary = state.get("debate_summary")

        print("[Risk Committee] Running Conservative / Neutral / Aggressive risk analysts...")
        analysts = [
            RiskAnalyst("Conservative"),
            RiskAnalyst("Neutral"),
            RiskAnalyst("Aggressive")
        ]

        recs = []
        for analyst in analysts:
            rec = analyst.assess_risk(ticker, synthesis, debate_summary)
            recs.append(rec)
            _dbg(f"Risk Analyst ({analyst.persona}) — Recommendation", rec, debug=debug)

        new_state = {**state, "risk_recommendations": recs}
        _print_state("risk_analysis", new_state, debug)
        return {"risk_recommendations": recs}

    workflow.add_node("risk_analysis", risk_analysis_node)

    def risk_committee_node(state: GraphState):
        ticker = state.get("current_ticker")
        recs = state.get("risk_recommendations", [])

        print("[Risk Committee] Chair summarizing risk recommendations...")
        chair = RiskCommitteeChair()
        summary = chair.summarize(ticker, recs)
        _dbg("Risk Committee Chair — Summary", summary, debug=debug)
        new_state = {**state, "risk_summary": summary}
        _print_state("risk_committee", new_state, debug)
        return {"risk_summary": summary}

    workflow.add_node("risk_committee", risk_committee_node)

    # ── 7. Portfolio Manager Final Decision ───────────────────────────────
    def pm_decision_node(state: GraphState):
        ticker = state.get("current_ticker")
        synthesis = state.get("synthesis")
        debate_summary = state.get("debate_summary")
        risk_summary = state.get("risk_summary")

        print("[Portfolio Manager] Making final investment decision...")
        pm = PortfolioManager()
        decision = pm.make_decision(ticker, synthesis, debate_summary, risk_summary)
        _dbg("Portfolio Manager — Full Decision", decision, debug=debug)

        # Save output — path relative to this file so it works on any machine
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{ticker.ticker}_report.json"), "w") as f:
            f.write(decision.model_dump_json(indent=2))

        new_state = {**state, "final_decisions": state.get("final_decisions", []) + [decision]}
        _print_state("pm_decision", new_state, debug)
        return {"final_decisions": [decision]}

    workflow.add_node("pm_decision", pm_decision_node)

    # ── Graph Edges ───────────────────────────────────────────────────────
    workflow.set_entry_point("quant_screening")

    workflow.add_conditional_edges(
        "quant_screening",
        check_outliers,
        {"end": END, "set_current_ticker": "set_current_ticker"}
    )

    workflow.add_edge("set_current_ticker", "specialist_analysis")
    workflow.add_edge("specialist_analysis", "synthesis")
    workflow.add_edge("synthesis", "debate_bull")
    workflow.add_edge("debate_bull", "debate_bear")
    workflow.add_edge("debate_bear", "debate_moderator")
    workflow.add_edge("debate_moderator", "risk_analysis")
    workflow.add_edge("risk_analysis", "risk_committee")
    workflow.add_edge("risk_committee", "pm_decision")
    workflow.add_edge("pm_decision", END)

    return workflow.compile()
