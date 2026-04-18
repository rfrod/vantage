from typing import List, Dict, Any, Optional, Annotated
import operator
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class OutlierTicker(BaseModel):
    ticker: str = Field(description="The stock ticker symbol")
    classification: str = Field(description="The outlier classification: '--', '-', '+', or '++'")
    
class QuantScreeningOutput(BaseModel):
    outliers: List[OutlierTicker] = Field(description="List of tickers that passed the outlier screening")

class SpecialistReport(BaseModel):
    agent_name: str = Field(description="Name of the specialist agent")
    findings: List[str] = Field(description="Key findings in bullet points")
    flags: List[str] = Field(description="Any risks, red flags, or uncertainties identified")

class SynthesisReport(BaseModel):
    ticker: str
    outlier_classification: str
    what_happened: List[str]
    key_technical_findings: List[str]
    key_fundamental_findings: List[str]
    key_news_findings: List[str]
    key_sentiment_findings: List[str]
    key_options_findings: List[str]
    major_risks: List[str]
    initial_directional_bias: str = Field(description="Bullish, Bearish, or Neutral")
    candidate_trade_expressions: List[str]

class DebateTurn(BaseModel):
    agent: str = Field(description="Bull or Bear")
    arguments: List[str] = Field(description="Concise, evidence-based arguments")

class DebateSummary(BaseModel):
    strongest_bull_arguments: List[str]
    strongest_bear_arguments: List[str]
    unresolved_uncertainties: List[str]
    current_score: str = Field(description="Which side is currently more convincing")

class RiskRecommendation(BaseModel):
    persona: str = Field(description="Conservative, Neutral, or Aggressive")
    whether_to_trade: bool
    trade_type: str = Field(
        description=(
            "One of: Buy Stock, Sell Cash-Secured Put, Buy Call, Buy Put, "
            "Buy Call Spread, Buy Put Spread, Reduce/Trim, or None"
        )
    )
    suggested_position_size_pct: float
    max_loss_framing: str
    stop_review_conditions: str
    invalidation_thesis: str

class RiskSummary(BaseModel):
    conservative_rec: RiskRecommendation
    neutral_rec: RiskRecommendation
    aggressive_rec: RiskRecommendation
    committee_consensus: str

class FinalDecision(BaseModel):
    ticker: str
    final_action: str = Field(
        description=(
            "One of: BUY STOCK, SELL CASH-SECURED PUT, BUY CALL, BUY PUT, "
            "BUY CALL SPREAD, BUY PUT SPREAD, REDUCE/TRIM, WATCHLIST, or REJECT"
        )
    )
    confidence_score: int = Field(description="Confidence score from 1 to 10")
    top_3_reasons: List[str]
    top_3_risks: List[str]
    recommended_position_size: str
    recommended_strike_expiration: str = Field(
        description=(
            "For options actions: describe the recommended strike(s) and expiration. "
            "For spreads: include both legs (e.g. 'Buy 150C / Sell 160C, 45 DTE'). "
            "For stock actions or non-trade outcomes: 'N/A'."
        )
    )
    spread_details: Optional[str] = Field(
        default=None,
        description=(
            "For BUY CALL SPREAD or BUY PUT SPREAD only: describe both legs, "
            "max profit, max loss, and breakeven. None for all other actions."
        )
    )
    what_to_monitor_next: List[str]
    audit_trail: List[str] = Field(description="Which agents influenced the decision most")

def _replace(old, new):
    """Reducer that replaces the existing list with the new one (used for per-ticker fields)."""
    return new


class GraphState(TypedDict):
    tickers: List[str]
    outliers: List[OutlierTicker]
    current_ticker: Optional[OutlierTicker]
    # Per-ticker fields reset by set_current_ticker_node:
    # specialist_reports is populated in one shot — use replace reducer
    specialist_reports: Annotated[List[SpecialistReport], _replace]
    synthesis: Optional[SynthesisReport]
    # debate_history and risk_recommendations are built incrementally across nodes
    # — use append reducer; reset to [] by set_current_ticker_node each new ticker
    debate_history: Annotated[List[DebateTurn], operator.add]
    debate_summary: Optional[DebateSummary]
    risk_recommendations: Annotated[List[RiskRecommendation], operator.add]
    risk_summary: Optional[RiskSummary]
    # Cross-ticker accumulator: appended across all tickers
    final_decisions: Annotated[List[FinalDecision], operator.add]
    errors: Annotated[List[str], operator.add]
