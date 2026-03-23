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
    trade_type: str = Field(description="Buy Stock, Sell Cash-Secured Put, or None")
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
    final_action: str = Field(description="BUY STOCK, SELL CASH-SECURED PUT, WATCHLIST, or REJECT")
    confidence_score: int = Field(description="Confidence score from 1 to 10")
    top_3_reasons: List[str]
    top_3_risks: List[str]
    recommended_position_size: str
    recommended_strike_expiration: str = Field(description="Strike/expiration if selling PUTs, otherwise 'N/A'")
    what_to_monitor_next: List[str]
    audit_trail: List[str] = Field(description="Which agents influenced the decision most")

class GraphState(TypedDict):
    tickers: List[str]
    outliers: List[OutlierTicker]
    current_ticker: Optional[OutlierTicker]
    specialist_reports: Annotated[List[SpecialistReport], operator.add]
    synthesis: Optional[SynthesisReport]
    debate_history: Annotated[List[DebateTurn], operator.add]
    debate_summary: Optional[DebateSummary]
    risk_recommendations: Annotated[List[RiskRecommendation], operator.add]
    risk_summary: Optional[RiskSummary]
    final_decisions: Annotated[List[FinalDecision], operator.add]
    errors: Annotated[List[str], operator.add]
