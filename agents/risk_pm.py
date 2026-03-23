import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from schemas.state import RiskRecommendation, RiskSummary, FinalDecision, OutlierTicker, SynthesisReport, DebateSummary
from dotenv import load_dotenv

from prompts.risk_pm import (
    RISK_ANALYST_CONSERVATIVE_PROMPT,
    RISK_ANALYST_NEUTRAL_PROMPT,
    RISK_ANALYST_AGGRESSIVE_PROMPT,
    RISK_ANALYST_COMMON_PROMPT,
    RISK_COMMITTEE_CHAIR_PROMPT,
    PORTFOLIO_MANAGER_PROMPT,
)

load_dotenv()


class RiskAnalyst:
    def __init__(self, persona: str, model_name: str = "gpt-4.1-mini"):
        self.persona = persona
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)

    def assess_risk(self, ticker: OutlierTicker, synthesis: SynthesisReport, debate_summary: DebateSummary) -> RiskRecommendation:
        if self.persona == "Conservative":
            base_prompt = RISK_ANALYST_CONSERVATIVE_PROMPT
        elif self.persona == "Neutral":
            base_prompt = RISK_ANALYST_NEUTRAL_PROMPT
        else:
            base_prompt = RISK_ANALYST_AGGRESSIVE_PROMPT

        system_prompt = base_prompt + RISK_ANALYST_COMMON_PROMPT

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Assess risk for {ticker} ({classification}).\n\nSynthesis: {synthesis}\n\nDebate Summary: {debate_summary}")
        ])

        chain = prompt | self.llm.with_structured_output(RiskRecommendation, method="function_calling")

        return chain.invoke({
            "ticker": ticker.ticker,
            "classification": ticker.classification,
            "synthesis": synthesis.model_dump_json(),
            "debate_summary": debate_summary.model_dump_json()
        })


class RiskCommitteeChair:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)

    def summarize(self, ticker: OutlierTicker, recs: list[RiskRecommendation]) -> RiskSummary:
        prompt = ChatPromptTemplate.from_messages([
            ("system", RISK_COMMITTEE_CHAIR_PROMPT),
            ("human", "Summarize risk recommendations for {ticker}:\n\n{recs}")
        ])

        chain = prompt | self.llm.with_structured_output(RiskSummary, method="function_calling")

        recs_text = "\n\n".join([f"--- {r.persona} ---\n{r.model_dump_json()}" for r in recs])

        return chain.invoke({
            "ticker": ticker.ticker,
            "recs": recs_text
        })


class PortfolioManager:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)

    def make_decision(self, ticker: OutlierTicker, synthesis: SynthesisReport, debate_summary: DebateSummary, risk_summary: RiskSummary) -> FinalDecision:
        prompt = ChatPromptTemplate.from_messages([
            ("system", PORTFOLIO_MANAGER_PROMPT),
            ("human", "Make final decision for {ticker} ({classification}).\n\nSynthesis: {synthesis}\n\nDebate Summary: {debate_summary}\n\nRisk Summary: {risk_summary}")
        ])

        chain = prompt | self.llm.with_structured_output(FinalDecision, method="function_calling")

        return chain.invoke({
            "ticker": ticker.ticker,
            "classification": ticker.classification,
            "synthesis": synthesis.model_dump_json(),
            "debate_summary": debate_summary.model_dump_json(),
            "risk_summary": risk_summary.model_dump_json()
        })
