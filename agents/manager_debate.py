import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from schemas.state import SynthesisReport, DebateTurn, DebateSummary, OutlierTicker, SpecialistReport
from dotenv import load_dotenv

from prompts.manager_debate import (
    RESEARCH_MANAGER_PROMPT,
    DEBATE_BULL_PROMPT,
    DEBATE_BEAR_PROMPT,
    DEBATE_MODERATOR_PROMPT,
)

load_dotenv()


class ResearchManager:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)

    def synthesize(self, ticker: OutlierTicker, reports: list[SpecialistReport]) -> SynthesisReport:
        prompt = ChatPromptTemplate.from_messages([
            ("system", RESEARCH_MANAGER_PROMPT),
            ("human", "Synthesize these reports for {ticker} (Outlier classification: {classification}):\n\n{reports}")
        ])

        chain = prompt | self.llm.with_structured_output(SynthesisReport, method="function_calling")

        reports_text = "\n\n".join([
            f"--- {r.agent_name} ---\nFindings: {r.findings}\nFlags: {r.flags}"
            for r in reports
        ])

        return chain.invoke({
            "ticker": ticker.ticker,
            "classification": ticker.classification,
            "reports": reports_text
        })


class DebateAgent:
    def __init__(self, side: str, model_name: str = "gpt-4.1-mini"):
        self.side = side
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)

    def generate_arguments(self, ticker: OutlierTicker, synthesis: SynthesisReport, history: list[DebateTurn]) -> DebateTurn:
        system_prompt = DEBATE_BULL_PROMPT if self.side == "Bull" else DEBATE_BEAR_PROMPT

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Ticker: {ticker} ({classification})\nSynthesis: {synthesis}\n\nDebate History:\n{history}\n\nProvide your next turn in the debate.")
        ])

        chain = prompt | self.llm.with_structured_output(DebateTurn, method="function_calling")

        history_text = "\n".join([f"{t.agent}: {t.arguments}" for t in history]) if history else "No history yet."

        return chain.invoke({
            "ticker": ticker.ticker,
            "classification": ticker.classification,
            "synthesis": synthesis.model_dump_json(),
            "history": history_text
        })


class DebateModerator:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)

    def summarize(self, ticker: OutlierTicker, history: list[DebateTurn]) -> DebateSummary:
        prompt = ChatPromptTemplate.from_messages([
            ("system", DEBATE_MODERATOR_PROMPT),
            ("human", "Summarize this debate for {ticker}:\n\n{history}")
        ])

        chain = prompt | self.llm.with_structured_output(DebateSummary, method="function_calling")

        history_text = "\n".join([f"{t.agent}: {t.arguments}" for t in history])

        return chain.invoke({
            "ticker": ticker.ticker,
            "history": history_text
        })
