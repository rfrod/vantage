from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from schemas.state import SpecialistReport, OutlierTicker
import os
from dotenv import load_dotenv

load_dotenv()

class BaseSpecialistAgent:
    def __init__(self, role_name: str, role_description: str, model_name: str = "gpt-4.1-mini"):
        self.role_name = role_name
        self.role_description = role_description
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        
    def _get_system_prompt(self) -> str:
        return f"""You are the {self.role_name}.
{self.role_description}

OUTPUT CONSTRAINTS:
- DO NOT write introductory preambles
- DO NOT write motivational filler
- DO NOT produce essays
- USE bullets when possible
- USE evidence
- BE concise
- BE auditable
- FLAG uncertainty clearly
- Distinguish facts from inferences

Format your output to strictly match the requested JSON schema."""

    def analyze(self, ticker_data: OutlierTicker, additional_context: dict = None) -> SpecialistReport:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "Analyze the ticker {ticker} with outlier classification {classification}.\nContext: {context}")
        ])
        
        chain = prompt | self.llm.with_structured_output(SpecialistReport, method="function_calling")
        
        context_str = str(additional_context) if additional_context else "No additional context."
        
        return chain.invoke({
            "ticker": ticker_data.ticker,
            "classification": ticker_data.classification,
            "context": context_str
        })
