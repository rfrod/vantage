# Vantage Investment Research System

**Vantage** is a multi-agent investment research and decision-support system designed to screen stock and options trading opportunities, mimicking the workflow of an institutional research desk.

## System Architecture

The system is built on **LangGraph** for workflow orchestration and uses **Pydantic** to ensure strict output structuring at every stage.

### Orchestration Flow (LangGraph)

1. **Quant Screening Agent**: The entry point of the system. It applies a standard deviation-based outlier detection routine (using Yahoo Finance data) to classify assets into outlier categories (`--`, `-`, `+`, `++`). Only assets that pass this filter move forward.
2. **Parallel Specialist Analysis**: For each outlier, the system triggers multiple specialist agents in parallel:
   - Market Analyst
   - Fundamentals Analyst
   - News Analyst
   - Sentiment Analyst
   - Options Strategist
   - Macro/Sector Analyst
   - Compliance Agent
   - Data Quality Agent
3. **Research Manager Synthesis**: The Research Manager receives the specialist reports and condenses them into a concise summary, focusing on practical, decision-oriented insights.
4. **Bull/Bear Debate**: An adversarial reasoning stage where a Bull Agent and a Bear Agent debate the case. Afterwards, a Debate Moderator summarizes the strong arguments and weaknesses of both sides.
5. **Risk Sizing**: Three risk analysts with different personas (Conservative, Neutral, Aggressive) recommend position sizing. The Risk Committee Chair summarizes these recommendations.
6. **Portfolio Manager Final Decision**: The Portfolio Manager evaluates all inputs (synthesis, debate, risk) and issues the final recommendation (e.g., BUY STOCK, SELL CASH-SECURED PUT, WATCHLIST, REJECT).

## Project Structure

The project adopts a modular architecture:

```
vantage/
├── agents/
│   ├── __init__.py
│   ├── base_specialist.py    # Base class for specialist agents
│   ├── manager_debate.py     # Research Manager and Debate agents
│   ├── quant_screener.py     # Quantitative outlier detection logic
│   ├── risk_pm.py            # Risk Agents and Portfolio Manager
│   └── specialists.py        # Specialist agent implementations (Market, Fundamentals, etc.)
├── schemas/
│   ├── __init__.py
│   └── state.py              # Pydantic models and LangGraph state schema
├── workflows/
│   ├── __init__.py
│   └── full_graph.py         # LangGraph definition and compilation
├── outputs/                  # Directory where final JSON reports are saved
├── main.py                   # Main execution script
└── README.md                 # Project documentation
```

## Dependencies

Key dependencies include:
- `langgraph` and `langchain`
- `langchain-openai` (using `gpt-4.1-mini` models by default)
- `yfinance` for market data
- `pydantic` for schema validation
- `python-dotenv` for environment variable management

## How to Run

To run the system in batch mode for a list of tickers:

```bash
cd vantage
python3 main.py AAPL TSLA NVDA
```

The system will execute the quantitative screening, process the discovered outliers through the full agent pipeline, and save the final Portfolio Manager decision report in the `outputs/` folder.

## Future Improvements

- **Real Tool Integration**: Connect specialist agents to actual search tools (e.g., Tavily), News APIs, and options data terminals.
- **Human-in-the-Loop**: Add checkpoints in the LangGraph where a human can approve the Research Manager's synthesis before moving to the risk phase.
- **Asynchronous Parallelism**: Implement `asyncio` so specialist agents can fetch data and process information in true parallel execution within the LangGraph node.
- **Full Batch Execution**: Map (Map-Reduce) the LangGraph execution to process multiple outliers simultaneously instead of sequentially.
