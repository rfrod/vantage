from langgraph.graph import StateGraph, END
from schemas.state import GraphState, OutlierTicker
from agents.quant_screener import QuantScreener

# This file will be expanded in subsequent phases

def create_workflow():
    workflow = StateGraph(GraphState)
    
    # Define nodes
    def quant_screening_node(state: GraphState):
        tickers = state.get("tickers", [])
        screener = QuantScreener()
        outliers = screener.screen_tickers(tickers)
        return {"outliers": outliers}
        
    workflow.add_node("quant_screening", quant_screening_node)
    
    # Placeholder for the next phase
    def dummy_node(state: GraphState):
        return {"current_ticker": state.get("outliers", [])[0] if state.get("outliers") else None}
        
    workflow.add_node("dummy", dummy_node)
    
    workflow.set_entry_point("quant_screening")
    workflow.add_edge("quant_screening", "dummy")
    workflow.add_edge("dummy", END)
    
    return workflow.compile()
