# langgraph_flow.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from agents import (
    query_rewriter_agent,
    retrieval_agent,
    fact_checker_agent,
    synthesis_agent
)

class AgentState(TypedDict):
    user_question: str
    rewritten_question: str
    retrieved_docs: List[str]
    fact_check: str
    answer: str

def get_graph():
    graph = StateGraph(AgentState)

    graph.add_node("rewrite", lambda s: {**s, "rewritten_question": query_rewriter_agent(s["user_question"])})
    graph.add_node("retrieve", lambda s: {**s, "retrieved_docs": retrieval_agent(s["rewritten_question"])})
    graph.add_node("factcheck", lambda s: {**s, "fact_check": fact_checker_agent(s["user_question"], s["retrieved_docs"])})
    graph.add_node("synthesize", lambda s: {**s, "answer": synthesis_agent(s["user_question"], s["retrieved_docs"])})

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "factcheck")
    graph.add_edge("factcheck", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()