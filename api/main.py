# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph_flow import get_graph

app = FastAPI(title="Multi-Agent RAG QA System")

graph = get_graph()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(q: Query):
    try:
        result = graph.invoke({"user_question": q.question})
        return {
            "rewritten_question": result["rewritten_question"],
            "top_docs": result["retrieved_docs"][:2],
            "fact_check": result["fact_check"],
            "answer": result["answer"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))