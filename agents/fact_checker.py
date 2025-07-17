# app/agents/fact_checker.py

from app.llm.mistral_loader import chat_with_mistral

def fact_checker_agent(question: str, retrieved_docs: list[str]) -> str:
    context = "\n\n".join(retrieved_docs)
    prompt = (
        f"Given the question:\n{question}\n\n"
        f"And the evidence:\n{context}\n\n"
        "Decide whether the evidence supports the answer. "
        "If yes, summarize the proof. If not, say it's insufficient."
    )
    return chat_with_mistral(prompt)