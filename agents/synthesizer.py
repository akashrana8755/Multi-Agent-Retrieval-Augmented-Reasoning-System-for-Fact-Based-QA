# app/agents/synthesizer.py

from app.llm.mistral_loader import chat_with_mistral

def synthesis_agent(question: str, retrieved_docs: list[str]) -> str:
    context = "\n\n".join(retrieved_docs)
    prompt = (
        f"Answer the question using the information below.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{context}\n\n"
        "Give a well-reasoned answer with traceable justification."
    )
    return chat_with_mistral(prompt)