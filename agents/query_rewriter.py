# app/agents/query_rewriter.py

from app.llm.mistral_loader import chat_with_mistral

def query_rewriter_agent(user_question: str) -> str:
    prompt = (
        "Rewrite the user's question to make it clearer for a document retrieval system. "
        "Focus on making it specific and search-friendly.\n\n"
        f"User question: {user_question}"
    )
    return chat_with_mistral(prompt)