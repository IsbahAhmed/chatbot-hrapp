# app/model_server.py
"""
Simple model server abstraction.
By default it will *not* call any external LLM.
It will return an answer built from retrieved context using a template.


To integrate a local LLM later, replace `generate_answer` implementation with calls to Ollama or TGI.
"""
from typing import List, Tuple




def generate_answer(question: str, retrieved: List[Tuple[str, float]]) -> str:

    if not retrieved:   
        return "I can only answer questions about official HR policies. Please contact HR."


# Check top similarity
    top_sim = retrieved[0][1]
    if top_sim < float(__import__('os').getenv('RELEVANCE_THRESHOLD','0.65')):
        return "I can only answer questions about official HR policies. Please rephrase or contact HR."


# Build concise answer by returning the most similar document followed by a safety note
    doc_text, sim = retrieved[0]
    answer = (
    f"Based on our HR policy documents (relevance {sim:.2f}):\n\n{doc_text}\n\n"
    "If this doesn't answer your question, contact HR directly."
    )
    return answer