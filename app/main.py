import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware import Middleware
from .app_security import RedactMiddleware, redact
from .retriever import Retriever
from .model_server import generate_answer
from dotenv import load_dotenv
import requests
import json

load_dotenv()

RELEVANCE_THRESHOLD = float(os.getenv('RELEVANCE_THRESHOLD', '0.1'))
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'deepseek-coder:1.3b')

app = FastAPI(middleware=[Middleware(RedactMiddleware)])
ret = Retriever()


class Query(BaseModel):
    query: str

@app.post('/ask')
async def ask(q: Query):
    # Redaction middleware already ran. Still do final scrub on the text we read here.
    user_query = redact(q.query)
    if not user_query.strip():
        raise HTTPException(status_code=400, detail='Empty query after redaction.')


    # Basic topic filter (optional). You can extend this if needed.
    allowed_topics = ['leave', 'overtime', 'compensation', 'salary', 'hours', 'policy', 'holiday']
    if not any(t in user_query.lower() for t in allowed_topics):
        return {'reply': 'Question outside allowed HR topics. Please ask about leave, overtime, compensation, or company policy.'}


    retrieved = ret.query(user_query, n_results=3)
    # retrieved is list of (doc, similarity)
    if not retrieved:
        return {'reply': 'No relevant HR documents found. Please contact HR.'}


    # Apply similarity threshold check
    if retrieved[0][1] < RELEVANCE_THRESHOLD:
        return {'reply': 'I can only answer questions that match official HR policies. Please rephrase or contact HR.'}


    answer = generate_llm_answer(user_query, retrieved)
    return {'reply': answer}

def generate_llm_answer(user_query: str, context_docs: list) -> str:

    try:
        # Prepare context from retrieved documents
        context = "\n\n".join([doc[0] for doc in context_docs])
        
        # Enhanced system prompt for HR policies
        system_prompt = """You are a professional HR assistant. Your role is STRICTLY LIMITED to answering questions based on the provided HR policy documents.

IMPORTANT RULES:
1. ONLY answer questions related to HR policies about leaves, overtime, compensation, salary, working hours, holidays, and company policies
2. If the question is outside HR topics, politely decline to answer
3. Base your answers ONLY on the provided context documents - do not use external knowledge
4. If the context doesn't contain relevant information, say you cannot answer from available policies
5. Keep responses professional, concise, and helpful (2-3 sentences maximum)
6. NEVER make up information or speculate beyond the provided context
7. If unsure, direct the user to contact HR directly

Context from HR Policy Documents:
{context}"""

        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "system", 
                        "content": system_prompt.format(context=context)
                    },
                    {
                        "role": "user", 
                        "content": user_query
                    }
                ],
                "stream": False,
              "options": {
                    "temperature": 0.1,
                    "num_predict": 150,  # ⬇️ REDUCE significantly for faster responses
                    "top_k": 20,
                    "top_p": 0.9,
                }
            },
            timeout=15  # 30 second timeout
        )
        
        if response.status_code == 200:
            return response.json()['message']['content']
        else:
            return f"Error: Unable to generate response. Status code: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to the AI service. Please make sure Ollama is running."
    except requests.exceptions.Timeout:
        return "Error: The AI service is taking too long to respond. Please try again."
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@app.on_event('startup')
async def startup_event():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            print(f"✅ Ollama connected successfully. Available models: {model_names}")
        else:
            print("⚠️  Ollama is running but returned an error")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("💡 Make sure to run: ./ollama serve")