import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware import Middleware
from .app_security import RedactMiddleware, redact
from .retriever import Retriever
from dotenv import load_dotenv
import requests
import json

load_dotenv()

RELEVANCE_THRESHOLD = float(os.getenv('RELEVANCE_THRESHOLD', '0.1'))
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'deepseek-coder:1.3b')

app = FastAPI(middleware=[Middleware(RedactMiddleware)])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ret = Retriever()

class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class Query(BaseModel):
    messages: list[Message]  # Full conversation history, last message is the new user query

@app.post('/ask')
async def ask(q: Query):
    if not q.messages:
        raise HTTPException(status_code=400, detail='No messages provided.')
    
    # The last message should be the new user query
    new_message = q.messages[-1]
    if new_message.role != 'user':
        raise HTTPException(status_code=400, detail='Last message must be from user.')
    
    # Redaction middleware already ran. Still do final scrub on the text we read here.
    user_query = redact(new_message.content)
    if not user_query.strip():
        raise HTTPException(status_code=400, detail='Empty query after redaction.')
    
    # Basic topic filter (optional). You can extend this if needed.
    # allowed_topics = ['leave', 'overtime', 'compensation', 'salary', 'hours', 'policy', 'holiday',"hello","hi"]
    # if not any(t in user_query.lower() for t in allowed_topics):
    #     return {'reply': 'Question outside allowed HR topics. Please ask about leave, overtime, compensation, or company policy.'}
    
    # Prepare conversation history (exclude the new message)
    history_messages = q.messages[:-1]
    
    # Generate a focused search query from the full conversation
    search_query = generate_search_query(user_query, history_messages)
    print(f"Generated search query: {search_query}")
    retrieved = ret.query(search_query, n_results=3)
    # retrieved is list of (doc, similarity)
    if not retrieved:
        return {'reply': 'No relevant information found please contact HR.'}
    
    # Apply similarity threshold check
    if retrieved[0][1] < RELEVANCE_THRESHOLD:
        return {'reply': 'I can only answer questions that match official HR policies. Please rephrase or contact HR.'}
 
    answer = generate_llm_answer(user_query, retrieved, history_messages)
    return {'reply': answer}

def generate_search_query(user_query: str, history_messages: list) -> str:
    """Generate a concise search query from the conversation history and latest user query."""
    if not history_messages:
        return user_query  # No history, use the query as-is
    
    try:
        # Truncate history if too long (e.g., last 10 messages to avoid token limits)
        truncated_history = history_messages[-10:] if len(history_messages) > 10 else history_messages
        
        # Convert Message objects to dicts for Ollama API
        history_dicts = [msg.model_dump() for msg in truncated_history]  # Use .dict() if on Pydantic v1
        
        # Prompt to generate a focused query
        system_prompt = """You are a helper for an HR chatbot. Based on the conversation history, generate a concise, specific search query (1-2 sentences) that captures the user's current intent for retrieving relevant HR policy documents. Focus on key topics like leaves, overtime, compensation, salary, working hours, or holidays. Ignore unrelated details."""
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(history_dicts)  # Now all are dicts
        messages.append({"role": "user", "content": user_query})
        
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Low creativity for consistent queries
                    "num_predict": 50,   # Short response
                    "top_k": 20,
                    "top_p": 0.9,
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            generated_query = response.json()['message']['content'].strip()
            return generated_query if generated_query else user_query  # Fallback
        else:
            return user_query  # Fallback on error
    
    except Exception:
        return user_query  # Fallback to original query

def generate_llm_answer(user_query: str, context_docs: list, history_messages: list) -> str:
    try:
        # Prepare context from retrieved documents
        context = "\n\n".join([doc[0] for doc in context_docs])
        history_dicts = [msg.model_dump() for msg in history_messages]
        # Enhanced system prompt for HR policies
        system_prompt = """You are an HR policy chatbot. Your ONLY knowledge comes from the context below.

CONTEXT:
{context}

RULES:
1. ONLY answer questions about leaves, overtime, compensation, salary, working hours, holidays
2. If the question is outside these topics, say "I can only answer HR policy questions about leaves, overtime, compensation, salary, working hours, and holidays"
3. If the answer isn't in the context, say "This information is not available in the policy documents"
4. Keep answers to 1-2 sentences maximum
5. NEVER add information not in the context
6. Consider the conversation history for context, but always prioritize the current query and provided context.

Current conversation:"""
        
        # Build the full messages list for Ollama
        messages = [
            {
                "role": "system", 
                "content": system_prompt.format(context=context)
            }
        ]
        # Add history
        messages.extend(history_dicts)
        # Add the new user query
        messages.append({
            "role": "user", 
            "content": user_query
        })
        
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 150,  # ⬇️ REDUCE significantly for faster responses
                    "top_k": 20,
                    "top_p": 0.9,
                }
            },
            timeout=300  # 30 second timeout
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