import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware import Middleware
from app_security import RedactMiddleware, redact
from retriever import Retriever
from dotenv import load_dotenv
import requests

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

try:
    # pyrefly: ignore [missing-import]
    from langchain_ollama import ChatOllama
except Exception:  
    ChatOllama = None  

try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover
    ChatGroq = None  # type: ignore

load_dotenv()

TEMPRETURE = 0.1
RELEVANCE_THRESHOLD = float(os.getenv('RELEVANCE_THRESHOLD', '0.1'))
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'deepseek-coder:1.3b')
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # "ollama" | "groq"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

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

def _history_to_text(history_messages: list[Message]) -> str:
    if not history_messages:
        return ""
    # Keep it short and stable for prompting.
    trimmed = history_messages[-10:] if len(history_messages) > 10 else history_messages
    return "\n".join(f"{m.role}: {m.content}" for m in trimmed)

def _build_ollama_llm():
    if ChatOllama is None:
        raise RuntimeError(
            "Missing dependency: langchain-ollama. Install it and restart the app."
        )
    # Uses local Ollama server (e.g. http://localhost:11434).
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        temperature=TEMPRETURE,
    )

def _build_groq_llm():
    if ChatGroq is None:
        raise RuntimeError("Missing dependency: langchain-groq. Install it and restart the app.")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Missing GROQ_API_KEY in environment.")
    return ChatGroq(model=GROQ_MODEL, temperature=TEMPRETURE)

def _build_llm():
    if LLM_PROVIDER == "groq":
        return _build_groq_llm()
    return _build_ollama_llm()

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
    
    
    history_messages = q.messages[:-1]
    
    # Generate a focused search query from the full conversation
    search_query = generate_search_query(user_query, history_messages)
    print(f"Generated search query: {search_query}")
    retrieved = ret.query(search_query, n_results=3)
    # retrieved is list of (doc, similarity)
    if not retrieved:
        return {'reply': 'No relevant information found please contact HR.'}
    
    if retrieved[0][1] < RELEVANCE_THRESHOLD:
        return {'reply': 'I can only answer questions that match official HR policies. Please rephrase or contact HR.', retrieved: retrieved }
 
    answer = generate_llm_answer(user_query, retrieved, history_messages)
    return {'reply': answer}

def generate_search_query(user_query: str, history_messages: list) -> str:
    """Generate a concise search query from conversation + latest user query."""
    if not history_messages:
        return user_query

    llm = _build_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helper for an HR chatbot.\n"
                "Given the conversation history and the user's latest question, write a concise retrieval query (1-2 sentences).\n"
                "Focus on HR policy topics like leave, overtime, compensation, salary, working hours, and holidays.\n"
                "Do not answer the user; only output the retrieval query.",
            ),
            ("human", "Conversation history:\n{history}\n\nLatest question:\n{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        history_text = _history_to_text(history_messages)
        generated = chain.invoke({"history": history_text, "question": user_query}).strip()
        return generated or user_query
    except Exception as e:
        print(f"Error generating search query: {e}")
        return user_query

def generate_llm_answer(user_query: str, context_docs: list, history_messages: list) -> str:
    try:
        llm = _build_llm()

        context = "\n\n".join([doc_text for (doc_text, _score) in context_docs])
        history_text = _history_to_text(history_messages)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an HR policy chatbot. Your ONLY knowledge comes from the CONTEXT.\n\n"
                    "CONTEXT:\n{context}\n\n"
                    "RULES:\n"
                    "1) ONLY answer questions that are relevent to HR policies.\n"
                    "2) If the question is outside these topics, say: "
                    "\"I can only answer HR policy questions about leaves, overtime, compensation, salary, working hours, and holidays\"\n"
                    "3) If the answer isn't in the context, say: "
                    "\"This information is not available in the policy documents\"\n"
                    "4) Keep answers to 1-2 sentences maximum.\n"
                    "5) NEVER add information not in the context.\n"
                    "6) The conversation history is optional context; prioritize the latest question + context.\n",
                ),
                ("human", "Conversation history:\n{history}\n\nQuestion:\n{question}"),
            ]
        )

        chain = (
            {"context": RunnablePassthrough(), "history": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain.invoke({"context": context, "history": history_text, "question": user_query}).strip()

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