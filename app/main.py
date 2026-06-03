import os
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware import Middleware
from app_security import RedactMiddleware, redact
from retriever import Retriever
from conversation_history import HISTORY_WINDOW, build_conversation_context
from session_store import SessionStore, StoredMessage
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
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 0.1))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-coder:1.3b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # "ollama" | "groq"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{8,128}$")

app = FastAPI(middleware=[Middleware(RedactMiddleware)])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ret = Retriever()
sessions = SessionStore()
_llm = None


class Query(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=8, max_length=128)


class ResetSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=128)


def _get_llm():
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


def _validate_session_id(session_id: str) -> str:
    session_id = session_id.strip()
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid session_id. Use 8-128 alphanumeric characters, hyphens, or underscores.",
        )
    return session_id


def _build_ollama_llm():
    if ChatOllama is None:
        raise RuntimeError(
            "Missing dependency: langchain-ollama. Install it and restart the app."
        )
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


@app.post("/ask")
async def ask(q: Query):
    session_id = _validate_session_id(q.session_id)
    user_query = redact(q.message)
    if not user_query.strip():
        raise HTTPException(status_code=400, detail="Empty message after redaction.")

    session = sessions.get_or_create(session_id)
    llm = _get_llm()

    history_text, conversation_summary, summarized_count = build_conversation_context(
        session.messages,
        llm,
        window=HISTORY_WINDOW,
        prior_summary=session.conversation_summary or None,
        summarized_message_count=session.summarized_message_count,
    )

    search_query = generate_search_query(user_query, history_text, llm)
    print(f"[{session_id}] search query: {search_query}")
    retrieved = ret.query(search_query, n_results=3)

    if not retrieved:
        reply = "No relevant information found please contact HR."
        _append_turn(session, user_query, reply, conversation_summary, summarized_count)
        return {"reply": reply}

    if retrieved[0][1] < RELEVANCE_THRESHOLD:
        reply = (
            "I can only answer questions that match official HR policies. "
            "Please rephrase or contact HR."
        )
        _append_turn(session, user_query, reply, conversation_summary, summarized_count)
        return {"reply": reply, "retrieved": retrieved}

    reply = generate_llm_answer(user_query, retrieved, history_text, llm)
    _append_turn(session, user_query, reply, conversation_summary, summarized_count)
    return {"reply": reply}


def _append_turn(session, user_query: str, reply: str, summary: str, summarized_count: int) -> None:
    session.messages.append(StoredMessage(role="user", content=user_query))
    session.messages.append(StoredMessage(role="assistant", content=reply))
    session.conversation_summary = summary
    session.summarized_message_count = summarized_count


@app.post("/session/reset")
async def reset_session(body: ResetSessionRequest):
    session_id = _validate_session_id(body.session_id)
    sessions.clear(session_id)
    return {"ok": True}


def generate_search_query(user_query: str, history_text: str, llm) -> str:
    if not history_text.strip():
        return user_query

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helper for an HR chatbot.\n"
                "Given the conversation context (summary of older turns + recent messages) "
                "and the user's latest question, write a concise retrieval query (1-2 sentences).\n"
                "Focus on HR policy topics like leave, overtime, compensation, salary, working hours, and holidays.\n"
                "Do not answer the user; only output the retrieval query.",
            ),
            ("human", "Conversation context:\n{history}\n\nLatest question:\n{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        generated = chain.invoke({"history": history_text, "question": user_query}).strip()
        return generated or user_query
    except Exception as e:
        print(f"Error generating search query: {e}")
        return user_query


def generate_llm_answer(
    user_query: str, context_docs: list, history_text: str, llm
) -> str:
    try:
        context = "\n\n".join([doc_text for (doc_text, _score) in context_docs])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an HR policy chatbot. Your ONLY knowledge comes from the CONTEXT.\n\n"
                    "CONTEXT:\n{context}\n\n"
                    "RULES:\n"
                    "1) ONLY answer questions that are relevent to HR policies.\n"
                    "2) If the question is outside these topics, say: "
                    '"I can only answer HR policy questions about leaves, overtime, compensation, salary, working hours, and holidays"\n'
                    "3) If the answer isn't in the context, say: "
                    '"This information is not available in the policy documents"\n'
                    "4) Keep answers to 1-2 sentences maximum.\n"
                    "5) NEVER add information not in the context.\n"
                    "6) Conversation context may include a summary of older turns and recent messages; "
                    "prioritize the latest question and CONTEXT.\n",
                ),
                ("human", "Conversation context:\n{history}\n\nQuestion:\n{question}"),
            ]
        )

        chain = (
            {
                "context": RunnablePassthrough(),
                "history": RunnablePassthrough(),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain.invoke(
            {"context": context, "history": history_text or "(none)", "question": user_query}
        ).strip()

    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to the AI service. Please make sure Ollama is running."
    except requests.exceptions.Timeout:
        return "Error: The AI service is taking too long to respond. Please try again."
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"


@app.on_event("startup")
async def startup_event():
    try:
        _get_llm()
    except Exception as e:
        print(f"LLM init warning: {e}")

    try:
        if LLM_PROVIDER == "ollama":
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                print(f"Ollama connected. Available models: {model_names}")
            else:
                print("Ollama is running but returned an error")
        else:
            print("Using Groq")
    except Exception as e:
        print(f"Cannot connect to {LLM_PROVIDER}: {e}")
        print(f"Make sure to run: {LLM_PROVIDER} serve")
