import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware import Middleware
from app_security import RedactMiddleware, redact
from retriever import Retriever
from model_server import generate_answer
from dotenv import load_dotenv

load_dotenv()

RELEVANCE_THRESHOLD = float(os.getenv('RELEVANCE_THRESHOLD', '0.65'))


app = FastAPI(middleware=[Middleware(RedactMiddleware)])
ret = Retriever()


class Query(BaseModel):
    query: str


@app.on_event('startup')
def startup_event():
# if chroma DB empty, guidance to run seed
    pass

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


    answer = generate_answer(user_query, retrieved)
    return {'reply': answer}