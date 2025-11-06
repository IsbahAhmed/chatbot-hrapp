# app/app_security.py
import re
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request


PII_PATTERNS = [
re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), # SSN style
re.compile(r"\bEMP-\d{3,}\b"),
]




def redact(text: str) -> str:
    if not text:
        return text
    for p in PII_PATTERNS:
        text = p.sub("[REDACTED]", text)
    return text




class RedactMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
    # Only handle text payloads
        if request.method in ("POST", "PUT"):
            body = await request.body()
            try:
                decoded = body.decode("utf-8")
            except Exception:
                decoded = None
            if decoded:
                safe = redact(decoded)
                request._body = safe.encode("utf-8")
        response = await call_next(request)
        return response