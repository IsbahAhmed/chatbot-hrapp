import threading
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StoredMessage:
    role: str
    content: str


@dataclass
class ChatSession:
    messages: List[StoredMessage] = field(default_factory=list)
    conversation_summary: str = ""
    summarized_message_count: int = 0


class SessionStore:
    """In-memory conversation store keyed by session_id."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> ChatSession:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ChatSession()
            return self._sessions[session_id]

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
