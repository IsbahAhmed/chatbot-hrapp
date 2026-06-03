import os
from typing import List, Optional, Protocol, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "10"))


class ChatMessage(Protocol):
    role: str
    content: str


def messages_to_langchain(messages: List[ChatMessage]) -> List[BaseMessage]:
    lc: List[BaseMessage] = []
    for m in messages:
        role = m.role if hasattr(m, "role") else m["role"]
        content = m.content if hasattr(m, "content") else m["content"]
        if role == "user":
            lc.append(HumanMessage(content=content))
        elif role == "assistant":
            lc.append(AIMessage(content=content))
    return lc


def langchain_messages_to_text(messages: List[BaseMessage]) -> str:
    lines: List[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"user: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"assistant: {m.content}")
    return "\n".join(lines)


def plain_messages_to_text(messages: List[ChatMessage]) -> str:
    lines: List[str] = []
    for m in messages:
        role = m.role if hasattr(m, "role") else m["role"]
        content = m.content if hasattr(m, "content") else m["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You summarize HR policy chatbot conversations.\n"
            "Preserve topics discussed "
            "specific numbers, and any unresolved questions.\n"
            "Write 3-6 concise sentences. Do not invent facts.",
        ),
        ("human", "{conversation}"),
    ]
)

_UPDATE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You maintain a running summary of an HR policy chatbot conversation.\n"
            "Merge the existing summary with the new messages. Keep 3-6 sentences total.\n"
            "Do not invent facts.",
        ),
        (
            "human",
            "Existing summary:\n{existing}\n\nNew messages to include:\n{new_messages}",
        ),
    ]
)


def build_conversation_context(
    history_messages: List[ChatMessage],
    llm,
    *,
    window: int = HISTORY_WINDOW,
    prior_summary: Optional[str] = None,
    summarized_message_count: int = 0,
) -> Tuple[str, str, int]:
    """
    LangChain-style history: last `window` messages verbatim; older messages folded into a summary.

    Returns:
        history_text — for retrieval / answer prompts
        conversation_summary — pass back on the next request
        summarized_message_count — how many leading messages are covered by the summary
    """
    if not history_messages:
        return "", prior_summary or "", 0

    if len(history_messages) <= window:
        recent_text = plain_messages_to_text(history_messages)
        return recent_text, prior_summary or "", 0

    older = history_messages[:-window]
    recent = history_messages[-window:]
    recent_text = plain_messages_to_text(recent)

    summary = (prior_summary or "").strip()
    covered = max(0, summarized_message_count)

    new_slice = older[covered:]
    if new_slice:
        new_text = plain_messages_to_text(new_slice)
        summarize_chain = _SUMMARY_PROMPT | llm | StrOutputParser()
        update_chain = _UPDATE_SUMMARY_PROMPT | llm | StrOutputParser()
        try:
            if summary:
                summary = update_chain.invoke(
                    {"existing": summary, "new_messages": new_text}
                ).strip()
            else:
                summary = summarize_chain.invoke({"conversation": new_text}).strip()
            covered = len(older)
        except Exception as e:
            print(f"Error summarizing conversation: {e}")
            if not summary:
                summary = new_text[:500]

    parts: List[str] = []
    if summary:
        parts.append(f"Summary of earlier conversation:\n{summary}")
    parts.append(f"Recent messages (last {window}):\n{recent_text}")
    history_text = "\n\n".join(parts)
    return history_text, summary, covered
