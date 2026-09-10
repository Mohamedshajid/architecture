from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .models import Capability, Task
from .scheduler import AdaptiveScheduler


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass
class Conversation:
    conversation_id: str
    messages: list[ChatMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


class ConversationStore:
    def __init__(self, max_messages: int = 24, max_characters: int = 24000):
        self.max_messages, self.max_characters = max_messages, max_characters
        self._items: dict[str, Conversation] = {}
        self._lock = threading.RLock()

    def get(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            item = self._items.get(conversation_id)
            return None if item is None else Conversation(item.conversation_id, list(item.messages), dict(item.metadata), item.updated_at)

    def append(self, conversation_id: str, messages: list[ChatMessage], metadata: dict[str, Any] | None = None) -> Conversation:
        with self._lock:
            conversation = self._items.setdefault(conversation_id, Conversation(conversation_id))
            conversation.messages.extend(messages)
            conversation.metadata.update(metadata or {})
            system = [m for m in conversation.messages if m.role == "system"][:1]
            rest = [m for m in conversation.messages if m.role != "system"][-self.max_messages + len(system):]
            conversation.messages = system + rest
            while sum(len(m.content) for m in conversation.messages) > self.max_characters and len(conversation.messages) > 1:
                if conversation.messages[0].role == "system" and len(conversation.messages) > 2:
                    conversation.messages.pop(1)
                else:
                    conversation.messages.pop(0)
            conversation.updated_at = time.time()
            return self.get(conversation_id)

    def clear(self, conversation_id: str) -> bool:
        with self._lock:
            return self._items.pop(conversation_id, None) is not None


class ChatService:
    def __init__(self, scheduler: AdaptiveScheduler, adapter_ids: list[str], store: ConversationStore | None = None):
        self.scheduler, self.adapter_ids, self.store = scheduler, adapter_ids, store or ConversationStore()

    async def respond(self, conversation_id: str, messages: list[ChatMessage], metadata: dict[str, Any] | None = None, deadline_seconds: float = 45) -> dict[str, Any]:
        conversation = self.store.append(conversation_id, messages, metadata)
        task = Task(conversation_id, "chat", Capability.CHAT, [{"role": m.role, "content": m.content} for m in conversation.messages], preferred_models=self.adapter_ids, priority=90, deadline=time.time() + deadline_seconds)
        result = await self.scheduler.run(task)
        assistant = ChatMessage("assistant", str(result.output.get("text", result.output)))
        self.store.append(conversation_id, [assistant])
        return {"conversation_id": conversation_id, "message": {"role": assistant.role, "content": assistant.content}, "model_id": result.model_id, "capability": Capability.CHAT.value, "provenance": result.provenance}
