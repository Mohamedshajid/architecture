import asyncio

from orchestrator.adapters import MockChatAdapter
from orchestrator.chat import ChatMessage, ChatService, ConversationStore
from orchestrator.models import Capability, ModelProfile, TaskStatus
from orchestrator.planner import TaskPlanner
from orchestrator.registry import ModelRegistry
from orchestrator.scheduler import AdaptiveScheduler


def test_chat_is_distinct_from_coding():
    planner = TaskPlanner()
    assert planner.plan("1", "How are you?").ready()[0].capability == Capability.CHAT
    assert planner.plan("2", "Write a Python function").ready()[0].capability == Capability.CODING


def test_conversation_store_bounds_history():
    store = ConversationStore(max_messages=3, max_characters=100)
    store.append("c", [ChatMessage("system", "be brief"), ChatMessage("user", "one"), ChatMessage("assistant", "two"), ChatMessage("user", "three")])
    conversation = store.get("c")
    assert conversation.messages[0].role == "system"
    assert len(conversation.messages) <= 3


def test_chat_service_uses_registry_adapter_and_preserves_context():
    profile = ModelProfile("chat-mini-01", "Chat", Capability.CHAT)
    registry = ModelRegistry([profile])
    scheduler = AdaptiveScheduler(registry, {profile.model_id: MockChatAdapter(profile)})
    service = ChatService(scheduler, [profile.model_id])
    result = asyncio.run(service.respond("c", [ChatMessage("user", "Hello")]))
    assert result["capability"] == "chat"
    assert service.store.get("c").messages[-1].role == "assistant"
