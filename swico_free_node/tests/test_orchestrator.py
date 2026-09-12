import asyncio
import time

import pytest
from pydantic import ValidationError

from orchestrator.adapters import MockAdapter
from orchestrator.models import Capability, Evidence, ModelProfile, VerificationStatus
from orchestrator.orchestrator import Orchestrator
from orchestrator.planner import TaskGraph, TaskPlanner
from orchestrator.aggregation import ResultAggregator
from orchestrator.registry import ModelRegistry
from orchestrator.scheduler import ResourceManager
from orchestrator.api import OrchestrateRequest


def make_orchestrator(ram=2048):
    profiles = [ModelProfile("coding", "Coding", Capability.CODING, ram_requirement_mb=128, quality_score=.8)]
    registry = ModelRegistry(profiles)
    return Orchestrator(registry, {"coding": MockAdapter(profiles[0])}, resources=ResourceManager(ram_limit_mb=ram))


def test_simple_request_uses_one_task():
    result = asyncio.run(make_orchestrator().run("help me debug this Python code"))
    assert result["status"] == "completed"
    assert result["metrics"]["task_count"] == 1
    assert result["results"][0]["capability"] == "coding"


def test_verification_is_explicit_and_evidence_is_provenance():
    result = asyncio.run(make_orchestrator().run("debug this", verify=True, evidence=[Evidence("doc-1", "debug this guidance", .9)]))
    assert result["verification_status"] == VerificationStatus.VERIFIED.value
    assert result["evidence"][0]["source_id"] == "doc-1"


def test_resource_manager_rejects_overcommit():
    result = asyncio.run(make_orchestrator(ram=64).run("debug this"))
    assert result["status"] == "failed"


def test_explicit_coding_capability_reaches_understanding_and_selects_coding_model():
    node = Orchestrator.with_defaults(mock_mode=True, resources=ResourceManager(ram_limit_mb=4096))
    result = asyncio.run(node.run("Return a short answer.", capability=Capability.CODING))
    assert result["understanding"]["capabilities"] == ["coding"]
    assert result["results"][0]["model_id"] == "coding-qwen2.5-coder-0.5b"


def test_explicit_chat_capability_reaches_understanding_and_selects_chat_model():
    node = Orchestrator.with_defaults(mock_mode=True, resources=ResourceManager(ram_limit_mb=4096))
    result = asyncio.run(node.run("Return a short answer.", capability=Capability.CHAT))
    assert result["understanding"]["capabilities"] == ["chat"]
    assert result["results"][0]["model_id"] == "chat-qwen3-0.6b"


def test_automatic_coding_detection_selects_coding_model():
    node = Orchestrator.with_defaults(mock_mode=True, resources=ResourceManager(ram_limit_mb=4096))
    result = asyncio.run(node.run("Write a Python function that adds two numbers."))
    assert result["understanding"]["capabilities"] == ["coding"]
    assert result["results"][0]["model_id"] == "coding-qwen2.5-coder-0.5b"


def test_automatic_chat_detection_selects_chat_model():
    node = Orchestrator.with_defaults(mock_mode=True, resources=ResourceManager(ram_limit_mb=4096))
    result = asyncio.run(node.run("What is the difference between RAM and storage?"))
    assert result["understanding"]["capabilities"] == ["chat"]
    assert result["results"][0]["model_id"] == "chat-qwen3-0.6b"


def test_invalid_capability_is_rejected_by_request_schema():
    with pytest.raises(ValidationError):
        OrchestrateRequest(prompt="Do something", capability="not-a-capability")


def test_planner_supports_parallel_multi_capability_tasks():
    graph = TaskPlanner().plan("multi", "Write Python code and explain the code.")
    ready = graph.ready()
    assert {task.capability for task in ready} == {Capability.CODING, Capability.CHAT}
    assert all(not task.dependencies for task in ready)


def test_planner_supports_sequential_analysis_then_synthesis():
    graph = TaskPlanner().plan("document", "Analyze this uploaded document and summarize the important points.")
    first = graph.ready()
    assert [task.capability for task in first] == [Capability.DOCUMENT_ANALYSIS]
    first[0].status = __import__("orchestrator.models", fromlist=["TaskStatus"]).TaskStatus.COMPLETED
    second = graph.ready()
    assert [task.capability for task in second] == [Capability.CHAT]
    assert second[0].dependencies == [first[0].task_id]


def test_task_graph_rejects_unknown_and_cyclic_dependencies():
    from orchestrator.models import Task

    with pytest.raises(ValueError, match="unknown dependencies"):
        TaskGraph([Task("a", "coding", Capability.CODING, "x", dependencies=["missing"])])
    a = Task("a", "coding", Capability.CODING, "x")
    b = Task("b", "chat", Capability.CHAT, "x", dependencies=[a.task_id])
    a.dependencies = [b.task_id]
    with pytest.raises(ValueError, match="dependency cycle"):
        TaskGraph([a, b])


def test_priority_and_deadline_are_carried_into_planned_tasks():
    graph = TaskPlanner().plan("priority", "Write code", deadline_seconds=12, priority=91)
    task = graph.ready()[0]
    assert task.priority == 91
    assert 10 < task.deadline - time.time() <= 12


def test_aggregator_retains_partial_failures():
    from orchestrator.models import Task, TaskResult, TaskStatus

    completed = TaskResult("one", "r", Capability.CODING, {"text": "ok"}, "coder", time.time(), time.time())
    failed = Task("two", "chat", Capability.CHAT, "x")
    failed.status, failed.error = TaskStatus.FAILED, "chat unavailable"
    aggregate = ResultAggregator().combine([completed], [failed])
    assert aggregate["model_count"] == 1
    assert aggregate["failed_tasks"][0]["error"] == "chat unavailable"


def test_verified_response_does_not_blindly_return_contradictory_model_text():
    node = make_orchestrator()
    evidence = [
        Evidence(
            "doc-1",
            "RAM is significantly faster than an SSD. RAM has much lower latency and higher access speed.",
            .95,
            {},
            "ram-doc",
            "ram-doc:0",
            .95,
        )
    ]
    result = asyncio.run(
        node.run(
            "debug RAM versus SSD",
            verify=True,
            evidence=evidence,
        )
    )
    assert result["verification_status"] == VerificationStatus.VERIFIED.value
    assert "RAM is significantly faster than an SSD" in result["text"]
