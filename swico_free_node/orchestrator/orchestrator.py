from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import replace
from typing import Any

from .adapters import ChatModelAdapter, CodingModelAdapter, DocumentAnalysisModelAdapter, DocumentCreationModelAdapter, E5ModelAdapter, ImageModelAdapter, KokoroModelAdapter, ModelAdapter, MockAdapter, MockChatAdapter, SpecializedUnavailableAdapter, STTModelAdapter, TTSModelAdapter, VideoModelAdapter, WhisperModelAdapter
from .chat import ChatMessage, ChatService
from .aggregation import EvidenceStore, ResultAggregator
from .controller import MainController
from .models import Capability, Evidence, Task, TaskResult, TaskStatus, VerificationStatus
from .planner import TaskPlanner
from .registry import CapabilityRegistry, ModelRegistry, default_profiles
from .scheduler import AdaptiveScheduler, ResourceManager
from .trirag import TriRAG
from .verification import VerificationGate
from .retrieval import Document, E5EmbeddingAdapter, Retriever


class Orchestrator:
    def __init__(self, registry: ModelRegistry, adapters: dict[str, ModelAdapter], max_retries: int = 2, resources: ResourceManager | None = None, retriever: Retriever | None = None, poll_interval_ms: int = 100):
        self.registry = registry
        self.capabilities = CapabilityRegistry(registry)
        self.adapters = adapters
        self.planner = TaskPlanner()
        self.controller = MainController()
        self.aggregator = ResultAggregator()
        self.evidence_store = EvidenceStore()
        self.verification_gate = VerificationGate()
        self.scheduler = AdaptiveScheduler(registry, adapters, resources, poll_interval_ms)
        self.retriever = retriever
        self.trirag = TriRAG(retriever)
        self.chat = ChatService(self.scheduler, self.capabilities.models_for(Capability.CHAT))
        self.max_retries = max_retries
        self.jobs: dict[str, dict[str, Any]] = {}

    @classmethod
    def with_defaults(cls, qwen_runtime: Any = None, e5_runtime: Any = None, max_output_tokens: int = 256, mock_mode: bool = False, coding_runtime: Any = None, stt_runtime: Any = None, tts_runtime: Any = None, resources: ResourceManager | None = None, poll_interval_ms: int = 100, model_paths: dict[str, object] | None = None) -> "Orchestrator":
        registry = ModelRegistry(default_profiles(model_paths))
        adapters: dict[str, ModelAdapter] = {}
        for profile in registry.all():
            if mock_mode:
                # Mock adapters do not instantiate a model and must not consume
                # real-model RAM reservations during API regression tests.
                profile = replace(profile, integration_status="MOCK", health_status="MOCK", ram_requirement_mb=0, cpu_requirement=1, heavyweight=False)
                registry.register(profile)
                adapters[profile.model_id] = MockChatAdapter(profile) if profile.capability == Capability.CHAT else MockAdapter(profile)
            elif profile.capability == Capability.CHAT and qwen_runtime is not None:
                profile = replace(
                    profile,
                    model_path=str(getattr(qwen_runtime, "model_path", profile.model_path or "")) or None,
                    integration_status="READY",
                    health_status="READY",
                    ram_requirement_mb=862,
                    measured_ram_mb=862,
                    cpu_requirement=2,
                    expected_latency_ms=1500,
                    quality_score=0.7,
                )
                registry.register(profile)
                adapters[profile.model_id] = ChatModelAdapter(profile, qwen_runtime, max_output_tokens)
            elif profile.capability == Capability.CODING and coding_runtime is not None:
                profile = replace(
                    profile,
                    model_path=str(getattr(coding_runtime, "model_path", profile.model_path or "")) or None,
                    integration_status="READY",
                    health_status="READY",
                    ram_requirement_mb=545,
                    measured_ram_mb=545,
                    cpu_requirement=2,
                    expected_latency_ms=1500,
                    quality_score=0.7,
                )
                registry.register(profile)
                adapters[profile.model_id] = CodingModelAdapter(profile, coding_runtime, max_output_tokens)
            elif profile.capability == Capability.STT and stt_runtime is not None:
                profile = replace(
                    profile,
                    model_path=str(getattr(stt_runtime, "model_path", profile.model_path or "")) or None,
                    integration_status="READY",
                    health_status="READY",
                    ram_requirement_mb=400,
                    measured_ram_mb=340,
                    cpu_requirement=2,
                    expected_latency_ms=5000,
                    quality_score=0.7,
                )
                registry.register(profile)
                adapters[profile.model_id] = WhisperModelAdapter(profile, stt_runtime)
            elif profile.capability == Capability.RETRIEVAL and e5_runtime is not None:
                profile = replace(
                    profile,
                    model_path=str(getattr(e5_runtime, "model_path", profile.model_path or "")) or None,
                    integration_status="READY",
                    health_status="READY",
                    ram_requirement_mb=700,
                    measured_ram_mb=668,
                    cpu_requirement=2,
                    expected_latency_ms=100,
                    quality_score=0.8,
                )
                registry.register(profile)
                adapters[profile.model_id] = E5ModelAdapter(profile, e5_runtime)
            elif profile.capability == Capability.TTS and tts_runtime is not None:
                profile = replace(
                    profile,
                    model_path=str(getattr(tts_runtime, "model_path", profile.model_path or "")) or None,
                    integration_status="READY",
                    health_status="READY",
                    ram_requirement_mb=1400,
                    measured_ram_mb=562,
                    cpu_requirement=2,
                    expected_latency_ms=10000,
                    quality_score=0.7,
                )
                registry.register(profile)
                adapters[profile.model_id] = KokoroModelAdapter(profile, tts_runtime)
            else:
                adapters[profile.model_id] = {Capability.IMAGE_GENERATION: ImageModelAdapter, Capability.VIDEO_GENERATION: VideoModelAdapter, Capability.STT: STTModelAdapter, Capability.TTS: TTSModelAdapter, Capability.DOCUMENT_CREATION: DocumentCreationModelAdapter, Capability.DOCUMENT_ANALYSIS: DocumentAnalysisModelAdapter}.get(profile.capability, SpecializedUnavailableAdapter)(profile)
        retriever = Retriever(E5EmbeddingAdapter(e5_runtime)) if e5_runtime is not None else None
        return cls(registry, adapters, resources=resources, retriever=retriever, poll_interval_ms=poll_interval_ms)

    async def run(self, prompt: str, *, request_id: str | None = None, verify: bool = False, evidence: list[Evidence] | None = None, deadline_seconds: float = 45, capability: Capability | str | None = None, priority: int | None = None) -> dict[str, Any]:
        request_id = request_id or str(uuid.uuid4())
        graph = self.planner.plan(request_id, prompt, verify, deadline_seconds, capability, priority)
        understanding = self.controller.understand(prompt, capability, priority=priority, deadline_seconds=deadline_seconds, verify=verify)
        self.jobs[request_id] = {"graph": graph, "started_at": time.time(), "cancelled": False}
        results: list[TaskResult] = []
        while True:
            ready = graph.ready()
            if not ready:
                break
            async def run_bounded(task):
                if task.dependencies:
                    dependency_text = []
                    for dependency_id in task.dependencies:
                        for previous in graph.tasks.values():
                            if previous.task_id == dependency_id and previous.result is not None:
                                value = previous.result.get("text", previous.result) if isinstance(previous.result, dict) else previous.result
                                dependency_text.append(str(value))
                    if dependency_text:
                        task.input = f"{prompt}\n\nContext from prerequisite tasks:\n" + "\n\n".join(dependency_text)
                for attempt in range(self.max_retries + 1):
                    try:
                        return await self.scheduler.run(task)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        if attempt >= self.max_retries:
                            raise
                        task.status = TaskStatus.PENDING
            batch = await asyncio.gather(*(run_bounded(task) for task in ready), return_exceptions=True)
            for task, result in zip(ready, batch):
                if isinstance(result, TaskResult):
                    results.append(result)
                else:
                    if task.status not in (TaskStatus.CANCELLED, TaskStatus.WAITING_FOR_RESOURCE):
                        task.status = TaskStatus.FAILED
                    task.error = str(result)
                    graph.cancel_dependents(task.task_id)
        status = VerificationStatus.NOT_REQUIRED
        ranked_evidence = self.trirag.retrieve(prompt, evidence or []) if evidence else []
        retrieval_requested = bool(re.search(r"\b(document|uploaded|source|according to|based on|policy|evidence|reference)\b", prompt.lower()))
        if retrieval_requested and not ranked_evidence and self.retriever is not None:
            ranked_evidence = self.trirag.retrieve_from_index(prompt)
        if verify:
            status, ranked_evidence = self.trirag.verify(prompt, results, ranked_evidence)
        aggregate = self.aggregator.combine(results, list(graph.tasks.values()))
        self.evidence_store.add(request_id, ranked_evidence)
        return {"request_id": request_id, "status": "completed" if all(r.error is None for r in results) and results and not aggregate["failed_tasks"] else "failed", "verification_status": self.verification_gate.decide(status).value, "understanding": understanding, "text": self.trirag.synthesize(results), "results": [{"task_id": r.task_id, "capability": r.capability.value, "model_id": r.model_id, "latency_ms": round(r.latency_ms, 2), "provenance": r.provenance} for r in results], "task_results": aggregate["tasks"], "evidence": [{"source_id": e.source_id, "document_id": e.document_id, "chunk_id": e.chunk_id, "score": e.score, "similarity": e.similarity, "content": e.content} for e in ranked_evidence], "aggregate": aggregate, "metrics": {"total_latency_ms": round((time.time() - self.jobs[request_id]["started_at"]) * 1000, 2), "task_count": len(graph.tasks), "completed_tasks": len(results)}}

    def index_document(self, document: Document):
        if self.retriever is None:
            raise RuntimeError("retrieval is unavailable")
        return self.retriever.index_document(document)

    async def chat_response(self, conversation_id: str, messages: list[ChatMessage], metadata=None, deadline_seconds: float = 45):
        return await self.chat.respond(conversation_id, messages, metadata, deadline_seconds)

    async def transcribe_audio(self, audio_path: str, language: str | None = None, *, request_id: str | None = None, deadline_seconds: float = 180) -> dict[str, Any]:
        request_id = request_id or str(uuid.uuid4())
        task = Task(request_id, "stt", Capability.STT, {"audio_path": audio_path, "language": language}, priority=90, deadline=time.time() + deadline_seconds)
        result = await self.scheduler.run(task)
        output = dict(result.output) if isinstance(result.output, dict) else {"transcript": str(result.output)}
        output.update({"request_id": request_id, "task_id": result.task_id, "model_id": result.model_id, "processing_time_seconds": result.latency_ms / 1000.0, "resource": self.scheduler.status()})
        return output

    def resource_preflight(self, model_id: str = "stt-whisper-base") -> dict[str, Any]:
        profile = self.registry.get(model_id)
        if profile is None:
            raise KeyError(f"unknown model: {model_id}")
        result = self.scheduler.resources.preflight(profile)
        adapter = self.adapters.get(model_id)
        result.update({
            "model_status": profile.integration_status,
            "health_status": profile.health_status,
            "lifecycle_state": getattr(adapter, "state", "NOT_INSTALLED"),
            "loaded": bool(adapter and getattr(adapter, "state", "") in ("LOADED", "EXECUTING")),
            "ready_for_load": result["admission"] == "READY_FOR_LOAD" and profile.integration_status in ("READY", "MOCK"),
        })
        return result

    def cancel(self, request_id: str) -> bool:
        job = self.jobs.get(request_id)
        if not job:
            return False
        job["cancelled"] = True
        self.scheduler.cancel(request_id)
        graph = job.get("graph")
        if graph is None:
            return True
        for task in graph.tasks.values():
            if task.status.value in ("pending", "ready", "waiting_for_dependency"):
                task.status = TaskStatus.CANCELLED
        return True
