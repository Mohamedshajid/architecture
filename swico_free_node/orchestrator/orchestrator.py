from __future__ import annotations

import asyncio
import re
import time
import urllib.parse
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
    def with_defaults(cls, qwen_runtime: Any = None, e5_runtime: Any = None, max_output_tokens: int = 256, mock_mode: bool = False, coding_runtime: Any = None, stt_runtime: Any = None, tts_runtime: Any = None, image_runtime: Any = None, resources: ResourceManager | None = None, poll_interval_ms: int = 100, model_paths: dict[str, object] | None = None) -> "Orchestrator":
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
                adapters[profile.model_id] = ChatModelAdapter(
                    profile,
                    qwen_runtime,
                    min(max_output_tokens, 128),
                )
            elif profile.capability == Capability.CODING and coding_runtime is not None:
                profile = replace(
                    profile,
                    model_path=str(getattr(coding_runtime, "model_path", profile.model_path or "")) or None,
                    integration_status="READY",
                    health_status="READY",
                    ram_requirement_mb=1120,
                    measured_ram_mb=1120,
                    cpu_requirement=2,
                    expected_latency_ms=1500,
                    quality_score=0.7,
                )
                registry.register(profile)
                adapters[profile.model_id] = CodingModelAdapter(profile, coding_runtime, max_output_tokens)
            elif profile.capability == Capability.IMAGE_GENERATION and image_runtime is not None:
                profile = replace(
                    profile,
                    model_path=str(getattr(image_runtime, "model_path", profile.model_path or "")) or None,
                    integration_status="READY",
                    health_status="READY",
                    ram_requirement_mb=4096,
                    measured_ram_mb=3580,
                    cpu_requirement=4,
                    expected_latency_ms=65000,
                    quality_score=0.7,
                    concurrency_limit=1,
                    load_mode="on_demand",
                )
                registry.register(profile)
                adapters[profile.model_id] = ImageModelAdapter(profile, image_runtime)
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
            elif profile.capability == Capability.DOCUMENT_CREATION and profile.model_path:
                profile = replace(
                    profile,
                    integration_status="READY",
                    health_status="READY",
                    ram_requirement_mb=512,
                    measured_ram_mb=724,
                    cpu_requirement=2,
                    expected_latency_ms=10000,
                    quality_score=0.7,
                )
                registry.register(profile)
                adapters[profile.model_id] = DocumentCreationModelAdapter(profile)
            elif profile.capability == Capability.DOCUMENT_ANALYSIS and profile.model_path:
                profile = replace(
                    profile,
                    integration_status="READY",
                    health_status="READY",
                    ram_requirement_mb=1024,
                    measured_ram_mb=1024,
                    cpu_requirement=2,
                    expected_latency_ms=15000,
                    quality_score=0.75,
                    concurrency_limit=1,
                    load_mode="on_demand",
                )
                registry.register(profile)
                adapters[profile.model_id] = DocumentAnalysisModelAdapter(profile)
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
                adapter_cls = {
                    Capability.VIDEO_GENERATION: VideoModelAdapter,
                    Capability.STT: STTModelAdapter,
                    Capability.TTS: TTSModelAdapter,
                    Capability.DOCUMENT_CREATION: DocumentCreationModelAdapter,
                    Capability.DOCUMENT_ANALYSIS: DocumentAnalysisModelAdapter,
                }.get(profile.capability)

                if adapter_cls is not None:
                    adapters[profile.model_id] = adapter_cls(profile)
                else:
                    adapters[profile.model_id] = SpecializedUnavailableAdapter(profile)
        retriever = Retriever(E5EmbeddingAdapter(e5_runtime)) if e5_runtime is not None else None
        return cls(registry, adapters, resources=resources, retriever=retriever, poll_interval_ms=poll_interval_ms)

    async def run(self, prompt: str, *, request_id: str | None = None, verify: bool = False, evidence: list[Evidence] | None = None, deadline_seconds: float = 45, capability: Capability | str | None = None, priority: int | None = None, output_format: str | None = None, document_path: str | None = None) -> dict[str, Any]:
        print("DEBUG ORCHESTRATOR RUN document_path =", repr(document_path))
        request_id = request_id or str(uuid.uuid4())

        # Detect the request before planning so automatic routing
        # can select the appropriate deadline.
        understanding = self.controller.understand(
            prompt,
            capability,
            priority=priority,
            deadline_seconds=deadline_seconds,
            verify=verify,
        )

        # Capability-specific deadlines.
        if deadline_seconds == 45:
            capability_deadlines = {
                Capability.CHAT: 45,
                Capability.CODING: 45,
                Capability.IMAGE_GENERATION: 180,
                Capability.STT: 180,
                Capability.TTS: 180,
                Capability.VIDEO_GENERATION: 600,
                Capability.DOCUMENT_CREATION: 120,
                Capability.DOCUMENT_ANALYSIS: 120,
            }

            detected = understanding.get("capabilities", [])

            if detected:
                try:
                    detected_capability = Capability(detected[0])
                    deadline_seconds = capability_deadlines.get(
                        detected_capability,
                        deadline_seconds,
                    )
                except (ValueError, TypeError):
                    pass

        understanding["deadline_seconds"] = deadline_seconds

        graph = self.planner.plan(
            request_id=request_id,
            prompt=prompt,
            verify=verify,
            deadline_seconds=deadline_seconds,
            capability=capability,
            priority=priority,
            output_format=output_format,
            document_path=document_path,
        )

        # Keep simple user-facing Chat requests concise to reduce CPU generation latency.
        if understanding.get("complexity") == "low":
            for task in graph.tasks.values():
                if task.capability == Capability.CHAT:
                    concise_instruction = (
                        "Answer briefly in 2–4 sentences. "
                        "Do not add unnecessary examples, bullet points, or extra explanation "
                        "unless the user requests them."
                    )
                    if isinstance(task.input, list):
                        messages = task.input
                    elif isinstance(task.input, dict):
                        messages = [
                            {
                                "role": "user",
                                "content": str(task.input.get("prompt", prompt)),
                            }
                        ]
                    else:
                        messages = [
                            {
                                "role": "user",
                                "content": str(task.input),
                            }
                        ]

                    task.input = {
                        "messages": [
                            {"role": "system", "content": concise_instruction},
                            *messages,
                        ],
                        "max_output_tokens": 64,
                    }

                elif task.capability == Capability.CODING:
                    coding_instruction = (
                        "Provide a concise, correct solution. "
                        "Return only the necessary code unless the user explicitly asks "
                        "for an explanation. Avoid unnecessary comments or extra text."
                    )

                    if isinstance(task.input, list):
                        messages = task.input
                    elif isinstance(task.input, dict):
                        messages = [
                            {
                                "role": "user",
                                "content": str(task.input.get("prompt", prompt)),
                            }
                        ]
                    else:
                        messages = [
                            {
                                "role": "user",
                                "content": str(task.input),
                            }
                        ]

                    task.input = {
                        "messages": [
                            {"role": "system", "content": coding_instruction},
                            *messages,
                        ],
                        "max_output_tokens": 96,
                    }

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
                        if isinstance(task.input, dict):
                            task.input = dict(task.input)

                            if task.capability == Capability.CHAT:
                                document_context = (
                                    "You are summarizing the output of a document-analysis model.\n"
                                    "Use ONLY the document-analysis content provided below.\n"
                                    "Do NOT add facts, topics, examples, or information that are not present "
                                    "in the document-analysis content.\n"
                                    "If the analysis is incomplete or unclear, say so.\n\n"
                                    "Original user request:\n"
                                    f"{prompt}\n\n"
                                    "Document-analysis content:\n"
                                    + "\n\n".join(dependency_text)
                                )

                                messages = task.input.get("messages")
                                if isinstance(messages, list):
                                    task.input["messages"] = [
                                        *messages,
                                        {"role": "user", "content": document_context},
                                    ]
                                else:
                                    task.input["messages"] = [
                                        {"role": "user", "content": document_context}
                                    ]

                            else:
                                task.input["prompt"] = (
                                    str(task.input.get("prompt", prompt))
                                    + "\n\nContext from prerequisite tasks:\n"
                                    + "\n\n".join(dependency_text)
                                )
                        else:
                            task.input = (
                                f"{prompt}\n\nContext from prerequisite tasks:\n"
                                + "\n\n".join(dependency_text)
                            )
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
        if (retrieval_requested or verify) and not ranked_evidence and self.retriever is not None:
            ranked_evidence = self.trirag.retrieve_from_index(prompt)
        grounded = None
        if verify:
            status, ranked_evidence = self.trirag.verify(prompt, results, ranked_evidence)
            outputs = [
                str(r.output.get("text", r.output)) if isinstance(r.output, dict) else str(r.output)
                for r in results
            ]
            grounded = self.trirag.grounded_synthesis(prompt, outputs, ranked_evidence)
        aggregate = self.aggregator.combine(results, list(graph.tasks.values()))
        self.evidence_store.add(request_id, ranked_evidence)
        return {"request_id": request_id, "status": "completed" if all(r.error is None for r in results) and results and not aggregate["failed_tasks"] else "failed", "verification_status": self.verification_gate.decide(status).value, "understanding": understanding, "text": grounded["text"] if grounded is not None else self.trirag.synthesize(results), "results": [{"task_id": r.task_id, "capability": r.capability.value, "model_id": r.model_id, "latency_ms": round(r.latency_ms, 2), "provenance": r.provenance} for r in results], "task_results": aggregate["tasks"], "evidence": [{"source_id": e.source_id, "document_id": e.document_id, "chunk_id": e.chunk_id, "score": e.score, "similarity": e.similarity, "content": e.content} for e in ranked_evidence], "aggregate": aggregate, "metrics": {"total_latency_ms": round((time.time() - self.jobs[request_id]["started_at"]) * 1000, 2), "task_count": len(graph.tasks), "completed_tasks": len(results)}}

    def index_document(self, document: Document):
        if self.retriever is None:
            raise RuntimeError("retrieval is unavailable")
        return self.retriever.index_document(document)

    async def chat_response(self, conversation_id: str, messages: list[ChatMessage], metadata=None, deadline_seconds: float = 45):
        return await self.chat.respond(conversation_id, messages, metadata, deadline_seconds)

    async def voice_response(self, audio_path: str, language: str | None = None, *, conversation_id: str | None = None, request_id: str | None = None, deadline_seconds: float = 180) -> dict[str, Any]:
        request_id = request_id or str(uuid.uuid4())
        started = time.perf_counter()

        # 1. Audio -> text using the existing Whisper STT pipeline.
        stt = await self.transcribe_audio(
            audio_path,
            language,
            request_id=request_id,
            deadline_seconds=deadline_seconds,
        )
        transcript = str(stt.get("transcript", "")).strip()
        if not transcript:
            raise ValueError("STT returned an empty transcript")

        # 2. Text -> answer using the existing Qwen chat pipeline.
        cid = conversation_id or request_id
        voice_system = ChatMessage(
            "system",
            (
                "You are responding through voice. "
                "Give a concise, natural spoken answer. "
                "Usually answer in 1–3 short sentences. "
                "Do not use markdown, bullet lists, headings, "
                "or unnecessary explanations. "
                "Only give more detail when the user explicitly asks for it."
            ),
        )

        chat = await self.chat_response(
            cid,
            [voice_system, ChatMessage("user", transcript)],
            metadata={"voice_request_id": request_id, "source": "voice"},
            deadline_seconds=min(float(deadline_seconds), 45.0),
        )
        answer_text = str(chat["message"]["content"]).strip()
        if not answer_text:
            raise ValueError("Chat returned an empty answer")

        # 3. Answer text -> audio using the existing Kokoro TTS adapter.
        tts_task = Task(
            request_id,
            "tts",
            Capability.TTS,
            {"text": answer_text, "voice": "af_heart", "speed": 1.0},
            priority=90,
            deadline=time.time() + min(float(deadline_seconds), 180.0),
        )
        tts_result = await self.scheduler.run(tts_task)
        tts_output = dict(tts_result.output) if isinstance(tts_result.output, dict) else {"audio_path": str(tts_result.output)}

        audio_path = str(tts_output.get("audio_path", ""))
        audio_url = ""
        if audio_path:
            audio_url = f"/v1/voice/audio?path={urllib.parse.quote(audio_path)}"
            tts_output["audio_url"] = audio_url

        return {
            "request_id": request_id,
            "status": "completed",
            "transcript": transcript,
            "answer": answer_text,
            "audio": tts_output,
            "stt": stt,
            "chat": chat,
            "tts": {
                "task_id": tts_result.task_id,
                "model_id": tts_result.model_id,
                "latency_ms": tts_result.latency_ms,
                "provenance": tts_result.provenance,
            },
            "total_processing_time_seconds": time.perf_counter() - started,
        }

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
