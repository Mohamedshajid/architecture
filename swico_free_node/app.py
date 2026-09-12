from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
import secrets
import threading
import time
import hashlib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

try:
    import psutil
except ImportError:  # pragma: no cover - installed on the Windows node
    psutil = None

try:
    from .config import NodeConfig
    from .e5_runtime import E5Runtime
    from .qwen_runtime import QwenRuntime
    from .whisper_runtime import WhisperRuntime
    from .schemas import EmbedRequest, GenerateRequest
    from .orchestrator import Orchestrator
    from .orchestrator.api import create_router
    from .orchestrator.api import OrchestrateRequest
    from .orchestrator.models import Evidence
    from .orchestrator.retrieval import Document
    from .orchestrator.chat import ChatMessage
    from .orchestrator.scheduler import ResourceManager
except ImportError:  # pragma: no cover - direct uvicorn execution from this folder
    from config import NodeConfig
    from e5_runtime import E5Runtime
    from qwen_runtime import QwenRuntime
    from whisper_runtime import WhisperRuntime
    from schemas import EmbedRequest, GenerateRequest
    from orchestrator import Orchestrator
    from orchestrator.api import create_router
    from orchestrator.api import OrchestrateRequest
    from orchestrator.models import Evidence
    from orchestrator.retrieval import Document
    from orchestrator.chat import ChatMessage
    from orchestrator.scheduler import ResourceManager


class GenerationCapacity:
    def __init__(self, active: int, queue: int) -> None:
        self._active_limit = active
        self._limit = active + queue
        self._active = 0
        self._waiting = 0
        self._condition = asyncio.Condition()

    async def acquire(self, *, timeout_seconds: float | None = None) -> float:
        queued_at = time.perf_counter()
        async with self._condition:
            if self._active + self._waiting >= self._limit:
                raise CapacityBusyError()
            self._waiting += 1
            acquired = False
            try:
                while self._active >= self._active_limit:
                    if timeout_seconds is None:
                        await self._condition.wait()
                    else:
                        remaining = timeout_seconds - (time.perf_counter() - queued_at)
                        if remaining <= 0:
                            raise CapacityWaitExpired()
                        try:
                            await asyncio.wait_for(self._condition.wait(), remaining)
                        except asyncio.TimeoutError as exc:
                            raise CapacityWaitExpired() from exc
                self._waiting -= 1
                self._active += 1
                acquired = True
                return (time.perf_counter() - queued_at) * 1000
            except BaseException:
                if not acquired:
                    self._waiting -= 1
                self._condition.notify_all()
                raise

    async def release(self) -> None:
        async with self._condition:
            self._active = max(0, self._active - 1)
            self._condition.notify(1)

    @property
    def waiting_or_active(self) -> int:
        return self._active + self._waiting

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def active_capacity(self) -> int:
        return self._active_limit

    @property
    def queue_capacity(self) -> int:
        return max(0, self._limit - self._active_limit)


class CapacityBusyError(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=429,
            detail={"code": "swico_free_busy", "message": "Swico Free is busy."},
            headers={"Retry-After": "1"},
        )


class CapacityWaitExpired(RuntimeError):
    pass


config: NodeConfig | None = None
qwen: QwenRuntime | None = None
e5: E5Runtime | None = None
stt_runtime: WhisperRuntime | None = None
capacity: GenerationCapacity | None = None
embedding_capacity: GenerationCapacity | None = None
orchestrator: Orchestrator | None = None
startup_metrics: dict[str, int] = {}
process_started_at = time.monotonic()
bearer = HTTPBearer(auto_error=False)
counter_lock = threading.Lock()
generation_counters = {
    "total_completed_generations": 0,
    "total_failed_generations": 0,
    "total_timed_out_generations": 0,
    "total_busy_rejections": 0,
}
process_peak_rss_bytes = 0
EMBEDDING_QUEUE_WAIT_SECONDS = 10.0
ABORT_GRACE_SECONDS = 2.0


class MockQwenRuntime:
    """Explicit development-only runtime; never selected unless MOCK_MODE is enabled."""
    def generate(self, messages, max_output_tokens, cancellation=None):
        if cancellation is not None and cancellation.is_set():
            raise RuntimeError("mock generation cancelled")
        prompt = str(messages[-1].get("content", ""))
        return f"[mock-qwen] {prompt[:max_output_tokens]}", {"finish_reason": "stop", "mock": True}

    def stream(self, messages, max_output_tokens, cancellation=None):
        text, usage = self.generate(messages, max_output_tokens, cancellation)
        for token in text.split(" "):
            if cancellation is not None and cancellation.is_set():
                break
            yield token + " ", {}
        yield "", usage


class MockE5Runtime:
    dimensions = 384

    def embed(self, texts, modes):
        vectors = []
        for text, mode in zip(texts, modes):
            digest = hashlib.sha256(f"{mode}:{text}".encode()).digest()
            vectors.append([((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(self.dimensions)])
        return vectors


class IndexDocumentRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=200_000)
    source: str = Field(min_length=1, max_length=1_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=24_000)
    top_k: int = Field(default=5, ge=1, le=20)


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(default=None, max_length=256)
    messages: list[dict[str, str]] = Field(default_factory=list, max_length=24)
    system_instruction: str | None = Field(default=None, max_length=4000)
    user_message: str | None = Field(default=None, max_length=12000)
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    deadline_seconds: float = Field(default=45, ge=1, le=300)


class STTRequest(BaseModel):
    audio_path: str = Field(min_length=1, max_length=1_000)
    language: str | None = Field(default=None, max_length=16)
    request_id: str | None = Field(default=None, max_length=256)
    deadline_seconds: float = Field(default=180, ge=1, le=300)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global config, qwen, e5, coding_runtime, stt_runtime, capacity, embedding_capacity, startup_metrics, orchestrator
    started = time.perf_counter()
    config = NodeConfig.from_environment()
    qwen_started = time.perf_counter()
    qwen = MockQwenRuntime() if config.mock_mode else QwenRuntime(
        config.qwen_gguf_path, threads=config.qwen_threads, batch_size=config.qwen_batch_size,
        context_size=config.qwen_context_size,
    )
    qwen_elapsed = time.perf_counter()
    e5_started = time.perf_counter()
    e5 = MockE5Runtime() if config.mock_mode else E5Runtime(config.e5_model_path, threads=config.e5_threads)
    e5_elapsed = time.perf_counter()
    coding_path = os.getenv("SWICO_CODING_MODEL_PATH", "").strip()
    coding_runtime = None
    if not config.mock_mode and coding_path and os.path.isfile(coding_path):
        coding_runtime = QwenRuntime(
            __import__('pathlib').Path(coding_path),
            threads=config.qwen_threads,
            batch_size=config.qwen_batch_size,
            context_size=config.qwen_context_size,
        )
    stt_path = os.getenv("SWICO_STT_MODEL_PATH", "").strip()
    stt_runtime = None
    if not config.mock_mode and stt_path and os.path.isdir(stt_path):
        stt_runtime = WhisperRuntime(__import__('pathlib').Path(stt_path), threads=2)
    capacity = GenerationCapacity(config.max_concurrent_generations, config.max_queue_size)
    embedding_capacity = GenerationCapacity(
        config.max_concurrent_embeddings, config.max_embedding_queue_size,
    )
    orchestrator = Orchestrator.with_defaults(
        qwen, e5, config.max_output_tokens, config.mock_mode,
        coding_runtime=coding_runtime,
        stt_runtime=stt_runtime,
        resources=ResourceManager(
            min_free_ram_mb=config.min_free_ram_mb,
            max_heavy_models_resident=config.max_heavy_models_resident,
            resident_ram_mb=0 if config.mock_mode else 668,
            resident_components=() if config.mock_mode else ("e5",),
        ),
        poll_interval_ms=config.resource_poll_interval_ms,
        model_paths=config.model_paths(),
    )
    startup_metrics = {
        "qwen_startup_ms": int((qwen_elapsed - qwen_started) * 1000),
        "e5_startup_ms": int((e5_elapsed - e5_started) * 1000),
        "model_startup_ms": int((e5_elapsed - started) * 1000),
    }
    yield


app = FastAPI(title="Swico Free inference node", docs_url=None, redoc_url=None, lifespan=lifespan)


def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    if config is None or credentials is None or credentials.scheme.lower() != "bearer" or not secrets.compare_digest(credentials.credentials, config.token):
        raise HTTPException(401, {"code": "unauthorized", "message": "Authentication required."}, headers={"WWW-Authenticate": "Bearer"})


@app.post("/v1/orchestrate")
async def orchestrate_endpoint(payload: OrchestrateRequest, _: None = Depends(require_auth)):
    if orchestrator is None:
        raise HTTPException(503, {"code": "swico_free_unavailable", "message": "Orchestrator is unavailable."})
    evidence = [Evidence(source_id=str(item.get("source_id", "unknown")), content=str(item.get("content", "")), score=float(item.get("score", 0))) for item in payload.evidence]
    if payload.background:
        request_id = payload.request_id or str(__import__('uuid').uuid4())
        task = asyncio.create_task(orchestrator.run(payload.prompt, request_id=request_id, verify=payload.verify, evidence=evidence, deadline_seconds=payload.deadline_seconds, capability=payload.capability, priority=payload.priority))
        orchestrator.jobs.setdefault(request_id, {"graph": None, "started_at": time.time(), "cancelled": False, "future": task})
        return {"request_id": request_id, "status": "accepted"}
    try:
        return await orchestrator.run(payload.prompt, request_id=payload.request_id, verify=payload.verify, evidence=evidence, deadline_seconds=payload.deadline_seconds, capability=payload.capability, priority=payload.priority)
    except (TimeoutError, RuntimeError) as exc:
        raise HTTPException(503, {"code": "orchestration_unavailable", "message": str(exc)}) from exc


@app.post("/v1/stt")
async def stt_endpoint(payload: STTRequest, _: None = Depends(require_auth)):
    if orchestrator is None:
        raise HTTPException(503, {"code": "swico_free_unavailable"})
    try:
        return await orchestrator.transcribe_audio(payload.audio_path, payload.language, request_id=payload.request_id, deadline_seconds=payload.deadline_seconds)
    except (TimeoutError, RuntimeError, ValueError) as exc:
        raise HTTPException(503, {"code": "stt_unavailable", "message": str(exc)}) from exc


@app.get("/v1/models")
async def models_endpoint(_: None = Depends(require_auth)):
    if orchestrator is None:
        raise HTTPException(503, {"code": "swico_free_unavailable"})
    return {"models": orchestrator.registry.as_dicts()}


@app.get("/v1/models/health")
async def models_health_endpoint(_: None = Depends(require_auth)):
    if orchestrator is None:
        raise HTTPException(503, {"code": "swico_free_unavailable"})
    return {"models": orchestrator.registry.health(orchestrator.adapters, orchestrator.scheduler.status())}


@app.get("/v1/resources/preflight")
async def resources_preflight_endpoint(model_id: str = Query(default="stt-whisper-base", max_length=128), _: None = Depends(require_auth)):
    if orchestrator is None:
        raise HTTPException(503, {"code": "swico_free_unavailable"})
    try:
        return orchestrator.resource_preflight(model_id)
    except KeyError as exc:
        raise HTTPException(404, {"code": "model_not_found", "message": str(exc)}) from exc


@app.get("/v1/capabilities")
async def capabilities_endpoint(_: None = Depends(require_auth)):
    if orchestrator is None:
        raise HTTPException(503, {"code": "swico_free_unavailable"})
    return {"capabilities": orchestrator.capabilities.as_dict()}


@app.post("/v1/chat")
async def chat_endpoint(payload: ChatRequest, _: None = Depends(require_auth)):
    if orchestrator is None:
        raise HTTPException(503, {"code": "swico_free_unavailable"})
    conversation_id = payload.conversation_id or str(__import__('uuid').uuid4())
    incoming = [ChatMessage(item.get("role", ""), item.get("content", "")) for item in payload.messages]
    if payload.system_instruction:
        incoming.insert(0, ChatMessage("system", payload.system_instruction))
    if payload.user_message:
        incoming.append(ChatMessage("user", payload.user_message))
    if not incoming or any(item.role not in {"system", "user", "assistant"} or not item.content.strip() for item in incoming):
        raise HTTPException(422, {"code": "invalid_chat_messages"})
    result = await orchestrator.chat_response(conversation_id, incoming, payload.metadata, payload.deadline_seconds)
    if not payload.stream:
        return result
    async def body():
        yield f"data: {json.dumps({'conversation_id': conversation_id, 'delta': result['message']['content'], 'model_id': result['model_id']}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/v1/chat/{conversation_id}")
async def get_chat_endpoint(conversation_id: str, _: None = Depends(require_auth)):
    if orchestrator is None:
        raise HTTPException(503, {"code": "swico_free_unavailable"})
    conversation = orchestrator.chat.store.get(conversation_id)
    if conversation is None:
        raise HTTPException(404, {"code": "conversation_not_found"})
    return {"conversation_id": conversation.conversation_id, "messages": [{"role": m.role, "content": m.content} for m in conversation.messages], "metadata": conversation.metadata}


@app.delete("/v1/chat/{conversation_id}")
async def delete_chat_endpoint(conversation_id: str, _: None = Depends(require_auth)):
    if orchestrator is None:
        raise HTTPException(503, {"code": "swico_free_unavailable"})
    return {"conversation_id": conversation_id, "deleted": orchestrator.chat.store.clear(conversation_id)}


@app.post("/v1/retrieval/index")
async def retrieval_index_endpoint(payload: IndexDocumentRequest, _: None = Depends(require_auth)):
    if orchestrator is None or orchestrator.retriever is None:
        raise HTTPException(503, {"code": "retrieval_unavailable", "message": "E5 retrieval is unavailable."})
    chunks = await asyncio.to_thread(orchestrator.index_document, Document(payload.document_id, payload.text, payload.source, payload.metadata))
    return {"document_id": payload.document_id, "chunks": [{"document_id": c.document_id, "chunk_id": c.chunk_id, "source": c.source, "chunk_index": c.chunk_index, "text": c.text, "page": c.page, "section": c.section, "metadata": c.metadata} for c in chunks], "count": orchestrator.retriever.count(), "mode": "mock" if config and config.mock_mode else "real"}


@app.post("/v1/retrieval/search")
async def retrieval_search_endpoint(payload: SearchRequest, _: None = Depends(require_auth)):
    if orchestrator is None or orchestrator.retriever is None:
        raise HTTPException(503, {"code": "retrieval_unavailable", "message": "E5 retrieval is unavailable."})
    try:
        results = await asyncio.to_thread(orchestrator.retriever.search, payload.query, payload.top_k)
    except Exception as exc:
        raise HTTPException(503, {"code": "retrieval_unavailable", "message": str(exc)}) from exc
    return {"results": [{"document_id": r.chunk.document_id, "chunk_id": r.chunk.chunk_id, "source": r.chunk.source, "text": r.chunk.text, "similarity": r.similarity, "rank": r.rank, "metadata": r.chunk.metadata} for r in results], "count": orchestrator.retriever.count(), "mode": "mock" if config and config.mock_mode else "real"}


@app.delete("/v1/retrieval/documents/{document_id}")
async def retrieval_delete_endpoint(document_id: str, _: None = Depends(require_auth)):
    if orchestrator is None or orchestrator.retriever is None:
        raise HTTPException(503, {"code": "retrieval_unavailable"})
    await asyncio.to_thread(orchestrator.retriever.delete_document, document_id)
    return {"document_id": document_id, "deleted": True, "count": orchestrator.retriever.count()}


@app.get("/v1/tasks/{request_id}")
async def task_status_endpoint(request_id: str, _: None = Depends(require_auth)):
    if orchestrator is None or request_id not in orchestrator.jobs:
        raise HTTPException(404, {"code": "task_not_found"})
    job = orchestrator.jobs[request_id]
    if job.get("graph") is None:
        return {"request_id": request_id, "cancelled": job["cancelled"], "status": "running" if not job["future"].done() else "completed"}
    return {"request_id": request_id, "cancelled": job["cancelled"], "tasks": [{"task_id": t.task_id, "capability": t.capability.value, "status": t.status.value, "model_id": t.model_id, "error": t.error} for t in job["graph"].tasks.values()]}


@app.post("/v1/tasks/{request_id}/cancel")
async def cancel_task_endpoint(request_id: str, _: None = Depends(require_auth)):
    if orchestrator is None or not orchestrator.cancel(request_id):
        raise HTTPException(404, {"code": "task_not_found"})
    return {"request_id": request_id, "cancelled": True}


def _increment_counter(name: str) -> None:
    with counter_lock:
        generation_counters[name] += 1


def _counter_snapshot() -> dict[str, int]:
    with counter_lock:
        return dict(generation_counters)


def _process_metrics() -> dict[str, float | int]:
    global process_peak_rss_bytes
    if psutil is None:
        return {}
    try:
        process = psutil.Process(os.getpid())
        rss = int(process.memory_info().rss)
        process_peak_rss_bytes = max(process_peak_rss_bytes, rss)
        return {
            "process_rss_mb": round(rss / (1024 * 1024), 2),
            "process_peak_rss_mb": round(process_peak_rss_bytes / (1024 * 1024), 2),
            "process_cpu_percent": round(float(process.cpu_percent(interval=None)), 2),
        }
    except OSError:
        return {}


def _timeout_error() -> HTTPException:
    return HTTPException(
        status_code=504,
        detail={
            "code": "swico_free_timeout",
            "message": "Swico Free could not finish within the response-time limit.",
        },
    )


def _queue_wait_error() -> HTTPException:
    return HTTPException(
        status_code=429,
        detail={
            "code": "swico_free_busy",
            "message": "Swico Free is busy. Please try again shortly.",
        },
        headers={"Retry-After": "1"},
    )


async def _acquire_generation(
    node: NodeConfig, limit: GenerationCapacity, request_started: float,
) -> float:
    max_queue_wait_seconds = float(getattr(node, "max_queue_wait_seconds", 15))
    max_total_request_seconds = float(getattr(node, "max_total_request_seconds", 45))
    timeout_seconds = min(
        max_queue_wait_seconds,
        max_total_request_seconds,
    )
    try:
        queue_wait_ms = await limit.acquire(timeout_seconds=timeout_seconds)
    except CapacityBusyError:
        _increment_counter("total_busy_rejections")
        raise
    except CapacityWaitExpired as exc:
        if time.perf_counter() - request_started >= max_total_request_seconds:
            _increment_counter("total_timed_out_generations")
            raise _timeout_error() from exc
        _increment_counter("total_busy_rejections")
        raise _queue_wait_error() from exc
    if time.perf_counter() - request_started >= max_total_request_seconds:
        await limit.release()
        _increment_counter("total_timed_out_generations")
        raise _timeout_error()
    return queue_wait_ms


def _runtime() -> tuple[NodeConfig, QwenRuntime, E5Runtime, GenerationCapacity, GenerationCapacity]:
    if config is None or qwen is None or e5 is None or capacity is None or embedding_capacity is None:
        raise HTTPException(503, {"code": "swico_free_unavailable", "message": "Swico Free is unavailable."})
    return config, qwen, e5, capacity, embedding_capacity


@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1_500_000:
        raise HTTPException(413, {"code": "request_too_large", "message": "Request is too large."})
    return await call_next(request)


@app.get("/health")
async def health(_: None = Depends(require_auth)) -> dict[str, Any]:
    node, _qwen, _e5, limit, embeddings = _runtime()
    return {
        "status": "ok", "ready": True,
        "uptime_seconds": round(max(0.0, time.monotonic() - process_started_at), 3),
        "active_generations": limit.active,
        "waiting_generations": limit.waiting,
        "active_embeddings": embeddings.active,
        "waiting_embeddings": embeddings.waiting,
        "generation_capacity": limit.active_capacity,
        "generation_queue_capacity": limit.queue_capacity,
        "active_or_queued_generations": limit.waiting_or_active,
        "max_queue_size": node.max_queue_size,
        "active_or_queued_embeddings": embeddings.waiting_or_active,
        "max_embedding_queue_size": node.max_embedding_queue_size,
        "max_queue_wait_seconds": getattr(node, "max_queue_wait_seconds", 15),
        "max_total_request_seconds": getattr(node, "max_total_request_seconds", 45),
        **_counter_snapshot(),
        **_process_metrics(),
        **startup_metrics,
    }


@app.post("/v1/embed")
async def embed(payload: EmbedRequest, _: None = Depends(require_auth)) -> dict[str, Any]:
    _node, _qwen, runtime, _capacity, embeddings = _runtime()
    try:
        await embeddings.acquire(timeout_seconds=EMBEDDING_QUEUE_WAIT_SECONDS)
    except CapacityBusyError:
        raise
    except CapacityWaitExpired as exc:
        raise _queue_wait_error() from exc
    try:
        vectors = await asyncio.to_thread(runtime.embed, payload.texts, payload.modes)
        return {"vectors": vectors, "dimensions": 384}
    finally:
        await embeddings.release()


@app.post("/v1/generate")
async def generate(payload: GenerateRequest, _: None = Depends(require_auth)) -> dict[str, Any]:
    node, runtime, _e5, limit, _embeddings = _runtime()
    request_started = time.perf_counter()
    queue_wait_ms = await _acquire_generation(node, limit, request_started)
    cancellation = threading.Event()
    generation_started = time.perf_counter()
    generation_task = asyncio.create_task(asyncio.to_thread(
        runtime.generate,
        [item.model_dump() for item in payload.messages],
        min(node.max_output_tokens, payload.max_output_tokens),
        cancellation,
    ))
    release_deferred = False
    try:
        remaining = float(getattr(node, "max_total_request_seconds", 45)) - (time.perf_counter() - request_started)
        if remaining <= 0:
            raise asyncio.TimeoutError
        text, usage = await asyncio.wait_for(
            asyncio.shield(generation_task), timeout=remaining,
        )
        wall_generation_ms = (time.perf_counter() - generation_started) * 1000
        usage = dict(usage)
        usage["queue_wait_ms"] = round(queue_wait_ms, 2)
        usage["generation_ms"] = round(wall_generation_ms, 2)
        usage["total_node_latency_ms"] = round((time.perf_counter() - request_started) * 1000, 2)
        finish_reason = str(usage.get("finish_reason") or "stop")
        _increment_counter("total_completed_generations")
        return {
            "text": text, "usage": usage,
            "finish_reason": finish_reason,
            "truncated": bool(usage.get("truncated")) or finish_reason == "length",
        }
    except asyncio.TimeoutError as exc:
        cancellation.set()
        _increment_counter("total_timed_out_generations")
        try:
            await asyncio.wait_for(asyncio.shield(generation_task), timeout=ABORT_GRACE_SECONDS)
        except asyncio.TimeoutError:
            release_deferred = True
            generation_task.add_done_callback(
                lambda _task: asyncio.create_task(limit.release())
            )
        except BaseException:
            pass
        raise _timeout_error() from exc
    except BaseException:
        if not cancellation.is_set():
            _increment_counter("total_failed_generations")
        raise
    finally:
        if not release_deferred:
            await limit.release()


@app.post("/v1/generate/stream")
async def generate_stream(request: Request, payload: GenerateRequest, _: None = Depends(require_auth)) -> StreamingResponse:
    node, runtime, _e5, limit, _embeddings = _runtime()
    request_started = time.perf_counter()
    queue_wait_ms = await _acquire_generation(node, limit, request_started)
    if time.perf_counter() - request_started >= float(getattr(node, "max_total_request_seconds", 45)):
        await limit.release()
        _increment_counter("total_timed_out_generations")
        raise _timeout_error()
    messages = [item.model_dump() for item in payload.messages]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    cancellation = threading.Event()
    deadline_expired = threading.Event()

    def worker() -> None:
        generation_started = time.perf_counter()
        terminal_usage: dict[str, Any] = {}
        visible_output = False
        try:
            for delta, usage in runtime.stream(
                messages,
                min(node.max_output_tokens, payload.max_output_tokens),
                cancellation,
            ):
                if usage:
                    terminal_usage.update(usage)
                elif delta:
                    visible_output = True
                loop.call_soon_threadsafe(queue.put_nowait, ("usage" if usage else "delta", usage or delta))
            wall_generation_ms = (time.perf_counter() - generation_started) * 1000
            terminal_usage.update({
                "queue_wait_ms": round(queue_wait_ms, 2),
                "generation_ms": round(wall_generation_ms, 2),
                "total_node_latency_ms": round((time.perf_counter() - request_started) * 1000, 2),
            })
            if deadline_expired.is_set():
                _increment_counter("total_timed_out_generations")
                loop.call_soon_threadsafe(queue.put_nowait, (
                    "timeout", {**terminal_usage, "visible_output": visible_output},
                ))
            elif cancellation.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, ("cancelled", terminal_usage))
            else:
                _increment_counter("total_completed_generations")
                loop.call_soon_threadsafe(queue.put_nowait, ("done", terminal_usage))
        except Exception:
            if deadline_expired.is_set():
                _increment_counter("total_timed_out_generations")
                loop.call_soon_threadsafe(queue.put_nowait, (
                    "timeout", {"visible_output": visible_output},
                ))
            else:
                _increment_counter("total_failed_generations")
                loop.call_soon_threadsafe(queue.put_nowait, (
                    "error", {"finish_reason": "error", "truncated": False},
                ))

    task = asyncio.create_task(asyncio.to_thread(worker))

    async def watch_disconnect() -> None:
        try:
            while not await request.is_disconnected():
                await asyncio.sleep(0.2)
            cancellation.set()
            queue.put_nowait(("cancel", None))
        except asyncio.CancelledError:
            raise

    disconnect_task = asyncio.create_task(watch_disconnect())

    async def watch_deadline() -> None:
        try:
            remaining = float(getattr(node, "max_total_request_seconds", 45)) - (time.perf_counter() - request_started)
            if remaining > 0:
                await asyncio.sleep(remaining)
            if not task.done():
                deadline_expired.set()
                cancellation.set()
        except asyncio.CancelledError:
            raise

    deadline_task = asyncio.create_task(watch_deadline())

    async def body():
        try:
            while True:
                kind, value = await queue.get()
                if kind == "delta":
                    yield f"data: {json.dumps({'delta': value}, ensure_ascii=False)}\n\n"
                elif kind == "usage":
                    yield f"data: {json.dumps({'usage': value}, ensure_ascii=False)}\n\n"
                elif kind == "error":
                    yield f"data: {json.dumps({'error': 'swico_free_unavailable', **(value or {})})}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                elif kind == "timeout":
                    if value and value.get("visible_output"):
                        yield f"data: {json.dumps({**value, 'visible_output': None, 'finish_reason': 'timeout', 'truncated': True, 'completion_status': 'incomplete'}, ensure_ascii=False)}\n\n"
                    else:
                        yield "data: {\"error\":\"swico_free_timeout\"}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                elif kind == "cancelled":
                    yield f"data: {json.dumps({**(value or {}), 'finish_reason': 'cancelled', 'truncated': False})}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                elif kind == "cancel":
                    break
                else:
                    if value:
                        yield f"data: {json.dumps({'usage': value}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
        finally:
            cancellation.set()
            disconnect_task.cancel()
            deadline_task.cancel()
            if task.done():
                await limit.release()
            else:
                async def finish_worker() -> None:
                    try:
                        await task
                    except BaseException:
                        pass
                    await limit.release()
                asyncio.create_task(finish_worker())

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    cfg = NodeConfig.from_environment()
    uvicorn.run("app:app", host=cfg.host, port=cfg.port, workers=1)
