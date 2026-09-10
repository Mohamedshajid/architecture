from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Capability(str, Enum):
    CHAT = "chat"
    CODING = "coding"
    IMAGE_GENERATION = "image_generation"
    STT = "stt"
    TTS = "tts"
    VIDEO_GENERATION = "video_generation"
    DOCUMENT_CREATION = "document_creation"
    DOCUMENT_ANALYSIS = "document_analysis"
    RETRIEVAL = "retrieval"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_FOR_RESOURCE = "waiting_for_resource"
    WAITING_FOR_DEPENDENCY = "waiting_for_dependency"


class ModelLifecycleState(str, Enum):
    NOT_INSTALLED = "NOT_INSTALLED"
    INSTALLED = "INSTALLED"
    LOADING = "LOADING"
    READY = "READY"
    LOADED = "LOADED"
    EXECUTING = "EXECUTING"
    UNLOADING = "UNLOADING"
    EVICTED = "EVICTED"
    LOAD_FAILED = "LOAD_FAILED"
    UNHEALTHY = "UNHEALTHY"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNCERTAIN = "uncertain"
    FAILED = "failed"
    NOT_REQUIRED = "not_required"


@dataclass
class LongRunningJob:
    task_id: str
    capability: Capability
    status: str = "pending"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    output_ref: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    name: str
    capability: Capability
    runtime: str = "mock"
    model_path: str | None = None
    model_identifier: str | None = None
    format: str | None = None
    quantization: str | None = None
    parameter_count: int | None = None
    ram_requirement_mb: int = 128
    measured_ram_mb: int | None = None
    cpu_requirement: int = 1
    gpu_requirement_mb: int = 0
    context_length: int = 4096
    max_output_tokens: int = 256
    expected_latency_ms: int = 1000
    quality_score: float = 0.5
    priority: int = 50
    concurrency_limit: int = 1
    load_mode: str = "on_demand"
    fallback_model: str | None = None
    supported_inputs: tuple[str, ...] = ("text",)
    supported_outputs: tuple[str, ...] = ("text",)
    enabled: bool = True
    health_status: str = "unknown"
    integration_status: str = "mock"
    input_types: tuple[str, ...] = ("text",)
    output_types: tuple[str, ...] = ("text",)
    streaming_supported: bool = False
    cancellation_supported: bool = True
    progress_supported: bool = False
    lifecycle_policy: str = "on_demand"
    heavyweight: bool = True
    long_running: bool = False
    unloadable: bool = True


@dataclass
class Task:
    request_id: str
    task_type: str
    capability: Capability
    input: Any
    dependencies: list[str] = field(default_factory=list)
    preferred_models: list[str] = field(default_factory=list)
    priority: int = 50
    deadline: float = field(default_factory=lambda: time.time() + 45)
    resource_estimate: dict[str, int] = field(default_factory=dict)
    verification_required: bool = False
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    model_id: str | None = None


@dataclass
class Evidence:
    source_id: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    document_id: str | None = None
    chunk_id: str | None = None
    similarity: float | None = None


@dataclass
class TaskResult:
    task_id: str
    request_id: str
    capability: Capability
    output: Any
    model_id: str
    started_at: float
    finished_at: float
    provenance: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        return (self.finished_at - self.started_at) * 1000
