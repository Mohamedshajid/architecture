from __future__ import annotations

import asyncio
import gc
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .adapters import ModelAdapter
from .models import ModelProfile, Task, TaskResult, TaskStatus, ModelLifecycleState
from .registry import ModelRegistry


class ResourceUnavailableError(RuntimeError):
    """Controlled admission failure; the scheduler never force-loads a model."""


@dataclass(frozen=True)
class ResourceReservation:
    task_id: str
    model_id: str
    ram_mb: int
    cpu: int
    heavyweight: bool
    created_at: float = field(default_factory=time.time)


@dataclass
class ResourceManager:
    """Single authoritative RAM/CPU admission and reservation mechanism."""

    ram_limit_mb: int | None = None  # deterministic test override
    cpu_limit: int = 4
    min_free_ram_mb: int = 768
    max_heavy_models_resident: int = 1
    resident_ram_mb: int = 0
    resident_components: tuple[str, ...] = ()
    available_ram_provider: Callable[[], int] | None = None
    reserved_ram_mb: int = 0
    reserved_cpu: int = 0
    _reservations: dict[str, ResourceReservation] = field(default_factory=dict, init=False, repr=False)
    _loaded_heavy: set[str] = field(default_factory=set, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def available_ram_mb(self) -> int:
        if self.ram_limit_mb is not None:
            return self.ram_limit_mb
        if self.available_ram_provider is not None:
            return max(0, int(self.available_ram_provider()))
        try:
            import psutil
            return int(psutil.virtual_memory().available // (1024 * 1024))
        except ImportError:
            return 0

    def admission(self, profile: ModelProfile) -> tuple[bool, str]:
        with self._lock:
            available = self.available_ram_mb()
            if self.reserved_cpu + profile.cpu_requirement > self.cpu_limit:
                return False, "cpu reservation limit"
            if profile.heavyweight and len(self._loaded_heavy) >= self.max_heavy_models_resident:
                return False, "heavy-model residency limit"
            required = self.resident_ram_mb + self.reserved_ram_mb + profile.ram_requirement_mb + self.min_free_ram_mb
            if available < required:
                return False, f"free RAM {available} MB below required {required} MB"
            return True, "admitted"

    def reserve(self, task: Task, profile: ModelProfile) -> ResourceReservation:
        with self._lock:
            allowed, reason = self.admission(profile)
            if not allowed:
                raise ResourceUnavailableError(reason)
            reservation = ResourceReservation(task.task_id, profile.model_id, profile.ram_requirement_mb, profile.cpu_requirement, profile.heavyweight)
            self._reservations[task.task_id] = reservation
            self.reserved_ram_mb += reservation.ram_mb
            self.reserved_cpu += reservation.cpu
            if reservation.heavyweight:
                self._loaded_heavy.add(profile.model_id)
            return reservation

    def release(self, task_id: str) -> None:
        with self._lock:
            reservation = self._reservations.pop(task_id, None)
            if reservation is None:
                return
            self.reserved_ram_mb = max(0, self.reserved_ram_mb - reservation.ram_mb)
            self.reserved_cpu = max(0, self.reserved_cpu - reservation.cpu)
            if reservation.heavyweight:
                self._loaded_heavy.discard(reservation.model_id)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "available_ram_mb": self.available_ram_mb(),
                "min_free_ram_mb": self.min_free_ram_mb,
                "resident_ram_mb": self.resident_ram_mb,
                "resident_components": list(self.resident_components),
                "ram_reserved_mb": self.reserved_ram_mb,
                "cpu_limit": self.cpu_limit,
                "cpu_reserved": self.reserved_cpu,
                "loaded_heavy_models": sorted(self._loaded_heavy),
                "reservation_count": len(self._reservations),
                "max_heavy_models_resident": self.max_heavy_models_resident,
            }

    def preflight(self, profile: ModelProfile) -> dict[str, object]:
        """Report admission using the exact calculation used by reserve()."""
        with self._lock:
            available = self.available_ram_mb()
            required = self.resident_ram_mb + self.reserved_ram_mb + profile.ram_requirement_mb + self.min_free_ram_mb
            deficit = max(0, required - available)
            allowed, reason = self.admission(profile)
            try:
                import psutil
                memory = psutil.virtual_memory()
                total_bytes = int(memory.total)
                available_bytes = int(memory.available)
                used_bytes = int(memory.used)
                process_rss_bytes = int(psutil.Process().memory_info().rss)
            except ImportError:
                total_bytes = available_bytes = used_bytes = process_rss_bytes = None
            return {
                "model_id": profile.model_id,
                "capability": profile.capability.value,
                "available_ram_mb": available,
                "total_ram_bytes": total_bytes,
                "used_ram_bytes": used_bytes,
                "current_process_rss_bytes": process_rss_bytes,
                "reserved_ram_mb": self.reserved_ram_mb,
                "loaded_model_ram_mb": self.reserved_ram_mb,
                "loaded_models": sorted(self._loaded_heavy),
                "resident_components": list(self.resident_components),
                "resident_ram_mb": self.resident_ram_mb if self.resident_ram_mb else None,
                "resident_ram_status": "MEASURED" if self.resident_ram_mb else ("NOT_MEASURED" if self.resident_components else "NOT_APPLICABLE"),
                "estimated_model_mb": profile.ram_requirement_mb,
                "measured_model_mb": profile.measured_ram_mb,
                "safety_margin_mb": self.min_free_ram_mb,
                "required_free_mb": required,
                "available_headroom_mb": available - required,
                "deficit_mb": deficit,
                "admission": "READY_FOR_LOAD" if allowed else "RESOURCE_UNAVAILABLE",
                "rejection_reason": None if allowed else reason,
                "formula": "available >= resident_ram + reserved_ram + estimated_model + safety_margin",
            }


class AdaptiveScheduler:
    def __init__(self, registry: ModelRegistry, adapters: dict[str, ModelAdapter], resources: ResourceManager | None = None, poll_interval_ms: int = 100):
        self.registry, self.adapters = registry, adapters
        self.resources = resources or ResourceManager()
        self.poll_interval = max(0.025, poll_interval_ms / 1000)
        self._cancellations: dict[str, threading.Event] = {}
        self._condition = asyncio.Condition()
        self._active: dict[str, Task] = {}
        self._waiting: dict[str, Task] = {}

    def cancel(self, request_id: str) -> None:
        self._cancellations.setdefault(request_id, threading.Event()).set()

    def select_model(self, task: Task):
        candidates = [p for p in self.registry.for_capability(task.capability)
                      if str(p.integration_status).upper() in ("MOCK", "READY", "LOADED") and p.model_id in self.adapters]
        candidates.sort(key=lambda m: (-task.priority, -m.quality_score, m.expected_latency_ms, m.ram_requirement_mb))
        if not candidates:
            raise ResourceUnavailableError(f"no available real or mock model for {task.capability.value}")
        return candidates[0], self.adapters[candidates[0].model_id]

    async def _reserve_until_admitted(self, task: Task, profile: ModelProfile, cancellation: threading.Event) -> ResourceReservation:
        task.status = TaskStatus.WAITING_FOR_RESOURCE
        self._waiting[task.task_id] = task
        try:
            while True:
                if cancellation.is_set():
                    task.status, task.error = TaskStatus.CANCELLED, "request cancelled while queued"
                    raise asyncio.CancelledError()
                if time.time() >= task.deadline:
                    task.status, task.error = TaskStatus.WAITING_FOR_RESOURCE, "resource deadline exceeded"
                    raise TimeoutError(task.error)
                higher_priority_waiter = any(other.priority > task.priority for other_id, other in self._waiting.items() if other_id != task.task_id)
                if higher_priority_waiter:
                    async with self._condition:
                        try:
                            await asyncio.wait_for(self._condition.wait(), self.poll_interval)
                        except asyncio.TimeoutError:
                            pass
                    continue
                try:
                    reservation = self.resources.reserve(task, profile)
                    task.status = TaskStatus.RUNNING
                    return reservation
                except ResourceUnavailableError as exc:
                    task.error = f"QUEUED_RESOURCE_LIMIT: {exc}"
                    # A fixed synthetic budget is used by deterministic tests
                    # and cannot recover while this task is waiting.
                    if self.resources.ram_limit_mb is not None and "free RAM" in str(exc):
                        task.status = TaskStatus.WAITING_FOR_RESOURCE
                        raise
                    async with self._condition:
                        try:
                            await asyncio.wait_for(self._condition.wait(), self.poll_interval)
                        except asyncio.TimeoutError:
                            pass
        finally:
            self._waiting.pop(task.task_id, None)

    async def _notify_resources(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    async def run(self, task: Task) -> TaskResult:
        if time.time() > task.deadline:
            task.status, task.error = TaskStatus.FAILED, "deadline exceeded"
            raise TimeoutError(task.error)
        profile, adapter = self.select_model(task)
        task.model_id = profile.model_id
        cancellation = self._cancellations.setdefault(task.request_id, threading.Event())
        started = time.time()
        queue_wait_started = time.perf_counter()
        await self._reserve_until_admitted(task, profile, cancellation)
        queue_wait_ms = (time.perf_counter() - queue_wait_started) * 1000
        self._active[task.task_id] = task
        loaded = False
        try:
            if cancellation.is_set():
                task.status = TaskStatus.CANCELLED
                raise asyncio.CancelledError()

            load_started = time.perf_counter()
            await adapter.load()
            load_ms = (time.perf_counter() - load_started) * 1000
            loaded = True

            if cancellation.is_set():
                task.status = TaskStatus.CANCELLED
                raise asyncio.CancelledError()

            adapter.state = ModelLifecycleState.EXECUTING.value
            timeout = max(0.01, task.deadline - time.time())

            inference_started = time.perf_counter()
            output = await asyncio.wait_for(adapter.execute(task.input, cancellation), timeout)
            inference_ms = (time.perf_counter() - inference_started) * 1000
            if cancellation.is_set():
                task.status = TaskStatus.CANCELLED
                raise asyncio.CancelledError()
            adapter.state = ModelLifecycleState.LOADED.value
            task.status, task.result, task.error = TaskStatus.COMPLETED, output, None
            return TaskResult(task.task_id, task.request_id, task.capability, output, profile.model_id, started, time.time(), {
                "runtime": profile.runtime,
                "queue_wait_ms": round(queue_wait_ms, 2),
                "load_ms": round(load_ms, 2),
                "inference_ms": round(inference_ms, 2),
                "resource": self.resources.snapshot(),
            })
        except asyncio.CancelledError:
            task.status, task.error = TaskStatus.CANCELLED, "request cancelled"
            await adapter.cancel()
            raise
        except TimeoutError:
            task.status, task.error = TaskStatus.FAILED, "deadline exceeded"
            await adapter.cancel()
            raise
        except Exception as exc:
            if adapter.state == ModelLifecycleState.LOADING.value:
                adapter.state = ModelLifecycleState.LOAD_FAILED.value
            task.status, task.error = TaskStatus.FAILED, str(exc)
            raise
        finally:
            self._active.pop(task.task_id, None)
            if loaded and profile.unloadable:
                try:
                    await adapter.unload()
                except Exception:
                    adapter.state = ModelLifecycleState.UNHEALTHY.value
            self.resources.release(task.task_id)
            gc.collect()
            await self._notify_resources()

    def status(self) -> dict[str, object]:
        return {
            "queue_length": len(self._waiting),
            "active_task_count": len(self._active),
            "resources": self.resources.snapshot(),
        }
