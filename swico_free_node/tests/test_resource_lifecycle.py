import asyncio
import threading
import time

import pytest

from orchestrator.adapters import ModelAdapter
from orchestrator.models import Capability, ModelLifecycleState, ModelProfile, Task, TaskStatus
from orchestrator.registry import ModelRegistry
from orchestrator.scheduler import AdaptiveScheduler, ResourceManager, ResourceUnavailableError


class RecordingAdapter(ModelAdapter):
    def __init__(self, profile, delay=0, failure=None):
        super().__init__(profile)
        self.delay = delay
        self.failure = failure
        self.loads = 0
        self.unloads = 0
        self.cancels = 0

    async def load(self):
        self.loads += 1
        await super().load()

    async def unload(self):
        self.unloads += 1
        await super().unload()

    async def execute(self, input_data, cancellation):
        self.state = ModelLifecycleState.EXECUTING.value
        for _ in range(20):
            if cancellation.is_set():
                raise asyncio.CancelledError()
            await asyncio.sleep(self.delay or 0)
        if self.failure:
            raise self.failure
        return input_data

    async def cancel(self):
        self.cancels += 1


def setup(*profiles, ram=4096, min_free=256, provider=None):
    registry = ModelRegistry(list(profiles))
    adapters = {p.model_id: RecordingAdapter(p, delay=0.005) for p in profiles}
    resources = ResourceManager(ram_limit_mb=ram if provider is None else None, min_free_ram_mb=min_free, available_ram_provider=provider)
    return AdaptiveScheduler(registry, adapters, resources, poll_interval_ms=10), adapters, resources


def task(request_id, capability=Capability.CODING, deadline=2, priority=50):
    return Task(request_id, capability.value, capability, request_id, priority=priority, deadline=time.time() + deadline)


def test_admission_preserves_safety_margin_and_reservations_are_atomic():
    profile = ModelProfile("m", "M", Capability.CODING, integration_status="READY", ram_requirement_mb=100, cpu_requirement=1)
    manager = ResourceManager(ram_limit_mb=400, min_free_ram_mb=250)
    t = task("one")
    reservation = manager.reserve(t, profile)
    assert reservation.task_id == t.task_id
    assert manager.snapshot()["ram_reserved_mb"] == 100
    with pytest.raises(ResourceUnavailableError):
        manager.reserve(task("two"), profile)
    manager.release(t.task_id)
    assert manager.snapshot()["reservation_count"] == 0


def test_model_switching_unloads_before_next_heavy_model():
    a = ModelProfile("a", "A", Capability.CHAT, integration_status="READY", ram_requirement_mb=100, cpu_requirement=2)
    b = ModelProfile("b", "B", Capability.CODING, integration_status="READY", ram_requirement_mb=100, cpu_requirement=2)
    scheduler, adapters, resources = setup(a, b)
    async def run():
        first, second = await asyncio.gather(scheduler.run(task("a", Capability.CHAT)), scheduler.run(task("b", Capability.CODING)))
        return first, second
    asyncio.run(run())
    assert adapters["a"].unloads == 1 and adapters["b"].unloads == 1
    assert resources.snapshot()["reservation_count"] == 0


def test_timeout_and_runtime_failure_release_all_resources():
    profile = ModelProfile("m", "M", Capability.CODING, integration_status="READY", ram_requirement_mb=100, cpu_requirement=1)
    scheduler, adapters, resources = setup(profile)
    with pytest.raises(TimeoutError):
        asyncio.run(scheduler.run(task("timeout", deadline=0.001)))
    assert resources.snapshot()["reservation_count"] == 0
    adapters["m"].failure = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        asyncio.run(scheduler.run(task("failure")))
    assert resources.snapshot()["reservation_count"] == 0


def test_running_cancellation_releases_reservation_and_unloads():
    profile = ModelProfile("m", "M", Capability.CODING, integration_status="READY", ram_requirement_mb=100, cpu_requirement=1)
    scheduler, adapters, resources = setup(profile)
    async def run():
        pending = asyncio.create_task(scheduler.run(task("cancel")))
        await asyncio.sleep(0.02)
        scheduler.cancel("cancel")
        with pytest.raises(asyncio.CancelledError):
            await pending
    asyncio.run(run())
    assert resources.snapshot()["reservation_count"] == 0
    assert adapters["m"].unloads == 1


def test_kokoro_profile_is_rejected_before_loading_when_budget_is_insufficient():
    profile = ModelProfile("kokoro", "Kokoro", Capability.TTS, integration_status="READY", ram_requirement_mb=1400, cpu_requirement=2)
    scheduler, adapters, resources = setup(profile, ram=2000, min_free=768)
    with pytest.raises(ResourceUnavailableError):
        asyncio.run(scheduler.run(task("tts", Capability.TTS, deadline=1)))
    assert adapters["kokoro"].loads == 0
    assert resources.snapshot()["reservation_count"] == 0


def test_resident_infrastructure_cost_is_included_in_admission():
    profile = ModelProfile("m", "M", Capability.CODING, integration_status="READY", ram_requirement_mb=100, cpu_requirement=1)
    manager = ResourceManager(ram_limit_mb=900, min_free_ram_mb=250, resident_ram_mb=600, resident_components=("e5",))
    with pytest.raises(ResourceUnavailableError):
        manager.reserve(task("e5-blocked"), profile)
    assert manager.snapshot()["resident_components"] == ["e5"]


def test_higher_priority_waiter_runs_before_lower_priority_waiter():
    profile = ModelProfile("m", "M", Capability.CODING, integration_status="READY", ram_requirement_mb=100, cpu_requirement=1)
    scheduler, adapters, resources = setup(profile, ram=5000)
    order = []
    original = adapters["m"].execute
    async def record(input_data, cancellation):
        order.append(input_data)
        return await original(input_data, cancellation)
    adapters["m"].execute = record

    async def run():
        blocker = asyncio.create_task(scheduler.run(task("blocker", priority=10)))
        await asyncio.sleep(0.01)
        low = asyncio.create_task(scheduler.run(task("low", priority=10)))
        high = asyncio.create_task(scheduler.run(task("high", priority=90)))
        await asyncio.gather(blocker, low, high)
    asyncio.run(run())
    assert order[0] == "blocker"
    assert order.index("high") < order.index("low")


def test_preflight_reports_required_memory_and_deficit_without_loading():
    profile = ModelProfile("whisper", "Whisper", Capability.STT, integration_status="READY", ram_requirement_mb=400, measured_ram_mb=340, cpu_requirement=2)
    registry = ModelRegistry([profile])
    adapter = RecordingAdapter(profile)
    resources = ResourceManager(ram_limit_mb=1000, min_free_ram_mb=768)
    scheduler = AdaptiveScheduler(registry, {"whisper": adapter}, resources)
    result = resources.preflight(profile)
    assert result["required_free_mb"] == 1168
    assert result["deficit_mb"] == 168
    assert result["admission"] == "RESOURCE_UNAVAILABLE"
    assert adapter.loads == 0
    assert scheduler.status()["resources"]["reservation_count"] == 0


def test_preflight_reports_ready_without_loading_when_budget_is_sufficient():
    profile = ModelProfile("coder", "Coder", Capability.CODING, integration_status="READY", ram_requirement_mb=545, cpu_requirement=2)
    adapter = RecordingAdapter(profile)
    resources = ResourceManager(ram_limit_mb=2000, min_free_ram_mb=768)
    result = resources.preflight(profile)
    assert result["admission"] == "READY_FOR_LOAD"
    assert result["deficit_mb"] == 0
    assert adapter.loads == 0


def test_preflight_includes_reservations_and_resident_infrastructure():
    profile = ModelProfile("qwen", "Qwen", Capability.CHAT, integration_status="READY", ram_requirement_mb=862, cpu_requirement=2)
    held_task = globals()["task"]("held")
    resources = ResourceManager(ram_limit_mb=3000, min_free_ram_mb=768, resident_ram_mb=300, resident_components=("e5",))
    resources.reserve(held_task, profile)
    result = resources.preflight(profile)
    assert result["reserved_ram_mb"] == 862
    assert result["resident_ram_mb"] == 300
    assert result["required_free_mb"] == 2792
    assert result["admission"] == "RESOURCE_UNAVAILABLE"
    resources.release(held_task.task_id)
