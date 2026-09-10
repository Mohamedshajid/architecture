from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Any

from .models import Capability, ModelLifecycleState, ModelProfile


class ModelAdapter(ABC):
    def __init__(self, profile: ModelProfile):
        self.profile = profile
        self.state = ModelLifecycleState.READY.value if profile.integration_status in ("READY", "MOCK") else ModelLifecycleState.INSTALLED.value
        self.last_load_time: float | None = None
        self.last_unload_time: float | None = None

    async def load(self) -> None:
        import time
        if self.state == ModelLifecycleState.LOADED.value:
            return
        self.state = ModelLifecycleState.LOADING.value
        try:
            self.state = ModelLifecycleState.LOADED.value
            self.last_load_time = time.time()
        except Exception:
            self.state = ModelLifecycleState.LOAD_FAILED.value
            raise

    async def unload(self) -> None:
        import time
        if self.state in (ModelLifecycleState.INSTALLED.value, ModelLifecycleState.READY.value, ModelLifecycleState.EVICTED.value):
            return
        self.state = ModelLifecycleState.UNLOADING.value
        self.state = ModelLifecycleState.EVICTED.value
        self.last_unload_time = time.time()

    async def health(self) -> str:
        return "ok"

    def estimate_resources(self, _input: Any) -> dict[str, int]:
        return {"ram_mb": self.profile.ram_requirement_mb, "cpu": self.profile.cpu_requirement}

    @abstractmethod
    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        raise NotImplementedError

    async def cancel(self) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {"model_id": self.profile.model_id, "capability": self.profile.capability.value, "runtime": self.profile.runtime, "state": self.state, "integration_status": self.profile.integration_status, "loaded": self.state in (ModelLifecycleState.LOADED.value, ModelLifecycleState.EXECUTING.value), "health": "ok" if self.state != ModelLifecycleState.UNHEALTHY.value else "unhealthy", "last_load_time": self.last_load_time, "last_unload_time": self.last_unload_time, "cancellation_supported": self.profile.cancellation_supported}


class MockAdapter(ModelAdapter):
    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        await asyncio.sleep(0)
        if cancellation.is_set():
            raise asyncio.CancelledError()
        text = input_data if isinstance(input_data, str) else str(input_data)
        return {"text": f"[{self.profile.capability.value} adapter] {text}", "mock": True}


class QwenTextAdapter(ModelAdapter):
    """Adapter that keeps orchestration above the existing protected Qwen runtime."""
    def __init__(self, profile: ModelProfile, runtime: Any, max_tokens: int = 256):
        super().__init__(profile)
        self.runtime = runtime
        self.max_tokens = max_tokens

    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        messages = input_data if isinstance(input_data, list) else [{"role": "user", "content": str(input_data)}]
        text, usage = await asyncio.to_thread(self.runtime.generate, messages, self.max_tokens, cancellation)
        return {"text": text, "usage": usage, "runtime": "qwen"}


class ChatModelAdapter(QwenTextAdapter):
    """Chat-specific contract over the shared text runtime."""
    async def stream(self, input_data: Any, cancellation: threading.Event):
        result = await self.execute(input_data, cancellation)
        yield result.get("text", ""), result.get("usage", {})


class CodingModelAdapter(QwenTextAdapter):
    """Coding-specific contract over a separately supplied Qwen runtime."""
    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        result = await super().execute(input_data, cancellation)
        result["runtime"] = "qwen-coding"
        return result


class WhisperModelAdapter(ModelAdapter):
    """Speech-to-text adapter over a separately supplied local Whisper runtime."""
    def __init__(self, profile: ModelProfile, runtime: Any):
        super().__init__(profile)
        self.runtime = runtime

    async def load(self) -> None:
        await super().load()
        if hasattr(self.runtime, "load"):
            try:
                await asyncio.to_thread(self.runtime.load)
            except Exception:
                self.state = "LOAD_FAILED"
                raise

    async def unload(self) -> None:
        if hasattr(self.runtime, "unload"):
            await asyncio.to_thread(self.runtime.unload)
        await super().unload()

    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        if isinstance(input_data, dict):
            audio_path = input_data.get("audio_path")
            language = input_data.get("language")
        else:
            audio_path, language = input_data, None
        if not audio_path:
            raise ValueError("STT input requires audio_path")
        return await asyncio.to_thread(
            self.runtime.transcribe, __import__("pathlib").Path(str(audio_path)), cancellation, language,
        )


class MockChatAdapter(MockAdapter):
    async def stream(self, input_data: Any, cancellation: threading.Event):
        result = await self.execute(input_data, cancellation)
        yield result.get("text", ""), {"mock": True}


class ModelUnavailableError(RuntimeError):
    pass


class SpecializedUnavailableAdapter(ModelAdapter):
    """Contract adapter for a selected model whose runtime/weights are not installed."""
    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        raise ModelUnavailableError(f"{self.profile.model_id} is not installed; configure its runtime and model path")


class ImageModelAdapter(SpecializedUnavailableAdapter):
    pass


class VideoModelAdapter(SpecializedUnavailableAdapter):
    async def progress(self, _task_id: str) -> dict[str, Any]:
        return {"status": "unavailable", "progress": 0}


class STTModelAdapter(SpecializedUnavailableAdapter):
    pass


class TTSModelAdapter(SpecializedUnavailableAdapter):
    pass


class KokoroModelAdapter(ModelAdapter):
    """Adapter boundary for a supplied Kokoro runtime; no implicit fallback."""
    def __init__(self, profile: ModelProfile, runtime: Any):
        super().__init__(profile)
        self.runtime = runtime

    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        request = input_data if isinstance(input_data, dict) else {"text": str(input_data)}
        text = str(request.get("text", "")).strip()
        if not text:
            raise ValueError("TTS input requires text")
        return await asyncio.to_thread(self.runtime.synthesize, text, request.get("voice", "af_heart"), request.get("speed", 1.0), cancellation)


class E5ModelAdapter(ModelAdapter):
    """Health/lifecycle boundary for the one E5 runtime used by retrieval."""
    def __init__(self, profile: ModelProfile, runtime: Any):
        super().__init__(profile)
        self.runtime = runtime
        if getattr(runtime, "loaded", False):
            self.state = "LOADED"

    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        if cancellation.is_set():
            raise asyncio.CancelledError()
        texts = input_data.get("texts", []) if isinstance(input_data, dict) else list(input_data)
        modes = input_data.get("modes", ["query"] * len(texts)) if isinstance(input_data, dict) else ["query"] * len(texts)
        return await asyncio.to_thread(self.runtime.embed, texts, modes)


class DocumentAnalysisModelAdapter(SpecializedUnavailableAdapter):
    pass


class DocumentCreationModelAdapter(SpecializedUnavailableAdapter):
    async def render(self, _spec: dict[str, Any], _format: str) -> Any:
        raise ModelUnavailableError("deterministic document renderer is not configured")
