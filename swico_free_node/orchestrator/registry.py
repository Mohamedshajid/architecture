from __future__ import annotations

from dataclasses import asdict
import os
from threading import RLock

from .models import Capability, ModelProfile


class ModelRegistry:
    def __init__(self, profiles: list[ModelProfile] | None = None):
        self._models: dict[str, ModelProfile] = {}
        self._lock = RLock()
        for profile in profiles or []:
            self.register(profile)

    def register(self, profile: ModelProfile) -> None:
        with self._lock:
            self._models[profile.model_id] = profile

    def get(self, model_id: str) -> ModelProfile | None:
        return self._models.get(model_id)

    def for_capability(self, capability: Capability) -> list[ModelProfile]:
        return [m for m in self._models.values() if m.capability == capability and m.enabled]

    def all(self) -> list[ModelProfile]:
        return list(self._models.values())

    def health(self, adapters: dict | None = None, scheduler_status: dict | None = None) -> dict[str, dict]:
        report = {}
        for model in self._models.values():
            adapter = (adapters or {}).get(model.model_id)
            status = adapter.status() if adapter is not None and hasattr(adapter, "status") else {}
            report[model.model_id] = {"model_id": model.model_id, "capability": model.capability.value, "runtime": model.runtime, "model_identifier": model.model_identifier, "model_path": model.model_path, "installed": bool(model.model_path) and model.integration_status != "NOT_INSTALLED", "health": model.health_status, "integration_status": model.integration_status, "state": status.get("state", model.health_status), "loaded": bool(status.get("loaded", False)), "resource_profile": {"estimated_ram_mb": model.ram_requirement_mb, "measured_ram_mb": model.measured_ram_mb, "cpu": model.cpu_requirement, "heavyweight": model.heavyweight, "long_running": model.long_running, "unloadable": model.unloadable}, "resource_status": "unknown" if model.ram_requirement_mb == 0 else "configured", "scheduler": scheduler_status or {}}
        return report

    def as_dicts(self) -> list[dict]:
        return [{**asdict(m), "installed": bool(m.model_path) and m.integration_status != "NOT_INSTALLED"} for m in self.all()]


class CapabilityRegistry:
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry

    def models_for(self, capability: Capability) -> list[str]:
        return [m.model_id for m in self.model_registry.for_capability(capability)]

    def as_dict(self) -> dict[str, list[str]]:
        return {cap.value: [{"model_id": model.model_id, "model_identifier": model.model_identifier, "status": model.integration_status, "model_path": model.model_path} for model in self.model_registry.for_capability(cap)] for cap in Capability if cap != Capability.RETRIEVAL}


def default_profiles(model_paths: dict[str, object] | None = None) -> list[ModelProfile]:
    specs = [
        ("embedding-e5-small", "multilingual-e5-small", Capability.RETRIEVAL, "intfloat/multilingual-e5-small", "SWICO_FREE_E5_MODEL_PATH", "transformers", ("text",), ("embedding",), False),
        ("chat-qwen3-1.7b", "Qwen3-1.7B", Capability.CHAT, "Qwen/Qwen3-1.7B", "SWICO_CHAT_MODEL_PATH", "qwen", ("text",), ("text",), True),
        ("coding-qwen2.5-coder-1.5b", "Qwen2.5-Coder-1.5B-Instruct", Capability.CODING, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "SWICO_CODING_MODEL_PATH", "qwen", ("text",), ("text",), True),
        ("image-small-sd-v0", "Small Stable Diffusion v0", Capability.IMAGE_GENERATION, "OFA-Sys/small-stable-diffusion-v0", "SWICO_IMAGE_MODEL_PATH", "diffusers", ("text",), ("image",), True),
        ("stt-whisper-base", "Whisper-base", Capability.STT, "openai/whisper-base", "SWICO_STT_MODEL_PATH", "whisper", ("audio",), ("text",), False),
        ("tts-kokoro-82m", "Kokoro-82M", Capability.TTS, "hexgrad/Kokoro-82M", "SWICO_TTS_MODEL_PATH", "kokoro", ("text",), ("audio",), False),
        ("video-wan2.1-t2v-1.3b", "Wan2.1 T2V-1.3B", Capability.VIDEO_GENERATION, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "SWICO_VIDEO_MODEL_PATH", "diffusers", ("text",), ("video",), False),
        ("document-create-smollm2-360m", "SmolLM2-360M-Instruct", Capability.DOCUMENT_CREATION, "HuggingFaceTB/SmolLM2-360M-Instruct", "SWICO_DOCUMENT_CREATE_MODEL_PATH", "transformers", ("text",), ("document_spec",), False),
        ("document-analysis-smoldocling-256m", "SmolDocling-256M", Capability.DOCUMENT_ANALYSIS, "ds4sd/SmolDocling-256M-preview", "SWICO_DOCUMENT_ANALYSIS_MODEL_PATH", "smoldocling", ("document", "text"), ("analysis",), False),
    ]
    profiles = []
    resource_defaults = {
        "embedding-e5-small": (700, 668, 2, True, False),
        "chat-qwen3-1.7b": (862, 862, 2, True, False),
        "coding-qwen2.5-coder-1.5b": (1120, 1120, 2, True, False),
        "stt-whisper-base": (400, 340, 2, True, False),
        # 1,400 MB is an admission estimate, deliberately above the observed
        # process RSS because Kokoro reduced system available RAM to ~78.8 MB.
        "tts-kokoro-82m": (1400, 562, 2, True, False),
        "image-small-sd-v0": (4096, 3580, 4, True, False),
        "video-wan2.1-t2v-1.3b": (2048, None, 4, True, True),
        "document-create-smollm2-360m": (512, None, 2, True, False),
        "document-analysis-smoldocling-256m": (512, None, 2, True, False),
    }
    expected_types = {
        "embedding-e5-small": "dir",
        "chat-qwen3-1.7b": "file",
        "coding-qwen2.5-coder-1.5b": "file",
        "stt-whisper-base": "dir",
        "tts-kokoro-82m": "dir",
    }
    for model_id, name, capability, identifier, env_name, runtime, inputs, outputs, streaming in specs:
        if model_paths is None:
            raw_path = os.getenv(env_name, "").strip()
            path = raw_path or None
        else:
            configured = model_paths.get(model_id)
            path = str(configured).strip() if configured else None
        expected = expected_types.get(model_id)
        installed = bool(path) and (
            os.path.isfile(path) if expected == "file" else
            os.path.isdir(path) if expected == "dir" else
            os.path.exists(path)
        )
        integration_status = "INSTALLED" if installed else "NOT_INSTALLED"
        estimate, measured, cpu, heavy, long_running = resource_defaults[model_id]
        profiles.append(ModelProfile(model_id=model_id, name=name, capability=capability, runtime=runtime, model_path=path, model_identifier=identifier, health_status=integration_status, integration_status=integration_status, ram_requirement_mb=estimate, measured_ram_mb=measured, cpu_requirement=cpu, expected_latency_ms=0, input_types=inputs, output_types=outputs, streaming_supported=streaming, progress_supported=capability == Capability.VIDEO_GENERATION, lifecycle_policy="on_demand", heavyweight=heavy, long_running=long_running))
    return profiles
