from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} is outside supported bounds")
    return value


@dataclass(frozen=True)
class NodeConfig:
    token: str
    qwen_gguf_path: Path
    e5_model_path: Path
    host: str
    port: int
    qwen_threads: int
    qwen_batch_size: int
    qwen_context_size: int
    e5_threads: int
    max_concurrent_generations: int
    max_queue_size: int
    max_queue_wait_seconds: int
    max_total_request_seconds: int
    max_output_tokens: int
    max_concurrent_embeddings: int
    max_embedding_queue_size: int
    max_heavy_models_resident: int = 1
    min_free_ram_mb: int = 768
    resource_poll_interval_ms: int = 100
    model_idle_timeout_sec: int = 0
    mock_mode: bool = False
    chat_model_path: Path | None = None
    coding_model_path: Path | None = None
    image_model_path: Path | None = None
    video_model_path: Path | None = None
    stt_model_path: Path | None = None
    tts_model_path: Path | None = None
    document_create_model_path: Path | None = None
    document_analysis_model_path: Path | None = None

    def model_paths(self) -> dict[str, Path | None]:
        return {
            "embedding-e5-small": self.e5_model_path,
            "chat-qwen3-1.7b": self.chat_model_path,
            "coding-qwen2.5-coder-1.5b": self.coding_model_path,
            "image-small-sd-v0": self.image_model_path,
            "video-wan2.1-t2v-1.3b": self.video_model_path,
            "stt-whisper-base": self.stt_model_path,
            "tts-kokoro-82m": self.tts_model_path,
            "document-create-smollm2-360m": self.document_create_model_path,
            "document-analysis-smoldocling-256m": self.document_analysis_model_path,
        }

    @classmethod
    def from_environment(cls) -> "NodeConfig":
        token = os.getenv("SWICO_FREE_NODE_TOKEN", "").strip()
        if len(token) < 32 or any(char.isspace() for char in token):
            raise RuntimeError("SWICO_FREE_NODE_TOKEN must be a strong token")
        mock_mode = os.getenv("SWICO_FREE_MOCK_MODE", "0").strip().lower() in {"1", "true", "yes"}
        qwen = Path(os.getenv("SWICO_FREE_QWEN_GGUF_PATH", "").strip()).expanduser()
        e5 = Path(os.getenv("SWICO_FREE_E5_MODEL_PATH", "").strip()).expanduser()
        optional_paths = {
            "chat_model_path": os.getenv("SWICO_CHAT_MODEL_PATH", "").strip(),
            "coding_model_path": os.getenv("SWICO_CODING_MODEL_PATH", "").strip(),
            "image_model_path": os.getenv("SWICO_IMAGE_MODEL_PATH", "").strip(),
            "video_model_path": os.getenv("SWICO_VIDEO_MODEL_PATH", "").strip(),
            "stt_model_path": os.getenv("SWICO_STT_MODEL_PATH", "").strip(),
            "tts_model_path": os.getenv("SWICO_TTS_MODEL_PATH", "").strip(),
            "document_create_model_path": os.getenv("SWICO_DOCUMENT_CREATE_MODEL_PATH", "").strip(),
            "document_analysis_model_path": os.getenv("SWICO_DOCUMENT_ANALYSIS_MODEL_PATH", "").strip(),
        }
        if mock_mode:
            qwen, e5 = Path("mock.gguf"), Path("mock-e5")
        else:
            if qwen.suffix.lower() != ".gguf" or not qwen.is_file():
                raise RuntimeError("SWICO_FREE_QWEN_GGUF_PATH must point to an existing GGUF file")
            if not e5.is_dir():
                raise RuntimeError("SWICO_FREE_E5_MODEL_PATH must point to a model directory")
        host = os.getenv("SWICO_FREE_NODE_HOST", "127.0.0.1").strip() or "127.0.0.1"
        if host != "127.0.0.1":
            raise RuntimeError("SWICO_FREE_NODE_HOST must remain 127.0.0.1")
        return cls(
            token=token, qwen_gguf_path=qwen, e5_model_path=e5,
            host=host,
            port=_int("SWICO_FREE_NODE_PORT", 8765, 1, 65_535),
            qwen_threads=_int("SWICO_FREE_QWEN_THREADS", 4, 1, 4),
            qwen_batch_size=_int("SWICO_FREE_QWEN_BATCH_SIZE", 128, 16, 256),
            qwen_context_size=_int("SWICO_FREE_QWEN_CONTEXT_SIZE", 1024, 512, 4096),
            e5_threads=_int("SWICO_FREE_E5_THREADS", 2, 1, 2),
            max_concurrent_generations=_int("SWICO_FREE_MAX_CONCURRENT_GENERATIONS", 1, 1, 1),
            max_queue_size=_int("SWICO_FREE_MAX_QUEUE_SIZE", 3, 0, 10),
            max_queue_wait_seconds=_int("SWICO_FREE_MAX_QUEUE_WAIT_SECONDS", 15, 1, 60),
            max_total_request_seconds=_int("SWICO_FREE_MAX_TOTAL_REQUEST_SECONDS", 45, 10, 90),
            max_output_tokens=_int("SWICO_FREE_MAX_OUTPUT_TOKENS", 256, 64, 512),
            max_concurrent_embeddings=_int("SWICO_FREE_MAX_CONCURRENT_EMBEDDINGS", 1, 1, 1),
            max_embedding_queue_size=_int("SWICO_FREE_MAX_EMBEDDING_QUEUE_SIZE", 4, 0, 4),
            max_heavy_models_resident=_int("SWICO_MAX_HEAVY_MODELS_RESIDENT", 1, 1, 1),
            min_free_ram_mb=_int("SWICO_MIN_FREE_RAM_MB", 768, 256, 4096),
            resource_poll_interval_ms=_int("SWICO_RESOURCE_POLL_INTERVAL_MS", 100, 25, 1000),
            model_idle_timeout_sec=_int("SWICO_MODEL_IDLE_TIMEOUT_SEC", 0, 0, 3600),
            mock_mode=mock_mode,
            **{name: (Path(value).expanduser() if value else None) for name, value in optional_paths.items()},
        )
