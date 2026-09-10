from orchestrator.adapters import ChatModelAdapter, CodingModelAdapter, ImageModelAdapter, KokoroModelAdapter, SpecializedUnavailableAdapter, WhisperModelAdapter
from orchestrator.models import Capability
from orchestrator.orchestrator import Orchestrator
from orchestrator.registry import CapabilityRegistry, ModelRegistry, default_profiles
from config import NodeConfig


EXPECTED = {
    "embedding-e5-small",
    "chat-qwen3-0.6b", "coding-qwen2.5-coder-0.5b", "image-mobilediffusion",
    "video-wan2.1-t2v-1.3b", "tts-kokoro-82m", "stt-whisper-base",
    "document-create-smollm2-360m", "document-analysis-smoldocling-256m",
}


def test_exact_selected_models_are_registered_without_claiming_installation(monkeypatch):
    for name in (
        "SWICO_FREE_E5_MODEL_PATH", "SWICO_CHAT_MODEL_PATH", "SWICO_CODING_MODEL_PATH",
        "SWICO_IMAGE_MODEL_PATH", "SWICO_VIDEO_MODEL_PATH", "SWICO_STT_MODEL_PATH",
        "SWICO_TTS_MODEL_PATH", "SWICO_DOCUMENT_CREATE_MODEL_PATH", "SWICO_DOCUMENT_ANALYSIS_MODEL_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    registry = ModelRegistry(default_profiles())
    profiles = {profile.model_id: profile for profile in registry.all()}
    assert EXPECTED == set(profiles)
    assert all(profile.integration_status == "NOT_INSTALLED" for profile in profiles.values())
    assert profiles["image-mobilediffusion"].runtime == "mobile_diffusion"


def test_config_loads_empty_optional_paths_as_none(monkeypatch):
    monkeypatch.setenv("SWICO_FREE_NODE_TOKEN", "t" * 32)
    monkeypatch.setenv("SWICO_FREE_MOCK_MODE", "true")
    for name in (
        "SWICO_CHAT_MODEL_PATH", "SWICO_CODING_MODEL_PATH", "SWICO_IMAGE_MODEL_PATH",
        "SWICO_VIDEO_MODEL_PATH", "SWICO_STT_MODEL_PATH", "SWICO_TTS_MODEL_PATH",
        "SWICO_DOCUMENT_CREATE_MODEL_PATH", "SWICO_DOCUMENT_ANALYSIS_MODEL_PATH",
    ):
        monkeypatch.setenv(name, "")
    config = NodeConfig.from_environment()
    assert all(getattr(config, field) is None for field in (
        "chat_model_path", "coding_model_path", "image_model_path", "video_model_path",
        "stt_model_path", "tts_model_path", "document_create_model_path", "document_analysis_model_path",
    ))


def test_existing_and_missing_optional_artifacts_have_correct_installation_state(tmp_path):
    coding = tmp_path / "coder.gguf"
    coding.write_bytes(b"test")
    stt = tmp_path / "whisper"
    stt.mkdir()
    missing = tmp_path / "missing"
    profiles = {p.model_id: p for p in default_profiles({
        "coding-qwen2.5-coder-0.5b": coding,
        "stt-whisper-base": stt,
        "image-mobilediffusion": missing,
    })}
    assert profiles["coding-qwen2.5-coder-0.5b"].integration_status == "INSTALLED"
    assert profiles["coding-qwen2.5-coder-0.5b"].model_path == str(coding)
    assert profiles["stt-whisper-base"].integration_status == "INSTALLED"
    assert profiles["stt-whisper-base"].model_path == str(stt)
    assert profiles["image-mobilediffusion"].integration_status == "NOT_INSTALLED"
    assert profiles["image-mobilediffusion"].model_path == str(missing)


def test_qwen_and_e5_paths_remain_discoverable(tmp_path):
    qwen = tmp_path / "qwen.gguf"
    qwen.write_bytes(b"test")
    e5 = tmp_path / "e5"
    e5.mkdir()
    profiles = {p.model_id: p for p in default_profiles({
        "chat-qwen3-0.6b": qwen,
        "embedding-e5-small": e5,
    })}
    assert profiles["chat-qwen3-0.6b"].integration_status == "INSTALLED"
    assert profiles["embedding-e5-small"].integration_status == "INSTALLED"


def test_public_capability_groups_remain_exactly_eight():
    registry = ModelRegistry(default_profiles())
    assert len([cap for cap in Capability if cap.value != "retrieval"]) == 8
    assert len(CapabilityRegistry(registry).as_dict()) == 8


def test_models_payload_and_health_expose_installation_and_lifecycle_state(tmp_path):
    coding = tmp_path / "coder.gguf"
    coding.write_bytes(b"test")
    stt = tmp_path / "whisper"
    stt.mkdir()
    node = Orchestrator.with_defaults(model_paths={"coding-qwen2.5-coder-0.5b": coding, "stt-whisper-base": stt})
    models = {model["model_id"]: model for model in node.registry.as_dicts()}
    health = node.registry.health(node.adapters, node.scheduler.status())
    assert models["coding-qwen2.5-coder-0.5b"]["installed"] is True
    assert models["coding-qwen2.5-coder-0.5b"]["model_path"] == str(coding)
    assert health["coding-qwen2.5-coder-0.5b"]["installed"] is True
    assert health["coding-qwen2.5-coder-0.5b"]["state"] == "INSTALLED"
    assert models["image-mobilediffusion"]["installed"] is False


def test_models_endpoints_preserve_auth_and_report_configured_artifacts(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_module

    coding = tmp_path / "coder.gguf"
    coding.write_bytes(b"test")
    stt = tmp_path / "whisper"
    stt.mkdir()
    token = "t" * 32
    monkeypatch.setenv("SWICO_FREE_NODE_TOKEN", token)
    monkeypatch.setenv("SWICO_FREE_MOCK_MODE", "true")
    monkeypatch.setenv("SWICO_CODING_MODEL_PATH", str(coding))
    monkeypatch.setenv("SWICO_STT_MODEL_PATH", str(stt))
    with TestClient(app_module.app) as client:
        assert client.get("/v1/models").status_code == 401
        response = client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
        health = client.get("/v1/models/health", headers={"Authorization": f"Bearer {token}"})
        capabilities = client.get("/v1/capabilities", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    models = {model["model_id"]: model for model in response.json()["models"]}
    assert models["coding-qwen2.5-coder-0.5b"]["installed"] is True
    assert models["stt-whisper-base"]["installed"] is True
    assert health.json()["models"]["coding-qwen2.5-coder-0.5b"]["state"] == "READY"
    assert health.json()["models"]["coding-qwen2.5-coder-0.5b"]["integration_status"] == "MOCK"
    assert len(capabilities.json()["capabilities"]) == 8


def test_mock_mode_marks_models_explicitly_and_real_mode_is_safe():
    mock = Orchestrator.with_defaults(mock_mode=True)
    assert all(profile.integration_status == "MOCK" for profile in mock.registry.all())
    real = Orchestrator.with_defaults()
    assert isinstance(real.adapters["image-mobilediffusion"], ImageModelAdapter)
    assert isinstance(real.adapters["chat-qwen3-0.6b"], SpecializedUnavailableAdapter)


def test_validated_qwen_runtime_connects_only_chat_adapter():
    class Runtime:
        model_path = "D:/Swico/models/chat-qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf"

    real = Orchestrator.with_defaults(qwen_runtime=Runtime())
    assert isinstance(real.adapters["chat-qwen3-0.6b"], ChatModelAdapter)
    assert real.registry.get("chat-qwen3-0.6b").integration_status == "READY"
    assert isinstance(real.adapters["coding-qwen2.5-coder-0.5b"], SpecializedUnavailableAdapter)


def test_validated_coding_runtime_connects_only_coding_adapter():
    class Runtime:
        model_path = "D:/Swico/models/coding-qwen2.5-coder-0.5b/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"

    real = Orchestrator.with_defaults(coding_runtime=Runtime())
    assert isinstance(real.adapters["coding-qwen2.5-coder-0.5b"], CodingModelAdapter)
    assert real.registry.get("coding-qwen2.5-coder-0.5b").integration_status == "READY"
    assert isinstance(real.adapters["chat-qwen3-0.6b"], SpecializedUnavailableAdapter)


def test_validated_stt_runtime_connects_only_whisper_adapter():
    class Runtime:
        model_path = "D:/Swico/models/stt-whisper-base"

    real = Orchestrator.with_defaults(stt_runtime=Runtime())
    assert isinstance(real.adapters["stt-whisper-base"], WhisperModelAdapter)
    assert real.registry.get("stt-whisper-base").integration_status == "READY"
    assert isinstance(real.adapters["chat-qwen3-0.6b"], SpecializedUnavailableAdapter)


def test_kokoro_runtime_connects_only_tts_adapter_without_claiming_synthesis():
    class Runtime:
        model_path = "D:/Swico/models/tts-kokoro-82m"

    real = Orchestrator.with_defaults(tts_runtime=Runtime())
    assert isinstance(real.adapters["tts-kokoro-82m"], KokoroModelAdapter)
    assert real.registry.get("tts-kokoro-82m").integration_status == "READY"
    assert real.registry.get("tts-kokoro-82m").measured_ram_mb == 562
    assert isinstance(real.adapters["chat-qwen3-0.6b"], SpecializedUnavailableAdapter)
