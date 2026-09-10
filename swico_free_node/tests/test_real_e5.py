import math
import os
from pathlib import Path

import pytest

from orchestrator.models import Capability, ModelProfile, Task
from orchestrator.retrieval import Document, E5EmbeddingAdapter, Retriever
from orchestrator.scheduler import ResourceManager
from orchestrator.trirag import TriRAG


pytestmark = pytest.mark.skipif(
    os.getenv("SWICO_RUN_REAL_E5_TESTS", "0") != "1",
    reason="set SWICO_RUN_REAL_E5_TESTS=1 to run local real-E5 integration tests",
)


@pytest.fixture(scope="module")
def real_e5():
    from e5_runtime import E5Runtime

    root = Path(os.getenv("SWICO_FREE_E5_MODEL_PATH", r"D:\Swico\models\multilingual-e5-small"))
    required = (root / "config.json", root / "model.safetensors", root / "tokenizer.json")
    if not root.is_dir() or not all(path.is_file() for path in required):
        pytest.skip("exact local multilingual-e5-small artifacts are not installed")
    profile = ModelProfile("embedding-e5-small", "multilingual-e5-small", Capability.RETRIEVAL, integration_status="READY", ram_requirement_mb=700, cpu_requirement=2, model_path=str(root), model_identifier="intfloat/multilingual-e5-small")
    resources = ResourceManager(min_free_ram_mb=768, max_heavy_models_resident=1)
    task = Task("real-e5-test", "retrieval", Capability.RETRIEVAL, {}, deadline=__import__("time").time() + 300)
    preflight = resources.preflight(profile)
    if preflight["admission"] != "READY_FOR_LOAD":
        pytest.skip(f"real E5 admission rejected: {preflight['rejection_reason']}")
    resources.reserve(task, profile)
    runtime = E5Runtime(root, threads=2)
    try:
        yield runtime, E5EmbeddingAdapter(runtime)
    finally:
        runtime.unload()
        resources.release(task.task_id)


def test_real_e5_artifacts_and_embedding_contract(real_e5):
    _runtime, adapter = real_e5
    vectors = adapter.embed_queries(["retrieval query", "second query"])
    vectors += adapter.embed_passages(["retrieval passage"])
    assert {len(vector) for vector in vectors} == {384}
    assert all(math.isfinite(value) for vector in vectors for value in vector)
    assert all(abs(math.sqrt(sum(value * value for value in vector)) - 1.0) < 1e-4 for vector in vectors)


def test_real_e5_retrieval_and_trirag_contract(real_e5):
    _runtime, adapter = real_e5
    retriever = Retriever(adapter)
    retriever.index_document(Document("doc-1", "The exact E5 model powers local retrieval.", "source-1", {"section": "models"}))
    retriever.index_document(Document("doc-2", "Unrelated gardening information.", "source-2"))
    results = retriever.search("exact E5 model retrieval", top_k=2)
    assert results and results[0].chunk.document_id == "doc-1"
    assert results[0].chunk.source == "source-1"
    evidence = TriRAG(retriever).retrieve_from_index("exact E5 model retrieval")
    assert evidence and evidence[0].document_id == "doc-1"
