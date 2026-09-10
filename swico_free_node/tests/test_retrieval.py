from orchestrator.models import Evidence
from orchestrator.retrieval import DeterministicChunker, Document, LocalVectorIndex, Retriever
from orchestrator.trirag import TriRAG


class FakeE5:
    def embed_passages(self, texts):
        return [[1.0 if "revenue" in text.lower() else 0.1] + [0.01] * 383 for text in texts]

    def embed_queries(self, texts):
        return [[1.0] + [0.01] * 383 for _ in texts]


def test_chunking_and_provenance_are_deterministic():
    doc = Document("d1", "alpha " * 50, "report.pdf", {"page": 2})
    chunker = DeterministicChunker(chunk_size=50, overlap=10, minimum_size=5)
    first, second = chunker.chunk(doc), chunker.chunk(doc)
    assert first == second and first[0].document_id == "d1" and first[0].chunk_id == "d1:0"
    assert first[0].source == "report.pdf" and first[0].page == 2


def test_index_retrieve_dimension_and_duplicate_policy():
    retriever = Retriever(FakeE5(), LocalVectorIndex(), DeterministicChunker(chunk_size=200, overlap=20, minimum_size=1))
    retriever.index_document(Document("d1", "Revenue was 100 million.", "a.txt"))
    retriever.index_document(Document("d2", "Other unrelated content.", "b.txt"))
    assert retriever.count() == 2
    assert retriever.search("revenue", 1)[0].chunk.document_id == "d1"
    retriever.index_document(Document("d1", "Revenue was 120 million.", "a.txt"))
    assert retriever.count() == 2


def test_trirag_support_contradiction_and_uncertainty():
    rag = TriRAG()
    supported = rag.verify_claims(["Revenue was 100 million"], [Evidence("a", "Revenue was 100 million", .9)])
    contradicted = rag.verify_claims(["Revenue was 200 million"], [Evidence("a", "Revenue was 100 million", .9)])
    uncertain = rag.verify_claims(["The weather is sunny"], [])
    assert supported["verification_status"] == "verified"
    assert contradicted["contradictions"]
    assert uncertain["verification_status"] == "uncertain"


def test_trirag_validates_deduplicates_and_preserves_provenance():
    rag = TriRAG()
    evidence = [
        Evidence("source-a", "Refunds are allowed within 30 days.", .4, {"page": 2}, "doc-1", "doc-1:0", .4),
        Evidence("source-a", "Refunds are allowed within 30 days.", .9, {"page": 2}, "doc-1", "doc-1:0", .9),
        Evidence("", "invalid", .9),
    ]
    ranked = rag.retrieve("What is the refund policy?", evidence)
    assert len(ranked) == 1
    assert ranked[0].score == .9
    assert ranked[0].document_id == "doc-1"
    assert ranked[0].chunk_id == "doc-1:0"
