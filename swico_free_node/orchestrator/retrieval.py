from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    document_id: str
    chunk_id: str
    source: str
    text: str
    chunk_index: int
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    similarity: float
    rank: int


class EmbeddingAdapter(Protocol):
    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


class E5EmbeddingAdapter:
    """Adapter over the one existing Swico E5Runtime instance."""
    dimensions = 384

    def __init__(self, runtime: Any):
        self.runtime = runtime

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "query")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "passage")

    def _embed(self, texts: list[str], mode: str) -> list[list[float]]:
        vectors = self.runtime.embed(texts, [mode] * len(texts))
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError("E5 adapter received an embedding with the wrong dimension")
        return vectors

    def health(self) -> str:
        return "ok" if self.runtime is not None else "unavailable"


class DeterministicChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 120, minimum_size: int = 40):
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size or minimum_size < 0:
            raise ValueError("invalid chunking configuration")
        self.chunk_size, self.overlap, self.minimum_size = chunk_size, overlap, minimum_size

    def chunk(self, document: Document) -> list[DocumentChunk]:
        text = document.text.strip()
        if not text:
            return []
        chunks: list[DocumentChunk] = []
        start, index = 0, 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
                if boundary > start + self.minimum_size:
                    end = boundary
            part = text[start:end].strip()
            if len(part) >= self.minimum_size or not chunks:
                chunks.append(DocumentChunk(document.document_id, f"{document.document_id}:{index}", document.source, part, index, document.metadata.get("page"), document.metadata.get("section"), dict(document.metadata)))
                index += 1
            if end >= len(text):
                break
            start = max(start + 1, end - self.overlap)
        return chunks


class LocalVectorIndex:
    dimensions = 384

    def __init__(self):
        self._items: dict[str, tuple[DocumentChunk, list[float]]] = {}

    def add(self, chunks: list[DocumentChunk], vectors: list[list[float]], replace_document: bool = True) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        if replace_document and chunks:
            self.delete(chunks[0].document_id)
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.dimensions:
                raise ValueError("vector dimension must be 384")
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:
                raise ValueError("zero-length vectors are not searchable")
            self._items[chunk.chunk_id] = (chunk, vector)

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        if len(vector) != self.dimensions:
            raise ValueError("query vector dimension must be 384")
        qnorm = math.sqrt(sum(value * value for value in vector))
        if qnorm == 0:
            return []
        scored = []
        for chunk, item in self._items.values():
            norm = math.sqrt(sum(value * value for value in item))
            scored.append((sum(a * b for a, b in zip(vector, item)) / (qnorm * norm), chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [SearchResult(chunk, score, rank) for rank, (score, chunk) in enumerate(scored[:max(0, top_k)], 1)]

    def delete(self, document_id: str) -> None:
        self._items = {key: value for key, value in self._items.items() if value[0].document_id != document_id}

    def clear(self) -> None:
        self._items.clear()

    def count(self) -> int:
        return len(self._items)

    def health(self) -> str:
        return "ok"


class Retriever:
    def __init__(self, embeddings: EmbeddingAdapter, index: LocalVectorIndex | None = None, chunker: DeterministicChunker | None = None):
        self.embeddings, self.index, self.chunker = embeddings, index or LocalVectorIndex(), chunker or DeterministicChunker()

    def index_document(self, document: Document) -> list[DocumentChunk]:
        chunks = self.chunker.chunk(document)
        if chunks:
            self.index.add(chunks, self.embeddings.embed_passages([chunk.text for chunk in chunks]))
        return chunks

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        vectors = self.embeddings.embed_queries([query])
        return self.index.search(vectors[0], top_k)

    def delete_document(self, document_id: str) -> None:
        self.index.delete(document_id)

    def clear(self) -> None:
        self.index.clear()

    def count(self) -> int:
        return self.index.count()

    def health(self) -> str:
        return self.index.health() if self.embeddings is not None else "unavailable"
