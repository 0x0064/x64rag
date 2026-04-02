from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class Source:
    source_id: str
    status: str = "completed"
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    chunk_count: int = 0
    embedding_model: str = ""
    file_hash: str | None = None
    stale: bool = False
    created_at: datetime | None = None
    knowledge_id: str | None = None
    source_type: str | None = None
    source_weight: float = 1.0


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    content: str
    page_number: int | None = None
    section: str | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]

    def __repr__(self) -> str:
        return f"SparseVector({len(self.indices)} non-zero entries)"


@dataclass
class VectorPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, Any]
    sparse_vector: SparseVector | None = None

    def __repr__(self) -> str:
        sparse = f", sparse={len(self.sparse_vector.indices)} entries" if self.sparse_vector else ""
        return f"VectorPoint(point_id={self.point_id!r}, vector=[{len(self.vector)} dims]{sparse}, payload=...)"


@dataclass
class VectorResult:
    point_id: str
    score: float
    payload: dict[str, Any]

    def __repr__(self) -> str:
        return f"VectorResult(point_id={self.point_id!r}, score={self.score:.4f}, payload=...)"


@dataclass
class RetrievedChunk:
    chunk_id: str
    source_id: str
    content: str
    score: float
    page_number: int | None = None
    section: str | None = None
    source_type: str | None = None
    source_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentMatch:
    source_id: str
    title: str
    excerpt: str
    score: float
    match_type: Literal["fulltext", "exact"]
    source_type: str | None = None

    def __repr__(self) -> str:
        return f"ContentMatch(source_id={self.source_id!r}, score={self.score:.4f}, match_type={self.match_type!r})"


@dataclass
class SourceStats:
    source_id: str
    total_chunks: int = 0
    total_pages: int = 0
    avg_chunk_size: int = 0
    processing_time: float = 0.0
    total_hits: int = 0
    grounded_hits: int = 0
    ungrounded_hits: int = 0
