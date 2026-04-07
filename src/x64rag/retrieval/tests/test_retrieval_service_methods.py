from types import SimpleNamespace
from unittest.mock import AsyncMock

from x64rag.retrieval.common.models import RetrievedChunk
from x64rag.retrieval.modules.retrieval.search.service import RetrievalService


def _mock_method(name: str, results: list[RetrievedChunk]) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        weight=1.0,
        search=AsyncMock(return_value=results),
    )


async def test_dispatch_single_method():
    vector = _mock_method(
        "vector",
        [
            RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.9),
        ],
    )
    service = RetrievalService(retrieval_methods=[vector], top_k=5)
    results = await service.retrieve(query="test")
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    vector.search.assert_called_once()


async def test_dispatch_multiple_methods_fused():
    vector = _mock_method(
        "vector",
        [
            RetrievedChunk(chunk_id="c1", source_id="s1", content="vector", score=0.9),
        ],
    )
    document = _mock_method(
        "document",
        [
            RetrievedChunk(chunk_id="c2", source_id="s2", content="doc", score=0.8),
        ],
    )
    service = RetrievalService(retrieval_methods=[vector, document], top_k=5)
    results = await service.retrieve(query="test")
    assert len(results) == 2
    ids = {r.chunk_id for r in results}
    assert "c1" in ids
    assert "c2" in ids


async def test_empty_method_list():
    service = RetrievalService(retrieval_methods=[], top_k=5)
    results = await service.retrieve(query="test")
    assert results == []


async def test_failed_method_returns_empty_others_succeed():
    vector = _mock_method(
        "vector",
        [
            RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.9),
        ],
    )
    graph = _mock_method("graph", [])
    service = RetrievalService(retrieval_methods=[vector, graph], top_k=5)
    results = await service.retrieve(query="test")
    assert len(results) == 1


async def test_tree_chunks_injected():
    vector = _mock_method(
        "vector",
        [
            RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.9),
        ],
    )
    tree_chunk = RetrievedChunk(chunk_id="tree-1", source_id="s2", content="tree", score=0.7)
    service = RetrievalService(retrieval_methods=[vector], top_k=5)
    results = await service.retrieve(query="test", tree_chunks=[tree_chunk])
    ids = {r.chunk_id for r in results}
    assert "tree-1" in ids


async def test_reranker_applied():
    vector = _mock_method(
        "vector",
        [
            RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.5),
            RetrievedChunk(chunk_id="c2", source_id="s2", content="text2", score=0.3),
        ],
    )
    reranker = AsyncMock()
    reranker.rerank = AsyncMock(
        return_value=[
            RetrievedChunk(chunk_id="c2", source_id="s2", content="text2", score=0.95),
        ]
    )
    service = RetrievalService(retrieval_methods=[vector], reranking=reranker, top_k=1)
    results = await service.retrieve(query="test")
    assert len(results) == 1
    assert results[0].chunk_id == "c2"
