# src/x64rag/retrieval/tests/test_vector_retrieval.py
from unittest.mock import AsyncMock

from x64rag.retrieval.common.models import SparseVector, VectorResult
from x64rag.retrieval.modules.retrieval.methods.vector import VectorRetrieval


async def test_dense_search():
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.9,
                payload={"content": "test", "source_id": "s1", "chunk_type": "child", "parent_id": None},
            ),
        ]
    )
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])

    method = VectorRetrieval(
        vector_store=vector_store,
        embeddings=embeddings,
        weight=1.0,
    )
    assert method.name == "vector"
    assert method.weight == 1.0

    results = await method.search(query="test", top_k=5)
    assert len(results) == 1
    vector_store.search.assert_called_once()


async def test_hybrid_search_with_sparse():
    vector_store = AsyncMock()
    vector_store.hybrid_search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.9,
                payload={"content": "test", "source_id": "s1", "chunk_type": "child", "parent_id": None},
            ),
        ]
    )
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])
    sparse = AsyncMock()
    sparse.embed_sparse_query = AsyncMock(return_value=SparseVector(indices=[1], values=[0.8]))

    method = VectorRetrieval(
        vector_store=vector_store,
        embeddings=embeddings,
        sparse_embeddings=sparse,
        weight=1.5,
    )
    results = await method.search(query="test", top_k=5)
    assert len(results) == 1
    vector_store.hybrid_search.assert_called_once()


async def test_bm25_enabled_fuses_results():
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.9,
                payload={"content": "matching content", "source_id": "s1", "chunk_type": "child", "parent_id": None},
            ),
        ]
    )
    vector_store.scroll = AsyncMock(
        return_value=(
            [
                VectorResult(
                    point_id="p1",
                    score=0.0,
                    payload={
                        "content": "matching content",
                        "source_id": "s1",
                        "chunk_type": "child",
                        "source_type": None,
                        "source_weight": 1.0,
                        "source_name": "",
                        "file_url": "",
                        "tags": [],
                        "page_number": None,
                        "section": None,
                    },
                ),
            ],
            None,
        )
    )
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])

    method = VectorRetrieval(
        vector_store=vector_store,
        embeddings=embeddings,
        bm25_enabled=True,
        bm25_max_indexes=16,
        weight=1.0,
    )
    results = await method.search(query="matching content", top_k=5)
    assert len(results) >= 1


async def test_error_returns_empty():
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(side_effect=RuntimeError("connection lost"))
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])

    method = VectorRetrieval(
        vector_store=vector_store,
        embeddings=embeddings,
        weight=1.0,
    )
    results = await method.search(query="test", top_k=5)
    assert results == []


async def test_name_and_weight_properties():
    method = VectorRetrieval(
        vector_store=AsyncMock(),
        embeddings=AsyncMock(),
        weight=2.5,
    )
    assert method.name == "vector"
    assert method.weight == 2.5


async def test_invalidate_cache():
    method = VectorRetrieval(
        vector_store=AsyncMock(),
        embeddings=AsyncMock(),
        bm25_enabled=True,
        weight=1.0,
    )
    await method.invalidate_cache(knowledge_id=None)
    await method.invalidate_cache(knowledge_id="kb-1")
