# src/x64rag/retrieval/tests/test_document_retrieval.py
from unittest.mock import AsyncMock

from x64rag.retrieval.common.models import ContentMatch
from x64rag.retrieval.modules.retrieval.methods.document import DocumentRetrieval


async def test_search_converts_matches():
    store = AsyncMock()
    store.search_content = AsyncMock(
        return_value=[
            ContentMatch(
                source_id="src-1",
                title="Manual",
                excerpt="Excerpt text",
                score=0.85,
                match_type="fulltext",
                source_type="manuals",
            ),
        ]
    )
    method = DocumentRetrieval(document_store=store, weight=0.8)
    assert method.name == "document"
    assert method.weight == 0.8

    results = await method.search(query="test", top_k=10, knowledge_id="kb-1")
    assert len(results) == 1
    assert results[0].chunk_id == "fulltext:src-1"
    assert results[0].content == "Excerpt text"
    assert results[0].source_metadata["match_type"] == "fulltext"
    store.search_content.assert_called_once_with(query="test", knowledge_id="kb-1", top_k=10)


async def test_search_empty_results():
    store = AsyncMock()
    store.search_content = AsyncMock(return_value=[])

    method = DocumentRetrieval(document_store=store)
    results = await method.search(query="nothing", top_k=5)
    assert results == []


async def test_error_returns_empty():
    store = AsyncMock()
    store.search_content = AsyncMock(side_effect=RuntimeError("db down"))

    method = DocumentRetrieval(document_store=store)
    results = await method.search(query="test", top_k=5)
    assert results == []
