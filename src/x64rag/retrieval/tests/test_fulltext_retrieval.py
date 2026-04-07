from types import SimpleNamespace
from unittest.mock import AsyncMock

from x64rag.retrieval.common.models import ContentMatch, RetrievedChunk
from x64rag.retrieval.modules.retrieval.methods.document import DocumentRetrieval
from x64rag.retrieval.modules.retrieval.search.service import RetrievalService


def _make_service(document_method=None):
    mock_vector = SimpleNamespace(
        name="vector",
        weight=1.0,
        top_k=None,
        search=AsyncMock(
            return_value=[
                RetrievedChunk(chunk_id="chunk-1", source_id="src-1", content="Some chunk content", score=0.8),
            ]
        ),
    )
    methods = [mock_vector]
    if document_method is not None:
        methods.append(document_method)
    return RetrievalService(
        retrieval_methods=methods,
        reranking=None,
        top_k=5,
    )


async def test_retrieve_with_document_store():
    mock_document = SimpleNamespace(
        name="document",
        weight=1.0,
        top_k=None,
        search=AsyncMock(
            return_value=[
                RetrievedChunk(
                    chunk_id="fulltext:src-2",
                    source_id="src-2",
                    content="The FBD-20254 filter specs...",
                    score=0.9,
                    source_type="manuals",
                    source_metadata={"title": "Manual X", "match_type": "exact"},
                ),
            ]
        ),
    )
    service = _make_service(document_method=mock_document)
    results = await service.retrieve(query="FBD-20254", knowledge_id="kb-1")
    assert len(results) == 2
    mock_document.search.assert_called_once()


async def test_retrieve_without_document_store():
    service = _make_service(document_method=None)
    results = await service.retrieve(query="test query", knowledge_id="kb-1")
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"


async def test_content_match_to_chunk_conversion():
    matches = [
        ContentMatch(
            source_id="src-1",
            title="Manual",
            excerpt="Excerpt text",
            score=0.85,
            match_type="fulltext",
            source_type="manuals",
        ),
    ]
    chunks = DocumentRetrieval._convert(matches)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "fulltext:src-1"
    assert chunks[0].content == "Excerpt text"
    assert chunks[0].source_metadata["match_type"] == "fulltext"
