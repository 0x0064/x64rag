from unittest.mock import AsyncMock

from x64rag.retrieval.common.models import ContentMatch, RetrievedChunk
from x64rag.retrieval.modules.retrieval.search.service import RetrievalService
from x64rag.retrieval.modules.retrieval.search.vector import VectorSearch


def _make_service(document_store=None):
    vector_search = AsyncMock(spec=VectorSearch)
    vector_search.search = AsyncMock(
        return_value=[
            RetrievedChunk(chunk_id="chunk-1", source_id="src-1", content="Some chunk content", score=0.8),
        ]
    )
    return RetrievalService(
        vector_search=vector_search,
        keyword_search=None,
        reranking=None,
        top_k=5,
        document_store=document_store,
    )


async def test_retrieve_with_document_store():
    doc_store = AsyncMock()
    doc_store.search_content = AsyncMock(
        return_value=[
            ContentMatch(
                source_id="src-2",
                title="Manual X",
                excerpt="The FBD-20254 filter specs...",
                score=0.9,
                match_type="exact",
                source_type="manuals",
            ),
        ]
    )
    service = _make_service(document_store=doc_store)
    results = await service.retrieve(query="FBD-20254", knowledge_id="kb-1")
    assert len(results) == 2
    doc_store.search_content.assert_called_once()


async def test_retrieve_without_document_store():
    service = _make_service(document_store=None)
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
    chunks = RetrievalService._content_matches_to_chunks(matches)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "fulltext:src-1"
    assert chunks[0].content == "Excerpt text"
    assert chunks[0].source_metadata["match_type"] == "fulltext"
