from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from x64rag.retrieval.modules.ingestion.chunk.service import IngestionService


def _mock_method(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        ingest=AsyncMock(),
        delete=AsyncMock(),
    )


def _make_service(methods=None, metadata_store=None):
    chunker = MagicMock()
    chunker.chunk = MagicMock(
        return_value=[
            MagicMock(
                content="chunk text",
                embedding_text="chunk text",
                page_number=1,
                section=None,
                chunk_index=0,
                context="",
                contextualized="",
                parent_id=None,
                chunk_type="child",
            ),
        ]
    )
    return IngestionService(
        chunker=chunker,
        ingestion_methods=methods or [],
        metadata_store=metadata_store,
    )


async def test_ingest_text_delegates_to_methods():
    vector = _mock_method("vector")
    document = _mock_method("document")
    service = _make_service(methods=[vector, document])

    source = await service.ingest_text(content="Hello world", metadata={"name": "test"})
    vector.ingest.assert_called_once()
    document.ingest.assert_called_once()
    assert source.chunk_count == 1


async def test_ingest_text_no_methods():
    service = _make_service(methods=[])
    source = await service.ingest_text(content="Hello world")
    assert source is not None
    assert source.chunk_count == 1


async def test_ingest_text_creates_metadata_source():
    meta_store = SimpleNamespace(
        create_source=AsyncMock(),
        list_sources=AsyncMock(return_value=[]),
    )
    service = _make_service(methods=[], metadata_store=meta_store)
    source = await service.ingest_text(content="Hello world")
    meta_store.create_source.assert_called_once()
    created = meta_store.create_source.call_args[0][0]
    assert created.source_id == source.source_id
    assert created.chunk_count == 1


async def test_ingest_text_fires_on_complete_callback():
    callback = AsyncMock()
    chunker = MagicMock()
    chunker.chunk = MagicMock(
        return_value=[
            MagicMock(
                content="chunk text",
                embedding_text="chunk text",
                page_number=1,
                section=None,
                chunk_index=0,
                context="",
                contextualized="",
                parent_id=None,
                chunk_type="child",
            ),
        ]
    )
    service = IngestionService(
        chunker=chunker,
        ingestion_methods=[],
        on_ingestion_complete=callback,
    )
    source = await service.ingest_text(content="Hello world", knowledge_id="k1")
    callback.assert_called_once_with("k1")
    assert source is not None


async def test_ingest_text_method_receives_correct_args():
    method = _mock_method("vector")
    service = _make_service(methods=[method])

    source = await service.ingest_text(
        content="Hello world",
        knowledge_id="k1",
        source_type="document",
        metadata={"name": "test-doc"},
    )

    call_kwargs = method.ingest.call_args[1]
    assert call_kwargs["source_id"] == source.source_id
    assert call_kwargs["knowledge_id"] == "k1"
    assert call_kwargs["source_type"] == "document"
    assert call_kwargs["title"] == "test-doc"
    assert call_kwargs["full_text"] == "Hello world"
    assert len(call_kwargs["chunks"]) == 1
    assert call_kwargs["tags"] == []
    assert call_kwargs["metadata"] == {"name": "test-doc"}


async def test_ingest_text_empty_chunks_raises():
    chunker = MagicMock()
    chunker.chunk = MagicMock(return_value=[])
    service = IngestionService(
        chunker=chunker,
        ingestion_methods=[],
    )
    import pytest

    from x64rag.retrieval.common.errors import EmptyDocumentError

    with pytest.raises(EmptyDocumentError):
        await service.ingest_text(content="Hello world")
