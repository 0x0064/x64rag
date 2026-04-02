from unittest.mock import AsyncMock, MagicMock

from x64rag.retrieval.modules.ingestion.chunk.service import IngestionService


def _make_service(document_store=None):
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1] * 128])
    embeddings.model = "test-model"

    chunker = MagicMock()
    chunker.chunk = MagicMock(
        return_value=[
            MagicMock(content="chunk 1", page_number=1, section=None, chunk_index=0),
        ]
    )

    vector_store = AsyncMock()
    vector_store.initialize = AsyncMock()
    vector_store.upsert = AsyncMock()

    metadata_store = AsyncMock()
    metadata_store.list_sources = AsyncMock(return_value=[])
    metadata_store.create_source = AsyncMock()

    return IngestionService(
        embeddings=embeddings,
        chunker=chunker,
        vector_store=vector_store,
        embedding_model_name="test:model",
        metadata_store=metadata_store,
        document_store=document_store,
    )


async def test_ingest_calls_document_store(tmp_path):
    doc_store = AsyncMock()
    doc_store.store_content = AsyncMock()
    service = _make_service(document_store=doc_store)

    test_file = tmp_path / "test.txt"
    test_file.write_text("Page one content.\n\nPage two content.")

    await service.ingest(file_path=test_file, knowledge_id="kb-1", source_type="manuals")

    doc_store.store_content.assert_called_once()
    kwargs = doc_store.store_content.call_args[1]
    assert kwargs["knowledge_id"] == "kb-1"
    assert kwargs["source_type"] == "manuals"
    assert "Page one content" in kwargs["content"]


async def test_ingest_text_calls_document_store():
    doc_store = AsyncMock()
    doc_store.store_content = AsyncMock()
    service = _make_service(document_store=doc_store)

    await service.ingest_text(
        content="Full manual text with FBD-20254.",
        knowledge_id="kb-1",
        source_type="manuals",
    )

    doc_store.store_content.assert_called_once()
    kwargs = doc_store.store_content.call_args[1]
    assert "FBD-20254" in kwargs["content"]


async def test_ingest_without_document_store(tmp_path):
    service = _make_service(document_store=None)

    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content.")

    source = await service.ingest(file_path=test_file, knowledge_id="kb-1")
    assert source is not None


async def test_structured_ingestion_has_document_store():
    """StructuredIngestionService stores document_store reference."""
    from x64rag.retrieval.modules.ingestion.analyze.service import StructuredIngestionService

    doc_store = AsyncMock()
    metadata_store = AsyncMock()
    embeddings = AsyncMock()
    embeddings.model = "test-model"

    service = StructuredIngestionService(
        embeddings=embeddings,
        vector_store=AsyncMock(),
        metadata_store=metadata_store,
        embedding_model_name="test:model",
        document_store=doc_store,
    )
    assert service._document_store is doc_store
