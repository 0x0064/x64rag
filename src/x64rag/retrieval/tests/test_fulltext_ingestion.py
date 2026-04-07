from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from x64rag.retrieval.modules.ingestion.chunk.service import IngestionService


def _make_method(name="document"):
    return SimpleNamespace(
        name=name,
        ingest=AsyncMock(),
        delete=AsyncMock(),
    )


def _make_service(ingestion_methods=None):
    chunker = MagicMock()
    chunker.chunk = MagicMock(
        return_value=[
            MagicMock(content="chunk 1", page_number=1, section=None, chunk_index=0),
        ]
    )

    metadata_store = AsyncMock()
    metadata_store.list_sources = AsyncMock(return_value=[])
    metadata_store.create_source = AsyncMock()

    if ingestion_methods is None:
        ingestion_methods = [_make_method("vector")]

    return IngestionService(
        chunker=chunker,
        ingestion_methods=ingestion_methods,
        embedding_model_name="test:model",
        metadata_store=metadata_store,
    )


async def test_ingest_calls_document_method(tmp_path):
    doc_method = _make_method("document")
    service = _make_service(ingestion_methods=[doc_method])

    test_file = tmp_path / "test.txt"
    test_file.write_text("Page one content.\n\nPage two content.")

    await service.ingest(file_path=test_file, knowledge_id="kb-1", source_type="manuals")

    doc_method.ingest.assert_called_once()
    kwargs = doc_method.ingest.call_args[1]
    assert kwargs["knowledge_id"] == "kb-1"
    assert kwargs["source_type"] == "manuals"
    assert "Page one content" in kwargs["full_text"]


async def test_ingest_text_calls_document_method():
    doc_method = _make_method("document")
    service = _make_service(ingestion_methods=[doc_method])

    await service.ingest_text(
        content="Full manual text with FBD-20254.",
        knowledge_id="kb-1",
        source_type="manuals",
    )

    doc_method.ingest.assert_called_once()
    kwargs = doc_method.ingest.call_args[1]
    assert "FBD-20254" in kwargs["full_text"]


async def test_ingest_without_methods(tmp_path):
    """Ingestion succeeds with an empty method list (no methods to dispatch)."""
    service = _make_service(ingestion_methods=[])

    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content.")

    source = await service.ingest(file_path=test_file, knowledge_id="kb-1")
    assert source is not None


async def test_structured_ingestion_has_document_method():
    """AnalyzedIngestionService stores document method in ingestion_methods."""
    from x64rag.retrieval.modules.ingestion.analyze.service import AnalyzedIngestionService

    doc_method = SimpleNamespace(name="document", ingest=AsyncMock(), delete=AsyncMock())
    metadata_store = AsyncMock()
    embeddings = AsyncMock()
    embeddings.model = "test-model"

    service = AnalyzedIngestionService(
        embeddings=embeddings,
        vector_store=AsyncMock(),
        metadata_store=metadata_store,
        embedding_model_name="test:model",
        ingestion_methods=[doc_method],
    )
    assert doc_method in service._ingestion_methods
    assert any(m.name == "document" for m in service._ingestion_methods)
