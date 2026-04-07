from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from x64rag.retrieval.modules.ingestion.chunk.service import IngestionService


def _make_method(name="vector"):
    return SimpleNamespace(
        name=name,
        ingest=AsyncMock(),
        delete=AsyncMock(),
    )


def _make_service(ingestion_methods=None, contextual_chunking=True):
    chunker = MagicMock()
    chunker.chunk = MagicMock(
        return_value=[
            MagicMock(
                content="chunk text",
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
        contextual_chunking=contextual_chunking,
    )


async def test_ingestion_calls_all_methods(tmp_path):
    """All registered ingestion methods are invoked during ingest."""
    method_a = _make_method("vector")
    method_b = _make_method("document")

    service = _make_service(ingestion_methods=[method_a, method_b])

    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content for ingestion.")

    await service.ingest(file_path=test_file, knowledge_id="kb-1")

    method_a.ingest.assert_called_once()
    method_b.ingest.assert_called_once()


async def test_ingestion_passes_correct_arguments(tmp_path):
    """Ingestion methods receive source_id, chunks, metadata, etc."""
    method = _make_method("vector")
    service = _make_service(ingestion_methods=[method])

    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content.")

    source = await service.ingest(
        file_path=test_file,
        knowledge_id="kb-1",
        source_type="manuals",
        metadata={"name": "Test Doc"},
    )

    method.ingest.assert_called_once()
    kwargs = method.ingest.call_args[1]
    assert kwargs["source_id"] == source.source_id
    assert kwargs["knowledge_id"] == "kb-1"
    assert kwargs["source_type"] == "manuals"
    assert kwargs["source_weight"] == 1.0
    assert kwargs["title"] == "Test Doc"
    assert kwargs["metadata"] == {"name": "Test Doc"}
    assert isinstance(kwargs["chunks"], list)
    assert len(kwargs["chunks"]) == 1


async def test_ingestion_creates_source(tmp_path):
    """The service creates a Source object and stores it via metadata_store."""
    method = _make_method("vector")
    service = _make_service(ingestion_methods=[method])

    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content.")

    source = await service.ingest(
        file_path=test_file,
        knowledge_id="kb-1",
        source_type="manuals",
        metadata={"name": "Test Doc"},
    )

    assert source.source_id is not None
    assert source.knowledge_id == "kb-1"
    assert source.source_type == "manuals"
    assert source.chunk_count == 1
    assert source.embedding_model == "test:model"

    service._metadata_store.create_source.assert_called_once()


async def test_ingestion_payload_has_context_fields(tmp_path):
    """Contextual chunking adds context fields to chunks before passing to methods."""
    method = _make_method("vector")
    service = _make_service(ingestion_methods=[method], contextual_chunking=True)

    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content.")

    await service.ingest(file_path=test_file, knowledge_id="kb-1", source_type="manuals", metadata={"name": "Test Doc"})

    method.ingest.assert_called_once()
    kwargs = method.ingest.call_args[1]
    chunks = kwargs["chunks"]
    chunk = chunks[0]
    assert hasattr(chunk, "context")
    assert hasattr(chunk, "contextualized")
    assert hasattr(chunk, "chunk_type")
