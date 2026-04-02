from unittest.mock import AsyncMock, MagicMock

from x64rag.retrieval.common.models import SparseVector
from x64rag.retrieval.modules.ingestion.chunk.service import IngestionService


def _make_service(sparse_embeddings=None, contextual_chunking=True):
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1] * 128])
    embeddings.model = "test-model"

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
        sparse_embeddings=sparse_embeddings,
        contextual_chunking=contextual_chunking,
    )


async def test_ingestion_with_sparse_embeddings(tmp_path):
    sparse = AsyncMock()
    sparse.embed_sparse = AsyncMock(return_value=[SparseVector(indices=[1, 5], values=[0.8, 0.3])])

    service = _make_service(sparse_embeddings=sparse)

    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content for ingestion.")

    await service.ingest(file_path=test_file, knowledge_id="kb-1")

    sparse.embed_sparse.assert_called_once()
    upsert_call = service._vector_store.upsert.call_args
    points = upsert_call[0][0] if upsert_call[0] else upsert_call[1]["points"]
    assert points[0].sparse_vector is not None


async def test_ingestion_without_sparse_embeddings(tmp_path):
    service = _make_service(sparse_embeddings=None)

    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content.")

    await service.ingest(file_path=test_file, knowledge_id="kb-1")

    upsert_call = service._vector_store.upsert.call_args
    points = upsert_call[0][0] if upsert_call[0] else upsert_call[1]["points"]
    assert points[0].sparse_vector is None


async def test_ingestion_payload_has_context_fields(tmp_path):
    service = _make_service(contextual_chunking=True)

    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content.")

    await service.ingest(file_path=test_file, knowledge_id="kb-1", source_type="manuals", metadata={"name": "Test Doc"})

    upsert_call = service._vector_store.upsert.call_args
    points = upsert_call[0][0] if upsert_call[0] else upsert_call[1]["points"]
    payload = points[0].payload
    assert "context" in payload
    assert "contextualized" in payload
    assert "chunk_type" in payload
