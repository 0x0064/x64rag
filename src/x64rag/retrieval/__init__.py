"""RAG — Retrieval-Augmented Generation SDK."""

from x64rag.retrieval.common.startup import check_baml as _check_baml

_check_baml()

from x64rag.retrieval.common.errors import ConfigurationError as ConfigurationError
from x64rag.retrieval.common.errors import DuplicateSourceError as DuplicateSourceError
from x64rag.retrieval.common.errors import EmbeddingError as EmbeddingError
from x64rag.retrieval.common.errors import EmptyDocumentError as EmptyDocumentError
from x64rag.retrieval.common.errors import GenerationError as GenerationError
from x64rag.retrieval.common.errors import IngestionError as IngestionError
from x64rag.retrieval.common.errors import IngestionInterruptedError as IngestionInterruptedError
from x64rag.retrieval.common.errors import ParseError as ParseError
from x64rag.retrieval.common.errors import RagError as RagError
from x64rag.retrieval.common.errors import RetrievalError as RetrievalError
from x64rag.retrieval.common.errors import SourceNotFoundError as SourceNotFoundError
from x64rag.retrieval.common.errors import StoreError as StoreError
from x64rag.retrieval.common.errors import TreeIndexingError as TreeIndexingError
from x64rag.retrieval.common.errors import TreeSearchError as TreeSearchError
from x64rag.retrieval.common.language_model import LanguageModelClient as LanguageModelClient
from x64rag.retrieval.common.language_model import LanguageModelProvider as LanguageModelProvider
from x64rag.retrieval.common.models import ContentMatch as ContentMatch
from x64rag.retrieval.common.models import RetrievedChunk as RetrievedChunk
from x64rag.retrieval.common.models import Source as Source
from x64rag.retrieval.common.models import SparseVector as SparseVector
from x64rag.retrieval.common.models import TreeIndex as TreeIndex
from x64rag.retrieval.common.models import TreeNode as TreeNode
from x64rag.retrieval.common.models import TreePage as TreePage
from x64rag.retrieval.common.models import TreeSearchResult as TreeSearchResult
from x64rag.retrieval.modules.evaluation.metrics import ExactMatch as ExactMatch
from x64rag.retrieval.modules.evaluation.metrics import F1Score as F1Score
from x64rag.retrieval.modules.evaluation.metrics import LLMJudge as LLMJudge
from x64rag.retrieval.modules.evaluation.models import JudgmentResult as JudgmentResult
from x64rag.retrieval.modules.evaluation.models import MetricResult as MetricResult
from x64rag.retrieval.modules.evaluation.retrieval_metrics import RetrievalPrecision as RetrievalPrecision
from x64rag.retrieval.modules.evaluation.retrieval_metrics import RetrievalRecall as RetrievalRecall
from x64rag.retrieval.modules.generation.models import QueryResult as QueryResult
from x64rag.retrieval.modules.generation.models import StepResult as StepResult
from x64rag.retrieval.modules.generation.models import StreamEvent as StreamEvent
from x64rag.retrieval.modules.ingestion.analyze.models import DiscoveredEntity as DiscoveredEntity
from x64rag.retrieval.modules.ingestion.analyze.models import DocumentSynthesis as DocumentSynthesis
from x64rag.retrieval.modules.ingestion.analyze.models import PageAnalysis as PageAnalysis
from x64rag.retrieval.modules.ingestion.base import BaseIngestionMethod as BaseIngestionMethod
from x64rag.retrieval.modules.ingestion.embeddings.facade import Embeddings as Embeddings
from x64rag.retrieval.modules.ingestion.embeddings.sparse.base import BaseSparseEmbeddings as BaseSparseEmbeddings
from x64rag.retrieval.modules.ingestion.embeddings.sparse.fastembed import (
    FastEmbedSparseEmbeddings as FastEmbedSparseEmbeddings,
)
from x64rag.retrieval.modules.ingestion.methods.document import DocumentIngestion as DocumentIngestion
from x64rag.retrieval.modules.ingestion.methods.graph import GraphIngestion as GraphIngestion
from x64rag.retrieval.modules.ingestion.methods.tree import TreeIngestion as TreeIngestion
from x64rag.retrieval.modules.ingestion.methods.vector import VectorIngestion as VectorIngestion
from x64rag.retrieval.modules.ingestion.vision.facade import Vision as Vision
from x64rag.retrieval.modules.namespace import MethodNamespace as MethodNamespace
from x64rag.retrieval.modules.retrieval.base import BaseRetrievalMethod as BaseRetrievalMethod
from x64rag.retrieval.modules.retrieval.judging import BaseRetrievalJudge as BaseRetrievalJudge
from x64rag.retrieval.modules.retrieval.judging import LLMRetrievalJudge as LLMRetrievalJudge
from x64rag.retrieval.modules.retrieval.methods.document import DocumentRetrieval as DocumentRetrieval
from x64rag.retrieval.modules.retrieval.methods.graph import GraphRetrieval as GraphRetrieval
from x64rag.retrieval.modules.retrieval.methods.vector import VectorRetrieval as VectorRetrieval
from x64rag.retrieval.modules.retrieval.refinement.abstractive import AbstractiveRefiner as AbstractiveRefiner
from x64rag.retrieval.modules.retrieval.refinement.base import BaseChunkRefiner as BaseChunkRefiner
from x64rag.retrieval.modules.retrieval.refinement.extractive import ExtractiveRefiner as ExtractiveRefiner
from x64rag.retrieval.modules.retrieval.search.reranking.base import BaseReranking as BaseReranking
from x64rag.retrieval.modules.retrieval.search.reranking.facade import Reranking as Reranking
from x64rag.retrieval.modules.retrieval.search.rewriting.base import BaseQueryRewriter as BaseQueryRewriter
from x64rag.retrieval.modules.retrieval.search.rewriting.hyde import HyDeRewriter as HyDeRewriter
from x64rag.retrieval.modules.retrieval.search.rewriting.multi_query import MultiQueryRewriter as MultiQueryRewriter
from x64rag.retrieval.modules.retrieval.search.rewriting.step_back import StepBackRewriter as StepBackRewriter
from x64rag.retrieval.server import GenerationConfig as GenerationConfig
from x64rag.retrieval.server import IngestionConfig as IngestionConfig
from x64rag.retrieval.server import PersistenceConfig as PersistenceConfig
from x64rag.retrieval.server import RagServer as RagServer
from x64rag.retrieval.server import RagServerConfig as RagServerConfig
from x64rag.retrieval.server import RetrievalConfig as RetrievalConfig
from x64rag.retrieval.server import TreeIndexingConfig as TreeIndexingConfig
from x64rag.retrieval.server import TreeSearchConfig as TreeSearchConfig
from x64rag.retrieval.stores.document.base import BaseDocumentStore as BaseDocumentStore
from x64rag.retrieval.stores.document.filesystem import FilesystemDocumentStore as FilesystemDocumentStore
from x64rag.retrieval.stores.document.postgres import PostgresDocumentStore as PostgresDocumentStore
from x64rag.retrieval.stores.graph.base import BaseGraphStore as BaseGraphStore
from x64rag.retrieval.stores.graph.models import GraphEntity as GraphEntity
from x64rag.retrieval.stores.graph.models import GraphPath as GraphPath
from x64rag.retrieval.stores.graph.models import GraphRelation as GraphRelation
from x64rag.retrieval.stores.graph.models import GraphResult as GraphResult
from x64rag.retrieval.stores.graph.neo4j import Neo4jGraphStore as Neo4jGraphStore
from x64rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore as SQLAlchemyMetadataStore
from x64rag.retrieval.stores.vector.qdrant import QdrantVectorStore as QdrantVectorStore

__all__ = [
    "RagServer",
    "RagServerConfig",
    "PersistenceConfig",
    "IngestionConfig",
    "RetrievalConfig",
    "GenerationConfig",
    "Source",
    "QueryResult",
    "RetrievedChunk",
    "ContentMatch",
    "StreamEvent",
    "SparseVector",
    "PageAnalysis",
    "DocumentSynthesis",
    "DiscoveredEntity",
    "GraphEntity",
    "GraphRelation",
    "GraphPath",
    "GraphResult",
    "JudgmentResult",
    "MetricResult",
    "StepResult",
    "RagError",
    "IngestionError",
    "ParseError",
    "EmptyDocumentError",
    "EmbeddingError",
    "IngestionInterruptedError",
    "RetrievalError",
    "GenerationError",
    "StoreError",
    "DuplicateSourceError",
    "SourceNotFoundError",
    "ConfigurationError",
    "Embeddings",
    "BaseSparseEmbeddings",
    "FastEmbedSparseEmbeddings",
    "Vision",
    "LanguageModelClient",
    "LanguageModelProvider",
    "Reranking",
    "BaseReranking",
    "BaseQueryRewriter",
    "HyDeRewriter",
    "MultiQueryRewriter",
    "StepBackRewriter",
    "BaseGraphStore",
    "Neo4jGraphStore",
    "QdrantVectorStore",
    "SQLAlchemyMetadataStore",
    "BaseDocumentStore",
    "PostgresDocumentStore",
    "FilesystemDocumentStore",
    "ExactMatch",
    "F1Score",
    "LLMJudge",
    "RetrievalRecall",
    "RetrievalPrecision",
    "BaseRetrievalJudge",
    "LLMRetrievalJudge",
    "BaseChunkRefiner",
    "ExtractiveRefiner",
    "AbstractiveRefiner",
    "TreeIndexingConfig",
    "TreeSearchConfig",
    "TreeNode",
    "TreePage",
    "TreeIndex",
    "TreeSearchResult",
    "TreeIndexingError",
    "TreeSearchError",
    "BaseRetrievalMethod",
    "BaseIngestionMethod",
    "MethodNamespace",
    "VectorRetrieval",
    "DocumentRetrieval",
    "GraphRetrieval",
    "VectorIngestion",
    "DocumentIngestion",
    "GraphIngestion",
    "TreeIngestion",
]
