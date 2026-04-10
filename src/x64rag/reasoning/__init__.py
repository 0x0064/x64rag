"""RAC — Reasoning-Augmented Classification SDK."""

from x64rag.reasoning.common.startup import check_baml as _check_baml

_check_baml()

from x64rag.reasoning.common.errors import AnalysisError as AnalysisError
from x64rag.reasoning.common.errors import ClassificationError as ClassificationError
from x64rag.reasoning.common.errors import ClusteringError as ClusteringError
from x64rag.reasoning.common.errors import ComplianceError as ComplianceError
from x64rag.reasoning.common.errors import ConfigurationError as ConfigurationError
from x64rag.reasoning.common.errors import EvaluationError as EvaluationError
from x64rag.reasoning.common.errors import ReasoningError as ReasoningError
from x64rag.reasoning.common.language_model import LanguageModelClient as LanguageModelClient
from x64rag.reasoning.common.language_model import LanguageModelProvider as LanguageModelProvider
from x64rag.reasoning.modules.analysis.models import AnalysisConfig as AnalysisConfig
from x64rag.reasoning.modules.analysis.models import AnalysisResult as AnalysisResult
from x64rag.reasoning.modules.analysis.models import ContextTrackingConfig as ContextTrackingConfig
from x64rag.reasoning.modules.analysis.models import DimensionDefinition as DimensionDefinition
from x64rag.reasoning.modules.analysis.models import DimensionResult as DimensionResult
from x64rag.reasoning.modules.analysis.models import Entity as Entity
from x64rag.reasoning.modules.analysis.models import EntityTypeDefinition as EntityTypeDefinition
from x64rag.reasoning.modules.analysis.models import IntentShift as IntentShift
from x64rag.reasoning.modules.analysis.models import Message as Message
from x64rag.reasoning.modules.analysis.models import RetrievalHint as RetrievalHint
from x64rag.reasoning.modules.analysis.service import AnalysisService as AnalysisService
from x64rag.reasoning.modules.classification.models import CategoryDefinition as CategoryDefinition
from x64rag.reasoning.modules.classification.models import Classification as Classification
from x64rag.reasoning.modules.classification.models import ClassificationConfig as ClassificationConfig
from x64rag.reasoning.modules.classification.models import ClassificationSetDefinition as ClassificationSetDefinition
from x64rag.reasoning.modules.classification.models import ClassificationSetResult as ClassificationSetResult
from x64rag.reasoning.modules.classification.service import ClassificationService as ClassificationService
from x64rag.reasoning.modules.clustering.comparison import ClusterChange as ClusterChange
from x64rag.reasoning.modules.clustering.comparison import ClusterComparison as ClusterComparison
from x64rag.reasoning.modules.clustering.comparison import compare_clusters as compare_clusters
from x64rag.reasoning.modules.clustering.models import Cluster as Cluster
from x64rag.reasoning.modules.clustering.models import ClusteringConfig as ClusteringConfig
from x64rag.reasoning.modules.clustering.models import ClusteringResult as ClusteringResult
from x64rag.reasoning.modules.clustering.models import TextWithMetadata as TextWithMetadata
from x64rag.reasoning.modules.clustering.service import ClusteringService as ClusteringService
from x64rag.reasoning.modules.compliance.models import ComplianceConfig as ComplianceConfig
from x64rag.reasoning.modules.compliance.models import ComplianceDimensionDefinition as ComplianceDimensionDefinition
from x64rag.reasoning.modules.compliance.models import ComplianceResult as ComplianceResult
from x64rag.reasoning.modules.compliance.models import Violation as Violation
from x64rag.reasoning.modules.compliance.service import ComplianceService as ComplianceService
from x64rag.reasoning.modules.evaluation.models import EvaluationConfig as EvaluationConfig
from x64rag.reasoning.modules.evaluation.models import EvaluationDimensionDefinition as EvaluationDimensionDefinition
from x64rag.reasoning.modules.evaluation.models import EvaluationPair as EvaluationPair
from x64rag.reasoning.modules.evaluation.models import EvaluationReport as EvaluationReport
from x64rag.reasoning.modules.evaluation.models import EvaluationResult as EvaluationResult
from x64rag.reasoning.modules.evaluation.service import EvaluationService as EvaluationService
from x64rag.reasoning.modules.pipeline.models import AnalyzeStep as AnalyzeStep
from x64rag.reasoning.modules.pipeline.models import ClassifyStep as ClassifyStep
from x64rag.reasoning.modules.pipeline.models import ComplianceStep as ComplianceStep
from x64rag.reasoning.modules.pipeline.models import EvaluateStep as EvaluateStep
from x64rag.reasoning.modules.pipeline.models import PipelineResult as PipelineResult
from x64rag.reasoning.modules.pipeline.models import PipelineServices as PipelineServices
from x64rag.reasoning.modules.pipeline.service import Pipeline as Pipeline
from x64rag.reasoning.protocols import BaseEmbeddings as BaseEmbeddings
from x64rag.reasoning.protocols import BaseSemanticIndex as BaseSemanticIndex

__all__ = [
    "AnalysisError",
    "ClassificationError",
    "ClusteringError",
    "ComplianceError",
    "ConfigurationError",
    "EvaluationError",
    "ReasoningError",
    "LanguageModelClient",
    "LanguageModelProvider",
    "BaseEmbeddings",
    "BaseSemanticIndex",
    "AnalysisConfig",
    "AnalysisResult",
    "AnalysisService",
    "DimensionDefinition",
    "DimensionResult",
    "Entity",
    "EntityTypeDefinition",
    "IntentShift",
    "Message",
    "RetrievalHint",
    "ContextTrackingConfig",
    "CategoryDefinition",
    "Classification",
    "ClassificationConfig",
    "ClassificationService",
    "ClassificationSetDefinition",
    "ClassificationSetResult",
    "Cluster",
    "ClusterChange",
    "ClusterComparison",
    "ClusteringConfig",
    "ClusteringResult",
    "ClusteringService",
    "TextWithMetadata",
    "compare_clusters",
    "EvaluationConfig",
    "EvaluationDimensionDefinition",
    "EvaluationPair",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationService",
    "ComplianceConfig",
    "ComplianceDimensionDefinition",
    "ComplianceResult",
    "ComplianceService",
    "Violation",
    "AnalyzeStep",
    "ClassifyStep",
    "ComplianceStep",
    "EvaluateStep",
    "Pipeline",
    "PipelineResult",
    "PipelineServices",
]
