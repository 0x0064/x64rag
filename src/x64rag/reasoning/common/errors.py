from x64rag.common.errors import ConfigurationError as ConfigurationError
from x64rag.common.errors import X64RagError


class ReasoningError(X64RagError):
    """Base exception for reasoning SDK errors."""


class ClassificationError(ReasoningError):
    """Error during text classification."""


class ClusteringError(ReasoningError):
    """Error during text clustering."""


class EvaluationError(ReasoningError):
    """Error during evaluation."""


class ComplianceError(ReasoningError):
    """Error during compliance checking."""


class AnalysisError(ReasoningError):
    """Error during text analysis."""
