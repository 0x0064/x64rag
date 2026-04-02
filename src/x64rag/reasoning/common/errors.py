from x64rag.common.errors import ConfigurationError as ConfigurationError
from x64rag.common.errors import X64RagError


class AceError(X64RagError):
    """Base exception for reasoning SDK errors."""


class ClassificationError(AceError):
    """Error during text classification."""


class ClusteringError(AceError):
    """Error during text clustering."""


class EvaluationError(AceError):
    """Error during evaluation."""


class ComplianceError(AceError):
    """Error during compliance checking."""


class AnalysisError(AceError):
    """Error during text analysis."""
