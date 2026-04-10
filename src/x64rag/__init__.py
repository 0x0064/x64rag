"""x64rag — Retrieval-Augmented Generation + Reasoning-Augmented Classification SDK."""

from importlib.metadata import version

__version__ = version("x64rag")

from x64rag.common.errors import ConfigurationError as ConfigurationError
from x64rag.common.errors import X64RagError as X64RagError
from x64rag.common.language_model import LanguageModelClient as LanguageModelClient
from x64rag.common.language_model import LanguageModelProvider as LanguageModelProvider
from x64rag.reasoning import *
from x64rag.retrieval import *
