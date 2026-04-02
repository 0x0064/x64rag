"""Shared error base classes for both retrieval and reasoning SDKs."""


class X64RagError(Exception):
    """Base exception for all x64rag SDK errors."""


class ConfigurationError(X64RagError):
    """Invalid SDK configuration."""
