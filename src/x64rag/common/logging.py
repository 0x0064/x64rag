import logging
import os


def get_logger(module: str) -> logging.Logger:
    logger = logging.getLogger(f"x64rag.{module}")
    if os.getenv("X64RAG_LOG_ENABLED", "false").lower() == "true":
        level = os.getenv("X64RAG_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logger.addHandler(handler)
            logger.propagate = False
    else:
        logger.setLevel(logging.CRITICAL)
    return logger
