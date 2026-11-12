import logging
from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    logger.info("Initialising embedding model: %s", settings.embedding_model)
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_host,
    )
