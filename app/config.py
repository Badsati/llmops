from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama
    ollama_host: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout: int = 120

    # ChromaDB
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "documents"

    # Embeddings (pulled via Ollama)
    embedding_model: str = "nomic-embed-text"

    # RAG
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 4

    # API
    api_title: str = "LLMOps Platform API"
    api_version: str = "1.0.0"

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
