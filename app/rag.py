import logging
import os
import tempfile
import time
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.config import settings
from app.embeddings import get_embeddings
from app.inference import generate
from app.metrics import (
    documents_indexed_total,
    rag_duration_seconds,
    rag_requests_total,
)

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on provided document context. "
    "Use only the information from the provided context to answer the question. "
    "If the context does not contain enough information to answer the question, say so clearly. "
    "Always cite the source document and page number when referencing information. "
    "Do not fabricate information that is not present in the context."
)


def _chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_vector_store() -> Chroma:
    return Chroma(
        client=_chroma_client(),
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
    )


async def index_pdf(file_bytes: bytes, filename: str) -> dict:
    logger.info("Indexing PDF: %s (%d bytes)", filename, len(file_bytes))

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            chunk.metadata["source"] = filename

        store = get_vector_store()
        store.add_documents(chunks)

        documents_indexed_total.inc()
        logger.info("Indexed %d chunks from %s", len(chunks), filename)
        return {"filename": filename, "pages": len(pages), "chunks": len(chunks)}

    finally:
        os.unlink(tmp_path)


async def retrieve_context(question: str, top_k: Optional[int] = None) -> list[Document]:
    k = top_k or settings.rag_top_k
    store = get_vector_store()
    return store.similarity_search(question, k=k)


async def query_rag(question: str, top_k: Optional[int] = None) -> dict:
    start_time = time.time()

    try:
        docs = await retrieve_context(question, top_k)

        if not docs:
            rag_requests_total.labels(status="no_results").inc()
            return {
                "answer": "No relevant documents were found in the knowledge base.",
                "sources": [],
                "context_used": False,
                "chunks_retrieved": 0,
            }

        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        prompt = f"Context from documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        result = await generate(prompt=prompt, system=RAG_SYSTEM_PROMPT)

        seen: set = set()
        sources = []
        for doc in docs:
            key = (doc.metadata.get("source", "unknown"), doc.metadata.get("page", "N/A"))
            if key not in seen:
                seen.add(key)
                sources.append({"filename": key[0], "page": key[1]})

        duration = time.time() - start_time
        rag_requests_total.labels(status="success").inc()
        rag_duration_seconds.observe(duration)

        return {
            "answer": result["text"],
            "sources": sources,
            "context_used": True,
            "chunks_retrieved": len(docs),
            "duration_seconds": duration,
        }

    except Exception as exc:
        duration = time.time() - start_time
        rag_requests_total.labels(status="error").inc()
        rag_duration_seconds.observe(duration)
        logger.error("RAG query failed: %s", exc)
        raise
