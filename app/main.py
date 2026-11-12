import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from app.config import settings
from app.inference import InferenceError, check_ollama_health, generate, list_models
from app.rag import index_pdf, query_rag
from app.agent import run_agent

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LLMOps Platform API v%s", settings.api_version)
    if not await check_ollama_health():
        logger.warning("Ollama not reachable at startup — requests will fail until it is ready")
    yield
    logger.info("Shutting down LLMOps Platform API")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic models ---

class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: Optional[str] = None
    system: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1, le=8192)


class InferenceResponse(BaseModel):
    text: str
    model: str
    tokens_generated: int
    prompt_tokens: int
    duration_seconds: float
    tokens_per_second: float


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=20)


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


# --- Endpoints ---

@app.get("/health")
async def health():
    ollama_up = await check_ollama_health()
    return {
        "status": "healthy" if ollama_up else "degraded",
        "ollama": "up" if ollama_up else "down",
        "version": settings.api_version,
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    try:
        return await generate(
            prompt=request.prompt,
            model=request.model,
            system=request.system,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/api/v1/models")
async def get_models():
    try:
        return {"models": await list_models()}
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/v1/documents")
async def upload_document(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        result = await index_pdf(content, file.filename or "document.pdf")
        return {"message": "Document indexed successfully", **result}
    except Exception as exc:
        logger.error("Failed to index document: %s", exc)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")


@app.post("/api/v1/rag/query")
async def rag_query(request: RAGQueryRequest):
    try:
        return await query_rag(request.question, request.top_k)
    except Exception as exc:
        logger.error("RAG query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


@app.post("/api/v1/agent/query")
async def agent_query(request: AgentQueryRequest):
    try:
        return await run_agent(request.query)
    except Exception as exc:
        logger.error("Agent query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent query failed: {exc}")
