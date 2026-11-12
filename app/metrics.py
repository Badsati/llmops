from prometheus_client import Counter, Gauge, Histogram

inference_requests_total = Counter(
    "llmops_inference_requests_total",
    "Total number of inference requests",
    ["model", "status"],
)

inference_duration_seconds = Histogram(
    "llmops_inference_duration_seconds",
    "Inference request duration in seconds",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

tokens_generated_total = Counter(
    "llmops_tokens_generated_total",
    "Total tokens generated",
    ["model"],
)

rag_requests_total = Counter(
    "llmops_rag_requests_total",
    "Total RAG pipeline requests",
    ["status"],
)

rag_duration_seconds = Histogram(
    "llmops_rag_duration_seconds",
    "RAG request duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

documents_indexed_total = Counter(
    "llmops_documents_indexed_total",
    "Total documents indexed into the vector store",
)

ollama_available = Gauge(
    "llmops_ollama_available",
    "Ollama service availability (1=up, 0=down)",
)

app_info = Gauge(
    "llmops_app_info",
    "Application metadata",
    ["version"],
)
app_info.labels(version="1.0.0").set(1)
