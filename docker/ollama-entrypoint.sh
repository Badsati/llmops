#!/bin/bash
set -e

# Start Ollama server in background
/bin/ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama server to be ready..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 3
done
echo "Ollama server ready."

# Pull inference model
INFERENCE_MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
echo "Pulling inference model: ${INFERENCE_MODEL}"
/bin/ollama pull "${INFERENCE_MODEL}"

# Pull embedding model
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
echo "Pulling embedding model: ${EMBED_MODEL}"
/bin/ollama pull "${EMBED_MODEL}"

echo "All models ready. Ollama is serving."
wait "${OLLAMA_PID}"
