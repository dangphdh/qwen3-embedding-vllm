# Copilot Instructions for Qwen3 Embedding Service

## Architecture Overview

This is a **single-file FastAPI microservice** (`app.py`) that wraps the Qwen3-0.6B Vietnamese embedding model with vLLM optimization. The service provides OpenAI-compatible embedding endpoints running in a GPU-accelerated Docker container.

**Key Architecture Pattern**: The vLLM `LLM` instance is initialized once at startup (module-level) with `task="embed"` mode, not per-request. This is critical for performance - never refactor to lazy-load the model.

## Critical Development Workflows

### Local Development
```bash
# Install dependencies (requires CUDA-enabled GPU)
pip install -r requirements.txt

# Run server locally (loads ~600MB model to GPU)
python app.py
```

### Docker Development
```bash
# Build and run with compose (preferred for testing GPU integration)
docker-compose up --build

# Check GPU utilization while running
docker exec qwen3-embedding-service nvidia-smi

# View real-time logs
docker-compose logs -f
```

### Testing the Service
```bash
# Health check
curl http://localhost:8000/health

# Test single embedding
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Xin chào Việt Nam"}'

# Test batch embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Câu 1", "Câu 2"]}'
```

## Project-Specific Conventions

### Environment Variables Pattern
All configs use `os.getenv()` with sensible defaults in `app.py`:
- `MODEL_NAME`: defaults to `"zzeennoo/Qwen3-0.6B-Embedding-finetuned-sts-vi"`
- `MAX_MODEL_LEN`: defaults to `8192` (model's context window)
- `GPU_MEMORY_UTILIZATION`: defaults to `0.9` (use 90% of GPU memory)

Override in `docker-compose.yml` environment section, not in code.

### API Endpoint Design
- `/v1/embeddings` - Primary OpenAI-compatible endpoint
- `/embeddings` - Alias that calls `/v1/embeddings` (for backwards compatibility)
- Both accept single string OR list of strings in `input` field

### Request/Response Handling
```python
# Input normalization pattern (line 73-74 in app.py)
texts = [request.input] if isinstance(request.input, str) else request.input
```
Always normalize to list before calling `llm.encode()`. The response format mimics OpenAI's embedding API structure with `object`, `data`, `model`, and `usage` fields.

### vLLM Integration Specifics
- Initialize with `task="embed"` parameter (line 30 in app.py)
- Use `llm.encode(texts)` not `llm.generate()` for embeddings
- Access embeddings via `embedding.outputs.embedding` from each output object
- Token counting is approximate using `len(text.split())` - good enough for usage stats

## Docker & GPU Configuration

### Base Image Strategy
Uses `vllm/vllm-openai:latest` as base (Dockerfile:1) which includes CUDA, cuDNN, and vLLM pre-installed. This is ~10GB but saves compilation time.

### GPU Memory Management
The `GPU_MEMORY_UTILIZATION=0.9` setting (docker-compose.yml:11) tells vLLM to reserve 90% of GPU memory. Lower this if running multiple services on same GPU or getting OOM errors.

### Volume Mounting Pattern
```yaml
volumes:
  - ./cache:/root/.cache/huggingface
```
Caches the downloaded model (~600MB) between container restarts. First startup takes ~2-3 minutes to download model.

### Health Check Implementation
The healthcheck (docker-compose.yml:18-22) curls `/health` endpoint. 60s `start_period` allows time for model loading on cold start.

## Logging & Error Handling

All endpoints use `logger.info()` for normal operations and `logger.error()` for failures. Logs appear in docker-compose output.

**Error Pattern**: Catch all exceptions at endpoint level, log with `logger.error()`, then raise `HTTPException` with appropriate status code (400 for bad input, 500 for internal errors).

## Model-Specific Notes

- **Vietnamese-optimized**: This model is fine-tuned for Vietnamese semantic text similarity (STS-VI task)
- **Embedding dimension**: Output vectors are model-dependent (typically 512 or 768 dimensions)
- **Context limit**: 8192 tokens max per input text
- **Trust remote code**: Required (`trust_remote_code=True` in line 31) as model uses custom code from HuggingFace

## Common Modification Patterns

### Adding a new endpoint
Follow the pattern in lines 64-86. Use `@app.post()` or `@app.get()`, add docstring, use Pydantic models for request/response validation.

### Changing the model
Update `MODEL_NAME` in docker-compose.yml. Ensure new model supports `task="embed"` in vLLM and has HuggingFace integration.

### Adjusting performance
Tune `MAX_MODEL_LEN` (lower = more memory for batch size) and `GPU_MEMORY_UTILIZATION` (lower = less reserved memory). Monitor with `nvidia-smi`.
