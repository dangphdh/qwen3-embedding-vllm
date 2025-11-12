import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from vllm import LLM, SamplingParams
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Qwen3 Embedding Service",
    description="Embedding service using Qwen3-0.6B-Embedding-finetuned-sts-vi with vLLM",
    version="1.0.0"
)

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "zzeennoo/Qwen3-0.6B-Embedding-finetuned-sts-vi")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))

# Initialize vLLM model
logger.info(f"Loading model: {MODEL_NAME}")
try:
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        task="embed",
        trust_remote_code=True
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Qwen3 Embedding Service",
        "model": MODEL_NAME,
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings for the given input text(s).
    Compatible with OpenAI embeddings API format.
    """
    try:
        # Normalize input to list
        texts = [request.input] if isinstance(request.input, str) else request.input
        
        if not texts:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        
        # Generate embeddings using vLLM
        logger.info(f"Generating embeddings for {len(texts)} text(s)")
        outputs = llm.encode(texts)
        
        # Format response
        data = []
        for idx, embedding in enumerate(outputs):
            data.append({
                "object": "embedding",
                "embedding": embedding.outputs.embedding,
                "index": idx
            })
        
        # Calculate token usage (approximate)
        total_tokens = sum(len(text.split()) for text in texts)
        
        response = {
            "object": "list",
            "data": data,
            "model": request.model or MODEL_NAME,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings")
async def create_embeddings_simple(request: EmbeddingRequest):
    """Simplified embeddings endpoint (alias for /v1/embeddings)"""
    return await create_embeddings(request)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
