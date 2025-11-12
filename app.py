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
    title="Qwen3 Embedding & Reranker Service",
    description="Embedding and Reranking service using Qwen3 models with vLLM",
    version="1.0.0"
)

# Model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B-GGUF")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "")  # Optional reranker model
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
DEVICE = os.getenv("DEVICE", "cpu")

# Initialize embedding model
logger.info(f"Loading embedding model: {EMBEDDING_MODEL} on {DEVICE}")
try:
    embedding_llm = LLM(
        model=EMBEDDING_MODEL,
        max_model_len=MAX_MODEL_LEN,
        device=DEVICE,
        task="embed",
        trust_remote_code=True
    )
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    raise

# Initialize reranker model (optional)
reranker_llm = None
if RERANKER_MODEL:
    logger.info(f"Loading reranker model: {RERANKER_MODEL} on {DEVICE}")
    try:
        reranker_llm = LLM(
            model=RERANKER_MODEL,
            max_model_len=MAX_MODEL_LEN,
            device=DEVICE,
            task="score",
            trust_remote_code=True
        )
        logger.info("Reranker model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load reranker model: {e}")
        logger.warning("Reranker endpoints will not be available")


class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: Optional[str] = None
    top_n: Optional[int] = None


class RerankResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Qwen3 Embedding & Reranker Service",
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL or "not configured",
        "reranker_available": reranker_llm is not None,
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
        outputs = embedding_llm.encode(texts)
        
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
            "model": request.model or EMBEDDING_MODEL,
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


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """
    Rerank documents based on relevance to a query.
    Returns documents sorted by relevance score.
    """
    if reranker_llm is None:
        raise HTTPException(
            status_code=503,
            detail="Reranker model not configured. Set RERANKER_MODEL environment variable."
        )
    
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents cannot be empty")
        
        # Prepare input pairs (query, document)
        pairs = [[request.query, doc] for doc in request.documents]
        
        # Generate scores using reranker
        logger.info(f"Reranking {len(request.documents)} document(s)")
        outputs = reranker_llm.encode(pairs)
        
        # Format response with scores
        results = []
        for idx, output in enumerate(outputs):
            # Get score from output (depending on model implementation)
            # vLLM score task typically returns a single score
            score = output.outputs.embedding[0] if hasattr(output.outputs, 'embedding') else 0.0
            results.append({
                "index": idx,
                "document": request.documents[idx],
                "relevance_score": float(score)
            })
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Apply top_n filter if specified
        if request.top_n is not None and request.top_n > 0:
            results = results[:request.top_n]
        
        # Format final response
        data = []
        for rank, result in enumerate(results):
            data.append({
                "index": result["index"],
                "relevance_score": result["relevance_score"],
                "document": result["document"]
            })
        
        # Calculate token usage (approximate)
        total_tokens = len(request.query.split()) + sum(len(doc.split()) for doc in request.documents)
        
        response = {
            "object": "list",
            "data": data,
            "model": request.model or RERANKER_MODEL,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank")
async def rerank_documents_simple(request: RerankRequest):
    """Simplified reranking endpoint (alias for /v1/rerank)"""
    return await rerank_documents(request)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
