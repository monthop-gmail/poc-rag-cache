import time
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.llm.gemini import get_embedding, get_embeddings_batch, get_gemini_response
from app.cache.redis_cache import RedisCache
from app.cache.semantic_cache import SemanticCache
from app.rag.knowledge_base import KnowledgeBase
from app.rag.chunking import extract_text_from_pdf, extract_text_from_txt, chunk_document
from app.rag.metrics import record, get_stats, StatsResponse
from app.config import DEFAULT_CHILD_SIZE, DEFAULT_PARENT_SIZE, DEFAULT_OVERLAP

logger = logging.getLogger("RAG")

router = APIRouter(prefix="/rag", tags=["RAG"])

# Service instances
redis_cache = RedisCache()
semantic_cache = SemanticCache()
knowledge_base = KnowledgeBase()


# --- Request/Response Models ---

class RAGQueryRequest(BaseModel):
    query: str
    threshold: float = 0.92
    rag_threshold: float = 0.7
    top_k: int = 5


class RAGQueryResponse(BaseModel):
    response: str
    cache_level: str
    latency_ms: float
    saved_api_call: bool
    sources: list[str] = []


class UploadResponse(BaseModel):
    filename: str
    chunks: int
    message: str


class SourcesResponse(BaseModel):
    sources: list[str]


class InvalidateCacheResponse(BaseModel):
    cleared_redis_keys: int
    cleared_semantic_cache: bool
    message: str


# --- Endpoints ---

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    child_size: int = DEFAULT_CHILD_SIZE,
    parent_size: int = DEFAULT_PARENT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
):
    """Upload a PDF or TXT file to the RAG knowledge base."""
    filename = file.filename or "unknown"

    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

    file_bytes = await file.read()

    # Extract text
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the file")

    # Invalidate stale cache if re-uploading same file
    if redis_cache.r and redis_cache.r.sismember("rag:sources", filename):
        logger.info(f"Re-upload detected for '{filename}', invalidating related cache")
        knowledge_base.delete_by_source(filename)
        _invalidate_cache()

    # Chunk with parent-child strategy
    chunk_results = chunk_document(text, child_size=child_size, parent_size=parent_size, overlap=overlap)
    logger.info(f"Uploading '{filename}': {len(chunk_results)} chunks (child={child_size}, parent={parent_size})")

    # Embed child chunks in batches
    child_texts = [cr.child_text for cr in chunk_results]
    vectors = get_embeddings_batch(child_texts)

    # Ingest
    knowledge_base.ingest_chunks(chunk_results, vectors, source_filename=filename)

    # Track source
    if redis_cache.r:
        redis_cache.r.sadd("rag:sources", filename)

    record("total_documents")
    record("total_chunks_ingested", len(chunk_results))

    return UploadResponse(
        filename=filename,
        chunks=len(chunk_results),
        message=f"Ingested {len(chunk_results)} chunks from {filename}"
    )


@router.post("/ask", response_model=RAGQueryResponse)
async def rag_ask(request: RAGQueryRequest):
    """Ask a question — checks cache first, then searches knowledge base."""
    start_time = time.time()
    user_query = request.query.strip()

    record("total_queries")

    # 1. L1: Redis exact match
    l1_response = redis_cache.get(user_query)
    if l1_response:
        latency = (time.time() - start_time) * 1000
        record("l1_hits")
        record("total_latency_ms", latency)
        return RAGQueryResponse(
            response=l1_response,
            cache_level="L1 (Redis)",
            latency_ms=latency,
            saved_api_call=True
        )

    # 2. L2: Semantic cache
    query_vector = get_embedding(user_query)
    l2_response = semantic_cache.query(query_vector, threshold=request.threshold)
    if l2_response:
        redis_cache.set(user_query, l2_response)
        latency = (time.time() - start_time) * 1000
        record("l2_hits")
        record("total_latency_ms", latency)
        return RAGQueryResponse(
            response=l2_response,
            cache_level="L2 (Semantic Cache)",
            latency_ms=latency,
            saved_api_call=True
        )

    # 3. RAG: Search knowledge base
    hits = knowledge_base.search(query_vector, limit=request.top_k, threshold=request.rag_threshold)

    if hits:
        context_parts = []
        for h in hits:
            section = h.get("section", "")
            source = h["source"]
            header = f"[{source} | {section}]" if section else f"[{source}]"
            context_parts.append(f"{header}:\n{h['text']}")
        context = "\n\n---\n\n".join(context_parts)
        prompt = (
            f"Based on the following context, answer the question. "
            f"Only use information from the provided context. "
            f"If the context does not contain enough information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_query}\n"
            f"Answer:"
        )
        sources = list(set(h["source"] for h in hits))
        cache_level = "MISS (RAG)"
    else:
        prompt = user_query
        sources = []
        cache_level = "MISS (No context)"

    # 4. Call Gemini
    try:
        logger.info(f"Cache Miss: Calling Gemini for '{user_query}' (RAG={bool(hits)})")
        response = get_gemini_response(prompt)

        # 5. Cache the result
        redis_cache.set(user_query, response)
        semantic_cache.upsert(query_vector, user_query, response)

        latency = (time.time() - start_time) * 1000
        if hits:
            record("rag_hits")
        else:
            record("cache_misses")
        record("total_latency_ms", latency)

        return RAGQueryResponse(
            response=response,
            cache_level=cache_level,
            latency_ms=latency,
            saved_api_call=False,
            sources=sources
        )
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources", response_model=SourcesResponse)
async def list_sources():
    """List all ingested document sources."""
    if redis_cache.r:
        sources = list(redis_cache.r.smembers("rag:sources"))
    else:
        sources = []
    return SourcesResponse(sources=sources)


@router.delete("/sources/{filename}")
async def delete_source(filename: str):
    """Delete a document and invalidate cache."""
    try:
        knowledge_base.delete_by_source(filename)
        if redis_cache.r:
            redis_cache.r.srem("rag:sources", filename)
        _invalidate_cache()
        return {"message": f"Deleted '{filename}' and invalidated cache"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def stats():
    """Get RAG system metrics."""
    return get_stats()


@router.post("/invalidate-cache", response_model=InvalidateCacheResponse)
async def invalidate_cache():
    """Manually clear all caches."""
    return _invalidate_cache()


def _invalidate_cache() -> InvalidateCacheResponse:
    cleared_keys = 0
    if redis_cache.r:
        cursor = 0
        while True:
            cursor, keys = redis_cache.r.scan(cursor, match="*", count=100)
            for key in keys:
                if not key.startswith("rag:"):
                    redis_cache.r.delete(key)
                    cleared_keys += 1
            if cursor == 0:
                break

    semantic_cache.clear_all()

    return InvalidateCacheResponse(
        cleared_redis_keys=cleared_keys,
        cleared_semantic_cache=True,
        message=f"Cache invalidated: {cleared_keys} Redis keys cleared, semantic cache reset"
    )
