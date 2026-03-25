import time
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from gemini_utils import get_embedding, get_gemini_response
from onyx_provider import RedisCache, OnyxCache
from rag_provider import KnowledgeBase
from document_processor import extract_text_from_pdf, extract_text_from_txt, chunk_document

logger = logging.getLogger("RAG")

router = APIRouter(prefix="/rag", tags=["RAG"])

# Shared cache instances
redis_cache = RedisCache()
onyx_cache = OnyxCache()
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


# --- Endpoints ---

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
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

    # Chunk with parent-child strategy
    chunk_results = chunk_document(text)
    logger.info(f"Uploading '{filename}': {len(chunk_results)} child chunks")

    # Embed child chunks (small, precise for search)
    vectors = [get_embedding(cr.child_text) for cr in chunk_results]

    # Ingest into knowledge base
    knowledge_base.ingest_chunks(chunk_results, vectors, source_filename=filename)

    # Track source in Redis set
    redis_cache.r.sadd("rag:sources", filename) if redis_cache.r else None

    return UploadResponse(
        filename=filename,
        chunks=len(chunk_results),
        message=f"Ingested {len(chunk_results)} chunks from {filename}"
    )


@router.post("/ask", response_model=RAGQueryResponse)
async def rag_ask(request: RAGQueryRequest):
    """Ask a question with RAG — checks cache first, then searches knowledge base."""
    start_time = time.time()
    user_query = request.query.strip()

    # 1. L1: Redis exact match
    l1_response = redis_cache.get(user_query)
    if l1_response:
        latency = (time.time() - start_time) * 1000
        logger.info(f"L1 Cache Hit: '{user_query}'")
        return RAGQueryResponse(
            response=l1_response,
            cache_level="L1 (Redis)",
            latency_ms=latency,
            saved_api_call=True
        )

    # 2. L2: Onyx semantic cache
    query_vector = get_embedding(user_query)
    l2_response = onyx_cache.query_cache(query_vector, threshold=request.threshold)
    if l2_response:
        redis_cache.set(user_query, l2_response)
        latency = (time.time() - start_time) * 1000
        logger.info(f"L2 Cache Hit: '{user_query}'")
        return RAGQueryResponse(
            response=l2_response,
            cache_level="L2 (Onyx Semantic Cache)",
            latency_ms=latency,
            saved_api_call=True
        )

    # 3. RAG: Search knowledge base
    hits = knowledge_base.search(query_vector, limit=request.top_k, threshold=request.rag_threshold)

    if hits:
        # Build augmented prompt with section context (parent chunks)
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
        # No relevant docs — ask Gemini directly
        prompt = user_query
        sources = []
        cache_level = "MISS (No context)"

    # 4. Call Gemini
    try:
        logger.info(f"Cache Miss: Calling Gemini for '{user_query}' (RAG={bool(hits)})")
        response = get_gemini_response(prompt)

        # 5. Cache the result
        redis_cache.set(user_query, response)
        onyx_cache.upsert_cache(query_vector, user_query, response)

        latency = (time.time() - start_time) * 1000
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
    """Delete a document and its chunks from the knowledge base."""
    try:
        knowledge_base.delete_by_source(filename)
        if redis_cache.r:
            redis_cache.r.srem("rag:sources", filename)
        return {"message": f"Deleted '{filename}' from knowledge base"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
