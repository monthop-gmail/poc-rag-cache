import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.cache.redis_cache import RedisCache
from app.cache.semantic_cache import SemanticCache
from app.llm.gemini import get_embedding, get_gemini_response
from app.rag.routes import router as rag_router

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SemanticCache")

app = FastAPI(title="Gemini Semantic Cache API")
app.include_router(rag_router)

# Global Cache Instances
redis_cache = RedisCache()
semantic_cache = SemanticCache()


class QueryRequest(BaseModel):
    query: str
    threshold: float = 0.92


class QueryResponse(BaseModel):
    response: str
    cache_level: str
    latency_ms: float
    saved_api_call: bool


@app.post("/ask", response_model=QueryResponse)
async def ask_gemini(request: QueryRequest):
    start_time = time.time()
    user_query = request.query.strip()

    # 1. Check L1: Redis (Exact Match)
    l1_response = redis_cache.get(user_query)
    if l1_response:
        latency = (time.time() - start_time) * 1000
        logger.info(f"L1 Cache Hit: '{user_query}'")
        return QueryResponse(
            response=l1_response,
            cache_level="L1 (Redis)",
            latency_ms=latency,
            saved_api_call=True
        )

    # 2. Check L2: Semantic Cache (Semantic Match)
    try:
        query_vector = get_embedding(user_query)
        l2_response = semantic_cache.query(query_vector, threshold=request.threshold)

        if l2_response:
            redis_cache.set(user_query, l2_response)
            latency = (time.time() - start_time) * 1000
            logger.info(f"L2 Cache Hit: '{user_query}'")
            return QueryResponse(
                response=l2_response,
                cache_level="L2 (Semantic Cache)",
                latency_ms=latency,
                saved_api_call=True
            )
    except Exception as e:
        logger.error(f"Semantic cache error: {e}")

    # 3. Cache Miss: Gemini Call
    try:
        logger.info(f"Cache Miss: Calling Gemini for '{user_query}'")
        response = get_gemini_response(user_query)

        # Save to both caches
        redis_cache.set(user_query, response)
        semantic_cache.upsert(query_vector, user_query, response)

        latency = (time.time() - start_time) * 1000
        return QueryResponse(
            response=response,
            cache_level="MISS",
            latency_ms=latency,
            saved_api_call=False
        )
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}
