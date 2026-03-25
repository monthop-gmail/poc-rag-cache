import os
import time
import logging
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from onyx_provider import OnyxCache, RedisCache
from dotenv import load_dotenv

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SemanticCache")

load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

EMBEDDING_MODEL = "models/text-embedding-004"
GENERATIVE_MODEL = "gemini-1.5-flash"

app = FastAPI(title="Gemini Semantic Cache API")

# Global Cache Instances
redis_cache = RedisCache()
onyx_cache = OnyxCache()

class QueryRequest(BaseModel):
    query: str
    threshold: float = 0.92

class QueryResponse(BaseModel):
    response: str
    cache_level: str
    latency_ms: float
    saved_api_call: bool

def get_embedding(text: str):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']

def get_gemini_response(prompt: str):
    model = genai.GenerativeModel(GENERATIVE_MODEL)
    response = model.generate_content(prompt)
    return response.text

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

    # 2. Check L2: Onyx (Semantic Match)
    try:
        query_vector = get_embedding(user_query)
        l2_response = onyx_cache.query_cache(query_vector, threshold=request.threshold)
        
        if l2_response:
            # Sync L2 back to L1 for future fast hits
            redis_cache.set(user_query, l2_response)
            
            latency = (time.time() - start_time) * 1000
            logger.info(f"L2 Cache Hit: '{user_query}'")
            return QueryResponse(
                response=l2_response,
                cache_level="L2 (Onyx)",
                latency_ms=latency,
                saved_api_call=True
            )
    except Exception as e:
        logger.error(f"Onyx error: {e}")

    # 3. Cache Miss: Gemini Call
    try:
        logger.info(f"Cache Miss: Calling Gemini for '{user_query}'")
        response = get_gemini_response(user_query)
        
        # Save to both caches
        redis_cache.set(user_query, response)
        onyx_cache.upsert_cache(query_vector, user_query, response)
        
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
