import os
import redis
from onyx_database import OnyxClient
from dotenv import load_dotenv

load_dotenv()

class RedisCache:
    """
    Cache ชั้นที่ 1 สำหรับ Exact Match (ค้นหาจาก Query ตรงๆ)
    """
    def __init__(self):
        host = os.getenv("REDIS_HOST", "redis")
        port = os.getenv("REDIS_PORT", "6379")
        try:
            self.r = redis.Redis(host=host, port=int(port), decode_responses=True)
        except Exception:
            self.r = None

    def get(self, query):
        if not self.r: return None
        return self.r.get(query)

    def set(self, query, response, expire=3600):
        if not self.r: return
        self.r.set(query, response, ex=expire)

class OnyxCache:
    """
    Cache ชั้นที่ 2 สำหรับ Semantic Search (ใช้ Vector)
    """
    def __init__(self):
        host = os.getenv("ONYX_HOST", "onyx")
        port = os.getenv("ONYX_PORT", "8080")
        self.client = OnyxClient(host=host, port=int(port))
        self.collection_name = "gemini_semantic_cache"
        
        try:
            self.client.create_collection(
                name=self.collection_name,
                dimension=768,
                metric="cosine"
            )
        except Exception:
            pass

    def upsert_cache(self, vector, query, response):
        self.client.upsert(
            collection=self.collection_name,
            points=[{
                "id": query,
                "vector": vector,
                "payload": {
                    "query": query,
                    "response": response
                }
            }]
        )

    def query_cache(self, vector, threshold=0.9):
        results = self.client.search(
            collection=self.collection_name,
            vector=vector,
            limit=1
        )
        if results and len(results) > 0:
            top_hit = results[0]
            if top_hit.score >= threshold:
                return top_hit.payload.get("response")
        return None
