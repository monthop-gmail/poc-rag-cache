import redis
from app.config import REDIS_HOST, REDIS_PORT, REDIS_CACHE_EXPIRE


class RedisCache:
    """L1 Cache — Exact Match (ค้นหาจาก Query ตรงๆ)"""

    def __init__(self):
        try:
            self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        except Exception:
            self.r = None

    def get(self, query: str) -> str | None:
        if not self.r:
            return None
        return self.r.get(query)

    def set(self, query: str, response: str, expire: int = REDIS_CACHE_EXPIRE):
        if not self.r:
            return
        self.r.set(query, response, ex=expire)
