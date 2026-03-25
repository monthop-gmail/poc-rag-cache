from onyx_database import OnyxClient
from app.config import ONYX_HOST, ONYX_PORT, ONYX_DIMENSION, ONYX_METRIC, SEMANTIC_CACHE_COLLECTION


class SemanticCache:
    """L2 Cache — Semantic Search (ใช้ Vector ค้นหาคำถามที่ความหมายคล้ายกัน)"""

    def __init__(self):
        self.client = OnyxClient(host=ONYX_HOST, port=ONYX_PORT)
        self.collection_name = SEMANTIC_CACHE_COLLECTION

        try:
            self.client.create_collection(
                name=self.collection_name,
                dimension=ONYX_DIMENSION,
                metric=ONYX_METRIC,
            )
        except Exception:
            pass

    def upsert(self, vector: list[float], query: str, response: str):
        self.client.upsert(
            collection=self.collection_name,
            points=[{
                "id": query,
                "vector": vector,
                "payload": {"query": query, "response": response}
            }]
        )

    def query(self, vector: list[float], threshold: float = 0.9) -> str | None:
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

    def clear_all(self):
        """Clear all entries — recreate the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.client.create_collection(
                name=self.collection_name,
                dimension=ONYX_DIMENSION,
                metric=ONYX_METRIC,
            )
        except Exception:
            pass
