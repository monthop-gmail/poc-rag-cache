from onyx_database import OnyxClient
from app.config import ONYX_HOST, ONYX_PORT, ONYX_DIMENSION, ONYX_METRIC, KNOWLEDGE_BASE_COLLECTION
from app.rag.chunking import ChunkResult


class KnowledgeBase:
    """
    RAG Knowledge Base with parent-child chunking.

    - child chunks (small) → ใช้ค้นหา (vector search)
    - parent chunks (large) → ส่งให้ LLM อ่าน (บริบทครบ)
    """

    def __init__(self):
        self.client = OnyxClient(host=ONYX_HOST, port=ONYX_PORT)
        self.collection_name = KNOWLEDGE_BASE_COLLECTION

        try:
            self.client.create_collection(
                name=self.collection_name,
                dimension=ONYX_DIMENSION,
                metric=ONYX_METRIC,
            )
        except Exception:
            pass

    def ingest_chunks(self, chunk_results: list[ChunkResult], vectors: list[list[float]], source_filename: str):
        points = []
        for cr, vector in zip(chunk_results, vectors):
            points.append({
                "id": f"{source_filename}_{cr.chunk_index}",
                "vector": vector,
                "payload": {
                    "child_text": cr.child_text,
                    "parent_text": cr.parent_text,
                    "section": cr.section,
                    "source_filename": source_filename,
                    "chunk_index": cr.chunk_index
                }
            })
        self.client.upsert(collection=self.collection_name, points=points)

    def search(self, vector: list[float], limit: int = 5, threshold: float = 0.7) -> list[dict]:
        """Search by child vector, return deduplicated parent texts."""
        results = self.client.search(
            collection=self.collection_name,
            vector=vector,
            limit=limit
        )
        hits = []
        seen_parents = set()

        if results:
            for hit in results:
                if hit.score < threshold:
                    continue

                parent_text = hit.payload.get("parent_text", "")

                if parent_text in seen_parents:
                    continue
                seen_parents.add(parent_text)

                hits.append({
                    "text": parent_text,
                    "child_text": hit.payload.get("child_text", ""),
                    "section": hit.payload.get("section", ""),
                    "score": hit.score,
                    "source": hit.payload.get("source_filename")
                })
        return hits

    def delete_by_source(self, source_filename: str):
        self.client.delete(
            collection=self.collection_name,
            filter={"source_filename": source_filename}
        )
