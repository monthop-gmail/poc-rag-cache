import os
from onyx_database import OnyxClient
from dotenv import load_dotenv
from document_processor import ChunkResult

load_dotenv()


class KnowledgeBase:
    """
    RAG Knowledge Base with parent-child chunking.

    - child chunks (small) → ใช้ค้นหา (vector search)
    - parent chunks (large) → ส่งให้ LLM อ่าน (บริบทครบ)

    ทั้งสอง เก็บใน collection เดียวกัน โดย child มี parent_text ใน payload
    """

    def __init__(self):
        host = os.getenv("ONYX_HOST", "onyx")
        port = os.getenv("ONYX_PORT", "8080")
        self.client = OnyxClient(host=host, port=int(port))
        self.collection_name = "knowledge_base"

        try:
            self.client.create_collection(
                name=self.collection_name,
                dimension=768,
                metric="cosine"
            )
        except Exception:
            pass

    def ingest_chunks(self, chunk_results: list[ChunkResult], vectors: list[list[float]], source_filename: str):
        """
        Ingest parent-child chunks into Onyx.
        Vector is computed from child_text (for search precision).
        Payload includes parent_text (for full context retrieval).
        """
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
        """
        Search by child vector, return deduplicated parent texts.
        This ensures the LLM sees full context, not just small fragments.
        """
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

                # Deduplicate: multiple children may share the same parent
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
