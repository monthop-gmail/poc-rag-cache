import os
from onyx_database import OnyxClient
from dotenv import load_dotenv

load_dotenv()


class KnowledgeBase:
    """
    RAG Knowledge Base — เก็บ chunks ของเอกสารใน Onyx (แยก collection จาก semantic cache)
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

    def ingest_chunks(self, chunks: list[str], vectors: list[list[float]], source_filename: str):
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append({
                "id": f"{source_filename}_{i}",
                "vector": vector,
                "payload": {
                    "text": chunk,
                    "source_filename": source_filename,
                    "chunk_index": i
                }
            })
        self.client.upsert(collection=self.collection_name, points=points)

    def search(self, vector: list[float], limit: int = 5, threshold: float = 0.7) -> list[dict]:
        results = self.client.search(
            collection=self.collection_name,
            vector=vector,
            limit=limit
        )
        hits = []
        if results:
            for hit in results:
                if hit.score >= threshold:
                    hits.append({
                        "text": hit.payload.get("text"),
                        "score": hit.score,
                        "source": hit.payload.get("source_filename")
                    })
        return hits

    def delete_by_source(self, source_filename: str):
        self.client.delete(
            collection=self.collection_name,
            filter={"source_filename": source_filename}
        )
