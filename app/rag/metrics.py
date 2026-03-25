from pydantic import BaseModel


class StatsResponse(BaseModel):
    total_queries: int
    l1_hits: int
    l2_hits: int
    rag_hits: int
    cache_misses: int
    cache_hit_rate: float
    avg_latency_ms: float
    total_documents: int
    total_chunks_ingested: int


# In-memory metrics store
_metrics = {
    "total_queries": 0,
    "l1_hits": 0,
    "l2_hits": 0,
    "rag_hits": 0,
    "cache_misses": 0,
    "total_latency_ms": 0.0,
    "total_documents": 0,
    "total_chunks_ingested": 0,
}


def record(key: str, value: float = 1):
    """Increment a metric by value."""
    _metrics[key] = _metrics.get(key, 0) + value


def get_stats() -> StatsResponse:
    total = _metrics["total_queries"]
    total_hits = _metrics["l1_hits"] + _metrics["l2_hits"]
    hit_rate = (total_hits / total * 100) if total > 0 else 0.0
    avg_latency = (_metrics["total_latency_ms"] / total) if total > 0 else 0.0

    return StatsResponse(
        total_queries=total,
        l1_hits=_metrics["l1_hits"],
        l2_hits=_metrics["l2_hits"],
        rag_hits=_metrics["rag_hits"],
        cache_misses=_metrics["cache_misses"],
        cache_hit_rate=round(hit_rate, 2),
        avg_latency_ms=round(avg_latency, 2),
        total_documents=_metrics["total_documents"],
        total_chunks_ingested=_metrics["total_chunks_ingested"],
    )
