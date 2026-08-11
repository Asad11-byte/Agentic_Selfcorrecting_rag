from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.config import settings


@lru_cache
def get_qdrant() -> QdrantClient:
    if not settings.qdrant_url:
        raise RuntimeError("QDRANT_URL is not set")
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


def ensure_collection() -> None:
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.jina_vector_size, distance=Distance.COSINE),
        )


@dataclass
class Chunk:
    id: str
    text: str
    source: str | None = None


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    client = get_qdrant()
    points = [
        PointStruct(id=chunk.id, vector=vector, payload={"text": chunk.text, "source": chunk.source})
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)


@dataclass
class RetrievedDoc:
    text: str
    source: str | None
    score: float


def search(query_vector: list[float], top_k: int = 5) -> list[RetrievedDoc]:
    client = get_qdrant()
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        RetrievedDoc(
            text=str(r.payload.get("text", "")) if r.payload else "",
            source=(r.payload.get("source") if r.payload else None),
            score=r.score,
        )
        for r in results
    ]
