from typing import Literal

import httpx

from app.config import settings

JINA_URL = "https://api.jina.ai/v1/embeddings"

Task = Literal["retrieval.passage", "retrieval.query"]


async def embed(texts: list[str], task: Task = "retrieval.passage") -> list[list[float]]:
    """Embed one or more strings with Jina AI.

    `task` follows Jina v3's asymmetric embedding convention: embed documents with
    "retrieval.passage" at ingest time and queries with "retrieval.query" at search
    time — this measurably improves retrieval quality over using the same task for both.
    """
    if not settings.jina_api_key:
        raise RuntimeError("JINA_API_KEY is not set")

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            JINA_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.jina_api_key}",
            },
            json={"model": settings.jina_embedding_model, "task": task, "input": texts},
        )
    res.raise_for_status()
    data = res.json()["data"]
    # Jina doesn't guarantee response order matches input order — sort by index.
    data.sort(key=lambda d: d["index"])
    return [d["embedding"] for d in data]


async def embed_one(text: str, task: Task = "retrieval.query") -> list[float]:
    vectors = await embed([text], task)
    return vectors[0]
