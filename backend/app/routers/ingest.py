import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.chunking import chunk_text
from app.services.embeddings import embed
from app.services.vectorstore import Chunk, ensure_collection, upsert_chunks

router = APIRouter()

BATCH_SIZE = 32  # stay well under Jina's per-request payload limits


class IngestDoc(BaseModel):
    text: str
    source: str | None = None


class IngestRequest(BaseModel):
    documents: list[IngestDoc]


@router.post("/ingest")
async def ingest_endpoint(body: IngestRequest):
    if not body.documents:
        raise HTTPException(status_code=400, detail="`documents` must be a non-empty list")

    try:
        ensure_collection()

        chunks: list[Chunk] = []
        for doc in body.documents:
            for text in chunk_text(doc.text):
                chunks.append(Chunk(id=str(uuid.uuid4()), text=text, source=doc.source))

        inserted = 0
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            vectors = await embed([c.text for c in batch], "retrieval.passage")
            upsert_chunks(batch, vectors)
            inserted += len(batch)

        return {"chunksInserted": inserted}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
