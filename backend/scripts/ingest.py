"""Usage: python -m scripts.ingest ./data
Reads all .txt/.md files in the given folder, chunks, embeds with Jina, upserts to Qdrant.
"""

import asyncio
import sys
import uuid
from pathlib import Path

from app.services.chunking import chunk_text
from app.services.embeddings import embed
from app.services.vectorstore import Chunk, ensure_collection, upsert_chunks

BATCH_SIZE = 32


async def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "./data")
    files = [f for f in folder.iterdir() if f.suffix in (".txt", ".md")]

    if not files:
        print(f"No .txt or .md files found in {folder}")
        return

    ensure_collection()

    total = 0
    for file in files:
        text = file.read_text(encoding="utf-8")
        chunks = [Chunk(id=str(uuid.uuid4()), text=t, source=file.name) for t in chunk_text(text)]

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            vectors = await embed([c.text for c in batch], "retrieval.passage")
            upsert_chunks(batch, vectors)
            total += len(batch)
        print(f"Ingested {len(chunks)} chunks from {file.name}")

    print(f"Done. {total} chunks inserted total.")


if __name__ == "__main__":
    asyncio.run(main())
