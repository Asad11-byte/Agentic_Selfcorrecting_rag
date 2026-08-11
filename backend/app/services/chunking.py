def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Naive fixed-size chunker with overlap. Good enough for a prototype —
    swap for a semantic/markdown-aware chunker once this is past the learning stage."""
    clean = text.replace("\r\n", "\n").strip()
    if len(clean) <= chunk_size:
        return [clean]

    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = end - overlap
    return chunks
