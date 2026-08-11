from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_generation_model: str = "llama-3.3-70b-versatile"
    groq_grader_model: str = "llama-3.1-8b-instant"

    jina_api_key: str = ""
    jina_embedding_model: str = "jina-embeddings-v3"
    jina_vector_size: int = 1024

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "documents"

    cors_origins: str = "http://localhost:5173"

    class Config:
        env_file = ".env"


settings = Settings()
