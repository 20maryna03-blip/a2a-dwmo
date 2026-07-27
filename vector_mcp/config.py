"""Configuration for the Vector MCP server (embeddings + ChromaDB)."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    VECTOR_MCP_PORT: int = Field(default=8006, description="Port for the Vector MCP server")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # HuggingFace embedding model (runs locally via sentence-transformers)
    HF_EMBED_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model for embeddings",
    )

    # ChromaDB storage path (relative to project root)
    VECTOR_DB_PATH: str = Field(
        default="data/vector_db",
        description="Local path for ChromaDB persistent storage",
    )

    # Public API endpoints used to seed the vector DB
    ARXIV_API_URL: str = Field(
        default="https://export.arxiv.org/api/query",
        description="arXiv Atom API base URL",
    )
    WIKIPEDIA_API_URL: str = Field(
        default="https://en.wikipedia.org/api/rest_v1",
        description="Wikipedia REST API base URL",
    )

    # Seed configuration
    SEED_TOPICS: str = Field(
        default="artificial intelligence,machine learning,natural language processing",
        description="Comma-separated topics to auto-seed the vector DB on first start",
    )
    SEED_PAPERS_PER_TOPIC: int = Field(
        default=10,
        description="Number of arXiv papers to fetch per topic during seeding",
    )
