"""Configuration for the Custom MCP (Knowledge Management) server."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    MCP_SERVER_PORT: int = Field(default=8004, description="Port for the Custom MCP server")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    RESEARCH_DB_PATH: str = Field(default="data/research.db", description="Path to SQLite database")
