"""
nexus/config/settings.py
------------------------
Central configuration for the entire Nexus kernel.
All values have sensible local defaults — no paid APIs required.
Override via environment variables or a .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """Settings for local Ollama LLM (replaces OpenAI / Anthropic)."""

    base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # Primary model — gemma3 is the correct Ollama tag for Gemma 3
    # If you pulled a specific variant, set OLLAMA_MODEL=gemma3:12b etc.
    model: str = Field(default="gemma3", alias="OLLAMA_MODEL")

    # Smaller/faster model used for MCTS intermediate rollouts to save time
    fast_model: str = Field(default="gemma3", alias="OLLAMA_FAST_MODEL")

    # Embedding model — runs locally via sentence-transformers (no Ollama needed)
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", alias="OLLAMA_EMBEDDING_MODEL"
    )

    # Generation parameters
    temperature: float = Field(default=0.2, alias="OLLAMA_TEMPERATURE")
    top_p: float = Field(default=0.9, alias="OLLAMA_TOP_P")
    num_ctx: int = Field(default=8192, alias="OLLAMA_NUM_CTX")  # context window
    timeout: int = Field(default=120, alias="OLLAMA_TIMEOUT")   # seconds

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Neo4jSettings(BaseSettings):
    """Graph database — stores hierarchical entity graph."""

    uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    password: str = Field(default="nexuspassword", alias="NEO4J_PASSWORD")
    database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # Query limits to prevent explosion
    max_depth: int = Field(default=3, alias="NEO4J_MAX_DEPTH")
    max_nodes_per_query: int = Field(default=500, alias="NEO4J_MAX_NODES")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class QdrantSettings(BaseSettings):
    """Vector database — stores episodic execution memory."""

    host: str = Field(default="localhost", alias="QDRANT_HOST")
    port: int = Field(default=6333, alias="QDRANT_PORT")
    grpc_port: int = Field(default=6334, alias="QDRANT_GRPC_PORT")

    # Collection names
    collection_execution_logs: str = Field(
        default="nexus_execution_logs", alias="QDRANT_COLLECTION_LOGS"
    )
    collection_tool_schemas: str = Field(
        default="nexus_tool_schemas", alias="QDRANT_COLLECTION_TOOLS"
    )

    # Vector dimension for all-MiniLM-L6-v2
    vector_size: int = Field(default=384, alias="QDRANT_VECTOR_SIZE")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class RedisSettings(BaseSettings):
    """Redis — Celery broker + MCTS state cache."""

    host: str = Field(default="localhost", alias="REDIS_HOST")
    port: int = Field(default=6379, alias="REDIS_PORT")
    db: int = Field(default=0, alias="REDIS_DB")
    password: str | None = Field(default=None, alias="REDIS_PASSWORD")

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ObservabilitySettings(BaseSettings):
    """OpenTelemetry + Jaeger tracing."""

    enabled: bool = Field(default=True, alias="OTEL_ENABLED")
    service_name: str = Field(default="nexus-kernel", alias="OTEL_SERVICE_NAME")

    # Jaeger OTLP endpoint
    otlp_endpoint: str = Field(
        default="http://localhost:4317", alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )

    # Sampling rate: 1.0 = trace everything (good for dev)
    sample_rate: float = Field(default=1.0, alias="OTEL_SAMPLE_RATE")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class MCTSSettings(BaseSettings):
    """Monte Carlo Tree Search hyperparameters."""

    # UCT exploration constant — higher = more exploration
    exploration_constant: float = Field(default=1.414, alias="MCTS_C")

    # Max tree depth before forced rollout
    max_depth: int = Field(default=50, alias="MCTS_MAX_DEPTH")

    # Simulations per planning step
    num_simulations: int = Field(default=20, alias="MCTS_NUM_SIMULATIONS")

    # Maximum total tool calls before kernel halts
    max_horizon: int = Field(default=1000, alias="MCTS_MAX_HORIZON")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class NexusSettings(BaseSettings):
    """Root settings — aggregates all subsystem configs."""

    # Kernel identity
    kernel_version: str = "0.1.0"
    environment: Literal["development", "production"] = Field(
        default="development", alias="NEXUS_ENV"
    )

    # FastAPI
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_reload: bool = Field(default=True, alias="API_RELOAD")

    # Subsystems (lazily instantiated so imports stay fast)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings
    )
    mcts: MCTSSettings = Field(default_factory=MCTSSettings)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> NexusSettings:
    """
    Returns a cached singleton of NexusSettings.
    Import this function everywhere — never instantiate NexusSettings directly.

    Usage:
        from nexus.config.settings import get_settings
        cfg = get_settings()
        print(cfg.ollama.model)  # "gemma3"
    """
    return NexusSettings()
