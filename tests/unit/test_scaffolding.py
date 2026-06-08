"""
tests/unit/test_scaffolding.py
-------------------------------
Smoke tests — verify the project scaffold is wired correctly
before touching any real infrastructure.
"""

import pytest
from nexus.config.settings import get_settings, NexusSettings


def test_settings_loads():
    """Settings singleton returns a valid NexusSettings object."""
    cfg = get_settings()
    assert isinstance(cfg, NexusSettings)


def test_settings_defaults():
    """All critical defaults are present and sane."""
    cfg = get_settings()
    assert cfg.ollama.model == "gemma3"
    assert cfg.ollama.base_url == "http://localhost:11434"
    assert cfg.neo4j.uri == "bolt://localhost:7687"
    assert cfg.qdrant.host == "localhost"
    assert cfg.redis.host == "localhost"
    assert cfg.mcts.exploration_constant == pytest.approx(1.414)
    assert cfg.mcts.max_horizon == 1000


def test_redis_url():
    """Redis URL is correctly constructed."""
    cfg = get_settings()
    assert cfg.redis.url == "redis://localhost:6379/0"


def test_settings_singleton():
    """get_settings() returns the same object every time (lru_cache)."""
    a = get_settings()
    b = get_settings()
    assert a is b


def test_ollama_client_import():
    """OllamaClient can be imported and instantiated without errors."""
    from nexus.tools.llm_client import OllamaClient, Message
    client = OllamaClient()
    assert client.model == "gemma3"


def test_message_model():
    """Message Pydantic model validates correctly."""
    from nexus.tools.llm_client import Message
    m = Message(role="user", content="Hello, Nexus.")
    assert m.role == "user"
    assert m.content == "Hello, Nexus."
