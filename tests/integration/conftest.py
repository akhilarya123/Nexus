"""
tests/integration/conftest.py
------------------------------
Pytest configuration for integration tests.

THE FIX for "Future attached to a different loop":
  pytest-asyncio by default creates a NEW event loop for every test.
  The Neo4j AsyncDriver (and its connection pool) binds its socket
  Futures to the loop that was active when connect() was called.
  When the next test runs on a NEW loop, the old Futures become
  invalid → RuntimeError: Future attached to a different loop.

  Solution: force ALL integration tests to share ONE event loop for
  the entire session. This is the standard pattern for async
  integration tests that hold persistent connections.
"""

import asyncio
import pytest


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use the default asyncio policy (required by pytest-asyncio >=0.23)."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def event_loop(event_loop_policy):
    """
    Single event loop shared across the ENTIRE test session.

    This is critical for integration tests because:
    - Neo4j AsyncDriver holds socket connections bound to a specific loop
    - Qdrant AsyncClient uses httpx which also binds to the creation loop
    - Without this, every test class gets a fresh loop, making all
      previously-created async objects invalid (Future attached to
      a different loop)
    """
    policy = event_loop_policy
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    # Graceful shutdown: run pending cleanup tasks
    try:
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        loop.close()