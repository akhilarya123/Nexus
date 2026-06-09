"""
tests/conftest.py
------------------
Session-scoped event loop for all tests to avoid "Future attached
to a different loop" errors when using async clients (Neo4j, Qdrant).

This ensures a single `asyncio` event loop is created and shared
across the entire pytest session.
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

    Ensures async clients (Neo4j, Qdrant, httpx) are created and used
    on the same loop to avoid "Future attached to a different loop".
    """
    policy = event_loop_policy
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    try:
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        loop.close()
