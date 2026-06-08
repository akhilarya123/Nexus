"""
tests/unit/test_embeddings.py
------------------------------
Unit tests for the LocalEmbedder.
Requires sentence-transformers installed but NO Docker services.

Run with: pytest tests/unit/test_embeddings.py -v
Note: First run downloads ~80MB model — subsequent runs are instant.
"""

import pytest
import numpy as np
from nexus.epistemic.embeddings import LocalEmbedder, get_embedder


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    """Single embedder instance shared across all tests in this module."""
    return get_embedder()


class TestLocalEmbedder:

    def test_embed_returns_correct_dimension(self, embedder):
        vec = embedder.embed("hello world")
        assert len(vec) == 384

    def test_embed_returns_list_of_floats(self, embedder):
        vec = embedder.embed("test sentence")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_embed_is_unit_normalised(self, embedder):
        vec = embedder.embed("some text")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_empty_string_returns_zero_vector(self, embedder):
        vec = embedder.embed("")
        assert len(vec) == 384
        assert all(v == 0.0 for v in vec)

    def test_whitespace_only_returns_zero_vector(self, embedder):
        vec = embedder.embed("   \n\t  ")
        assert all(v == 0.0 for v in vec)

    def test_identical_texts_give_identical_vectors(self, embedder):
        v1 = embedder.embed("postgres database connection")
        v2 = embedder.embed("postgres database connection")
        assert v1 == v2

    def test_different_texts_give_different_vectors(self, embedder):
        v1 = embedder.embed("postgres database")
        v2 = embedder.embed("kubernetes deployment yaml")
        assert v1 != v2

    def test_similar_texts_have_high_similarity(self, embedder):
        v1 = embedder.embed("connect to postgres database")
        v2 = embedder.embed("establish connection to postgresql")
        sim = embedder.similarity(v1, v2)
        assert sim > 0.7, f"Expected high similarity, got {sim:.3f}"

    def test_unrelated_texts_have_lower_similarity(self, embedder):
        v1 = embedder.embed("kubernetes pod deployment")
        v2 = embedder.embed("baking chocolate chip cookies recipe")
        sim = embedder.similarity(v1, v2)
        assert sim < 0.6, f"Expected low similarity, got {sim:.3f}"

    def test_embed_batch_matches_individual(self, embedder):
        texts = [
            "postgres connection timeout",
            "kubernetes pod restart",
            "redis cache miss",
        ]
        batch_vecs = embedder.embed_batch(texts)
        assert len(batch_vecs) == len(texts)
        for text, batch_vec in zip(texts, batch_vecs):
            single_vec = embedder.embed(text)
            # Allow tiny floating point differences
            diff = np.linalg.norm(np.array(batch_vec) - np.array(single_vec))
            assert diff < 1e-4, f"Batch vs single mismatch for '{text}': diff={diff}"

    def test_embed_batch_empty_input(self, embedder):
        result = embedder.embed_batch([])
        assert result == []

    def test_embed_batch_large_input(self, embedder):
        texts = [f"log line number {i} with some content" for i in range(150)]
        vecs = embedder.embed_batch(texts)
        assert len(vecs) == 150
        assert all(len(v) == 384 for v in vecs)

    def test_singleton_returns_same_instance(self):
        a = get_embedder()
        b = get_embedder()
        assert a is b

    def test_similarity_with_zero_vector(self, embedder):
        v1 = [0.0] * 384
        v2 = embedder.embed("some text")
        sim = embedder.similarity(v1, v2)
        assert sim == 0.0  # should not crash, just return 0
