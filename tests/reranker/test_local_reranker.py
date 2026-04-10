"""Tests for LocalReranker (unit via stub + optional slow integration)."""

import numpy as np
import pytest

from zotero_arxiv_daily.reranker.local import LocalReranker


def test_local_reranker(config, monkeypatch):
    class _Similarity:
        def __init__(self, value):
            self._value = value

        def numpy(self):
            return self._value

    class _FakeSentenceTransformer:
        def __init__(self, model_name, trust_remote_code):
            assert model_name == config.reranker.local.model
            assert trust_remote_code is True

        def encode(self, texts, **kwargs):
            return np.array([[float(i + 1)] for i, _ in enumerate(texts)])

        def similarity(self, s1_feature, s2_feature):
            return _Similarity(s1_feature @ s2_feature.T)

    pytest.importorskip("sentence_transformers")
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        _FakeSentenceTransformer,
    )

    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)


@pytest.mark.slow
def test_local_reranker_integration(config):
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)
