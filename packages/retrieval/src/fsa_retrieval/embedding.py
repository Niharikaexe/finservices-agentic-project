"""Embeddings behind an interface.

Two implementations, and the interface is the point: `HashingEmbedder` runs offline
with no model download and no API key, so the ACL suite and the retrieval tests are
part of the ordinary unit test run. `AzureOpenAIEmbedder` is what production uses.

Swapping them must not change a line of the retriever — that is the same argument as
pgvector versus Qdrant, and it is what makes "we can move providers" a true statement
rather than an aspiration.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(Protocol):
    def fit(self, corpus: list[str]) -> None: ...
    def encode(self, texts: list[str]) -> npt.NDArray[np.float64]: ...
    @property
    def dimensions(self) -> int: ...


class HashingEmbedder:
    """TF-IDF over character and word n-grams, L2-normalised.

    Not a semantic model, and that is stated plainly rather than hidden: it matches on
    lexical overlap. For a policy corpus — where the query is "meals limit" and the
    chunk says "Meals: claims must not exceed" — lexical overlap is most of the signal,
    and it makes every test deterministic and offline.

    The properties that matter for the retriever are the same either way: fixed
    dimensionality, L2-normalised vectors, so cosine similarity is a dot product.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self._dimensions = dimensions
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            ngram_range=(1, 2),
            max_features=dimensions,
            stop_words="english",
        )
        self._fitted = False

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def fit(self, corpus: list[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def encode(self, texts: list[str]) -> npt.NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("embedder not fitted; call fit(corpus) during ingestion")
        matrix = self._vectorizer.transform(texts).toarray().astype(np.float64)
        # Pad to a fixed width so the store's dimensionality never depends on how many
        # distinct terms the corpus happened to contain.
        if matrix.shape[1] < self._dimensions:
            matrix = np.pad(matrix, ((0, 0), (0, self._dimensions - matrix.shape[1])))
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.asarray(matrix / np.maximum(norms, 1e-12), dtype=np.float64)


class AzureOpenAIEmbedder:
    """Production embedder. Not exercised in tests — no key, and no network in CI.

    Cache aggressively in real use: the policy corpus barely changes, so re-embedding
    it on every ingestion run is pure cost (ARCHITECTURE.md §19).
    """

    def __init__(self, endpoint: str, deployment: str, dimensions: int = 1536) -> None:
        self._endpoint = endpoint
        self._deployment = deployment
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def fit(self, corpus: list[str]) -> None:
        """No-op — a hosted embedder has nothing to fit. Present so the two
        implementations are drop-in interchangeable."""
        del corpus

    def encode(self, texts: list[str]) -> npt.NDArray[np.float64]:
        raise NotImplementedError(
            "wire to Azure OpenAI embeddings in M1 deployment; HashingEmbedder covers local and CI"
        )
