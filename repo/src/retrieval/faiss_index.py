from __future__ import annotations

import pickle
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except ImportError:  # pragma: no cover - optional dependency
    faiss = None
    _HAS_FAISS = False


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix if norm == 0.0 else matrix / norm
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class HNSWIndex:
    """HNSW wrapper with cosine similarity via L2 normalize + inner product."""

    def __init__(
        self,
        dim: int,
        m: int = 32,
        ef_search: int = 128,
        ef_construction: int = 200,
        use_faiss: Optional[bool] = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.m = m
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.ids: List[str] = []
        self._use_faiss = _HAS_FAISS if use_faiss is None else bool(use_faiss) and _HAS_FAISS
        self._index = None
        self._vectors: Optional[np.ndarray] = None
        if self._use_faiss:
            self._index = self._create_faiss_index()

    def _create_faiss_index(self):
        index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = self.ef_search
        index.hnsw.efConstruction = self.ef_construction
        return index

    def build(self, vectors: np.ndarray, ids: List[str]) -> None:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            raise ValueError("Vector dimension mismatch")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids length must match vectors")

        vectors = _l2_normalize(vectors.astype(np.float32))
        self.ids = list(ids)

        if self._use_faiss:
            self._index = self._create_faiss_index()
            self._index.add(vectors)
            self._vectors = None
        else:
            self._vectors = vectors.copy()

    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            raise ValueError("Vector dimension mismatch")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids length must match vectors")

        vectors = _l2_normalize(vectors.astype(np.float32))
        if not self.ids:
            self.build(vectors, ids)
            return

        if self._use_faiss:
            if self._index is None:
                self._index = self._create_faiss_index()
            self._index.add(vectors)
        else:
            if self._vectors is None:
                self._vectors = vectors.copy()
            else:
                self._vectors = np.vstack([self._vectors, vectors])
        self.ids.extend(ids)

    def search(self, queries: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[List[Optional[str]]]]:
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.shape[1] != self.dim:
            raise ValueError("Query dimension mismatch")
        if not self.ids:
            raise ValueError("Index is empty")

        queries = _l2_normalize(queries.astype(np.float32))
        top_k = min(top_k, len(self.ids))

        if self._use_faiss:
            scores, indices = self._index.search(queries, top_k)
        else:
            if self._vectors is None:
                raise ValueError("Index vectors missing")
            scores = queries @ self._vectors.T
            indices = np.argpartition(-scores, range(top_k), axis=1)[:, :top_k]
            row_indices = np.arange(scores.shape[0])[:, None]
            sorted_order = np.argsort(-scores[row_indices, indices], axis=1)
            indices = indices[row_indices, sorted_order]
            scores = scores[row_indices, indices]

        id_lists: List[List[Optional[str]]] = []
        for row in indices:
            mapped = [self.ids[i] if i >= 0 else None for i in row]
            id_lists.append(mapped)
        return scores, id_lists

    def save(self, path: str) -> None:
        if self._use_faiss:
            faiss.write_index(self._index, path)
            return
        payload = {
            "dim": self.dim,
            "m": self.m,
            "ef_search": self.ef_search,
            "ef_construction": self.ef_construction,
            "vectors": self._vectors,
            "ids": list(self.ids),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(
        cls,
        path: str,
        use_faiss: Optional[bool] = None,
        m: int = 32,
        ef_search: int = 128,
        ef_construction: int = 200,
    ) -> "HNSWIndex":
        use_faiss = _HAS_FAISS if use_faiss is None else bool(use_faiss) and _HAS_FAISS
        if use_faiss:
            index = faiss.read_index(path)
            obj = cls(
                dim=index.d,
                m=m,
                ef_search=ef_search,
                ef_construction=ef_construction,
                use_faiss=True,
            )
            obj._index = index
            return obj
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(
            dim=int(payload["dim"]),
            m=int(payload.get("m", m)),
            ef_search=int(payload.get("ef_search", ef_search)),
            ef_construction=int(payload.get("ef_construction", ef_construction)),
            use_faiss=False,
        )
        obj._vectors = payload.get("vectors")
        obj.ids = list(payload.get("ids") or [])
        return obj
