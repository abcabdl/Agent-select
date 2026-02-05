from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.constraints import Constraints, filter_candidates
from core.query_builder import build_role_query
from core.registry import SQLiteRegistry
from retrieval.embedder import BaseEmbedder, build_embedder
from retrieval.faiss_index import HNSWIndex
from retrieval.mmr import mmr_rerank
from retrieval.build_index import build_index


class CandidateRequest(BaseModel):
    task_text: str
    role: str
    constraints: Optional[Dict[str, Any]] = None
    kind: str = Field(default="agent")
    top_n: int = Field(default=50, ge=1)
    top_k: int = Field(default=10, ge=1)
    mmr_lambda: float = Field(default=0.5)


class CandidateResponse(BaseModel):
    card_id: str
    score: float
    brief_tags: Dict[str, Any]


def _brief_tags(card) -> Dict[str, Any]:
    return {
        "domain_tags": card.domain_tags,
        "role_tags": card.role_tags,
        "tool_tags": card.tool_tags,
        "modalities": card.modalities,
        "output_formats": card.output_formats,
    }


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix if norm == 0.0 else matrix / norm
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _card_text(card) -> str:
    if card.embedding_text:
        return card.embedding_text
    name = card.name or ""
    desc = card.description or ""
    text = f"{name} {desc}".strip()
    return text or name or desc


def get_candidates(
    task_text: str,
    role: str,
    constraints: Optional[Dict[str, Any]],
    kind: str,
    top_n: int,
    top_k: int,
    mmr_lambda: float,
    registry: SQLiteRegistry,
    index: HNSWIndex,
    embedder: BaseEmbedder,
) -> List[Dict[str, Any]]:
    constraints_obj = Constraints(**constraints) if constraints else None
    query_text = build_role_query(task_text, role, constraints_obj)
    query_vec = embedder.embed([query_text])

    if constraints_obj is not None:
        with SQLiteRegistry(registry.db_path) as local_registry:
            cards = local_registry.list({"kind": kind})
        filtered_cards = filter_candidates(cards, constraints_obj)
        if not filtered_cards:
            return []

        candidate_texts = [_card_text(card) for card in filtered_cards]
        candidate_vecs = embedder.embed(candidate_texts).astype(np.float32)
        query_vec_f = query_vec.astype(np.float32)
        query_norm = _l2_normalize(query_vec_f)[0]
        candidate_norms = _l2_normalize(candidate_vecs)
        query_scores = candidate_norms @ query_norm

        if top_n and len(filtered_cards) > top_n:
            top_idx = np.argsort(-query_scores)[:top_n]
            filtered_cards = [filtered_cards[i] for i in top_idx]
            candidate_vecs = candidate_vecs[top_idx]
            query_scores = query_scores[top_idx]

        selected_indices, selected_scores = mmr_rerank(
            query_vec_f[0],
            candidate_vecs,
            lambda_param=mmr_lambda,
            top_k=top_k,
        )

        responses: List[Dict[str, Any]] = []
        for rank_idx, candidate_idx in enumerate(selected_indices):
            card = filtered_cards[candidate_idx]
            score = float(query_scores[candidate_idx]) if candidate_idx < len(query_scores) else float(
                selected_scores[rank_idx]
            )
            responses.append(
                {
                    "card_id": card.id,
                    "score": score,
                    "brief_tags": _brief_tags(card),
                }
            )
        return responses

    scores, id_rows = index.search(query_vec, top_k=top_n)
    ids = id_rows[0]
    score_row = scores[0]

    candidates_list = []
    with SQLiteRegistry(registry.db_path) as local_registry:
        for idx, card_id in enumerate(ids):
            if card_id is None:
                continue
            card = local_registry.get(card_id)
            if card is None:
                continue
            if card.kind != kind:
                continue
            candidates_list.append((card, float(score_row[idx])))

    filtered_cards = [card for card, _ in candidates_list]
    if constraints_obj is not None:
        filtered_cards = filter_candidates(filtered_cards, constraints_obj)

    if not filtered_cards:
        return []

    score_map = {card.id: score for card, score in candidates_list}
    candidate_texts = [_card_text(card) for card in filtered_cards]
    candidate_vecs = embedder.embed(candidate_texts)

    selected_indices, selected_scores = mmr_rerank(
        query_vec[0],
        candidate_vecs,
        lambda_param=mmr_lambda,
        top_k=top_k,
    )

    responses: List[Dict[str, Any]] = []
    for rank_idx, candidate_idx in enumerate(selected_indices):
        card = filtered_cards[candidate_idx]
        score = score_map.get(card.id, float(selected_scores[rank_idx]))
        responses.append(
            {
                "card_id": card.id,
                "score": score,
                "brief_tags": _brief_tags(card),
            }
        )
    return responses


def create_app(
    registry: SQLiteRegistry,
    index: HNSWIndex,
    embedder: BaseEmbedder,
) -> FastAPI:
    app = FastAPI()

    @app.post("/candidates", response_model=List[CandidateResponse])
    def candidates(request: CandidateRequest) -> List[CandidateResponse]:
        payload = get_candidates(
            task_text=request.task_text,
            role=request.role,
            constraints=request.constraints,
            kind=request.kind,
            top_n=request.top_n,
            top_k=request.top_k,
            mmr_lambda=request.mmr_lambda,
            registry=registry,
            index=index,
            embedder=embedder,
        )
        return [CandidateResponse(**item) for item in payload]

    return app


def _build_default_app() -> FastAPI:
    try:
        db_path = os.getenv("SEARCH_DB", "registry.sqlite")
        index_dir = os.getenv("SEARCH_INDEX_DIR", "./index")
        dim = int(os.getenv("SEARCH_DIM", "256"))
        seed = int(os.getenv("SEARCH_SEED", "7"))

        index_path = os.path.join(index_dir, "faiss.index")

        registry = SQLiteRegistry(db_path)
        if os.path.exists(index_path):
            index = HNSWIndex.load(index_path)
        else:
            index = build_index(db_path=db_path, kind="agent", out_dir=index_dir, dim=dim, seed=seed)
        embedder_kind = os.getenv("SEARCH_EMBEDDER", "sentence-transformer")
        embedder_model = os.getenv("SEARCH_EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embedder_device = os.getenv("SEARCH_EMBEDDER_DEVICE", None)
        embedder_normalize = os.getenv("SEARCH_EMBEDDER_NORMALIZE", "false").lower() in {"1", "true", "yes"}
        embedder = build_embedder(
            kind=embedder_kind,
            dim=dim,
            seed=seed,
            model_name=embedder_model,
            device=embedder_device,
            normalize=embedder_normalize,
        )
        if embedder.dim != index.dim:
            raise ValueError(
                f"Embedder dim {embedder.dim} != index dim {index.dim}. "
                "Rebuild index with matching embedder/model."
            )
        return create_app(registry=registry, index=index, embedder=embedder)
    except Exception:
        app = FastAPI()

        @app.get("/health")
        def health() -> Dict[str, Any]:
            return {"ok": False}

        return app


app = _build_default_app()
