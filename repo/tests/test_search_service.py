import importlib
import sys
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def test_search_service_candidates(tmp_path) -> None:
    cards_mod = _import_from_src("core.cards")
    registry_mod = _import_from_src("core.registry")
    build_index_mod = _import_from_src("retrieval.build_index")
    embedder_mod = _import_from_src("retrieval.embedder")
    service_mod = _import_from_src("retrieval.search_service")

    AgentCard = cards_mod.AgentCard
    SQLiteRegistry = registry_mod.SQLiteRegistry
    build_index = build_index_mod.build_index
    DummyEmbedder = embedder_mod.DummyEmbedder
    create_app = service_mod.create_app

    db_path = tmp_path / "search.sqlite"
    registry = SQLiteRegistry(str(db_path))
    for i in range(20):
        card = AgentCard(
            id=f"agent-{i}",
            name=f"Agent {i}",
            kind="agent",
            version="1.0",
            updated_at=datetime(2026, 1, 1),
            domain_tags=["finance" if i % 2 == 0 else "health"],
            role_tags=["researcher"],
            description="test agent",
            embedding_text=f"agent {i} domain finance" if i % 2 == 0 else f"agent {i} domain health",
        )
        registry.register(card)

    out_dir = tmp_path / "index"
    dim = 32
    index = build_index(
        db_path=str(db_path),
        kind="agent",
        out_dir=str(out_dir),
        dim=dim,
        batch_size=10,
        seed=9,
    )
    embedder = DummyEmbedder(dim=dim, seed=9)
    app = create_app(registry=registry, index=index, embedder=embedder)
    client = TestClient(app)

    response = client.post(
        "/candidates",
        json={
            "task_text": "analyze finance trends",
            "role": "researcher",
            "constraints": {"domain_tags": ["finance"]},
            "kind": "agent",
            "top_n": 10,
            "top_k": 5,
            "mmr_lambda": 0.5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) <= 5

    registry.close()
