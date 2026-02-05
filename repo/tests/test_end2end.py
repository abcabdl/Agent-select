import importlib
import sys
from pathlib import Path


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def test_demo_runner_end2end(tmp_path, monkeypatch) -> None:
    demo_mod = _import_from_src("execution.demo_runner")
    monkeypatch.chdir(tmp_path)

    result = demo_mod.run_demo(
        task_text="quick test",
        db_path=str(tmp_path / "demo.sqlite"),
        index_dir=str(tmp_path / "index"),
        tool_count=4,
        domains=["finance", "health"],
        n_per_domain=2,
        roles=["planner", "researcher"],
        dim=16,
        seed=3,
        workflow_version="test-v1",
    )

    log_path = result.get("log_path")
    assert log_path is not None
    assert Path(log_path).exists()
