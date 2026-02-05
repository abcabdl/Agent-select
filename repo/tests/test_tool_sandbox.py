import importlib
import sys
from pathlib import Path


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def test_tool_sandbox_execution(tmp_path) -> None:
    registry_mod = _import_from_src("core.registry")
    tool_gen_mod = _import_from_src("generation.tool_generator")
    executor_mod = _import_from_src("execution.tool_executor")

    SQLiteRegistry = registry_mod.SQLiteRegistry
    register_tool = tool_gen_mod.register_tool
    ToolExecutor = executor_mod.ToolExecutor

    db_path = tmp_path / "tools.sqlite"
    with SQLiteRegistry(str(db_path)) as registry:
        card = register_tool(registry, "text_cleaner")
        executor = ToolExecutor(registry, timeout_s=1.0)
        result = executor.run_tool(card.id, {"text": " Hello, World! "})

    assert result["ok"] is True
    assert result["output"]["text"] == "hello world"
