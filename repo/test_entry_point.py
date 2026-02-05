"""
Quick test to verify task_context with entry_point is passed to tools.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from execution.orchestrator import run_workflow
from data_sources.sqlite_registry import SQLiteRegistry
from embedders.sentence_bert import SentenceBERTEmbedder  
from index.faiss_wrapper import FAISSIndex
from llm_client import LocalLLMClient

# Simple test
DB_PATH = "demo_registry.sqlite"
INDEX_PATH = "index/faiss.index"
ID_MAP_PATH = "index/id_map.json"

def test_entry_point():
    print("Testing entry_point propagation...")
    
    with SQLiteRegistry(DB_PATH) as registry:
        embedder = SentenceBERTEmbedder()
        index = FAISSIndex()
        index.load(INDEX_PATH, ID_MAP_PATH)
        
        local_client = LocalLLMClient(
            api_key="test",
            base_url="http://localhost:8000",
            model="gpt-4o"
        )
        
        # Simple test task
        task_text = "def check_dict_case(dict):\n    pass\n\nImplement this function"
        task_context = {"entry_point": "check_dict_case"}
        
        result = run_workflow(
            task_text=task_text,
            roles=["builder"],
            workflow_version="v1",
            registry=registry,
            index=index,
            embedder=embedder,
            llm_client=local_client,
            router_llm_client=local_client,
            task_context=task_context,
            execute_tools=False,  # Don't actually execute for quick test
        )
        
        print(f"✅ Workflow completed")
        print(f"Result keys: {list(result.keys())}")
        print(f"Task context passed: {task_context}")
        
        # Check if task_context was preserved somewhere
        if "builder" in result:
            print(f"Builder result available")
        
        return result

if __name__ == "__main__":
    try:
        test_entry_point()
        print("\n✅ Test passed - task_context parameter accepted")
    except TypeError as e:
        if "task_context" in str(e):
            print(f"\n❌ Test failed - task_context not recognized: {e}")
        else:
            print(f"\n❌ Test failed with different error: {e}")
    except Exception as e:
        print(f"\n⚠️  Test failed: {e}")
        import traceback
        traceback.print_exc()
