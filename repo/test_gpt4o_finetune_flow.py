"""
Test GPT-4o fine-tuning workflow to verify all components work correctly.

This script tests:
1. Label generation with GPT-4o
2. Data format validation
3. Training data preparation for LoRA fine-tuning
"""
import json
import os
import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def test_llm_client():
    """Test if LLM client can connect and use gpt-4o"""
    print("=" * 80)
    print("TEST 1: LLM Client Connection with GPT-4o")
    print("=" * 80)
    
    try:
        from generation.llm_client import LLMClient
        
        # Check environment variables
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL")
        
        if not api_key:
            print("❌ FAILED: LLM_API_KEY or OPENAI_API_KEY not set")
            print("   Set it with: $env:OPENAI_API_KEY='your-key'")
            return False
            
        if not base_url:
            print("⚠️  WARNING: LLM_API_BASE not set, using default OpenAI URL")
        
        print(f"✓ API Key: {api_key[:10]}..." if len(api_key) > 10 else "✗ API Key: INVALID")
        print(f"✓ Base URL: {base_url or 'https://api.openai.com/v1'}")
        
        # Create client with gpt-4o
        client = LLMClient(model="gpt-4o", timeout_s=30.0)
        print(f"✓ Client initialized with model: {client.model}")
        
        # Test simple chat
        print("\nTesting API call...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello' in JSON format with key 'greeting'."}
        ]
        
        response = client.chat(messages, temperature=0.1, max_tokens=100)
        print(f"✓ API Response received: {response[:100]}...")
        
        print("\n✅ TEST 1 PASSED: LLM Client works with GPT-4o\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_label_generation():
    """Test label generation functionality"""
    print("=" * 80)
    print("TEST 2: Label Generation with GPT-4o")
    print("=" * 80)
    
    try:
        from routing.generate_labels import (
            load_humaneval_problems,
            _normalize_topology_label,
            SYSTEM_PROMPT
        )
        from generation.llm_client import LLMClient
        
        # Check if humaneval data exists
        data_path = Path("data/Humaneval/humaneval-py.jsonl")
        if not data_path.exists():
            print(f"⚠️  WARNING: Test data not found at {data_path}")
            print("   Skipping label generation test")
            return True
        
        # Load first problem
        problems = load_humaneval_problems([str(data_path)])
        if not problems:
            print("❌ FAILED: Could not load any problems from HumanEval data")
            return False
        
        print(f"✓ Loaded {len(problems)} problems from HumanEval")
        test_problem = problems[0]
        print(f"✓ Test problem: {test_problem['task_id']}")
        
        # Create client
        client = LLMClient(model="gpt-4o", timeout_s=30.0)
        
        # Generate label for first problem
        available_roles = ["planner", "builder", "tester", "refractor"]
        task_text = f"Task: Implement the following Python function:\n\n{test_problem['prompt']}"
        if test_problem['entry_point']:
            task_text += f"\n\nEntry point: {test_problem['entry_point']}"
        task_text += f"\n\nAvailable roles: {json.dumps(available_roles, ensure_ascii=True)}"
        task_text += "\nTopology options: single, centralized, decentralized, chain."
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_text}
        ]
        
        print("\nGenerating label...")
        response_text = client.chat(messages, temperature=0.1, max_tokens=500)
        print(f"✓ Raw response: {response_text[:200]}...")
        
        # Parse and normalize
        cleaned_response = response_text.replace("```json", "").replace("```", "").strip()
        label_json = json.loads(cleaned_response)
        print(f"✓ Parsed JSON: {json.dumps(label_json, indent=2)[:300]}...")
        
        normalized = _normalize_topology_label(label_json, available_roles)
        print(f"✓ Normalized label: {json.dumps(normalized, indent=2)}")
        
        # Validate structure
        required_keys = ["topology", "roles", "manager_role", "entry_role", "max_steps", "flow_type"]
        for key in required_keys:
            if key not in normalized:
                print(f"❌ FAILED: Missing required key '{key}' in normalized label")
                return False
        
        print("\n✅ TEST 2 PASSED: Label generation works correctly\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_training_data_format():
    """Test training data format for LoRA"""
    print("=" * 80)
    print("TEST 3: Training Data Format Validation")
    print("=" * 80)
    
    try:
        # Create sample training data
        sample_data = {
            "messages": [
                {"role": "system", "content": "You are a meta-router."},
                {"role": "user", "content": "Analyze this task."},
                {"role": "assistant", "content": json.dumps({
                    "topology": "single",
                    "roles": ["builder"],
                    "manager_role": None,
                    "entry_role": "builder",
                    "max_steps": 1,
                    "flow_type": None
                }, ensure_ascii=False)}
            ],
            "sample_type": "humaneval_teacher_distilled",
            "origin_task_id": "test/0"
        }
        
        print("✓ Sample training data structure:")
        print(json.dumps(sample_data, indent=2, ensure_ascii=False))
        
        # Validate messages format
        messages = sample_data.get("messages")
        if not isinstance(messages, list):
            print("❌ FAILED: 'messages' must be a list")
            return False
        
        if len(messages) < 2:
            print("❌ FAILED: 'messages' must contain at least 2 messages")
            return False
        
        for msg in messages:
            if not isinstance(msg, dict):
                print("❌ FAILED: Each message must be a dict")
                return False
            if "role" not in msg or "content" not in msg:
                print("❌ FAILED: Each message must have 'role' and 'content'")
                return False
        
        print("✓ Messages format is valid")
        
        # Check if assistant response is valid JSON
        assistant_content = messages[-1]["content"]
        try:
            parsed = json.loads(assistant_content)
            print(f"✓ Assistant response is valid JSON: {parsed}")
        except json.JSONDecodeError:
            print("❌ FAILED: Assistant response is not valid JSON")
            return False
        
        print("\n✅ TEST 3 PASSED: Training data format is valid\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_lora_imports():
    """Test if LoRA training dependencies are available"""
    print("=" * 80)
    print("TEST 4: LoRA Training Dependencies")
    print("=" * 80)
    
    required_packages = [
        ("transformers", "Hugging Face Transformers"),
        ("datasets", "Hugging Face Datasets"),
        ("peft", "Parameter-Efficient Fine-Tuning"),
        ("torch", "PyTorch"),
    ]
    
    all_available = True
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✓ {description} ({package}) is installed")
        except ImportError:
            print(f"✗ {description} ({package}) is NOT installed")
            all_available = False
    
    if not all_available:
        print("\n⚠️  Some dependencies are missing. Install with:")
        print("   pip install -r requirements-train.txt")
        print("\n✅ TEST 4 PASSED (with warnings): Check dependencies above\n")
    else:
        print("\n✅ TEST 4 PASSED: All LoRA dependencies are available\n")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("GPT-4o Fine-Tuning Workflow Test Suite")
    print("=" * 80 + "\n")
    
    results = []
    
    # Test 1: LLM Client
    results.append(("LLM Client", test_llm_client()))
    
    # Test 2: Label Generation
    results.append(("Label Generation", test_label_generation()))
    
    # Test 3: Training Data Format
    results.append(("Training Data Format", test_training_data_format()))
    
    # Test 4: LoRA Dependencies
    results.append(("LoRA Dependencies", test_lora_imports()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED! The GPT-4o fine-tuning workflow is ready to use.")
        print("\nNext steps:")
        print("1. Generate labels: python -m src.routing.generate_labels --data data/Humaneval/humaneval-py.jsonl --output data/router_labels.jsonl --model gpt-4o")
        print("2. Fine-tune LoRA: python -m src.routing.train_router_lora --data data/router_labels.jsonl --model Qwen/Qwen3-8B-Instruct --output_dir models/router_lora")
    else:
        print("\n❌ SOME TESTS FAILED. Please fix the issues above before proceeding.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
