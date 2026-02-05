# Agent-router

This repo provides a minimal agent routing stack with registry, retrieval, reranking, and orchestration.

## Create a virtual environment

PowerShell:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

bash:

```
python -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```
python -m pip install -r requirements.txt --progress-bar off
```

## End-to-end flow: generate agents → train reranker → query & execute

All commands assume you are in `Agent-router/repo`.

### 1) Generate tools/agents + build index + run demo (writes runs/<task_id>.jsonl)

```
python -m src.execution.demo_runner
```

### 1b) Generate real tools via LLM (themes -> tools -> agents)

Set env vars (PowerShell example):

```
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Then run:

```
python -m src.execution.demo_runner `
  --real_tools `
  --task_text "分析公司季度经营风险并给出建议" `
  --theme_count 4 `
  --tools_per_theme 4 `
  --agents_per_theme 6 `
  --tool_output_dir generated_tools
```

Domain-driven generation (no task_text needed for planning):

```
python -m src.execution.demo_runner `
  --real_tools `
  --domains "finance,health,retail,ops" `
  --tools_per_theme 4 `
  --agents_per_theme 6 `
  --tool_output_dir generated_tools
```

If `--domains` is omitted, a built-in multi-domain list is used (same categories as the batch generator).

Generate one domain at a time (sequential batches):

```
python -m src.execution.demo_runner `
  --real_tools `
  --domain_start 0 `
  --domain_limit 1 `
  --tools_per_theme 3 `
  --agents_per_theme 10 `
  --tool_output_dir generated_tools
```

Control how many tools each agent uses (default = all tools in the theme):

```
python -m src.execution.demo_runner `
  --real_tools `
  --domain_start 0 `
  --domain_limit 1 `
  --tools_per_theme 6 `
  --agents_per_theme 10 `
  --tools_per_agent 3 `
  --tool_output_dir generated_tools
```

Increase LLM timeout/retries if the API is slow:

```
python -m src.execution.demo_runner `
  --real_tools `
  --domain_start 0 `
  --domain_limit 1 `
  --tools_per_theme 3 `
  --agents_per_theme 10 `
  --llm_timeout 180 `
  --llm_retries 4
```

Debug LLM raw responses (saves JSON to runs/llm_debug):

```
python -m src.execution.demo_runner `
  --real_tools `
  --domain_start 0 `
  --domain_limit 1 `
  --tools_per_theme 3 `
  --agents_per_theme 10 `
  --debug_llm
```

Batch large tool counts to avoid truncated JSON:

```
python -m src.execution.demo_runner `
  --real_tools `
  --domain_start 0 `
  --domain_limit 1 `
  --tools_per_theme 50 `
  --tools_batch_size 10 `
  --agents_per_theme 30
```

Notes:
- Uses OpenAI-compatible Chat Completions via `httpx` (see `generation/llm_client.py`).
- You can override API base/model/key via `LLM_API_BASE`, `LLM_MODEL`, `LLM_API_KEY`.
- Generated tool code is stored in the registry and optionally written to `generated_tools/`.
- By default, real-tool generation is domain-driven (uses `--domains`); use `--plan_themes` to derive themes from `--task_text`.

Optional custom demo parameters:

```
@'
from execution.demo_runner import run_demo
result = run_demo(
    task_text="生成样例数据用于训练",
    db_path="demo_registry.sqlite",
    index_dir="index",
    tool_count=20,
    domains=["finance","health","tech","ops"],
    n_per_domain=50,
    roles=["planner","researcher","builder","checker"],
    dim=64,
    seed=7,
    workflow_version="v1",
)
print(result["log_path"])
'@ | python -
```

### 2) Export training data from runs

```
New-Item -ItemType Directory -Force data,models | Out-Null

python -m src.routing.export_training_data `
  --runs runs `
  --db demo_registry.sqlite `
  --out data/train.jsonl
```

### 3) Train reranker

```
python -m src.routing.train_router `
  --data data/train.jsonl `
  --out models/reranker.json `
  --epochs 5 `
  --lr 0.1
```

### 3b) Batch-generate runs + auto-train reranker (one command)

```
python -m src.routing.batch_train `
  --queries_file data/queries.txt `
  --db demo_registry.sqlite `
  --index_dir index `
  --llm_base_url "https://az.gptplus5.com/v1" `
  --llm_model gpt-4o `
  --llm_api_key YOUR_KEY `
  --train_out models/reranker.json
```

### 3c) Generate router SFT data with Qwen (labeler) + LoRA fine-tune

Install training deps:

```
python -m pip install -r requirements-train.txt
```

Generate labeled router data (Qwen decides best agent among candidates):

```
python -m src.routing.generate_router_data `
  --queries_file data/queries.txt `
  --db demo_registry.sqlite `
  --index_dir index `
  --llm_base_url "https://az.gptplus5.com/v1" `
  --llm_model qwen3-8b `
  --llm_api_key YOUR_KEY `
  --out data/router_sft.jsonl
```

LoRA fine-tune Qwen on router data:

```
python -m src.routing.train_router_lora `
  --data data/router_sft.jsonl `
  --model Qwen/Qwen3-8B-Instruct `
  --output_dir models/router_lora `
  --epochs 1 `
  --batch_size 1 `
  --grad_accum 8 `
  --lr 1e-4 `
  --use_4bit
```

Iterative training (run multiple rounds, reusing the latest reranker each round):

```
python -m src.routing.batch_train `
  --queries_file data/queries.txt `
  --db demo_registry.sqlite `
  --index_dir index `
  --llm_base_url "https://az.gptplus5.com/v1" `
  --llm_model gpt-4o `
  --llm_api_key YOUR_KEY `
  --train_out models/reranker.json `
  --iterations 3
```

### 4) Query → match agents + collaboration mode → execute workflow

```
@'
import os
from core.registry import SQLiteRegistry
from retrieval.faiss_index import HNSWIndex
from retrieval.embedder import DummyEmbedder
from execution.orchestrator import run_workflow

db_path = "demo_registry.sqlite"
index_path = os.path.join("index","faiss.index")

index = HNSWIndex.load(index_path)
embedder = DummyEmbedder(dim=64, seed=7)

with SQLiteRegistry(db_path) as registry:
    result = run_workflow(
        task_text="请分析公司季度经营风险，并给出可执行建议",
        roles=["planner","researcher","builder","checker"],
        constraints_per_role={
            "researcher": {"domain_tags": ["finance"]},
            "builder": {"tool_tags": ["basic_stats"]},
        },
        workflow_version="v1",
        registry=registry,
        index=index,
        embedder=embedder,
        top_n=20,
        top_k=5,
        rerank_top_m=3,
        mmr_lambda=0.5,
        reranker_model_path="models/reranker.json",
        bandit_db_path="models/bandit.sqlite",
    )

print(result["answer"])
print("log_path:", result["log_path"])
'@ | python -
```

### Rebuild index after adding/updating agents

```
python -m src.retrieval.build_index --db demo_registry.sqlite --kind agent --out index --dim 64
```

### 5) Direct CLI: run a single query

```
python -m src.execution.run_query --query "请分析公司季度经营风险，并给出可执行建议"
```

Enable tool auto-fill (LLM fills missing tool params before execution):

```
python -m src.execution.run_query `
  --query "Analyze quarterly risk" `
  --auto_fill `
  --auto_fill_timeout 30
```

Use real LLM for role outputs:

```
python -m src.execution.run_query `
  --query "Analyze quarterly risk" `
  --llm_base_url "https://az.gptplus5.com/v1" `
  --llm_model gpt-4o `
  --llm_api_key YOUR_KEY
```

Auto-install common Python libraries before executing tools:

```
python -m src.execution.run_query `
  --query "Analyze quarterly risk" `
  --auto_install_common_libs
```

Increase tool execution timeout (default 1s):

```
python -m src.execution.run_query `
  --query "Analyze quarterly risk" `
  --tool_timeout 5
```

Print tool execution results:

```
python -m src.execution.run_query `
  --query "Analyze quarterly risk" `
  --print_tools
```

Optional flags (examples):

```
python -m src.execution.run_query `
  --query "分析财务风险" `
  --db demo_registry.sqlite `
  --index_dir index `
  --roles "planner,researcher,builder,checker" `
  --constraints "{\"researcher\":{\"domain_tags\":[\"finance\"]}}" `
  --reranker_model models/reranker.json `
  --bandit_db models/bandit.sqlite
```

