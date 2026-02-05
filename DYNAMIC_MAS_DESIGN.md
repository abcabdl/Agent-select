# 动态多智能体结构与路由系统设计 (Dynamic Multi-Agent Topology & Routing)

本文档旨在不修改现有代码库的前提下，为您提供一套完整的**设计思路**，将当前的“固定流程”升级为“动态拓扑”系统。

核心目标：根据 Query 的复杂度与领域，动态决定使用 **单体(Single)**、**中心化(Centralized)** 或 **去中心化(Decentralized)** 结构，并定义它们之间的连接方式。

---

## 1. 总体架构图

```mermaid
graph TD
    UserQuery[用户 Query] --> MetaRouter[Meta-Router (元路由)]
    MetaRouter --> |简单任务| SingleAgent[单体模式]
    MetaRouter --> |复杂分解| Centralized[中心化模式]
    MetaRouter --> |创意/接力| Decentralized[去中心化模式]
    
    subgraph SingleAgent
        Solver[全能助手]
    end
    
    subgraph Centralized
        Manager[项目经理] <--> |分发/聚合| WorkerA[专家 A]
        Manager <--> |分发/聚合| WorkerB[专家 B]
    end
    
    subgraph Decentralized
        NodeA[节点 A] --> |消息传递| NodeB[节点 B]
        NodeB --> |消息传递| NodeC[节点 C]
    end
```

---

## 2. 核心组件设计

### 2.1. Meta-Router (元路由层) - "大脑"

**位置**：在 `run_workflow` 之前引入。
**职责**：分析 Task，输出 **Topology Config (拓扑配置)**。

**实现思路**：
利用 LLM (如 GPT-4/DeepSeek) 分析 Query 的特征：
*   如果是一次性问答 -> **Pattern A (Single)**
*   如果是需要协调多个步骤 (e.g. "先查这个，再算那个，最后写报告") -> **Pattern B (Centralized)**
*   如果是开放式创作 (e.g. "多角度辩论") 或流水线处理 -> **Pattern C (Decentralized/Chain)**

**输出数据结构示例 (JSON)**：
```json
{
    "topology": "centralized",
    "roles": {
        "manager": "ProjectManager",
        "workers": ["DataAnalyst", "PythonCoder"]
    },
    "flow_type": "star"
}
```

### 2.2. Topology Executor (拓扑执行器) - "身体"

**位置**：替代或改造 `orchestrator.py` 中的 `for role in role_list` 线性循环。

#### 模式 A: 单体模式 (Single Agent)
*   **逻辑**：保持现状。
*   **流程**：`retrieve(Agent) -> execute(Agent) -> return`。
*   **适用**：简单查询、工具调用。

#### 模式 B: 中心化模式 (Centralized / Star)
*   **逻辑**：引入 `Loop` 机制。
*   **流程**：
    1.  **Manager** (Routing Agent) 先运行。
    2.  Manager 的输出不是最终结果，而是一个 **Plan/Delegate 指令**。
    3.  系统解析 Manager 的指令，提取出“下一步给谁”。
    4.  调用子 Agent (Worker) 并将结果返回给 Manager (Context Update)。
    5.  重复直至 Manager 输出 "FINISH"。
*   **连接方式**：
    *   Manager -> Worker: 通过 `Function Call` 或结构化指令 (JSON)。
    *   Worker -> Manager: 通过 `return value` (写入 Context 的 `previous_steps` 字段)。

#### 模式 C: 去中心化模式 (Decentralized / Mesh or Chain)
*   **逻辑**：基于 Graph (图) 的状态机。
*   **流程**：
    1.  定义节点 (Nodes) 和 边 (Edges)。
    2.  每个 Agent 运行后，查看“路由表”决定将消息发给哪个邻居。
    3.  或者由 Agent 自身在输出末尾指定 `@NextAgent` (类似聊天群组)。
*   **连接方式**：
    *   **Shared Blackboard (黑板模式)**: 所有 Agent 读写同一个 `context` 字典 (您现有的 `results` 字典就是这种雏形)。
    *   **Message Passing**: Agent A 的 `output` 直接作为 Agent B 的 `input`。

---

## 3. 具体实现路径建议 (Roadmap)

### 第一步：定义拓扑描述语言 (TDL)

在 `src/core` 下定义一种用于描述任务结构的数据类。

```python
# 伪代码思路
class TopologyType(Enum):
    SINGLE = "single"
    CENTRALIZED = "centralized"
    CHAIN = "chain"

class WorkflowConfig:
    type: TopologyType
    roles: List[str]  # 参与的角色列表
    entry_point: str  # 第一个执行的角色
    max_steps: int    # 防止死循环
```

### 第二步：改造 `orchestrator.py`

将原本写死的 `role_list = [...]` 逻辑抽象化。

**原逻辑**：
```python
role_list = ["planner", "researcher", "builder", "checker"]
for role in role_list:
   # ... run agent ...
```

**新逻辑 (伪代码)**：
```python
# 1. 先决定怎么跑
config = meta_router.plan(task_text) 

# 2. 根据模式产生执行器
if config.type == TopologyType.CENTRALIZED:
    result = run_centralized_loop(config, task_text)
elif config.type == TopologyType.SINGLE:
    result = run_single_agent(config, task_text)
else:
    result = run_chain(config, task_text)

# --- 实现细节: 中心化 Loop ---
def run_centralized_loop(config, task_text):
    history = []
    manager_output = run_agent(config.manager_role, history)
    
    while not manager_output.is_done():
        # Manager 决定下一个是谁
        next_role = manager_output.get_next_role()
        worker_output = run_agent(next_role, task=manager_output.instruction)
        
        # 结果回传
        history.append({
            "from": next_role,
            "result": worker_output
        })
        
        # Manager 根据 Worker 结果再次决策
        manager_output = run_agent(config.manager_role, history)
```

### 第三步：Agent 间的“硬连接”与“软连接”

**硬连接 (Hard-coded)**：
*   在 `run_chain` 函数里写死顺序：`A -> B -> C`。
*   简单，适合固定 SOP (标准作业程序)。

**软连接 (Soft/Semantic)**：
*   利用 Embeddings 或 LLM 动态路由。
*   在此模式下，Agent A 完成工作后，系统计算 `output` 的 Embedding，去向量库匹配“最适合处理这个结果的下一个 Agent”。
*   这就利用了您现在的 `routing` 模块！
    *   `Top-N Agent Retrieval` 不仅仅用于选第一个人，也可以用于**选下一个人**。

---

## 4. 总结与建议

1.  **复用现有资产**：您现有的 `context["upstream"]` 其实就是一个很好的通信总线。
2.  **由简入繁**：
    *   先实现 **"Pattern Selector"** (Meta-Router)：只区分 "简单(Single)" 和 "复杂(Fixed Chain)"。
    *   再实现 **"Centralized Loop"**：让 Planner 变成真正的 Manager，可以多次调用 Builder。
    *   最后尝试 **"Dynamic Chain"**：Agent A 的输出作为 Query，去 Retrieve Agent B。

这个设计能够让您的 Agent 库“活”起来，从单向流水线变成真正的多智能体协作系统。
