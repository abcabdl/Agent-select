# 针对当前训练与动态架构的建议报告

## 1. 关于损失函数 (Loss Function) 的深度分析

您目前的理解非常精准：
> **当前状态**：本质是 **Causal LM Cross-Entropy (Next-Token Prediction)**。
> **机制**：由 `train_router_lora.py` 通过 mask 掉 system/user prompt，只让模型学习预测 assistant 的输出（即 JSON 字符串）。

### 此方案的优缺点
*   **优点 ✅**：实现极其简单，天然支持 JSON 这种结构化输出，不需要魔改模型结构（加分类头）。
*   **缺点 ❌**：
    *   **权重平均化**：模型预测 `task_type: "single"` 中的 `"single"`（关键决策）时，loss 权重和预测一个逗号 `,` 或引号 `"` 是一样的。模型可能会学会了完美的 JSON 格式，但选错了拓扑。
    *   **缺乏对比**：交叉熵只告诉模型“什么是对的”，没告诉模型“为什么 A 比 B 好”。

### 改进建议 (Actionable Advice)

#### A. 短期方案：保持现状但增强数据 (Data Augmentation)
既然 Loss 改起来麻烦，就在数据侧加强。在 JSON 输出中增加 **"reasoning" (思维链)** 字段。
*   **改前**：`{"topology": "centralized"}`
*   **改后**：`{"analysis": "任务涉及多步依赖，单体难以覆盖...", "topology": "centralized"}`
*   **原理**：强制模型先生成推理过程，利用 CoT (Chain of Thought) 提高后续 `"centralized"`token 的预测准确率。这也是 DeepSeek/OpenAI o1 的核心思路。

#### B. 中期方案：引入 DPO (Direct Preference Optimization)
这是解决“选拓扑”问题的**大杀器**。
*   **不再只做填空题**：SFT（您现在的做法）是教模型“怎么说话”。DPO 是教模型“怎么做选择”。
*   **构建 DPO 数据**：
    *   **Prompt**: 用户 Query
    *   **Chosen (赢家)**: `{Structure: Single}` (虽然简单但省钱且能做对)
    *   **Rejected (输家)**: `{Structure: Centralized}` (杀鸡用牛刀，或过度复杂导致失败)
*   **效果**：DPO 会直接拉大好选项和坏选项在 Logits 层面的差距，比单纯的交叉熵有效得多。

---

## 2. 关于“去中心化软连接”的实现现状

**结论：目前的 `orchestrator.py` 中** **尚未实现** **完全的去中心化软连接。**

### 证据 (Code Evidence)
我在 `src/execution/orchestrator.py` (Line 200+) 中看到的是一个**固定流水线**：

```python
# orchestrator.py
role_list = roles or ["planner", "researcher", "builder", "checker"]
for role in role_list:
    # 1. 检索候选人 (get_candidates)
    # 2. 路由选择 (Select)
    # 3. 执行工具 (Tool Execute)
```

**当前的连接方式是**：
*   **硬连接 (Hard-coded)**：依靠 `roles` 列表的顺序 (`planner` -> `researcher` -> ...)，前一个人的输出 (`results`) 被塞到下一个人的 context 里。
*   **检索逻辑**：虽然有 `get_candidates`，但它是基于 `User Query + Role Name` 去找人，而不是基于 `上一轮 Agent 的输出` 去找人。

### 如何实现“软连接” (How to Implement)

您需要打破 `for role in role_list` 的循环，改成 **While Loop + Dynamic Retrieval**。

**伪代码逻辑建议**：

```python
# 改造 orchestrator.py

current_context = task_text
current_agent_output = ""
history = []

while not is_task_done():
    # --- 软连接核心逻辑 ---
    # 1. 决定下一步要做什么 (Router / LLM Decision)
    #    输入: 历史记录 + 上一步结果
    #    输出: 下一步需要的 "能力描述" (e.g., "需要一个能画图的Agent")
    next_step_query = router_llm.decide_next_step(history, current_agent_output)
    
    if next_step_query == "FINISH":
        break

    # 2. 动态检索 (Soft Connection)
    #    用生成的 "能力描述" 去向量库里找最匹配的 Agent
    candidates = get_candidates(task_text=next_step_query, ...) 
    selected_agent = select_best(candidates)

    # 3. 执行
    current_agent_output = selected_agent.run(current_context)
    history.append(current_agent_output)
```

**关键点**：
*   **从“按角色找人”变为“按需求找人”**。
*   **软连接的本质**：是一个 **Embedding Search** 过程，Query 不再是用户的原始问题，而是**当前解决问题的中间状态**。
