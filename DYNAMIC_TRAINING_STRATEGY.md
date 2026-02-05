# 动态多智能体系统的 Qwen 微调训练策略 (Dynamic MAS Training Strategy)

这是针对您将系统升级为“动态拓扑”后，如何微调 Qwen 模型（作为 Router 或 Meta-Router）的全面思考与建议。

核心挑战在于：**从单纯的“分类问题”（选哪个 Agent）变成了“规划问题”（用什么阵型 + 选哪些 Agent + 怎么连接）。**

---

## 1. 训练目标重构 (Redefining Objectives)

在固定流程中，你的训练目标是：`Query -> Agent_ID`。
在动态系统中，Qwen 模型需要学习两个层面的能力（建议合二为一或分阶段训练）：

### A. Meta-Router (宏观规划)
*   **输入**: 用户原始 Query。
*   **输出**: 结构化配置 (JSON)，包含拓扑类型、关键角色、甚至执行步骤。
*   **示例**:
    ```json
    {
        "thought": "这是一个复杂的数据分析任务，需要代码编写和审核。",
        "topology": "centralized",
        "manager_role": "ProjectManager",
        "workers": ["PythonCoder", "DataAnalyst"]
    }
    ```

### B. Dynamic Router (微观路由/下一跳)
*   **输入**: Query + **当前对话历史/中间产物**。
*   **输出**: 下一步该调用的 Agent ID 或 "Terminate"。
*   **示例**:
    *   Input: "用户问：分析附件。历史：Coder 已经处理了数据，生成了 result.csv。"
    *   Output: `{"next_action": "call_agent", "agent_id": "ChartPlotter"}`

---

## 2. 数据构建策略 (Data Engineering) - 最关键的一步

没有现成的数据集包含“动态拓扑选择”，由于人工标注成本极高，建议采用 **Model-Generated + Simulation (合成数据 + 仿真筛选)** 的策略。

### 阶段 1: 构造“合成黄金数据” (Synthetic Ground Truth)
利用强大的闭源模型（如 GPT-4o / DeepSeek-V3）作为“教师”，生成训练数据。

1.  **Seed Queries (种子构建)**:
    *   准备 500-1000 个不同复杂度的 Query (简单问答, 多步推理, 创意写作, 代码工程)。
2.  **Oracle Annotation (先知标注)**:
    *   让 GPT-4 充当“架构师”，针对每个 Query 生成最佳的 Agent 组合方案。
    *   *Prompt*: "给定这个任务，请设计一个多智能体协作流程来解决它。是单兵作战好，还是组建一个团队？输出 JSON 配置。"
3.  **Result Filtering (仿真验证)**:
    *   (可选但推荐) 将这些方案在你的系统中实际跑一遍。
    *   如果任务成功且成本/时间合理 -> **正样本**。
    *   如果任务失败或甚至不如单体效果好 -> **负样本** (用于 DPO)。

### 阶段 2: 轨迹收集 (Trajectory Collection)
针对微观路由（Next Agent Prediction），需要记录“状态-动作”对。

*   让系统在现有规则下运行（哪怕是随机探索）。
*   记录成功的链路：`State(t) -> Action(t) -> State(t+1)`。
*   将这些成功的链路切片，转换成 SFT (Supervised Fine-Tuning) 格式的数据。

---

## 3. Qwen 模型微调具体方案

建议使用 **Instruction Tuning (指令微调)** 格式。

### 数据格式示例 (ChatML)

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "你是一个元路由助手。请根据用户请求，选择最优的智能体协作拓扑结构 (single/centralized/decentralized) 并分配角色。"
      },
      {
        "role": "user",
        "content": "帮我写一个贪吃蛇游戏，并写出测试用例，最后打包成exe。"
      },
      {
        "role": "assistant",
        "content": "{\"topology\": \"centralized\", \"reasoning\": \"涉及代码编写、测试和打包多个环节，需要项目经理协调。\", \"roles\": [\"GameDesigner\", \"PythonDev\", \"TestEngineer\", \"Packager\"]}"
      }
    ]
  }
]
```

### 训练流程

1.  **SFT (监督微调)**:
    *   **基座**: Qwen-7B-Chat 或 Qwen-14B-Chat。
    *   **数据量**: 1k - 10k 条高质量 (Query -> Plan) 数据。
    *   **目的**: 让模型学会输出合法的 JSON 格式，并理解不同拓扑的适用场景。

2.  **DPO (直接偏好优化) - 进阶**:
    *   如果 SFT 后模型还在“乱指挥”（例如简单问题非要用复杂团队），使用 DPO 修正。
    *   **数据对**:
        *   $x$: "查询天气"
        *   $y_w$ (Win): `{Single Agent}`
        *   $y_l$ (Lose): `{Centralized Team of 5}` (过度设计，浪费资源)
    *   通过 DPO 告诉模型：**“在能解决问题的前提下，越简单越好”**。

---

## 4. 解决“Context Length”问题

在动态路由中，如果通过“历史记录”来决定下一步，Context 会非常长（包含大量 Agent 的中间代码/输出）。

**建议策略**:
1.  **Summary Token**: 训练 Router 时，不要把完整的 Agent 输出喂给它。而是让上一个 Agent 输出一个 `Summary` 或 `Status`。
2.  **Sliding Window**: 只关注最近 2-3 步的交互。
3.  **Training Sample Truncation**: 构造训练数据时，随机裁剪历史记录，强迫模型根据残缺信息做决策（提高鲁棒性）。

---

## 5. 总结：落地步骤

1.  **冷启动 (Cold Start)**:
    *   不要一开始就上模型训练。
    *   先写 **Heuristics (启发式规则)** 代码：
        *   `if "compare" in query -> Decentralized`
        *   `if code_lines > 100 -> Centralized`
        *   `else -> Single`
    *   用规则跑通流程，收集数据。

2.  **数据积累**:
    *   记录规则系统运行下来的 Log。
    *   根据 Log 中的成功/失败案例，清洗出数据集。

3.  **模型替代 (Model Replacement)**:
    *   用积累的数据微调 Qwen。
    *   用微调后的 Qwen 替换掉规则代码。

4.  **持续迭代 (Active Learning)**:
    *   模型如果对某些 Query 只有 0.6 的置信度，交给人工或更强的模型（API）仲裁，并将仲裁结果加入训练集。
