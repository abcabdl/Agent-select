# 前沿 Baseline 推荐 (New Baselines for 2024-2025)

针对您的动态路由与大规模 Agent 检索系统，以下是最新的、极具竞争力的 Baseline，它们比 MetaGPT/ChatDev 更难以战胜，但也更能证明您工作的含金量。

## 1. 动态拓扑与自进化 (Dynamic Topology & Self-Evolution)
**核心竞品**：这些工作直接挑战您的“动态组队”和“工具生成”概念。

### **EvoAgent (ICLR 2025 / arXiv 2024)**
*   **全名**: *EvoAgent: Self-Evolving Agents via Evolutionary Algorithms*
*   **核心机制**: 不仅仅是生成工具，而是像生物进化一样，通过变异（Mutation）和交叉（Crossover）来进化 Agent 的 Prompt 和 工具库。
*   **对比点**: 您的系统是 "Search + Generate"，EvoAgent 是 "Evolve"。对比在长周期任务中，谁的 Agent 更能适应新环境。
*   **GitHub**: `EvoAgentX/EvoAgentX`

### **OASIS (Camel-AI, 2024)**
*   **全名**: *OASIS: Open Agent Social Interaction Simulations*
*   **核心机制**: 专注于百万级 (Million-Scale) 智能体交互。虽然偏向社会模拟，但它处理“大规模 Agent 检索与互动”的技术栈与您高度重合。
*   **对比点**: 您的 "Million-Agent Retrieval" 机制可以直接 PK OASIS 的检索效率和关联准确度。
*   **GitHub**: `camel-ai/oasis`

### **MiRAGE (Submitted to ACL 2025)**
*   **全名**: *MiRAGE: A Multiagent Framework for Generating Multimodal Multihop Question-Answer Dataset*
*   **核心机制**: 专注于多模态、多跳推理的 Agent 协作。验证了 Agent 在处理复杂、非结构化数据时的协作能力。
*   **对比点**: 如果您的系统能处理多模态任务（如 GAIA），MiRAGE 是最新的多模态协作 Baseline。

---

## 2. 也是 "Search-Based" 的竞品 (Search-based Agents)
**核心竞品**：这些工作也认为“与其自己根据 Prompt 扮演，不如去网上/库里找专家”。

### **MindSearch (InternLM, 2024)**
*   **全名**: *MindSearch: An LLM-based Multi-agent Framework of Web Search Engine* (类 Perplexity Pro)
*   **核心机制**: 动态构建搜索图 (Graph)，根据 Query 动态决定还要查什么，调用数以百计的 Search Agent 并行工作。
*   **对比点**: 它的 "Dynamic Graph Construction" 和您的 "Dynamic Routing" 非常像。对比在复杂信息收集任务上的**召回率**和**规划合理性**。
*   **GitHub**: `InternLM/MindSearch`

### **AppAgent / MobileAgent (2024)**
*   **核心机制**: 这些 Agent 不自己写代码，而是学会“操作”现有的 App 或工具。
*   **对比点**: 如果您的系统包含“操作已有工具”的能力，可以对比它们在工具使用上的灵活性。

---

## 3. 企业级/生产级框架 (Production-Grade Frameworks)
**核心竞品**：这些是工业界的最新标准。

### **Claude-Flow / Anthropic Swarms (2024-2025)**
*   **背景**: 随着 Claude 3.5 Sonnet 的代码能力爆发，基于 Claude 的 MCP (Model Context Protocol) 协议构建的 Agent Swarms 非常火。
*   **对比点**: 对比 **Pass@1 代码通过率**。您的系统（可能基于 Qwen/GPT-4）vs 基于 Claude 3.5 的最新工流。

### **Google ADK (Agent Development Kit)**
*   **背景**: Google 最新的官方 Agent 框架，强调 "Code-First" 和 "GenAI"。
*   **对比点**: 架构设计的优雅程度和开发者友好度。

---

## 总结：如何挑选？

如果您想发顶会 (ICLR/NeurIPS/ACL)，建议死磕前两类：
1.  **打 EvoAgent**: 证明您的“检索+生成”比它的“遗传进化”更高效、收敛更快。
2.  **打 MindSearch**: 证明您的通用性更强，不仅仅是做搜索，还能做代码、做数学。
3.  **打 OASIS**: 证明在百万级 Agent 库中，您的检索算法 (Embedding + LoRA Router) 能更精准地找到“对的人”。
