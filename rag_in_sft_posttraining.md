# RAG 在 SFT / 后训练 中的顶会工作综述 (2023-2025)

> 本文梳理近两年（2023-2025）顶级会议中，将 **检索增强生成（RAG）** 集成到 **监督微调（SFT）/ 后训练（Post-training）** 阶段的关键工作。

---

## 一、RAG 与 SFT 深度融合的后训练方法

### 1.1 RAFT: Retrieval Augmented FineTuning
- **论文**：*RAFT: Adapting Language Model to Domain Specific RAG*
- **作者**：Tianjun Zhang (UC Berkeley) et al.
- **会议**：**COLM 2024**
- **arXiv**：[2403.10131](https://arxiv.org/abs/2403.10131)
- **核心思路**：
  - 在 SFT 时**同时给模型喂检索到的正文档（oracle docs）和干扰文档（distractor docs）**，训练模型在已知领域中区分相关与不相关的检索信息
  - 类比「开卷考试」：SFT 是死记硬背（闭卷），传统 RAG 是不加甄别地看所有资料（开卷但不筛选），RAFT 是教会模型在开卷时学会辨别
  - 训练数据格式：Question + 多篇 retrieved documents（含正例和负例）+ Answer (CoT reasoning)
- **关键贡献**：首次将 RAG 文档直接编码到 SFT 训练数据中，使模型学会 selective reading

---

### 1.2 RA-DIT: Retrieval-Augmented Dual Instruction Tuning
- **论文**：*RA-DIT: Retrieval-Augmented Dual Instruction Tuning*
- **作者**：Xi Victoria Lin, Xilun Chen (Meta AI) et al.
- **会议**：**ICLR 2024**
- **arXiv**：[2310.01352](https://arxiv.org/abs/2310.01352)
- **核心思路**：
  - **双阶段调优**：① 更新 LLM 使其学会利用检索到的知识（Language Model Fine-Tuning）→ ② 更新 Retriever 使其更符合 LM 偏好（Retriever Fine-Tuning）
  - LM Fine-Tuning 阶段：用 (query, retrieval, answer) 三元组做 SFT，让 LM 学会有条件地利用检索内容
  - Retriever Fine-Tuning：以 LM 偏好为 reward signal 优化检索器
- **关键贡献**：提出 retrieval-augmented instruction tuning 的标准范式；实验证明 LM + retriever 协同调优显著优于各自独立优化

---

### 1.3 Self-RAG: Self-Reflective Retrieval-Augmented Generation
- **论文**：*Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*
- **作者**：Akari Asai, Zeqiu Wu (UW / Allen AI) et al.
- **会议**：**ICLR 2024** (Oral)
- **arXiv**：[2310.11511](https://arxiv.org/abs/2310.11511)
- **核心思路**：
  - 在 SFT 数据中引入**特殊反思 token**（`<Retrieve>`, `<IsRel>`, `<IsSup>`, `<IsUse>`），训练 LM 学会：
    1. 自适应判断何时需要检索
    2. 评估检索结果的相关性/支撑性/有用性
    3. 根据评估结果修正生成内容
  - 用 Critic 模型生成带有反思标注的 SFT 训练数据
- **关键贡献**：将 retrieval decision 和 self-critique 显式编码为训练 token，实现 on-demand retrieval + self-reflection

---

### 1.4 InstructRetro: Instruction Tuning after Retrieval-Augmented Pretraining
- **论文**：*InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining*
- **作者**：Boxin Wang, Wei Ping (NVIDIA) et al.
- **会议**：**ICML 2024**
- **arXiv**：[2310.07713](https://arxiv.org/abs/2310.07713)
- **核心思路**：
  - 先进行 **Retrieval-Augmented Continue Pretraining**（Retro ← chunked cross-attention），再进行 Instruction Tuning
  - 探究不同规模的 Retro-fitting 策略：从零预训练 vs 从标准 LM 继续训练
  - 证明 retrieval-augmented pretraining + instruction tuning 的组合显著优于纯 pretrain 或纯 retrieve-at-inference
- **关键贡献**：系统性研究检索增强预训练与指令微调的配合关系，提供 Scaling 分析

---

### 1.5 RAG-DDR: Differentiable Data Rewards for End-to-End RAG Optimization
- **论文**：*RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards*
- **作者**：(匿名 → GitHub: OpenMatch/RAG-DDR)
- **会议**：**ICLR 2025**
- **OpenReview**：[Pnktu2PBXD](https://openreview.net/forum?id=Pnktu2PBXD)
- **核心思路**：
  - 指出现有 RAG 的 instruction tuning 方法会导致 **各模块过拟合训练信号**，忽略模块间的数据偏好差异
  - 提出 DDR 方法：用 rollout 采样潜在响应作为扰动，评估对整个 RAG 系统性能的影响，通过可微分奖励端到端训练所有 RAG 组件
  - 使生成模块更有效地从文档中提取关键信息，缓解参数化记忆与外部知识间的冲突
- **关键贡献**：首个真正端到端的 RAG 后训练优化方法，对齐 retriever-generator 数据偏好

---

### 1.6 HIRAG: Hierarchical-Thought Instruction-Tuning RAG
- **论文**：*HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation*
- **会议**：**EMNLP 2025** (Findings)
- **arXiv**：[2507.05714](https://arxiv.org/abs/2507.05714)
- **核心思路**：
  - 提出层次化思维链 + RAG 指令微调方法
  - **"先思考再回答"** 策略：利用多层递进式 Chain-of-Thought 增强 open-book 能力
  - 在 SFT 中嵌入层次化推理路径，多层思维粒度由粗到细
- **关键贡献**：将层次化 CoT 推理引入 RAG 指令微调，提升模型利用检索知识进行复杂推理的能力

---

### 1.7 InstructRAG: Self-Synthesized Rationales for RAG Denoising
- **论文**：*InstructRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales*
- **作者**：Zhepei Wei et al.
- **会议**：**EMNLP 2024** / **ICLR 2025** submission
- **核心思路**：
  - 让 LM 显式学习如何从检索文档中「去噪」提取答案
  - 生成自合成的 rationale（推理路径），解释 ground-truth 答案如何从检索文档中推导
  - 这些 rationale 可作为 **in-context 示例** 或 **SFT 训练数据** 来训练模型
- **关键贡献**：用 self-synthesized rationale 桥接检索文档与答案生成，实现 denoising 能力的可训练化

---

## 二、检索增强指令数据构造方法

### 2.1 SAIL: Search-Augmented Instruction Learning
- **论文**：*SAIL: Search-Augmented Instruction Learning*
- **作者**：Hongyin Luo, Yung-Sung Chuang (MIT CSAIL) et al.
- **会议**：**EMNLP 2023**
- **arXiv**：[2305.15264](https://arxiv.org/abs/2305.15264)  
- **核心思路**：
  - 在指令微调时，**用搜索引擎实时检索相关文本**，将其嵌入 instruction prompt
  - 训练时模型学习利用检索结果的模式，推理时同样用搜索增强
  - 无需人工标注，自动构造搜索增强的 instruction 数据
- **关键贡献**：首次将 search engine 检索融入 instruction tuning pipeline

---

### 2.2 RAG-Instruct: 多样化 RAG 指令数据合成
- **论文**：*RAG-Instruct: Learning to Generate Diverse and High-Quality RAG Instruction Data*
- **作者**：FreedomIntelligence 团队
- **会议**：**EMNLP 2025**
- **GitHub**：[FreedomIntelligence/RAG-Instruct](https://github.com/FreedomIntelligence/RAG-Instruct)
- **核心思路**：
  - 基于任意源语料，合成多样化高质量 RAG 指令数据
  - 定义 **五种 RAG 范式**（不同的 query-document 关系），增强模型跨任务泛化
  - Instruction Simulation 技术丰富指令多样性
- **关键贡献**：系统化的 RAG instruction 数据合成框架

---

## 三、组件级 RAG 微调方法

### 3.1 REPLUG: Retrieval-Augmented Black-Box Language Models
- **论文**：*REPLUG: Retrieval Augments Black-Box Language Models*
- **作者**：Weijia Shi et al.
- **会议**：**NAACL 2024**
- **arXiv**：[2301.12652](https://arxiv.org/abs/2301.12652)
- **核心思路**：
  - 将 retriever 视为 LM 的插拔组件，以 **LM 作为无监督监督信号** 来训练 retriever
  - 将多个 retrieved documents prepend 到输入，对 LM 输出做 ensemble
  - 训练一个小模型作为 retriever（通过 LM likelihood 评分来更新）
- **关键贡献**：提出以 LM 自身反馈训练 retriever 的范式，适用于黑盒 API 模型

---

### 3.2 RECOMP: Compression + Selective Augmentation
- **论文**：*RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation*
- **作者**：Fangyuan Xu et al.
- **会议**：(已被广泛引用)
- **arXiv**：[2310.04408](https://arxiv.org/abs/2310.04408)
- **核心思路**：
  - 训练 **Extractive Compressor** 和 **Abstractive Compressor**，将检索文档压缩为简洁摘要再注入 context
  - 训练目标是使 LM 在压缩后仍保持最终任务性能
  - 支持 **Selective Augmentation**：当检索到的文档完全无关时输出空字符串
- **关键贡献**：通过训练 compressor 模型，在压缩率低至 6% 时仍保持性能

---

### 3.3 CRAG: Corrective Retrieval Augmented Generation
- **论文**：*Corrective Retrieval Augmented Generation*
- **作者**：Shi-Qi Yan et al.
- **会议**：2024
- **arXiv**：[2401.15884](https://arxiv.org/abs/2401.15884)
- **核心思路**：
  - 训练 **Retrieval Evaluator** 评估检索质量，对低质量检索做自动化纠正
  - 引入 Web Search 作为 fallback 知识源（当 static retrieval 质量不足时）
  - 将所有检索结果重组为结构化知识后输入生成器
- **关键贡献**：将 retriever evaluation + correction 显式训练进 pipeline，实现鲁棒的检索纠错

---

### 3.4 ARES: 自动化 RAG 评估（通过微调）
- **论文**：*ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems*
- **作者**：Jon Saad-Falcon, Omar Khattab (Stanford) et al.
- **会议**：**NeurIPS 2024**
- **arXiv**：[2311.09476](https://arxiv.org/abs/2311.09476)
- **核心思路**：
  - 用 LLM 自动生成合成训练数据，**微调轻量级分类器**评估 RAG 的 Context Relevance、Answer Faithfulness、Answer Relevance
  - 完全无需人工标注
- **关键贡献**：用微调小模型做 RAG 三轴评估，大幅降低评估成本

---

### 3.5 Finetune-RAG: 微调 LLM 抵御 RAG 幻觉
- **论文**：*Fine-Tuning Language Models to Resist Hallucination in RAG*
- **作者**：(2025)
- **arXiv**：[2505.10792](https://arxiv.org/abs/2505.10792)
- **核心思路**：
  - 用**现实不完美检索数据**构造 SFT，训练模型在检索结果质量不佳时仍不产生幻觉
  - 教模型在检索信息不充分时说 "我不知道"，而非编造
- **关键贡献**：针对性微调使 RAG 系统更鲁棒，降低检索失败导致的幻觉

---

### 3.6 Training LMs to Generate Text with Citations via Fine-grained Rewards
- **论文**：*Training Language Models to Generate Text with Citations via Fine-grained Rewards*
- **作者**：Chengyu Huang, Zeqiu Wu et al.
- **会议**：**ACL 2024**
- **核心思路**：
  - 训练 LLM 在生成答案时**准确标注内联引用（in-text citations）**
  - 使用细粒度奖励（Fine-grained Rewards）监督引用质量
- **关键贡献**：将 citation generation 能力通过 SFT + 奖励训练到 LLM 中

---

## 四、RAG + 推理增强的结合

### 4.1 RAT: Retrieval Augmented Thoughts
- **论文**：*RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation*
- **作者**：Zihao Wang et al.
- **会议**：2024
- **arXiv**：[2403.05313](https://arxiv.org/abs/2403.05313)
- **核心思路**：
  - 在生成每一步 Chain-of-Thought 后**检索相关信息迭代修正该步思考**
  - 大幅缓解长程生成中的幻觉问题
  - 支持 Code Generation、Math Reasoning、Creative Writing、Embodied Planning
- **关键贡献**：将 step-by-step retrieval 融入 CoT 推理过程

---

### 4.2 PlanRAG: Plan-then-Retrieval
- **论文**：*PlanRAG: A Plan-then-Retrieval Augmented Generation for LLMs as Decision Makers*
- **作者**：Myeonghwa Lee, Min-Soo Kim et al.
- **会议**：**NAACL 2024**
- **arXiv**：[2406.12430](https://arxiv.org/abs/2406.12430)
- **核心思路**：
  - 先规划决策方案（Plan），再根据计划生成检索查询（Retrieval）
  - 提出 DQA（Decision QA） Benchmark
  - 迭代式 plan→retrieve→analyze→decide 循环
- **关键贡献**：将 RAG 与决策规划相结合，先计划再检索的模式

---

## 五、RAG vs SFT 对比分析

### 5.1 Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs
- **论文**：*Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs*
- **作者**：Oded Ovadia, Menachem Brief et al.
- **会议**：**EMNLP 2024**
- **arXiv**：[2312.05934](https://arxiv.org/abs/2312.05934)
- **核心发现**：
  - 系统比较 fine-tuning 和 RAG 两种知识注入方法
  - **RAG 在大多数知识密集型任务上持续优于 fine-tuning**，无论是对已知知识还是全新知识
  - LLM 很难通过 unsupervised fine-tuning 学会新事实，需在训练时重复暴露同一事实的多种变体
- **关键贡献**：为 fine-tuning vs RAG 之争提供了系统性实证证据

---

## 六、论文全景表

| 论文 | 会议 | 年份 | 核心思路 |
|------|------|------|----------|
| **RAFT** | COLM | 2024 | SFT 数据中混合正/负检索文档，教会模型区分相关信息 |
| **RA-DIT** | ICLR | 2024 | LM-Retriever 双阶段指令调优 |
| **Self-RAG** | ICLR (Oral) | 2024 | 训练反思 tokens 实现按需检索与自我批判 |
| **InstructRetro** | ICML | 2024 | Retrieval-Augmented Pretrain → Instruction Tuning |
| **RAG-DDR** | ICLR | 2025 | 可微分数据奖励，端到端对齐 RAG 模块偏好 |
| **HIRAG** | EMNLP (Findings) | 2025 | 层次化 CoT + RAG 指令微调 |
| **InstructRAG** | EMNLP | 2024 | 自合成 Rationale 去噪 SFT 训练 |
| **SAIL** | EMNLP | 2023 | 搜索引擎检索融入 Instruction Tuning |
| **RAG-Instruct** | EMNLP | 2025 | 五种 RAG 范式合成多样化指令数据 |
| **REPLUG** | NAACL | 2024 | 以 LM likelihood 训练 retriever |
| **RECOMP** | — | 2023 | 训练 compressor 压缩检索文档 |
| **CRAG** | — | 2024 | 训练 Retrieval Evaluator 评估+纠正检索 |
| **ARES** | NeurIPS | 2024 | 微调分类器做自动化 RAG 评估 |
| **Finetune-RAG** | — | 2025 | 不完美检索下的抗幻觉 SFT |
| **Citation Tuning** | ACL | 2024 | 细粒度奖励训练 LLM 正确标注引用 |
| **RAT** | — | 2024 | 逐步 CoT + 检索修正 |
| **PlanRAG** | NAACL | 2024 | 先规划再检索的决策生成 |
| **FT vs RAG** | EMNLP | 2024 | RAG 系统性优于 Fine-tuning 做知识注入 |

---

## 七、趋势总结

1. **从推理时增强到训练时融合**：RAG 不再只是 inference-time 的 prompt 拼接，而是被深度整合进 SFT/RL 后训练阶段
2. **协同优化替代独立模块**：RA-DIT、RAG-DDR 等方法表明 retriever + generator 需要联合优化
3. **数据质量与多样性意识**：RAG-Instruct 等工作强调合成数据的覆盖度和多样性对训练效果至关重要
4. **鲁棒性与抗幻觉**：Finetune-RAG、CRAG 等方法关注检索失败时的退化控制
5. **可解释性增强**：Self-RAG、Citation Tuning、InstructRAG 等方法让模型学会解释其检索决策
6. **Scaling 律的探究**：InstructRetro 等研究了检索增强在不同模型规模下的效果变化

> *延伸阅读推荐*：[RAG Survey (Gao et al., 2024)](https://arxiv.org/abs/2402.19473) 提供了全面的 RAG 技术综述。
