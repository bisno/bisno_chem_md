# 预训练 → SFT → RL 阶段间分布对齐与数据筛选工作综述

> **范围**：2024–2026 年顶会论文（NeurIPS, ICML, ICLR, ACL, EMNLP, AAAI, CVPR 等）  
> **聚焦**：预训练(Pretraining)、监督微调(SFT)、强化学习(RL/RLHF/DPO) 三者两两之间的 (1) 分布对齐 与 (2) 数据筛选  
> **整理日期**：2026-05-07

---

## 目录

1. [Pretraining ↔ SFT](#1-pretraining--sft)
   - [1.1 分布对齐](#11-分布对齐)
   - [1.2 数据筛选](#12-数据筛选)
2. [Pretraining ↔ RL](#2-pretraining--rl)
   - [2.1 分布对齐](#21-分布对齐)
   - [2.2 数据筛选](#22-数据筛选)
3. [SFT ↔ RL](#3-sft--rl)
   - [3.1 分布对齐](#31-分布对齐)
   - [3.2 数据筛选](#32-数据筛选)
4. [总结与趋势](#4-总结与趋势)

---

## 1. Pretraining ↔ SFT

### 1.1 分布对齐

#### 1.1.1 Entropic Distribution Matching for Supervised Fine-tuning of LLMs
- **会议**：NeurIPS 2024
- **作者**：—
- **链接**：[NeurIPS 2024](https://neurips.cc/virtual/2024/107589)
- **核心思想**：指出交叉熵(CE)损失在 SFT 中因其激进的分布匹配策略（迫使模型生成分布紧密拟合经验数据分布）而导致过拟合和输出多样性受限。提出用**熵正则化的分布匹配**替代标准 CE 损失，在保持任务性能的同时提升生成多样性。
- **关键词**：distribution matching, entropy regularization, SFT loss design, overfitting mitigation

#### 1.1.2 Amuro & Char: Analyzing the Relationship between Pre-Training and Fine-Tuning of LLMs
- **会议**：ACL 2025 (RepL4NLP Workshop)
- **链接**：[ACL Anthology](https://aclanthology.org/2025.repl4nlp-1.11.pdf)
- **核心思想**：系统分析预训练与微调阶段模型行为的关系，研究从预训练到微调的过渡中表征和能力的演化，揭示预训练阶段的数据/任务如何影响 SFT 后的对齐效果。
- **关键词**：pretraining-finetuning relationship, capability transfer, representation analysis

#### 1.1.3 Tracing the Representation Geometry of Language Models from Pretraining to Post-training
- **作者**：Melody Zixuan Li, Kumar Krishna Agrawal, Arna Ghosh, Komal Kumar Teru, Adam Santoro, Guillaume Lajoie, Blake A. Richards
- **链接**：[arXiv 2025.09](https://arxiv.org/search/?query=Tracing+the+Representation+Geometry+of+Language+Models)
- **核心思想**：采用谱方法（RankMe、特征谱）追踪从预训练到后训练阶段表征几何的变化，揭示分布对齐在表征层面的几何学表现。
- **关键词**：representation geometry, effective rank, spectral analysis, pretraining-to-posttraining

#### 1.1.4 The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning (URIAL)
- **会议**：ICLR 2024
- **链接**：[OpenReview](https://openreview.net/forum?id=wxJ0eXwwda)
- **核心思想**：证明通过策略性提示和上下文学习(ICL)可显著缩小无调优方法与有调优方法之间的对齐差距。对"对齐调优一定要改变模型分布"这一假设提出质疑，表明 prompt-level distribution bridging 是有效的替代路径。
- **关键词**：in-context alignment, tuning-free alignment, prompt-based distribution bridging

#### 1.1.5 The Best Instruction-Tuning Data are Those That Fit
- **链接**：[arXiv 2025.02](https://arxiv.org/search/?query=The+Best+Instruction-Tuning+Data+are+Those+That+Fit)
- **核心思想**：研究什么样的 SFT 数据是最有效的——发现与 base model 的预训练分布"拟合得好"的数据（而非绝对质量最高）往往带来更好的 SFT 效果，即预训练→SFT 的数据分布匹配度比数据本身的质量更重要。
- **关键词**：data-model fit, distribution compatibility, SFT data quality

#### 1.1.6 Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, and Training Factors Shape LLM Alignment Quality
- **会议**：EMNLP 2025 (Main)
- **作者**：Yuto Harada et al.
- **链接**：[arXiv 2506.14681](https://arxiv.org/abs/2506.14681) | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1138/)
- **核心思想**：在 1000+ 个 SFT 模型上系统实验，发现**困惑度(perplexity)**一致地预测 SFT 效果，超越训练数据与基准之间的表面相似度；中间层的权重变化与性能增益最相关。这为预训练到 SFT 的分布差异提供了定量度量。
- **关键词**：perplexity as alignment metric, layer-wise analysis, pretraining-SFT compatibility

---

### 1.2 数据筛选

#### 1.2.1 What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
- **会议**：ICLR 2024 (或相关顶会)
- **作者**：Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, Junxian He
- **链接**：[arXiv 2312](https://arxiv.org/abs/2312.15672)
- **核心思想**：系统研究指令调优中自动数据选择的标准，比较了多种评分指标（复杂度、多样性、质量等），提出了综合性的数据筛选框架——DEITA (Data-Efficient Instruction Tuning for Alignment)。
- **关键词**：data selection metrics, instruction tuning, quality vs diversity, DEITA

#### 1.2.2 Rethinking Data Selection at Scale: Random Selection is Almost All You Need
- **会议**：EMNLP 2024 或 NeurIPS 2024
- **作者**：Tingyu Xia, Bowen Yu, Kai Dang, An Yang, Yuan Wu, Yuan Tian, Yi Chang, Junyang Lin
- **链接**：[arXiv 2410.09335](https://arxiv.org/abs/2410.09335)
- **核心思想**：在百万级 SFT 数据池上验证了几乎所有的 self-scoring 数据选择方法，发现它们的性能与随机选择相当——一个令人深省的发现。指出在 SFT 阶段，数据多样性比单纯追求高质量数据更重要。长度过滤是稳定有效的改进手段。
- **关键词**：large-scale data selection, diversity over quality, random selection baseline

#### 1.2.3 S2L: Scalable Data Selection for Fine-tuning Large Language Models by Summarizing Training Trajectories of Small Models
- **会议**：NeurIPS 2024
- **作者**：Yu Yang, Siddharth Mishra, J. Chiang, Baharan Mirzasoleiman
- **链接**：[arXiv 2403.07384](https://arxiv.org/abs/2403.07384)
- **核心思想**：通过小模型的训练轨迹(summarized training trajectories)来指导大模型的 SFT 数据选择，仅用 11% 数据达到全量数据性能，平均优于 SOTA 方法 4.7%。本质是用小模型的分布变化来预测大模型的数据效用。
- **关键词**：training trajectory, data selection proxy, small-to-large transfer

#### 1.2.4 The Harder The Better: Maintaining Supervised Fine-tuning Generalization with Less but Harder Data
- **作者**：Zhaoyang Shang, Sibo Wei, Jianbin Guo, Rui Zhou, Lifeng Dong, Yin Luo
- **链接**：[arXiv 2025.10](https://arxiv.org/search/?query=The+Harder+The+Better+Maintaining+Supervised+Fine-tuning)
- **核心思想**：提出用更少但更困难的数据进行 SFT 可维持甚至提升泛化能力。"难度"由 base model 在这些样本上的 loss/perplexity 衡量——困难样本本质上是与预训练分布差距更大的样本，选择它们有助于弥补分布 gap。
- **关键词**：hard example selection, generalization, difficulty scoring

#### 1.2.5 IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment
- **作者**：Chenlin Ming et al.
- **链接**：[arXiv 2025.05](https://arxiv.org/search/?query=IDEAL+Data+Equilibrium+Adaptation)
- **核心思想**：针对多能力对齐场景的数据均衡适配方法，通过均衡不同能力维度的数据分布来避免 SFT 中的灾难性遗忘和能力倾斜。
- **关键词**：data equilibrium, multi-capability alignment, balanced data selection

#### 1.2.6 Self-Filter: Your Vision-Language Model Itself Is a Strong Filter
- **会议**：ACL 2024
- **作者**：Ruibo Chen et al.
- **链接**：[arXiv 2402.12501](https://arxiv.org/abs/2402.12501)
- **核心思想**：利用 VLM 自身作为数据过滤器。通过联合训练一个难度评分网络，选择最具挑战性的指令样本，仅用约 15% 数据达到全量数据性能。该方法可迁移到纯文本 LLM。
- **关键词**：self-filtering, difficulty-based selection, VLM instruction tuning

#### 1.2.7 Cream of the Crop: Harvesting Rich, Scalable and Transferable Multi-Modal Data for Instruction Fine-Tuning
- **作者**：Mengyao Lyu et al.
- **链接**：[arXiv 2025.03](https://arxiv.org/search/?query=Cream+of+the+Crop+Harvesting+Rich+Scalable)
- **核心思想**：研究预训练模型如何从大规模多模态数据中筛选最适合 SFT 的样本，假设高质量的种子数据有益于 LLM 训练。
- **关键词**：multi-modal data selection, seed data quality, transferable data

#### 1.2.8 Rethinking Overlooked Aspects in Vision-Language Models
- **作者**：Yuan Liu et al.
- **链接**：[arXiv 2405.11850](https://arxiv.org/abs/2405.11850)
- **核心思想**：发现仅增加预训练数据量并不能保证性能提升，甚至可能导致退化；建立了定位最高效 SFT 数据集的 pipeline，表明并非所有 SFT 数据都是必要的。
- **关键词**：data efficiency, pretraining data scaling, SFT data selection pipeline

#### 1.2.9 Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality Pretraining Data
- **作者**：—
- **链接**：[arXiv 2505.05427](https://arxiv.org/abs/2505.05427)
- **核心思想**：通过验证策略优化正负样本选择，构建高效的数据过滤 pipeline，确保预训练数据的高质量，间接影响后续 SFT 的效果。
- **关键词**：pretraining data filtering, data verification, quality selection

#### 1.2.10 Data Selection for Language Models via Importance Resampling (DSIR)
- **会议**：NeurIPS 2024
- **链接**：—
- **核心思想**：通过重要性重采样从大规模通用语料中选择与目标分布匹配的预训练数据，是一种基于分布对齐的数据筛选方法。
- **关键词**：importance resampling, distribution matching, pretraining data selection

---

## 2. Pretraining ↔ RL

### 2.1 分布对齐

#### 2.1.1 Why Reinforcement Fine-Tuning Enables MLLMs Preserve Prior Knowledge Better: A Data Perspective
- **作者**：Zhihao Zhang et al.
- **链接**：[arXiv 2025.06](https://arxiv.org/search/?query=Why+Reinforcement+Fine-Tuning+Enables+MLLMs+Preserve+Prior+Knowledge)
- **核心思想**：从数据视角分析为什么强化微调(RFT)比 SFT 更能保留预训练的先验知识——RL 优化信号更温和，不像 CE loss 那样激进地改变模型分布，因此预训练→RL 的分布偏移小于预训练→SFT。
- **关键词**：prior knowledge preservation, distribution shift, RFT vs SFT

#### 2.1.2 Reasoning Curriculum: Bootstrapping Broad LLM Reasoning from Math
- **作者**：Bo Pang et al.
- **链接**：[arXiv 2025.10](https://arxiv.org/search/?query=Reasoning+Curriculum+Bootstrapping+Broad+LLM+Reasoning)
- **核心思想**：通过课程学习(curriculum)从数学推理逐步过渡到通用推理的 RL 训练，实现预训练到 RL 阶段的平滑分布过渡。
- **关键词**：curriculum RL, reasoning bootstrapping, distribution curriculum

#### 2.1.3 Leveraging Robust Optimization for LLM Alignment under Distribution Shifts
- **作者**：Zhu et al.
- **链接**：[arXiv](https://www.semanticscholar.org/paper/Leveraging-Robust-Optimization-for-LLM-Alignment-Zhu-Liu/30ca68af36578d238964df1c5c596e33521e00a5)
- **核心思想**：提出一种分布感知的鲁棒优化框架，在偏好对齐(DPO)中处理预训练到 RL 阶段的分布偏移，通过鲁棒优化缓解分布不匹配问题。
- **关键词**：robust optimization, distribution shift, preference alignment, DPO

#### 2.1.4 Alignment as Distribution Learning: Your Preference Model is Explicitly...
- **链接**：[arXiv 2506.01523](https://arxiv.org/pdf/2506.01523)
- **核心思想**：将对齐重新定义为从偏好反馈中的分布学习，显式建模目标语言模型的信息如何通过偏好数据渗透。提供了预训练→RL 阶段分布对齐的理论视角。
- **关键词**：alignment as distribution learning, preference feedback, theoretical distribution analysis

---

### 2.2 数据筛选

#### 2.2.1 ActiveUltraFeedback: Efficient Preference Data Generation using Active Learning
- **作者**：Davit Melikidze et al.
- **链接**：[arXiv 2026.03](https://arxiv.org/search/?query=ActiveUltraFeedback+Efficient+Preference+Data)
- **核心思想**：使用主动学习高效生成偏好数据用于 RLHF。通过主动选择最具信息量的 prompt 来最大化偏好数据的效用，减少预训练→RL 阶段的数据需求。
- **关键词**：active learning, preference data generation, data efficiency

#### 2.2.2 Decomposing the Delta: What Do Models Actually Learn from Preference Pairs?
- **作者**：Chia-Hsuan Lee et al.
- **链接**：[arXiv 2026.04](https://arxiv.org/search/?query=Decomposing+the+Delta+What+Do+Models+Actually+Learn)
- **核心思想**：解构偏好优化（DPO/KTO）中模型从偏好对中学到的内容，分析偏好对的质量与组成如何影响 RL 阶段的知识获取。
- **关键词**：preference pair analysis, DPO learning dynamics, data composition

#### 2.2.3 Prethink-RL: Pre-thinking Before Each Step for Reinforcement Fine-Tuning
- **作者**：—
- **链接**：[arXiv 2025](https://arxiv.org/search/?query=Prethink-RL)
- **核心思想**：在 RL 微调的每一步之前引入"预思考"机制来筛选和准备数据，确保 RL 训练数据的质量和相关性。
- **关键词**：pre-thinking, RL data preparation, step-wise data selection

#### 2.2.4 Crowd-SFT: Crowdsourcing for LLM Alignment
- **会议**：IEEE Conference 2024
- **链接**：[IEEE](https://ieeexplore.ieee.org/document/11143104)
- **核心思想**：通过多模型选择过程改进 RL 对齐的数据收集——基于点系统跟踪个体贡献，以去中心化方式实现对齐数据的筛选。
- **关键词**：crowdsourcing, multi-model selection, decentralized alignment

---

## 3. SFT ↔ RL

### 3.1 分布对齐

#### 3.1.1 Implicit Reward as the Bridge: A Unified View of SFT and DPO Connections
- **会议**：NeurIPS 2025
- **链接**：[NeurIPS 2025 Paper](https://papers.neurips.cc/paper_files/paper/2025/file/6599417a0b34d6ab7836cf86f8dd138c-Paper-Conference.pdf)
- **核心思想**：通过严格数学推导证明 SFT 和 DPO 在相同的最优策略-奖励子空间中运行，SFT 是隐式奖励学习的特例。这为 SFT→RL 的分布对齐提供了统一的理论框架。
- **关键词**：unified theory, implicit reward, SFT-DPO connection, optimal policy subspace

#### 3.1.2 Getting More Juice Out of the SFT Data: Reward Learning from Human Demonstration Improves SFT for LLM Alignment
- **会议**：NeurIPS 2024
- **作者**：Jiaxiang Li, Siliang Zeng, Hoi-To Wai, Chenliang Li, Alfredo Garcia, Mingyi Hong
- **链接**：[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e0c9b65fb3e41aaa86576df3ec33ad2e-Abstract-Conference.html)
- **核心思想**：从 SFT 数据中学习奖励函数（将人类示范转化为偏好信号），实现 SFT 到 RL 的自然过渡。本质上是在 SFT 和 RL 之间架起分布对齐的桥梁——利用 SFT 数据辅助 RL 阶段。
- **关键词**：reward from demonstration, SFT-to-RL bridge, data reuse

#### 3.1.3 Diverse Preference Learning for Capabilities and Alignment
- **会议**：ICLR 2025
- **链接**：[OpenReview](https://openreview.net/forum?id=pOq9vDIYev)
- **核心思想**：讨论 RLHF/DPO 等对齐算法如何降低 LLM 的输出多样性，导致输出过于均质化。提出多样化偏好学习以在 SFT 后保持分布多样性。
- **关键词**：diversity preservation, mode collapse, preference learning diversity

#### 3.1.4 When Weak LLMs Speak with Confidence, Preference Alignment Gets Stronger
- **作者**：Amirabbas Afzali et al.
- **链接**：[arXiv 2026.03](https://arxiv.org/search/?query=When+Weak+LLMs+Speak+with+Confidence)
- **核心思想**：利用弱模型的置信度信号改进偏好对齐——当弱模型对某些偏好判断有信心时，这些信号可以更有效地指导 SFT→RL 的分布调整。
- **关键词**：confidence-guided alignment, weak-to-strong, distribution adjustment

#### 3.1.5 Elo-Evolve: A Co-evolutionary Framework for Language Model Alignment
- **作者**：Jing Zhao et al.
- **链接**：[arXiv 2026.02](https://arxiv.org/search/?query=Elo-Evolve+A+Co-evolutionary+Framework)
- **核心思想**：提出共同进化框架，使 SFT 和 RL 阶段的数据和模型协同进化，动态调整 SFT→RL 的分布过渡。
- **关键词**：co-evolution, SFT-RL co-optimization, dynamic alignment

#### 3.1.6 ORPO: Monolithic Preference Optimization without Reference Model
- **会议**：ACL 2024 / NeurIPS 2024
- **作者**：Jiwoo Hong, Noah Lee, James Thorne
- **核心思想**：将 SFT 和偏好优化合并为单一训练阶段（不需要 reference model），通过 odds ratio 惩罚项直接在 SFT 过程中融合偏好信号，消除 SFT→RL 的阶段间分布跳变。
- **关键词**：combined SFT-preference optimization, reference-free alignment, odds ratio

#### 3.1.7 CPO: Contrastive Preference Optimization
- **会议**：NeurIPS 2024
- **作者**：—
- **核心思想**：对比偏好优化方法，通过对比学习框架统一 SFT 和 RL 的目标，减少两阶段间的分布差异。
- **关键词**：contrastive alignment, unified SFT-RL, distribution consistency

---

### 3.2 数据筛选

#### 3.2.1 Less is More: Improving LLM Alignment via Preference Data Selection
- **会议**：NeurIPS 2025
- **链接**：[GitHub](https://github.com/xiangtanshi/DPO-Data-Selection)
- **核心思想**：提出偏好数据选择方法用于改善 LLM 对齐——并非所有偏好对都同等有用，通过筛选最优偏好数据子集来提升 DPO 效果。核心发现：少量高质量偏好数据优于大量低质量数据。
- **关键词**：preference data selection, quality over quantity, DPO data filtering

#### 3.2.2 M3PO: Multimodal-Model-Guided Preference Optimization for Visual Instruction Following
- **作者**：Rui Gao et al.
- **链接**：[arXiv 2508.12458](https://arxiv.org/abs/2508.12458)
- **核心思想**：提出 M3P-Score 用于智能选择最优偏好样本对，融合了多模态对齐分数(MAS)和模型自洽性置信度，高效利用 DPO 的"高学习价值"样本。
- **关键词**：preference pair scoring, self-consistency, hard negative selection

#### 3.2.3 GRAIL: Graph Edit Distance and Node Alignment using LLM-Generated Code
- **会议**：ICML 2025
- **链接**：[ICML 2025](https://icml.cc/virtual/2025/papers.html)
- **核心思想**：利用 LLM 生成代码进行图结构对齐，提供了一种利用 RL 阶段生成的代码作为 SFT→RL 过渡数据的新途径。
- **关键词**：code generation for alignment, graph alignment, LLM-as-generator

#### 3.2.4 HelpSteer 2: Open-source dataset for training top-performing reward models
- **会议**：NeurIPS 2024 (Datasets & Benchmarks)
- **作者**：Zhilin Wang et al.
- **链接**：[NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/02fd91a387a6a5a5751e81b58a75af90-Abstract-Datasets_and_Benchmarks_Track.html)
- **核心思想**：开源高质量偏好数据集用于训练顶级奖励模型，提供精心筛选的偏好标注数据以改善 SFT→RL 阶段的奖励建模质量。
- **关键词**：reward model dataset, preference annotation quality, open-source benchmark

#### 3.2.5 EventRL: Enhancing Event Extraction with Outcome Supervision for LLMs
- **作者**：Jun Gao et al.
- **链接**：[arXiv 2402.11430](https://arxiv.org/abs/2402.11430)
- **核心思想**：使用结果监督和特定奖励函数，直接比较 RL vs SFT 在事件抽取上的效果。RL 在分布外(out-of-distribution)事件类型上显著优于 SFT。
- **关键词**：outcome supervision, RL vs SFT comparison, OOD generalization

#### 3.2.6 SciInstruct: a Self-Reflective Instruction Annotated Dataset for Training Scientific Language Models
- **会议**：NeurIPS 2024 (Datasets & Benchmarks)
- **作者**：Dan Zhang et al.
- **链接**：[NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/02ee6b7295f720407b56c457b34c54d5-Abstract-Datasets_and_Benchmarks_Track.html)
- **核心思想**：自反思指令标注数据集，通过自我反思机制筛选高质量的 SFT 数据，间接改善 SFT 到 RL 的数据质量。
- **关键词**：self-reflective annotation, scientific domain, data quality

---

## 4. 总结与趋势

### 4.1 分布对齐方向的关键趋势

| 阶段过渡 | 核心挑战 | 代表方法 |
|----------|----------|----------|
| **Pretraining → SFT** | CE loss 过度匹配导致过拟合与多样性丧失；预训练分布与 SFT 数据分布不匹配 | Entropic Distribution Matching (NeurIPS 2024), 数据-模型拟合度优先于绝对质量, Perplexity 作为兼容性度量 |
| **Pretraining → RL** | RL 优化信号需要从预训练分布平稳过渡，避免灾难性遗忘 | Curriculum RL, 鲁棒优化抗分布偏移, RFT 比 SFT 更温和 |
| **SFT → RL** | SFT 与 RL 优化目标不一致；SFT 数据如何辅助 RL 阶段 | 统一理论框架 (NeurIPS 2025), SFT 数据学奖励 (NeurIPS 2024), ORPO/CPO 融合两阶段 |

### 4.2 数据筛选方向的关键趋势

| 阶段过渡 | 核心挑战 | 代表方法 |
|----------|----------|----------|
| **Pretraining → SFT** | 大规模数据池中如何筛选少量高价值 SFT 数据 | 随机选择 baseline 的颠覆性发现、S2L 小模型代理、困难样本优先、数据多样性 > 质量 |
| **Pretraining → RL** | 高效生成/筛选偏好数据以启动 RL 训练 | Active Learning 偏好生成、偏好对质量分析、预思考机制 |
| **SFT → RL** | 偏好数据的质量与数量权衡；SFT 数据如何复用 | Less is More (NeurIPS 2025)、M3P-Score 评分、HelpSteer2 高质量偏好数据集 |

### 4.3 总体趋势

1. **统一框架趋势**：越来越多的研究尝试将 SFT 和 RL 统一在同一理论框架下（如 Implicit Reward as Bridge），消除阶段间的人为分界。
2. **分布感知训练**：从"一刀切"的 CE loss / DPO loss 向分布感知的损失函数演进，考虑预训练→SFT→RL 的分布连续性。
3. **数据效率觉醒**：多个研究（如 Rethinking Data Selection at Scale）表明在足够大规模下，复杂的数据筛选方法未必优于简单基线，多样性和数据-模型匹配度是关键。
4. **从数据质量到数据适配**：关注点从"绝对数据质量"转向"数据与当前阶段模型的适配度"（如 The Best Instruction-Tuning Data are Those That Fit）。
5. **课程化过渡**：通过课程学习实现预训练→SFT→RL 的平滑过渡，而非硬切换。
6. **小模型代理**：用小模型的训练轨迹/困惑度等信号指导大模型的数据筛选（S2L 等）。
7. **RL 阶段数据筛选兴起**：随着 RLHF/DPO/GRPO 的广泛应用，偏好数据的选择、生成、质量控制成为热门方向。

---

> **备注**：本文档基于 arXiv、NeurIPS、ICML、ICLR、ACL/EMNLP、Semantic Scholar 等公开资源整理。部分论文尚未正式出版，请以最终出版版本为准。欢迎补充和纠正。
