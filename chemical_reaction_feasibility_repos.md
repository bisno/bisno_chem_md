# 化学反应可行性相关开源仓库汇总

> 收集了 GitHub 上与化学反应可行性（逆合成预测、合成路线规划、反应条件推荐、反应产率预测等）相关的开源项目，包含仓库地址及简要描述。

---

## 一、单步逆合成预测

### [kaist-amsg/LocalRetro](https://github.com/kaist-amsg/LocalRetro)
基于局部反应模板的逆合成预测方法，通过对反应中心局部环境的建模来预测逆合成断键位置，具有较好的泛化能力。

### [vsomnath/graphretro](https://github.com/vsnath/graphretro)
基于图神经网络模型的逆合成预测方法（NeurIPS 2021），将反应物和产物表示为分子图，通过图编辑的方式预测逆向合成路线。

### [wangyu-sd/RetroExplainer](https://github.com/wangyu-sd/RetroExplainer)
化学知识引导的可解释逆合成预测模型，在实现高精度逆合成预测的同时，提供决策过程的可解释性，帮助化学家理解模型的预测依据。

### [igashov/RetroBridge](https://github.com/igashov/RetroBridge)
基于马尔可夫桥模型（Markov Bridge）的逆合成预测方法，通过扩散生成模型在产物和反应物之间建立概率映射。

### [coleygroup/rxn-ebm](https://github.com/coleygroup/rxn-ebm)
基于能量的化学选择性建模（Energy-Based Modeling），用于对逆合成候选路径进行打分排序，可结合化学选择性信息进行更准确的反应可行性评估。

---

## 二、多步合成路线规划

### [coleygroup/desp](https://github.com/coleygroup/desp)
双端合成规划（Double-Ended Synthesis Planning），NeurIPS 2024 Spotlight 论文。在指定起始物料和目标分子的约束下进行双向搜索，提出完整的合成路线方案。

### [MolecularAI/aizynthfinder](https://github.com/MolecularAI/aizynthfinder)
基于蒙特卡洛树搜索（MCTS）的逆合成路线规划工具。集成了策略网络和价值网络，支持大规模合成路线搜索，是药物发现和化学合成领域的重要工具。

### [reymond-group/MultiStepRetrosynthesisTTL](https://github.com/reymond-group/MultiStepRetrosynthesisTTL)
使用 Transformer 循环（Transformer-TTL）进行多步逆合成预测，通过迭代式 Transformer 架构进行深度搜索，实现端到端的多步合成路线规划。

### [Laboratoire-de-Chemoinformatique/SynPlanner](https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner)
集成蒙特卡洛树搜索与图神经网络的开源逆合成规划工具。支持数据清洗、反应规则提取、策略/价值网络训练、MCTS 规划、路线可视化等完整流程。

---

## 三、合成规划综合平台

### [ASKCOS/ASKCOS](https://github.com/ASKCOS/ASKCOS)
MIT 开发的计算机辅助合成设计综合软件包（已迁移至 GitLab）。提供逆合成预测、正向反应预测、反应条件推荐、产率预测等多种功能的 Web 服务。

---

## 四、反应预测模型与工具

### [MolecularAI/aizynthmodels](https://github.com/MolecularAI/aizynthmodels)
统一的合成预测模型训练、评估和使用框架。整合了 Chemformer、aizynthtrain、route-distances 等项目的代码，提供训练和评估分子合成预测模型的标准化工具。

### [MolecularAI/Chemformer](https://github.com/MolecularAI/Chemformer)
基于 BART Transformer 的分子预训练模型（JCIM/Nat. Mach. Intell.）。在正向反应预测、逆合成预测、分子优化和性质预测等任务上展示了预训练的优势。

### [rxn4chemistry/rxn-reaction-preprocessing](https://github.com/rxn4chemistry/rxn-reaction-preprocessing)
IBM Research 发布的化学反应数据预处理工具，用于清洗、标准化和格式化化学反应数据，为后续的机器学习模型训练做准备。

### [connorcoley/rdchiral](https://github.com/connorcoley/rdchiral)
RDKit RunReactants 的封装增强库，改进了立体化学处理（J. Cheminform. 2019）。提供 C++ 加速版本（约 10 倍加速），广泛应用于逆合成预测、反应模板提取等流程。

---

## 五、反应条件推荐与产率预测

### [mrodobbe/Rxn-INSIGHT](https://github.com/mrodobbe/Rxn-INSIGHT)
开源的化学反应分类、命名和条件推荐算法（J. Cheminform. 2024）。基于反应相似性和流行度向用户推荐催化剂、溶剂和试剂等反应条件。

### [Coughy1991/Reaction_condition_recommendation](https://github.com/Coughy1991/Reaction_condition_recommendation)
基于机器学习的有机反应条件预测工具（ACS Cent. Sci. 2018），支持预测反应的催化剂、溶剂、试剂和反应温度。

### [schwallergroup/bh-hte-ood](https://github.com/schwallergroup/bh-hte-ood)
Buchwald-Hartwig 反应的分布外预测（OOD Prediction）基准数据集和代码（ChemRxiv 2025）。包含大规模高通量实验数据集，用于评估反应产率预测模型的泛化能力。

### [sustainable-processes/ORDerly](https://github.com/sustainable-processes/orderly)
从 Open Reaction Database (ORD) 中提取和清洗化学反应数据（JCIM 2024）。提供反应产物预测、条件预测、逆合成和产率预测的基准数据集。

---

## 六、反应数据库与评估框架

### [open-reaction-database/ord-data](https://github.com/open-reaction-database/ord-data)
开放式化学反应数据库，由学术界和工业界合作构建（JACS 2021）。包含大量结构化的化学反应数据，支持机器学习在反应预测和合成规划中的应用。

### [open-reaction-database/ord-schema](https://github.com/open-reaction-database/ord-schema)
Open Reaction Database 的数据格式定义和验证工具，提供标准化的反应数据表示方案。

### [ischemist/project-procrustes](https://github.com/ischemist/project-procrustes)
RetroCast：多步逆合成路线统一格式和评估框架。支持 10+ 种逆合成模型输出的标准化适配（AiZynthFinder、Retro*、SynPlanner、ASKCOS 等），实现公平的多模型对比分析。

### [OptiMaL-PSE-Lab/EvalRetro](https://github.com/OptiMaL-PSE-Lab/EvalRetro)
单步逆合成算法的统一评估框架（Digital Discovery 2024）。提供标准化的评估流程，包括 top-k 准确率、逆合成合理性、向前验证等多维度的算法评价指标。

### [reymond-group/CASP-and-dataset-performance](https://github.com/reymond-group/CASP-and-dataset-performance)
计算机辅助合成规划（CASP）工具和反应数据集分析框架（Chemical Science 2019）。提供反应模板提取、模板验证、数据集拆分和性能分析功能。

---

## 七、相关 Schwaller Group 项目

### [schwallergroup/steer](https://github.com/schwallergroup/steer)
基于 LLM 的合成化学推理评估框架，评估大型语言模型在预测化学反应机理步骤方面的能力。

### [schwallergroup/minerva](https://github.com/schwallergroup/minerva)
高通量实验（HTE）反应优化的开源工具，支持自动化反应条件筛选和优化流程。

### [schwallergroup/synthelite](https://github.com/schwallergroup/synthelite)
基于大语言模型（LLM）的逆合成框架，探索使用 LLM 进行合成路线规划的新范式。

### [schwallergroup/mhnn](https://github.com/schwallergroup/mhnn)
分子超图神经网络（Molecular Hypergraph Neural Network），利用超图表示化学反应中的多分子关系，用于反应预测任务。

---

## 汇总表

| 仓库 | 类别 | 核心功能 |
|------|------|----------|
| [kaist-amsg/LocalRetro](https://github.com/kaist-amsg/LocalRetro) | 单步逆合成 | 局部反应模板逆合成预测 |
| [vsomnath/graphretro](https://github.com/vsnath/graphretro) | 单步逆合成 | 图模型逆合成预测 (NeurIPS 2021) |
| [wangyu-sd/RetroExplainer](https://github.com/wangyu-sd/RetroExplainer) | 单步逆合成 | 可解释逆合成预测 |
| [igashov/RetroBridge](https://github.com/igashov/RetroBridge) | 单步逆合成 | 马尔可夫桥逆合成模型 |
| [coleygroup/rxn-ebm](https://github.com/coleygroup/rxn-ebm) | 单步逆合成 | 能量模型逆合成排序 |
| [coleygroup/desp](https://github.com/coleygroup/desp) | 多步路线规划 | 双端合成规划 (NeurIPS 2024) |
| [MolecularAI/aizynthfinder](https://github.com/MolecularAI/aizynthfinder) | 多步路线规划 | MCTS 逆合成路线搜索 |
| [reymond-group/MultiStepRetrosynthesisTTL](https://github.com/reymond-group/MultiStepRetrosynthesisTTL) | 多步路线规划 | Transformer 迭代逆合成 |
| [Laboratoire-de-Chemoinformatique/SynPlanner](https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner) | 多步路线规划 | MCTS+GNN 合成规划 |
| [ASKCOS/ASKCOS](https://github.com/ASKCOS/ASKCOS) | 综合平台 | 计算机辅助合成综合平台 |
| [MolecularAI/aizynthmodels](https://github.com/MolecularAI/aizynthmodels) | 模型训练 | 合成预测模型训练框架 |
| [MolecularAI/Chemformer](https://github.com/MolecularAI/Chemformer) | 反应预测 | 预训练Transformer分子模型 |
| [rxn4chemistry/rxn-reaction-preprocessing](https://github.com/rxn4chemistry/rxn-reaction-preprocessing) | 数据预处理 | 反应数据清洗与标准化 |
| [connorcoley/rdchiral](https://github.com/connorcoley/rdchiral) | 工具库 | 反应立体化学处理 (C++加速) |
| [mrodobbe/Rxn-INSIGHT](https://github.com/mrodobbe/Rxn-INSIGHT) | 条件推荐 | 反应分类与条件推荐 |
| [Coughy1991/Reaction_condition_recommendation](https://github.com/Coughy1991/Reaction_condition_recommendation) | 条件推荐 | ML 反应条件预测 |
| [schwallergroup/bh-hte-ood](https://github.com/schwallergroup/bh-hte-ood) | 产率预测 | BH 反应 OOD 产率预测 |
| [sustainable-processes/ORDerly](https://github.com/sustainable-processes/orderly) | 数据处理 | ORD 数据清洗与基准 |
| [open-reaction-database/ord-data](https://github.com/open-reaction-database/ord-data) | 数据库 | 开放化学反应数据库 |
| [open-reaction-database/ord-schema](https://github.com/open-reaction-database/ord-schema) | 数据库 | ORD 数据模式定义 |
| [ischemist/project-procrustes](https://github.com/ischemist/project-procrustes) | 评估工具 | 多步逆合成统一评估 |
| [OptiMaL-PSE-Lab/EvalRetro](https://github.com/OptiMaL-PSE-Lab/EvalRetro) | 评估工具 | 单步逆合成标准化评估 |
| [reymond-group/CASP-and-dataset-performance](https://github.com/reymond-group/CASP-and-dataset-performance) | 评估工具 | CASP 模板提取与数据分析 |
| [schwallergroup/steer](https://github.com/schwallergroup/steer) | 推理评估 | LLM 反应机理评估 |
| [schwallergroup/minerva](https://github.com/schwallergroup/minerva) | 实验优化 | HTE 反应条件优化 |
| [schwallergroup/synthelite](https://github.com/schwallergroup/synthelite) | LLM方法 | LLM 逆合成框架 |
| [schwallergroup/mhnn](https://github.com/schwallergroup/mhnn) | 反应预测 | 分子超图神经网络 |

---

*更新日期：2026-05-08*
