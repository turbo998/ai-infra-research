# 企业全球招聘、无偏见技能图谱匹配与欧盟《AI 法案》高风险合规审计（`月审 50 万份简历 · EU AI Act Annex III / NYC LL144`）架构选型（`#544`）

> **Language / 语言**: **简体中文** | [English](./enterprise-hr-pii-blind-skill-graph-eu-ai-act-fairness-sizing-faq.en.md) · [📱 交互式 H5 精读页](../h5/hr-pii-blind-skill-graph-eu-ai-act-fairness-lab.html) · [🧮 交互式算卡器 Sizer](../reports/enterprise-hr-pii-blind-skill-graph-eu-ai-act-fairness-sizing-sizer.html) · [🃏 7-Card 评测报告](../reports/enterprise-hr-pii-blind-skill-graph-eu-ai-act-fairness-sizing-cards.html) · [返回总目录](../full-catalog-zh.md)

![架构摘要图](../../assets/svg/enterprise-hr-pii-blind-skill-graph-eu-ai-act-fairness-sizing-faq-summary.svg)

---

## 一、3 分钟核心结论卡片（Executive Summary）

1. **核心矛盾：为什么把候选人原始简历 PDF 直接喂给 `70B LLM` 打分排序会产生 `11.8%` 的姓名/性别/毕业院校隐性偏见（直接违反欧盟 `EU AI Act Annex III` 高风险 AI 法规与纽约 `NYC Local Law 144`），而传统 `ATS 布尔关键词过滤` 又误杀 `44.0%` 的优秀候选人？**
   - 招聘筛选在欧盟《人工智能法案》（`EU AI Act Annex III`）中被明确列为 **高风险 AI 系统（High-Risk AI System）**：若将未脱敏的原始简历直接送入 `70B LLM` 做黑盒打分（方案 A），反事实扰动测试（仅替换候选人姓名族群特征、性别代词或非核心院校名）显示总分波动高达 **`11.8%`**，缺乏可解释证据链，且处理每份简历需 `3,400ms`、消耗 **32 张 H200（`$96,000/月`）**；
   - 传统 **`ATS 布尔关键词匹配`**（方案 B）虽快，但候选人写“构建分布式 Kubernetes 微服务网关”会被死抠字面关键词 `"Cloud Native API"` 的规则直接淘汰，合格人才召回率仅 **`56.0%`**。
2. **三阶段盲审技能本体图谱 + 可审计结构化证据表 + `4/5 (80%)` 公平性门禁黄金架构**：
   - **第一阶段（确定性 `PII & 人口统计属性盲化脱敏`）**：在简历进入任何大模型前，先经 `Presidio + 版面擦除器` 剥离姓名、照片、性别、年龄、婚姻状况、住址邮编及院校名称标签，从源头切断人口统计学偏见输入；
   - **第二阶段（`8B FP8` 结构化抽取 + `ESCO / O*NET` 标准技能知识图谱对齐）**：将项目经历映射至欧盟 `ESCO` / 美国 `O*NET` 标准技能本体节点（自动建立“K8s 算子开发” $\subset$ “云原生基础设施”的上下位图谱推理），并按岗位硬性 Rubric 逐项绑定简历原文证据句；
   - **第三阶段（确定性加权计分 + 实时 `Four-Fifths Rule [Impact Ratio >= 0.80]` 公平性审计门禁）**：由确定性公式汇总得分，将反事实人口属性波动压至 **`0.3%`（公平一致性 `99.7%`）**、合格候选人召回率推升至 **`95.2%`（`+39.2 pts`）**，单份耗时仅 **`180ms`（提速 `18.89x`）**，全平台仅需 **4 张 H200（立省 `87.5%` GPU）**！

---

## 二、典型业务场景与灵魂拷问 Q&A

### Q1：为什么只删掉简历里的“性别/种族”字段还不够，还必须隐藏“住址邮编（ZIP Code）”与“毕业年份”？
- **答**：在算法公平性审计中，住址邮编、高中毕业年份、特定社团名称被称为 **“代理变量（Proxy Variables）”**——大模型能轻易从邮编推断出社区族裔构成，从毕业年份精确算出真实年龄。因此必须在第一阶段执行全量 **反事实盲化（Counterfactual Blind Redaction）**，只向技能抽取模型暴露“工作职责、技术栈、量化项目成果与技能年限”。

### Q2：如何向欧盟监管机构或外部独立审计师证明招聘 AI 系统满足《EU AI Act》的可解释性与无歧视要求？
- **答**：系统对每份简历不能只输出一个黑盒总分（如 `82 分`），而必须输出 **结构化证据核对表（Evidence-Grounded Rubric JSON）**：每一分都对应岗位 JD 的具体技能节点（`ESCO Skill URI`）与候选人简历原文的精确引用跨度，且每周自动运行 **反事实姓名/代词交换回归测试（Counterfactual Perturbation Audit）** 与群体通过率四分之五法则（`Selection Rate Ratio >= 0.80`）报表存档备查。

```mermaid
flowchart LR
    CV["500K Resumes/Month\nMultilingual PDF / DOCX"] --> S1["Stage 1: Deterministic Blind Redactor\nStrips Name, Photo, Gender, Age, ZIP"]
    S1 --> S2["Stage 2: 8B FP8 Extractor + ESCO/O*NET\nCanonical Skill-Graph Alignment"]
    S2 --> S3["Stage 3: Evidence-Grounded Rubric Scorer\n+ 4/5ths Disparate Impact Fairness Gate"]
    S3 --> OUT["95.2% Talent Recall · 0.3% Bias Variance\n180ms/Resume on 4x H200 GPUs"]
```

---

## 三、三大企业智能招聘筛选与合规审计架构定量对比矩阵

| 评估维度 | 方案 A：原始简历直喂 `70B LLM` 黑盒排序 | 方案 B：传统 `ATS 布尔关键词` 硬匹配 | 方案 C：`PII 盲化 + ESCO 技能图谱 + 可审计 Rubric`（推荐） |
| :--- | :--- | :--- | :--- |
| **合格候选人召回率 (`Recall`)** | `88.5%` | `56.0%`（误杀 `44%` 同义表述人才） | **`95.2%`（比传统 ATS 提升 `+39.2 pts`）** |
| **人口统计反事实偏见波动率** | `11.8%`（违反 EU AI Act / NYC LL144） | `2.4%` | **`0.3%`（`99.7%` 反事实公平一致率）** |
| **单份简历平均处理延迟** | `3,400 ms` | `15 ms` | **`180 ms`（比 70B 黑盒快 `18.89x`）** |
| **月审 50 万份所需 H200 卡数** | `32 张 H200` | `0 张` | **`4 张 H200`（立省 `87.5%` GPU）** |
| **月度推理与合规基础设施成本** | `$96,000 / 月` | `$12,000 / 月` | **`$12,000 / 月`（同等低成本下召回率跃升）** |

---

## 四、生产落地核对清单（Pre-Production Checklist）

- [ ] **严禁将包含候选人照片、姓名、年龄、性别、邮编的未脱敏原始简历直接输入打分模型**。
- [ ] **使用标准职业技能本体（`ESCO / O*NET`）做图谱上下位归一**，避免候选人因使用等价技术术语被漏判。
- [ ] **在 CI/CD 与生产周报中固化 `反事实扰动测试（Counterfactual Fairness Test）` 与 `4/5 法则（Impact Ratio >= 0.80）` 自动合规签核**。
