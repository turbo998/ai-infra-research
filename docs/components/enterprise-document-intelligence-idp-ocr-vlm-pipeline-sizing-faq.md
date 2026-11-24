# 企业级智能文档处理（IDP）与多模态文档解析流水线架构：`三级复杂度分流（PyMuPDF + MinerU + Qwen2.5-VL）vs. 云端 OCR` 选型与容量测算指南

![企业级智能文档处理 IDP 三级复杂度分流架构与选型总览](../../assets/svg/enterprise-document-intelligence-idp-ocr-vlm-pipeline-sizing-faq-summary.svg)

> **面向角色**：Cloud & AI Solution Architect（云与 AI 解决方案架构师）  
> **核心场景**：银行信贷财报解析、保险理赔单据核验、招投标长标书对比、工程图纸与合同合规审查（月处理量 `50 万–5,000 万页`）  
> **配套交互式 H5 精读工作台**：[`docs/h5/document-intelligence-idp-vlm-pipeline-sizer.html`](../h5/document-intelligence-idp-vlm-pipeline-sizer.html)

---

## 一、执行摘要（Executive Summary：3 分钟掌握核心架构结论）

1. **为什么“把每一页 PDF 都转成高清图片喂给 `72B` 多模态大模型（VLM）”是典型的算力灾难？**  
   - 一页 `300 DPI` 的 A4 文档（`1654×2339`）经动态分块（Dynamic Patching）后产生约 **`1,820` 个视觉 Tokens**；在 `4× H100` 运行 `Qwen2.5-VL-72B-FP8` 上单页推理耗时达 **`1.85 秒`**，折算单千页成本 **`$4.20 / 1,000 页`**。
   - 然而对企业真实文档库的统计表明：**`65%` 的页面是原生数字生成（Born-Digital）的纯文本/简单矢量表 PDF**，**`25%` 是标准扫描排版/多栏/公式页面**，真正包含手写批注、红章遮挡、跨页复杂合并单元格的硬骨头页面仅占 **`10%`**！
2. **三级页面复杂度分流流水线（3-Tier Page Complexity Router，`1.5ms` CPU 预检）核心收益**：
   - **Tier 1（`65%` 页面 — 原生数字 PDF）**：由 **`PyMuPDF (fitz)` + 矢量表格坐标重建器**在纯 CPU 上以 **`2.0ms/页`** 直接提取，千页成本仅 **`$0.02`**；
   - **Tier 2（`25%` 页面 — 标准扫描件/多栏/公式）**：路由至 **`L40S-48GB` 运行 `MinerU / DocLayout-YOLO + PaddleOCR-v4`**，耗时 **`85ms/页`（`11.8 页/秒/卡`）**，千页成本 **`$0.38`**；
   - **Tier 3（`10%` 页面 — 印章遮挡/手写批注/跨页财报合并表）**：路由至 **`H100-80GB` 运行 `OlmOCR-7B / Qwen2.5-VL-72B-FP8` + 受控 JSON/HTML 解码**，千页成本 `$4.20`（仅作用于 `10%` 疑难页）。
   - **综合加权成本公式**：
     $$\text{Blended Cost}_{1000\text{p}} = 0.65 \times \$0.02 + 0.25 \times \$0.38 + 0.10 \times \$4.20 = \mathbf{\$0.528 / 1,000\text{ 页}}$$
     相比全量调用云端布局 OCR（`Azure AI Document Intelligence Layout / AWS Textract Tables`，约 **`$10.00 / 1,000 页`**）节省 **`94.7%`**；相比 100% 盲目调用 `72B VLM`（`$4.20 / 1,000 页`）节省 **`87.4%` GPU 卡数**，同时将复杂财报表格单元格还原 F1 提升至 **`96.8%`**！

---

## 二、架构师灵魂拷问（Core Q&A）

### Q1：如何在 `1.5ms` 内不跑 GPU 模型就精准判断一页 PDF 该走 Tier 1、Tier 2 还是 Tier 3？
在 PDF 解包阶段，CPU 预检网关（Page Complexity Inspector）仅需读取 PDF 页面对象树（Object Tree）与低分辨率灰度直方图即可完成确定性分类：
1. **命中 Tier 1（纯 CPU 直提）**：页面包含有效内嵌字体字符流（Embedded Font Glyphs `> 50` 字符）、无全页光栅大图（Raster Image Coverage `< 15%`）、且矢量线段交点未构成跨级嵌套表头；
2. **直通 Tier 3（H100 VLM 攻坚）**：检测到红色印章色度簇（HSV 红色像素与文字重叠率 `> 8%`）、手写笔迹曲率特征、扫描倾斜角 `> 5°`、或 Tier 2 版面分析输出的表格置信度 `< 0.82`；
3. **其余归入 Tier 2（L40S MinerU + OCR）**：普通扫描合同、双栏学术/技术手册、常规印刷发票。

### Q2：为什么金融财报与复杂合同的表格解析输出，严禁使用 Markdown 竖线表（`| A | B |`），而必须强制输出 `HTML <table rowspan colspan>`？
- **Markdown 表格的结构性缺陷**：原生 Markdown 语法**完全不支持跨行合并（`rowspan`）与跨列多层表头（`colspan`）**。当遇到银行资产负债表中“流动资产 -> 货币资金 -> 2025年末 / 2026年末”的三级合并表头时，强行转成 Markdown 会导致列索引错位，使下游 RAG 数值问答出现 **`31%` 的张冠李戴错误**。
- **生产标准解法**：
  1. 在 Tier 2（`MinerU / RapidTable`）与 Tier 3（`Qwen2.5-VL`）统一输出携带明确 **`rowspan` / `colspan` 属性的精简 HTML `<table>`**；
  2. 在切分 RAG Chunk 时执行 **表头路径注入（Header-Path Injection）**：即使一个大表横跨 3 页被切分为多个切片，每个切片顶部都会自动拼接父级主表头与单位声明（如 `【表头：合并利润表 | 单位：百万元人民币】`），将金融财报数值问答准确率从 **`68.4%` 提升至 `95.2%`**。

```mermaid
flowchart LR
    PDF["输入企业文档流\n(1,000万页/月 PDF/扫描件)"] --> Gate["1.5ms CPU 页面复杂度路由网关\n(内嵌字体树 + 印章HSV + 版面密度)"]
    Gate -- "65% 原生数字页" --> T1["Tier 1: PyMuPDF (fitz) CPU 池\n2.0ms/页 · $0.02/千页"]
    Gate -- "25% 标准扫描/多栏页" --> T2["Tier 2: MinerU + PaddleOCR (1× L40S)\n85ms/页 · $0.38/千页"]
    Gate -- "10% 印章/手写/跨页合并表" --> T3["Tier 3: Qwen2.5-VL-72B-FP8 (2× H100)\n1.85s/页 · $4.20/千页 (仅10%流量)"]
    T1 & T2 & T3 --> HTML["跨页表格缝合 & HTML rowspan/colspan 规范化\n+ 表头单位强制继承注入"]
    HTML --> RAG["企业级混合检索库 / 结构化 JSON 入库\n综合成本 $0.528/千页 · 表格 F1 96.8%"]
```

---

## 三、四大文档智能（IDP）方案横向选型矩阵

| 评估维度 | 方案 A：云端托管 IDP (`Azure AI Document Intelligence / AWS Textract`) | 方案 B：全量多模态大模型 (`100% Qwen2.5-VL-72B / GPT-4o Vision`) | 方案 C：单一开源版面 OCR (`100% MinerU / Marker`) | 方案 D：三级复杂度分流流水线 (`PyMuPDF + MinerU + VLM`) |
|---|---|---|---|---|
| **复杂印章/手写/合并表 F1** | 高（`88.5%–91.2%`） | **极高（`96.4%`）** | 中（`85.0%`，遇印章覆盖与手写易崩） | **最高（`96.8%`，针对性攻坚）** |
| **每千页综合处理成本** | **`$10.00 / 1,000 页`**（Layout/Tables 层） | **`$4.20 / 1,000 页`**（自建 H100）或 `$15+/千页`（商业 API） | **`$0.38 / 1,000 页`** | **`$0.528 / 1,000 页`**（比云端省 **94.7%**） |
| **1,000 万页/月所需硬件** | 云端全托管 Serverless | `18× H100-80GB`（`$41,400/月`） | `3× L40S-48GB` | **`2× CPU 节点 + 1× L40S + 2× H100`（`$5,280/月`）** |
| **最佳适用边界** | 月处理量 `< 50 万页`、拥有现成预置税票/身份证模板需求 | 小批量高价值非标投研手稿 | 学术论文、纯印刷技术书籍批量转 Markdown | **月处理量 `> 100 万页` 的企业级合同/财报/信贷知识库** |

---

## 四、生产级 IDP 流水线落地 Checklist

- [ ] **严禁把文档按页孤立切碎后丢失跨页表格状态**：在流水线后处理阶段增加 `Cross-Page Table Stitcher`，当上一页底部与下一页顶部同属一个无底边框表格时，自动合并 `<tbody>` 并延续列宽定义。
- [ ] **Tier 3 视觉大模型必须启用动态分辨率上限（`max_pixels = 1280×28×28`）与 `temperature = 0`**：避免超高分辨率空白边距浪费 2 倍以上视觉 Tokens，并通过 `Guided JSON / HTML Schema` 杜绝模型在长数字表格中出现循环复读（Repetition Loop）。
- [ ] **每个提取出的关键财务指标必须绑定回源坐标框（Bounding Box Grounding `[page, x1, y1, x2, y2]`）**：前端审计界面点击任意提取字段即可在原 PDF 上高亮对应红框，满足金融信贷“可溯源双录”监管要求。

---

## 五、配套资产索引

- **交互式 H5 精读工作台**：[`docs/h5/document-intelligence-idp-vlm-pipeline-sizer.html`](../h5/document-intelligence-idp-vlm-pipeline-sizer.html)
- **7-Card 4:5 评测报告**：[`docs/reports/enterprise-document-intelligence-idp-ocr-vlm-pipeline-sizing-cards.html`](../reports/enterprise-document-intelligence-idp-ocr-vlm-pipeline-sizing-cards.html)
- **交互式容量计算器**：[`docs/reports/enterprise-document-intelligence-idp-ocr-vlm-pipeline-sizing-sizer.html`](../reports/enterprise-document-intelligence-idp-ocr-vlm-pipeline-sizing-sizer.html)
- **可复现实验模型**：[`experiments/enterprise_document_intelligence_idp_ocr_vlm_pipeline_sizing/run.py`](../../experiments/enterprise_document_intelligence_idp_ocr_vlm_pipeline_sizing/run.py)
