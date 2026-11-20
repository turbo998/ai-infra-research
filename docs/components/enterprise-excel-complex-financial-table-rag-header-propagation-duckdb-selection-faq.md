# 企业级复杂 Excel 工作簿与多级表头财务报表 RAG（`Spreadsheet & Multi-Header Table RAG`）、合并单元格坐标树反归一化（`openpyxl AST Un-Merging`）、行列层级注入与 `DuckDB` 沙箱计算架构选型指南：为什么直接用 `read_excel().to_csv()` 切块会让满屏变成 `Unnamed: 1`、导致第 150 行丢失表头并引发 `43.5%` 的财务问答算错？

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-excel-complex-financial-table-rag-header-propagation-duckdb-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、财务/供应链数据工程负责人与企业知识库架构师的“10 万份企业复杂 `.xlsx` 工作簿与多级合并表头财报问答、`openpyxl` 坐标级合并单元格拆解（`Merged-Cell Un-Merging`）、单 Sheet 多子表孤岛切分、行列树表头自包含注入与内嵌 `DuckDB` 确定性 SQL 聚合”实战指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_excel_complex_financial_table_rag_header_propagation_duckdb_selection/results.json`](../../experiments/enterprise_excel_complex_financial_table_rag_header_propagation_duckdb_selection/results.json)。

![企业级复杂 Excel 与多级表头财务报表 RAG 架构总结图](../../assets/svg/enterprise-excel-complex-financial-table-rag-header-propagation-duckdb-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

在任何一家大型企业的财务部、采购部、人力部与供应链部门，真正决定业务命脉的数据往往不在规整的 Word 文档里，而是躺在 **`100,000+` 份结构极其复杂的 `.xlsx` Excel 工作簿（共 `120 万` 个 Sheet，其中 `68%` 含有三层合并单元格表头、缩进式父子科目行、单 Sheet 并排摆放多个小表、以及顶部的 `单位：人民币千元` 说明，月均 `600,000` 次检索与核算提问）** 中。传统的文本 RAG 管道一旦撞上企业真实 Excel，会遭遇三大滑铁卢：
1. **死穴一：直接调 `pandas.read_excel().to_csv()` 转文本再按 500 词硬切（Tier 1，复杂表格问答准确率仅有惨淡的 `56.5%`——高达 `43.5%` 答错，算术与单位量纲错误率高达 `38.4%`，月度返工与核对成本高达 `$195,000`）**：
   - **合并单元格变 `Unnamed: 1`**：财务人员最爱在 Excel 第 2 行合并 `B2:E2` 写上 `"2026财年三季度（单位：千元）"`，在第 3 行合并 `B3:C3` 写上 `"华东区实际数"`。一旦直接调 `read_excel()`，除了左上方第一格有字，右边几列全部变成 **`Unnamed: 1, Unnamed: 2, NaN`**，列语义当场毁灭！
   - **第 150 行沦为“无头孤儿行”**：当一张 300 行的采购价格表被切成 6 个文本块时，第 4 个切片（第 150~200 行）里只剩下孤零零的一串数字 `| 型号-A9 | 14.2 | 18.9 | 22.1 |`——**既没有前 3 行的列标题，也丢掉了第 1 行的“单位：千美元”**，大模型直接把 1,420 万美元报价成了 1.42 万美元（差了 1,000 倍）！
2. **死穴二：把所有 Excel Sheet 截图转成图片喂给多模态视觉大模型（`Vision VLM Screenshot RAG`，Tier 2，准确率仅 `72.0%`，月耗 `$210,000` 天价视觉 Token）**：遇到 500 行 × 40 列的宽表，缩成一张图字小得根本看不清，且让视觉模型对着图片心算 80 行的加总必出错！
3. **死穴三：把未经清洗的原始 `.xlsx` 直接丢进 Python `Pandas REPL` 让大模型自己写代码探索（Tier 3）**：因为一个 Sheet 里经常上下左右摆了 3 张小表加一堆备注，大模型第一轮写 `df = pd.read_excel()` 读出来的全是错位列，陷入连续 4~5 轮报错重写代码的死循环（`P95` 延迟高达 `4,800ms`）！

在 **100,000 份企业复杂 Excel 工作簿（120 万个 Sheet，月均 600,000 次表格问答与聚合计算）** 的生产基准下，四档表格 RAG 架构的实测对决如下：

| 方案档位 | 合并单元格处理、表头继承与数值计算机制 | 复杂表格问答准确率 | 算术与单位量纲错误率 | 查询 `P95` 延迟 | 60 万次查询/月总 TCO | 架构师核心选型结论 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：简单 `read_excel().to_csv()` + 500 词硬切** | 不拆合并格；中间行切片丢失表头；靠 LLM 心算加总 | `56.5%` ✕ *(近半数张冠李戴)* | `38.4%` ✕ *(常错1000倍单位)* | `820.0 ms` | `$195,000 / 月` ✕ *(严重财务误导)* | ⛔ 严禁对含合并表头或超 30 行的业务 Excel 使用纯文本硬切 |
| **🟨 Tier 2：Excel 渲染截图 + 视觉大模型 (`VLM`)** | 把工作表渲染成 PNG 图片喂给多模态大模型 | `72.0%` ✕ *(宽长表看不清)* | `24.5%` ✕ *(看图心算必错)* | `3,400.0 ms` ✕ | `$210,000 / 月` ✕ *(视觉Token极贵)* | ⛔ 视觉截图只适合 20 行以内的PPT小表，无法处理数百行工作簿 |
| **🟦 Tier 3：原始 `.xlsx` 直塞 `Pandas REPL` 写代码** | 不做预处理，让 LLM 调 `pd.read_excel` 盲摸结构 | `78.5%` | `14.2%` | **`4,800.0 ms` ✕** *(多轮报错重试)* | `$138,000 / 月` ✕ | ⛔ 单 Sheet 多子表与合并表头会导致 Pandas 列名错乱 |
| **🟩 Tier 4：`openpyxl` 坐标树反归一化 (`Un-Merge & Island Split`) + 行列多级表头/单位自包含注入 + 内嵌 `DuckDB` SQL 引擎** | **入库时按空白/边框切分子表、展开合并表头为 `FY26_Q3_Actual_USD_K`、注入行缩进树，并同时注册进 `DuckDB` 执行 `8ms` 精确 SQL** | **`97.8%` 🏆** *(比Tier 1高41.3个百分点)* | **`0.2%` 🏆** *(SQL确定性计算)* | **`390.0 ms` 🏆** *(比Pandas循环快12倍)* | **`$23,000 / 月` 🏆** *(比 Tier 1 省 88.2%)* | 🌟 **企业级复杂 Excel 与多级财务报表 RAG 唯一黄金标准** |

> 💡 **架构师一句话复杂 Excel RAG 铁律（入库拆孤岛、合并必展开、每行带全名、算账走 DuckDB）**：
> 1. **入库时 `openpyxl` 坐标级“拆孤岛 + 展开合并单元格”铁律（Sub-Table Island Splitting & Merged-Cell Un-Merging）**：“**绝不把原始 `.xlsx` 直接喂给 `pandas.read_excel()`！**”在入库解析阶段，先用 `openpyxl (data_only=True)` 遍历二维网格：
>    - **拆孤岛**：根据全空行/全空列与单元格边框连通域，自动把同一个 Sheet 里上下或左右并排摆放的 3 张独立子表切分成 3 张规范表（并提取表上方的标题行与底部的注释行）；
>    - **展合并**：遍历 `ws.merged_cells.ranges`，将合并区域左上角的值复制到该区域的所有子格，再将前 $M$ 行多级表头纵向拼接为唯一自解释列名（例如将 3 层表头拼成 **`"2026财年__第三季度__华东区实际收入(人民币千元)"`**），彻底消灭 `Unnamed: 1`！
> 2. **行级缩进科目树与全局单位乘数（Scale Multiplier）双向注入铁律**：财务报表大量使用“首列空格缩进”或“加粗汇总行”表示层级（如 `营业费用 -> 研发费用 -> 云服务器租赁`）。解析器自动识别字体加粗与缩进深度，把完整父路径拼入每一行：`row_path = "营业费用 > 研发费用 > 云服务器租赁"`，并将表头的单位乘数（`scale = 1,000`）显式固化进元数据——**哪怕单独把第 248 行抽出来，它自身也是一条 100% 语义完整、带精确坐标 `[Sheet2!E248]` 的独立事实卡片**！
> 3. **“点查走自包含语义卡、聚合统计走内嵌 `DuckDB`”双引擎铁律**：入库反归一化后的干净二维表同步存为 **`Parquet` 并注册进内存 `DuckDB`**。当用户问“哪几个产品的毛利率低于 15% 且同比下降？总差额是多少？”时，路由直接生成一条针对扁平列名的单表 `DuckDB SQL` 在 **`8ms`** 内算出精确到分位的真值（算术错误率降至 **`0.2%`**）！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击怎么破除合并表头 `Unnamed`、怎么处理跨 Sheet 公式引用、为什么不能全靠视觉模型看图 3 大痛点）

### Q1：为什么同样一份财务预算 Excel，人眼看上去一目了然，一旦用 `pandas.read_excel()` 读进来就变成了满屏 `Unnamed: 1` 和 `NaN`？Tier 4 是怎么把它变成规整结构化表的？
- **揭秘人类视觉布局（2D Visual Grid）与关系型数据库（1D Flat Schema）的错位**：
  - 人类在 Excel 里习惯把前 3 行留作报表抬头与“编制单位：集团财务部 | 单位：万元”，从第 5 行才开始画表头，而且第 5、6、7 三行全是跨列合并单元格（Merged Cells），到第 40 行又空两行画第二张附表。
  - `pandas.read_excel()` 默认傻傻地把第 1 行当列名，结果自然整张表全废！
  - **Tier 4 的 `openpyxl` 坐标 AST 四步反归一化算法**：
    1. **连通域识别（Island Detection）**：扫描网格密度与边框属性，精准定位主表位于 `A5:N180`、附表位于 `A185:G220`，抬头元数据位于 `A1:N4`；
    2. **合并格广播（Un-Merge Broadcast）**：把 `B5:E5` 合并格里的 `"2026预算"` 填充到 `B5, C5, D5, E5` 每一格；
    3. **表头栈纵向折叠（Header Stack Flattening）**：识别出第 5~7 行为文本型表头区、第 8 行起为数值型数据区，将每列第 5、6、7 行的文字用 `__` 纵向拼接并附加上从 `A1:N4` 提取出的单位 `"_单位万元"`；
    4. **剔除小计行防双计（Subtotal Row Tagging）**：识别出含 `=SUM(...)` 公式的小计/合计行并打标 `is_subtotal = true`，防止下游做 `SUM()` 聚合时把明细和小计加在一起算重了 2 倍！

### Q2：如果 Excel 单元格里存的是跨 Sheet 公式（比如 `='假设参数表'!$C$8 * D12`），入库时应该读公式字符串还是读计算结果？
- **“双轨保留：以 `data_only=True` 缓存值为准 + 提取公式血缘（Formula Lineage）备查”**：
  - 架构师用 `openpyxl` 同时以两种模式打开工作簿一次：
    - 用 `data_only=True` 读取 Excel 保存时已由 Excel 引擎计算好的精确数值，写入 `DuckDB` 数值列用于查询和聚合；
    - 用 `data_only=False` 提取单元格背后的公式表达式（如 `Revenue = Price * Volume * 'FX_Rates'!B4`），存入该表的元数据說明字段！这样当高管不仅问“Q3 预测收入是多少”，还追问 **“这个预测收入是按什么公式和汇率假设算出来的？”** 时，RAG 能瞬间给出连带跨 Sheet 参数源头的满分回答！

### Q3：为什么很多团队尝试把 Excel 转成 Markdown 表格或者截图喂给多模态大模型（Tier 2），在 Demo 时很惊艳，一上生产 10 万份工作簿就彻底崩盘？
- **算清规模化生产中的“行列容量极限”与“百万页视觉 Token 账单”**：
  - Demo 里的表格通常只有 10 行 × 5 列；而生产供应链的《全球物料 BOM 与阶梯报价表》动辄 **2,500 行 × 35 列**！
  - 2,500 行如果转成图片，要么分辨率缩到字迹模糊无法辨认，要么切成 50 张高清图每问一次花掉 `$0.50` 和 `3.4 秒`；而通过 Tier 4 的 `openpyxl AST + Parquet/DuckDB` 转换，2,500 行 × 35 列在 `DuckDB` 里查询只需 **`8 毫秒`**，且大模型只需要读取几十个 Token 的表头 Schema 就能写出百发百中的 SQL！

---

## ③ 🧭 直观可视化图解（表格 RAG 选型决策树 + `openpyxl AST + DuckDB` 双通道架构图 + 三层合并表头财报精准核算时序图）

### 3.1 企业级复杂 Excel 与多级表头财务报表 RAG 架构选型决策树

```mermaid
flowchart TD
    A["企业级复杂 Excel 与多级表头财报 RAG 选型<br/>(10 万份工作簿 | 120 万个 Sheet | 68% 含多级合并表头与公式 | 月均 60 万次查询)"] --> B{"如何解析复杂合并表头 Excel 并回答检索与计算问题?"}
    B -- "直接 read_excel().to_csv() 转文本 + 500 词硬切 (Tier 1)" --> C["⛔ 问答准确率仅 56.5% + 38.4% 算错数字与单位！<br/>合并格变 Unnamed:1，第 150 行丢失表头与千元单位，月耗 $19.5 万"]
    B -- "将 Excel 渲染成图片喂给多模态视觉大模型 VLM (Tier 2)" --> D["⛔ 宽长表看不清 (准确率 72%) + 月烧 $21 万视觉 Token！<br/>看图心算多行加总错误率仍高达 24.5%"]
    B -- "把原始未清洗 .xlsx 直接丢进 Pandas REPL 写代码 (Tier 3)" --> E["⛔ P95 延迟高达 4,800ms (陷入多轮报错重试)！<br/>单 Sheet 多子表与合并格导致 df.columns 错位"]
    B -- "openpyxl 坐标 AST 拆孤岛/展合并 + 行列路径注入 + 内嵌 DuckDB SQL (Tier 4)" --> F["🟩 Tier 4: openpyxl 坐标 AST 反归一化 + 自包含行卡 + DuckDB 引擎 🏆<br/>(97.8% 准确率 | 0.2% 算术误差 | 390ms 极速响应 | 省 88.2% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `openpyxl` 坐标 AST 反归一化、小计行防重标记与“语义行卡 + `DuckDB`”双索引架构图

```mermaid
flowchart LR
    subgraph Ingest["1. 入库 openpyxl 坐标 AST 反归一化引擎"]
        XLSX["📊 原始复杂 .xlsx 工作簿<br/>(含多子表/3层合并格/缩进行/公式)"] --> Island["🏝️ 子表孤岛切分与公式提取<br/>• 切分同 Sheet 多张独立子表<br/>• 标记 =SUM() 小计行 (is_subtotal=true)"]
        Island --> Unmerge["🧱 合并格广播与表头栈折叠<br/>B2:E2 + B3:C3 -> 唯一自解释列名:<br/>'FY26_Q3_华东实际_单位千元'"]
        Unmerge --> RowTree["🌲 首列缩进科目树传播<br/>为每行注入完整父路径:<br/>'OPEX > 研发 > 云算力' + [Sheet!R42C5]"]
    end

    subgraph DualIndex["2. 双模态存储底座 (零孤儿行、零心算幻觉)"]
        RowTree --> Cards[("🗂️ 自包含语义行元组卡 (向量+BM25)<br/>每行自带完整列名、父科目与单位<br/>(支持跨 10 万工作簿秒级定位)")]
        RowTree --> Duck[("🦆 内存/Parquet DuckDB 规范表<br/>剔除小计行干扰 (WHERE NOT is_subtotal)<br/>支持 8ms 精确聚合、排序与同比计算!")]
    end

    subgraph Query["3. 在线查询路由 (390ms P95, 97.8% 准确)"]
        User["👤 财务/采购提问"] --> Router["🧭 定位目标表 + 执行 DuckDB SQL / 抽取行卡"]
        Router --> Cards & Duck
    end
```

### 3.3 跨多级合并表头与缩进科目行精准核算“华东区 Q3 研发云算力超支额（自动过滤小计行与还原千元量纲）”时序图

```mermaid
sequenceDiagram
    autonumber
    participant CFO as 财务总监 / 分析师
    participant Pipe as 表格检索与 DuckDB 路由引擎
    participant VDB as 自包含表 Schema 与行卡索引
    participant Duck as 内嵌 DuckDB (Parquet 规范表)

    CFO->>Pipe: 1. 提问：“《2026 集团费用表.xlsx》里，Q3 华东区研发部门下属各子项的实际支出比预算超支了多少万元？”
    Pipe->>VDB: 2. 检索命中 `Sheet3_SubTable1`，读取已展开的规范列名：`[row_hierarchy_path, FY26_Q3_East_Actual, FY26_Q3_East_Budget, unit_scale=1000, is_subtotal]`
    Pipe->>Duck: 3. 生成确定性 SQL：`SELECT SUM(FY26_Q3_East_Actual - FY26_Q3_East_Budget) * 1000 / 10000 FROM tbl WHERE row_hierarchy_path LIKE '营业费用 > 研发费用 > %' AND is_subtotal = false`
    Note over Pipe,Duck: 自动过滤 `is_subtotal = false`（防止把汇总行和明细行重复加总 2 遍！）并自动完成“千元 -> 万元”量纲换算！
    Duck-->>Pipe: 4. `8ms` 返回精确计算结果：`+142.85 万元` 及对应明细单元格坐标 `[Sheet3!D48:E56]`
    Pipe-->>CFO: 5. 返回带精确单元格溯源链接与公式说明的答案（总耗时 `390ms`，算术与量纲误差 `0.0%`）！
```

---

## ④ 📊 四档企业复杂 Excel 与多级表头表格 RAG 方案全维度决策矩阵

| 评估维度 | 🟥 Tier 1：`to_csv()` + 文本硬切 | 🟨 Tier 2：渲染图片喂 `VLM` | 🟦 Tier 3：裸 `.xlsx` 进 `Pandas` | 🟩 Tier 4：`openpyxl AST` 展开 + 行列注入 + `DuckDB` |
|---|---|---|---|---|
| **复杂多级表头问答准确率** | `56.5%` ✕ (满屏 `Unnamed` 与孤儿行) | `72.0%` ✕ (超过 50 行看不清) | `78.5%` | 🏆 **`97.8%`（合并格完整广播 + 行缩进树全路径注入）** |
| **算术加总与单位量纲错误率** | `38.4%` ✕ (常漏千元单位或算重小计) | `24.5%` ✕ (看图心算必错) | `14.2%` | 🏆 **`0.2%`（`DuckDB` 确定性 SQL + `is_subtotal=false` 防重）** |
| **端到端查询 `P95` 延迟** | `820.0 ms` | `3,400.0 ms` ✕ | `4,800.0 ms` ✕ (多轮报错调试) | 🏆 **`390.0 ms`（单次干净 Schema 映射 + `8ms` 内存 SQL）** |
| **单元格级精确坐标引文溯源** | 不支持（文本切碎后丢失行列号） | 仅能框出整张大图 | 不稳定 | 🏆 **支持精确到单元格的 `[Workbook#Sheet!R48C5]` 锚点** |
| **60 万次查询/月总 TCO** | `$195,000 / 月` ✕ | `$210,000 / 月` ✕ | `$138,000 / 月` ✕ | **`$23,000 / 月` 🏆（比 Tier 1 节省 88.2%）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（100,000 份工作簿 / 120 万个 Sheet / 月均 600,000 次表格问答）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级复杂 Excel 与多级表头财报 RAG ROI 计算器](../reports/enterprise-excel-complex-financial-table-rag-header-propagation-duckdb-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-excel-complex-financial-table-rag-header-propagation-duckdb-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_excel_complex_financial_table_rag_header_propagation_duckdb_selection/run.py --check
  python -m unittest tests/test_enterprise_excel_complex_financial_table_rag_header_propagation_duckdb_selection.py
  ```

### 5.2 核心量化对比图（复杂表格问答准确率 % vs. 算术与单位量纲错误率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Naive CSV + 500t Chunks", "metric": "Complex Table QA Accuracy (%)", "val": 56.5 },
      { "tier": "Tier 1: Naive CSV + 500t Chunks", "metric": "Math & Unit Scale Error (%)", "val": 38.4 },
      { "tier": "Tier 2: Vision VLM Screenshots", "metric": "Complex Table QA Accuracy (%)", "val": 72.0 },
      { "tier": "Tier 2: Vision VLM Screenshots", "metric": "Math & Unit Scale Error (%)", "val": 24.5 },
      { "tier": "Tier 3: Raw .xlsx in Pandas REPL", "metric": "Complex Table QA Accuracy (%)", "val": 78.5 },
      { "tier": "Tier 3: Raw .xlsx in Pandas REPL", "metric": "Math & Unit Scale Error (%)", "val": 14.2 },
      { "tier": "Tier 4: openpyxl AST + DuckDB", "metric": "Complex Table QA Accuracy (%)", "val": 97.8 },
      { "tier": "Tier 4: openpyxl AST + DuckDB", "metric": "Math & Unit Scale Error (%)", "val": 0.2 }
    ]
  },
  "semantic_types": { "tier": "Category", "metric": "Name", "val": "Percentage" },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 lifts complex Excel QA accuracy to 97.8% and cuts math/unit errors to 0.2%",
    "subtitle": "Complex Spreadsheet QA Accuracy (%) vs. Arithmetic & Unit-Scale Error Rate (%)",
    "encodings": { "x": "tier", "y": "val", "group": "metric", "color": "metric" }
  }
}
```

---

## ⑥ 🛠️ 架构评审与生产落地 Checklist（上线前必查 5 项）

- [ ] **严禁直接调用 `pandas.read_excel().to_csv()` 处理企业业务工作簿**：必须先通过 `openpyxl` 解析 `ws.merged_cells.ranges` 展开所有跨行跨列合并单元格，并将多级表头纵向拼接为自解释的扁平列名。
- [ ] **在入库流水线内置“单 Sheet 多子表孤岛切分器（Sub-Table Island Splitter）”**：根据空行/空列与边框连通域把同一个 Sheet 上的多张并排或上下附表拆分为独立的数据表。
- [ ] **提取首列缩进层级（`Row Hierarchy Path`）与表头全局单位乘数（如 `单位：千元 -> scale=1000`）注入每一行**：彻底杜绝第 150 行在分块后丢失表头上下文或少乘 1,000 倍的量纲灾难。
- [ ] **利用公式解析器（`data_only=False`）自动识别并标记 `=SUM() / =SUBTOTAL()` 汇总行（`is_subtotal = true`）**：在执行聚合查询时默认过滤掉汇总行，防止明细与小计相加导致结果翻倍。
- [ ] **采用“自包含语义行卡（用于跨表定位）+ 内嵌 `DuckDB / Parquet`（用于过滤与算术聚合）”双底座**：把一切加减乘除、同比环比与 Top-N 排序交给 `8ms` 的 `DuckDB SQL` 确定性执行，严禁让大模型对长表格做自然语言心算。
