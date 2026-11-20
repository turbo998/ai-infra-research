# 企业级遗留系统现代化改造工厂（`COBOL / DB2 PL/SQL / Java 8` 升级至云原生 `Java 21`）、`AST` 调用图叶子优先切分、黄金输入输出录像对账（`Golden I/O Tape Harness`）与渐进式绞杀者模式（`Strangler-Fig Shadowing`）架构选型指南：为什么把万行老代码直接丢给大模型翻译会有 `31.5%` 的定点小数截断与逻辑暗病，以及为什么“绝不能让同一个大模型既写新代码又写单元测试”？

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-legacy-code-modernization-cobol-java-migration-agent-factory-selection-faq.en.md)  
> **定位**：面向银行/保险/电信/制造业核心系统架构师、云与 AI 解决方案架构师（Solution Architect）与研发效能负责人的“百万行遗留代码现代化迁移流水线（Code Modernization Agent Factory）、`ProLeap / Tree-sitter AST` 拓扑依赖切分、`COMP-3` 定点小数精度铁律、基于生产真实流量录像的黄金 I/O 差分验证（`Golden Tape Differential Replay`）与双跑影子切流”实战指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_legacy_code_modernization_cobol_java_migration_agent_factory_selection/results.json`](../../experiments/enterprise_legacy_code_modernization_cobol_java_migration_agent_factory_selection/results.json)。

![企业级遗留系统现代化改造工厂架构总结图](../../assets/svg/enterprise-legacy-code-modernization-cobol-java-migration-agent-factory-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

在银行核心账务、保险精算批处理、电信计费与大型 ERP 数据库存储过程下移（如 `500,000 行` 主机 `COBOL / CICS`、Oracle/DB2 `PL/SQL` 包或老旧单体 `Java 6/8` 迁移至云原生 `Java 21 Spring Boot + PostgreSQL`）的战役中，许多企业以为“有了百万上下文大模型，把老代码文件往里一塞就能直接输出新系统”。实测表明，这种天真做法会一头撞上三大工程铁壁：
1. **死路一：把几千行老代码直接塞进 Prompt 让大模型整篇翻译（`Whole-File LLM Translation`，Tier 1，编译失败率达 `44%`，潜伏逻辑与金额精度缺陷率高达 `31.5%`，50 万行代码返工与排障总成本高达 `$620,000`）**：
   - 老主机 `COBOL` 里的 `PIC S9(7)V99 COMP-3` 是压缩十进制（Packed Decimal），默认在做中间除法时遵循的是 **“向零截断（Truncate）”而非四舍五入**！大模型如果随手把它翻译成 Java 的 `double` 或默认 `HALF_UP` 的 `BigDecimal`，在全行 1,000 万个账户的夜间结息批处理中就会产生 **几十万笔“差一分钱”的对账事故**！此外，未按调用图拓扑顺序乱翻，会导致上层模块引用的底层 `COPYBOOK` 数据结构全是幻觉出来的假字段！
2. **死路二：仅靠程序员在 IDE 里用 Copilot 逐段手工复制粘贴改写（Tier 2，50 万行代码耗时仍长达 `14.5 个月`，花费 `$490,000`）**：程序员 75% 的时间并不是在敲新代码，而是在苦苦考古“30 年前离职老员工写的这段奇葩 `GOTO` 分支到底代表什么业务规则”以及手工补写测试数据！
3. **死路三：让同一个大模型既负责把老代码翻成新代码，又负责给新代码生成 JUnit 单元测试（`Self-Graded LLM Unit Tests`，Tier 3，逃逸逻辑缺陷率仍高达 `14.2%`）**：
   - **这是当前业界最隐蔽的“自己批改自己作业”反模式！** 当大模型因为误解了某条隐式 `PL/SQL` 游标边界条件而写错了 Java 代码时，它在紧接着生成单元测试 `assertEquals(expected, actual)` 时，**会把一模一样的错误理解写进测试断言里**——结果 100% 单元测试全绿通过，一上线生产立刻暴雷！

在 **500,000 行遗留核心代码（含 420 个关联模块 / `COPYBOOK` / 存储过程）升级至云原生 `Java 21`** 的企业基准下，四档代码现代化改造架构的实测对决如下：

| 方案档位 | 代码切分、类型映射与等价性验证机制 | 生产输入输出等价率 (`I/O Parity`) | 逃逸金额/逻辑缺陷率 | 50 万行迁移总工期 | 50 万行总迁移与返工 TCO | 架构师核心选型结论 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：整文件直接塞给大模型裸翻 (`Raw Whole-File LLM`)** | 不解析 AST 调用树；无生产黄金录像对账 | `68.5%` ✕ *(大量编译与精度错)* | `31.5%` ✕ *(COMP-3差一分钱)* | `11.0 个月` *(陷入人工修Bug泥潭)* | `$620,000` ✕ *(天价排障与对账返工)* | ⛔ 严禁用于任何涉及资金核算、计费或核心状态流转的系统改造 |
| **🟨 Tier 2：程序员用 IDE 助手半手工逐文件改写** | 人工读老代码 + 聊天窗口辅助写新片段 | `87.2%` | `12.8%` ✕ | `14.5 个月` ✕ *(工期极度漫长)* | `$490,000` ✕ *(昂贵人力工时)* | ⛔ 缺乏自动化流水线与回归线束，无法规模化啃下百万行遗产 |
| **🟦 Tier 3：Agent 自动翻译 + 让同一个 LLM 自写单元测试** | 让翻译代码的 LLM 自己编造 JUnit 测试用例 | `85.8%` | **`14.2%` ✕** *(自己批自己作业!)* | `5.5 个月` | `$340,000` ✕ | ⛔ 测试通过率 100% 全是假象，模型把误解同时写进了代码和断言 |
| **🟩 Tier 4：`AST` 叶子优先拓扑切分 + 确定性 `COMP-3/BigDecimal` 守卫 + `10,000条` 生产黄金 I/O 录像差分沙箱 + 绞杀者影子伴飞** | **先用 `ProLeap/Tree-sitter` 按调用图自底向上翻译；用老系统录制的 1 万条真实输入输出磁带做逐字段 `$0.00` 差分自动修复** | **`99.8%` 🏆** *(字节级输入输出对齐)* | **`0.2%` 🏆** *(比Tier 1降31.3个百分点)* | **`2.4 个月` 🏆** *(比手工快 6.0 倍!)* | **`$85,000` 🏆** *(比 Tier 1 省 86.3%)* | 🌟 **企业级遗留系统现代化 AI 改造工厂唯一推荐标准** |

> 💡 **架构师一句话遗留代码改造工厂铁律（AST 排先后、小数定铁律、测试用录像、上线靠双跑）**：
> 1. **`AST` 调用图自底向上（Leaf-First Topological Order）切分铁律**：“**绝不先翻译上层调用方，必须从零依赖的底层叶子节点开始往上长！**”先用确定性语法分析器（`ProLeap COBOL Parser / ANTLR PL-SQL / Tree-sitter`）构建全库 420 个模块的调用与数据结构依赖图（Call & Data DAG）：第 1 步先将所有底层 `COPYBOOK` 数据字典确定性地编译为强类型 Java `Record / POJO`（强制将 `PIC S9..V99 COMP-3` 映射为带精确位数与截断模式的 `BigDecimal`）；第 2 步翻译最底层的公共计息/校验子程序并锁定接口签名；第 3 步把已编译通过的子程序接口签名作为上下文注入给上层主程序 Agent——编译一次通过率直接拉升至 **`99.4%`**！
> 2. **“绝不让大模型自己出考卷批作业”——生产黄金 I/O 录像差分沙箱铁律（Recorded `Golden I/O Tape` Harness）**：验证新代码对不对的唯一权威标准，是 **老系统在生产环境真实跑出来的输入输出录像带（Golden Tapes，如从主机 `VSAM / CICS / DB2` 录制的 10,000 组边界交易报文与批处理快照）**！改造工厂自动把这 10,000 组真实输入同时喂给老系统镜像与新生成的 Java 21 容器，逐个字段比对输出 JSON 与数据库变更集——但凡有一个字段差了 `0.01 元`，差分引擎立刻把具体的输入报文和差异字段塞回给 **Repair Agent（修复智能体）** 自动定位重修，直到 10,000 条黄金用例 **100% 零差异通过**！
> 3. **规格先行（Spec-First）与绞杀者模式双跑切流铁律（Strangler-Fig Dark-Traffic Shadowing）**：先从老代码提取人类可读的《业务规则说明书（Business Rule Spec）》交由领域专家签字并入库 RAG，再按模块在网关层开启 14 天生产流量影子双跑比对，确认 `99.8%+` 账目分毫不差后再切换主流量！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击为什么 COBOL 翻成 Java 会差一分钱、为什么全绿的 AI 单元测试不能信、万行大文件怎么切 3 大痛点）

### Q1：为什么银行核心的 COBOL 计息或保险精算代码，让大模型直接翻译成 Java 后看起来逻辑完全一样，但跑批对账时却有 `31.5%` 的账户金额对不上？
- **揭秘主机与现代编程语言在“定点十进制算术（Fixed-Point Arithmetic）、隐式溢出截断与 EBCDIC 排序”上的三大底层鸿沟**：
  1. **舍入与截断模式差异**：COBOL 的 `COMPUTE A = B / C` 在未显式声明 `ROUNDED` 关键字时，对超出小数位的部分执行的是 **直接砍掉（`RoundingMode.DOWN`）**；而大模型默认生成的 Java 代码往往用 `double` 或者 `BigDecimal.divide(..., RoundingMode.HALF_UP)`（四舍五入），一万笔交易里就会有几千笔差 1 分钱！
  2. **高位溢出静默截断（High-Order Truncation）**：COBOL 的 `PIC 9(4)` 变量存入 `12345` 时会静默截断保留低 4 位 `2345`，而 Java `int` 会保留 `12345`；
  3. **字符集排序顺序（Collating Sequence）**：主机 `EBCDIC` 编码中 **小写字母 < 大写字母 < 数字（`'a' < 'A' < '0'`）**，而现代 `ASCII / UTF-8` 中 **数字 < 大写字母 < 小写字母（`'0' < 'A' < 'a'`）**！如果老代码里有 `IF ACCOUNT-CODE > "A000"` 的范围判断，直接照翻成 Java 字符串比较就会把所有数字开头的账号漏掉！
  - **Tier 4 的确定性转译守卫（Deterministic Runtime Shim）**：架构师不靠大模型“自觉注意”，而是提供一套标准的基础设施库 `CobolRuntimeShim`（封装了严格对齐主机语义的 `PackedDecimal`、`FixedStringEbcdicComparator`），并在 AST 检查器里 **严禁生成代码出现原生 `float / double`**！

### Q2：为什么 Tier 3 中“让 Agent 写完 Java 代码后再顺手写一套 JUnit 测试，覆盖率达到 90% 且全部跑通”，上线后依然有 `14.2%` 的严重业务故障？
- **剖析“同源幻觉共振（Correlated Hallucination in Self-Generated Tests）”**：
  - 举例：老 PL/SQL 存储过程里有一段处理“闰年 2 月 29 日按 365 天还是 366 天计息”的晦涩分支。大模型在读这段老代码时理解反了，把新 Java 代码写成了按 360 天计息；
  - 紧接着你让同一个大模型为这段 Java 代码写单元测试，大模型自然会算好一个按 360 天算出来的值填进 `assertEquals(expected, actual)` 里——**测试当然 100% 绿灯通过，但这只是证明了“大模型自己跟自己达成了一致”，根本没有证明“新代码和老主机达成了一致”**！
  - **Tier 4 黄金铁律**：测试断言的数据源必须 **100% 来自老系统真实运行产生的历史输入输出录像带（Golden I/O Tapes）**，加上基于属性的形式化边界生成器（Property-Based Fuzzing，用随机生成的负数、零、极大值、闰年日期同时跑老代码模拟器和新 Java 代码做黑盒对拍），彻底斩断自娱自乐的假测试！

### Q3：遇到一个 18,000 行、里面包含 60 个 `PERFORM THRU` 段落且共用一堆全局变量的巨型面条式 COBOL / PL-SQL 程序，怎么切分才能不破坏上下文？
- **基于控制流图与数据流读写集（`Read/Write Variable Set Analysis`）的程序切片（Program Slicing）**：
  - 绝不能按“每 500 行切一刀”的物理行号硬切！
  - Tier 4 使用静态数据流分析器扫描每个 `PERFORM` 段落实际读取（`READ_SET`）和修改（`WRITE_SET`）了哪些全局 `WORKING-STORAGE` 变量，自动将该段落重构提取为显式传参的纯函数签名 `OutputSubState paragraphName(InputSubState in)`，将一个 18,000 行的全局面条文件拆解成 25 个内聚、可独立做黄金 I/O 对拍的微模块！

---

## ③ 🧭 直观可视化图解（遗留系统改造选型决策树 + `AST + 黄金录像对拍工厂` 架构图 + 差一分钱自动捕获重修闭环时序图）

### 3.1 企业级遗留系统代码现代化改造工厂架构选型决策树

```mermaid
flowchart TD
    A["企业级遗留系统代码现代化改造选型<br/>(50 万行 COBOL / PL-SQL / Java 8 | 420 个关联模块 | 金融/电信核心批处理与交易)"] --> B{"如何将老旧核心代码升级为云原生 Java 21 并保障 100% 账目等价?"}
    B -- "整文件直接塞给大模型裸翻，无 AST 拓扑与录像对账 (Tier 1)" --> C["⛔ 31.5% 金额与逻辑暗病 + 44% 编译失败！<br/>COMP-3 舍入与 EBCDIC 排序错误导致巨额对账事故，返工耗资 $62 万"]
    B -- "程序员在 IDE 里用 Copilot 半手工逐段改写 (Tier 2)" --> D["⛔ 50 万行代码耗时长达 14.5 个月 ($49 万)！<br/>75% 时间花在人工考古业务规则与手写测试上"]
    B -- "Agent 自动翻译 + 让同一个大模型自写单元测试 (Tier 3)" --> E["⛔ 自己批改自己作业，仍有 14.2% 逃逸逻辑缺陷！<br/>大模型把对老代码的误解同时写进了新代码与 JUnit 断言里"]
    B -- "AST 叶子优先拓扑切分 + COMP-3 精度守卫 + 1万条生产黄金 I/O 录像对拍工厂 (Tier 4)" --> F["🟩 Tier 4: AST 拓扑切分 + 生产黄金录像差分闭环 + 绞杀者影子伴飞 🏆<br/>(99.8% I/O 等价率 | 仅 2.4 个月交付 | 逃逸缺陷降至 0.2% | 省 86.3% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `AST` 拓扑切分、规格提取、确定性小数守卫与 `10,000条` 生产黄金录像对拍工厂架构图

```mermaid
flowchart LR
    subgraph Step1["1. AST 依赖图与叶子优先切分"]
        Legacy["🏛️ 50 万行遗留代码<br/>(COBOL / PL-SQL / Java 8)"] --> AST["🌳 ProLeap / Tree-sitter AST<br/>构建调用图与变量读写集 DAG<br/>按自底向上 (Leaf-First) 排序"]
    end

    subgraph Step2["2. 双轨生成：业务规格 + 强约束转译"]
        AST --> Spec["📘 业务规则提取 Agent<br/>输出人类可读 Markdown 规格书"]
        AST & Spec --> Trans["⚙️ 代码转译 Agent + AST Linter<br/>• 严禁 float/double<br/>• 强制 COMP-3 -> BigDecimal(DOWN)<br/>• EBCDIC 排序兼容层"]
    end

    subgraph Step3["3. 独立生产黄金 I/O 录像差分闭环 (严禁自出考卷!)"]
        Trans --> Sandbox["🧪 10,000 条生产黄金录像对拍沙箱<br/>(老主机真实输入输出 vs 新 Java 21 输出)"]
        Sandbox -- "发现 $0.01 字段差异" --> Repair["🔧 自动差分修复 Agent<br/>喂入精确报错报文重修直至 0 差异"]
        Sandbox -- "10,000 条 100% 零差异通过" --> Shadow["🚀 绞杀者网关 14 天影子双跑切流<br/>(99.8% 生产等价率!)"]
    end
```

### 3.3 黄金 I/O 录像沙箱自动抓出“未声明 `ROUNDED` 的截断差一分钱 Bug”并驱动 Agent 自愈时序图

```mermaid
sequenceDiagram
    autonumber
    participant DAG as AST 拓扑调度器
    participant Trans as 代码转译 Agent
    participant Harness as 1万条生产黄金录像对拍器 (Golden Tape)
    participant Repair as 差分修复 Agent
    participant Gate as 绞杀者影子网关 (Strangler Proxy)

    DAG->>Trans: 1. 派发叶子模块 `INT-CALC-04`（含 `COMPUTE DAILY-INT = BAL * RATE / 365`）及已编译 `COPYBOOK` 接口
    Trans->>Harness: 2. 生成第一版 Java 21 代码（误用了 `RoundingMode.HALF_UP` 四舍五入）并通过 Maven 编译
    Harness->>Harness: 3. 回放 10,000 条老主机真实结息录像带 -> 发现第 #412 条（余额 `$10,000.55`）老主机输出 `$1.36`，新代码输出 `$1.37`（差 `+$0.01`）！
    Harness->>Repair: 4. 拒绝通过！将第 #412 条输入报文、老主机 `$1.36` 真值与对应的 COBOL 第 284 行源码发给修复 Agent
    Repair->>Harness: 5. 识别出该行 COBOL 未带 `ROUNDED` 关键字 -> 将 Java 代码修正为 `RoundingMode.DOWN`（向零截断）
    Harness->>Gate: 6. 重新回放全部 10,000 条黄金录像 -> `10,000 / 10,000` 字段与金额 100% 分毫不差，准予进入影子伴飞池！
```

---

## ④ 📊 四档企业遗留代码现代化改造方案全维度决策矩阵

| 评估维度 | 🟥 Tier 1：整文件塞给 LLM 裸翻 | 🟨 Tier 2：IDE Copilot 半手工改写 | 🟦 Tier 3：Agent 写代码 + 自写单测 | 🟩 Tier 4：`AST` 拓扑切分 + 1万条生产黄金录像工厂 |
|---|---|---|---|---|
| **功能与金额输入输出等价率** | `68.5%` ✕ (频繁算错小数) | `87.2%` | `85.8%` | 🏆 **`99.8%`（经 10,000 条真实生产录像逐字段核验）** |
| **逃逸至生产的逻辑/精度缺陷率** | `31.5%` ✕ (灾难级账目不平) | `12.8%` ✕ | `14.2%` ✕ (假单测漏检) | 🏆 **`0.2%`（确定性 `COMP-3` 映射 + 黑盒差分对拍）** |
| **50 万行代码改造总工期** | `11.0 个月` (后期修 Bug 泥潭) | `14.5 个月` ✕ (极慢) | `5.5 个月` | 🏆 **`2.4 个月`（比人工 IDE 改写快 6.0 倍）** |
| **业务规则文档资产沉淀** | 无 | 零散笔记 | 仅代码注释 | 🏆 **自动沉淀 100% 双向关联的《业务规则说明书 RAG 库》** |
| **50 万行代码总迁移 TCO** | `$620,000` ✕ | `$490,000` ✕ | `$340,000` ✕ | **`$85,000` 🏆（比 Tier 1 节省 86.3%）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（500,000 行遗留代码 / 420 个模块 / 10,000 条黄金 I/O 对拍向量）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级遗留系统现代化改造工厂 ROI 计算器](../reports/enterprise-legacy-code-modernization-cobol-java-migration-agent-factory-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-legacy-code-modernization-cobol-java-migration-agent-factory-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_legacy_code_modernization_cobol_java_migration_agent_factory_selection/run.py --check
  python -m unittest tests/test_enterprise_legacy_code_modernization_cobol_java_migration_agent_factory_selection.py
  ```

### 5.2 核心量化对比图（生产输入输出等价率 % vs. 逃逸金额/逻辑缺陷率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Whole-File LLM Rewrite", "metric": "Functional I/O Equivalence (%)", "val": 68.5 },
      { "tier": "Tier 1: Whole-File LLM Rewrite", "metric": "Escaped Logic/Decimal Bugs (%)", "val": 31.5 },
      { "tier": "Tier 2: Manual IDE Copilot", "metric": "Functional I/O Equivalence (%)", "val": 87.2 },
      { "tier": "Tier 2: Manual IDE Copilot", "metric": "Escaped Logic/Decimal Bugs (%)", "val": 12.8 },
      { "tier": "Tier 3: LLM Self-Graded Unit Tests", "metric": "Functional I/O Equivalence (%)", "val": 85.8 },
      { "tier": "Tier 3: LLM Self-Graded Unit Tests", "metric": "Escaped Logic/Decimal Bugs (%)", "val": 14.2 },
      { "tier": "Tier 4: AST Slicer + 10K Golden Tape", "metric": "Functional I/O Equivalence (%)", "val": 99.8 },
      { "tier": "Tier 4: AST Slicer + 10K Golden Tape", "metric": "Escaped Logic/Decimal Bugs (%)", "val": 0.2 }
    ]
  },
  "semantic_types": { "tier": "Category", "metric": "Name", "val": "Percentage" },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 reaches 99.8% I/O equivalence and cuts escaped bugs from 31.5% to 0.2%",
    "subtitle": "Functional I/O Equivalence (%) vs. Escaped Financial/Logic Defect Rate (%)",
    "encodings": { "x": "tier", "y": "val", "group": "metric", "color": "metric" }
  }
}
```

---

## ⑥ 🛠️ 架构评审与生产落地 Checklist（上线前必查 5 项）

- [ ] **严禁把未切分的大文件直接丢给大模型乱序翻译**：必须先通过 `ProLeap / ANTLR / Tree-sitter` 生成全库调用与数据依赖图（DAG），严格按“叶子模块优先（Leaf-First）”顺序自底向上转译并锁定编译接口。
- [ ] **通过 AST 静态检查器强制执行定点十进制与字符集兼容铁律**：在金融/计费迁移中全面封杀原生 `float / double`，强制将 `COBOL COMP-3 / NUMERIC` 映射为显式指定标度与截断模式（`RoundingMode.DOWN` vs `HALF_UP`）的 `BigDecimal`。
- [ ] **严禁把“同一个大模型自己生成的单元测试通过”当作上线准入依据（No Self-Graded Homework）**：必须录制老系统至少 `10,000` 条真实生产输入输出报文（Golden I/O Tapes）构建独立的黑盒差分沙箱。
- [ ] **建立“黄金录像对拍 -> 精确字段 Diff 反馈 -> Repair Agent 自动重修”的闭环工厂**：让机器在沙箱里自动完成 99% 的边界对齐，仅将极少数老系统本身的陈年死代码交由领域专家裁决。
- [ ] **按业务子域采用绞杀者模式（Strangler Fig）执行至少 14 天的生产暗流量影子双跑（Dark-Traffic Shadowing）**：在网关层异步比对新老系统响应与账务结果，确认 `99.8%+` 对齐后再按 `1% -> 20% -> 100%` 灰度切流。
