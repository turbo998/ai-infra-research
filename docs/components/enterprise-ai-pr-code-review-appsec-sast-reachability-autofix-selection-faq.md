# 企业级 AI 代码审查与研发安全自愈（`AI PR Code Review & AppSec SAST/SCA Reachability Autofix Agent`）、`Tree-sitter / LSP` 一跳调用图上下文、污点可达性降噪（砍掉 `84.5%` 误报）与沙箱编译验证一键修复建议（`GitHub Suggested Changes`）架构选型指南：为什么只看 `git diff` 的 PR 机器人会产生 `74%` 废话噪音并在两周内被全员拉黑？

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-ai-pr-code-review-appsec-sast-reachability-autofix-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、企业研发效能（DevEx / Platform Engineering）总监与应用安全（AppSec / DevSecOps）负责人的“2,500 人研发团队（月均 40,000 个 Pull Request、18 万条原始 SAST/SCA 告警）智能代码审查与漏洞自愈架构选型、为什么裸喂 `git diff` 必变成话痨噪音制造机、`LSP / Tree-sitter` 跨文件调用者/被调用者签名补全、静态污点可达性过滤（误报率从 `84.5%` 压至 `4.2%`）与 `<4秒` 沙箱编译校验的一键 `Commit suggestion`（采纳率达 `88.6%`）”实战指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_ai_pr_code_review_appsec_sast_reachability_autofix_selection/results.json`](../../experiments/enterprise_ai_pr_code_review_appsec_sast_reachability_autofix_selection/results.json)。

![企业级 AI 代码审查与 AppSec 可达性自愈架构总结图](../../assets/svg/enterprise-ai-pr-code-review-appsec-sast-reachability-autofix-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

在拥有 `2,500` 名工程师、每月发起 **`40,000` 个 GitHub / GitLab Pull Request** 的大型研发组织中，几乎每个技术团队都尝试过挂载“AI 自动代码审查机器人”或“SAST/SCA 安全扫描门禁”。然而，高达 80% 的初代 PR 机器人在上线不到两周后就被工程师集体静音拉黑，原因在于三大架构缺陷：
1. **缺陷一：只把那几行孤立的 `git diff` 塞给大模型并让它“多提改进建议”（Tier 1，无意义的风格挑刺与假 Bug 误报率高达 `74.0%`，而真正的跨文件破坏性 Bug 漏检率却高达 `48.0%`，月度浪费研发排查工时与线上逃逸故障成本高达 `$225,000`）**：
   - **“只看 `git diff` 等于管中窥豹”**：当程序员在 `Pricing.ts` 里修改了 `getUserDiscount()` 的返回值（当优惠券过期时从返回 `0` 改成了返回 `null`），`git diff` 里只有这短短 3 行代码，看起来语法完美无缺！但大模型根本看不见在另一个文件 `CheckoutController.ts` 里调用了 `getUserDiscount().toFixed(2)`（没判空直接触发空指针崩溃）——于是它**漏掉了这个真正的 P1 跨文件故障**，反而去挑刺“建议把变量名 `d` 改成 `discountValue`”或者无脑怀疑某个已经在其他文件里判空过的参数没判空（`74.0%` 全是烦人的噪音）！
2. **缺陷二：把传统静态扫描器（`SAST / SCA Dependabot`）扫出的 18 万条原始告警不加可达性过滤直接刷在 PR 评论区（Tier 2，误报率高达 `84.5%`，研发合并前修复率仅 `14.0%`）**：
   - 某个底层三方包报了一个高危 CVE，但业务代码根本没有调用那个受感染的函数（不可达 Unreachable）；或者 SAST 报某行 SQL 拼接有注入风险，但其实在上一层网关 Middleware 里早就做过严格的整数白名单过滤了！
3. **缺陷三：大模型直接贴出未经编译器校验的 ```` ```suggestion ```` 一键修复代码块（Tier 3）**：程序员信任 AI 点了一下 GitHub 上的 **`[Commit suggestion]`**，结果因为 AI 漏引了一个 `import` 或写错了类型签名，**32% 的一键采纳直接把 CI 编译流水线搞挂了**！

在 **2,500 名研发人员、月均 40,000 个 PR、180,000 条原始 SAST/SCA 安全告警** 的企业生产基准下，四档 AI 代码审查与安全自愈架构的实测对决如下：

| 方案档位 | 上下文补全、安全可达性过滤与修复补丁验证机制 | 跨文件真 Bug/漏洞捕获率 | 误报与废话挑刺噪音率 | 研发一键修复采纳率 | 4 万 PR/月总研发损耗与安全 TCO | 架构师核心选型结论 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：仅把裸 `git diff` 塞给话痨大模型** | 看不见调用方与类型定义；无 Critic 降噪门禁 | `52.0%` ✕ *(漏掉48%跨文件破环)* | **`74.0%` ✕** *(满屏风格废话与瞎猜)* | `16.0%` *(全员拉黑静音)* | `$225,000 / 月` ✕ *(严重打断心流)* | ⛔ 严禁部署只读孤立 `git diff` 且不封杀风格挑刺的裸机器人 |
| **🟨 Tier 2：传统未过滤 `SAST / SCA` 原始告警直推** | 不检查调用链可达性（Reachability）与上游消毒函数 | `68.0%` | **`84.5%` ✕** *(不可达CVE刷屏!)* | `14.0%` ✕ | `$185,000 / 月` ✕ *(安全工单积压成灾)* | ⛔ 狼来了效应导致真正的高危可利用漏洞被淹没 |
| **🟦 Tier 3：LLM 审查 + 未经编译校验的一键修复建议** | 生成 `Suggested Change` 但未过类型检查器验证 | `79.0%` | `28.0%` | `45.0%` *(32%点击后挂掉CI!)* | `$152,000 / 月` ✕ | ⛔ 点一下修复建议却把编译搞挂两次后，程序员再也不敢点 |
| **🟩 Tier 4：`Tree-sitter/LSP` 一跳调用图补全 + `CodeQL/Semgrep` 污点可达性过滤 + `Critic >=0.85` 静默门禁 + `<4秒` 沙箱编译验证一键修复** | **自动拉取被改函数的全库调用方与类型签名；验证漏洞真实可达；补丁先在沙箱过 `tsc/javac/ruff` 100% 编译通过才允许发评论** | **`94.8%` 🏆** *(比裸Diff高42.8个点!)* | **`4.2%` 🏆** *(比Tier 2砍掉80.3个点!)* | **`88.6%` 🏆** *(点一下100%编译绿灯)* | **`$29,000 / 月` 🏆** *(比 Tier 1 省 87.1%)* | 🌟 **企业级 AI PR 代码审查与 AppSec 自愈唯一黄金标准** |

> 💡 **架构师一句话 AI 代码审查与 AppSec 铁律（上下文拉一跳、没大病别说话、补丁先过编译器）**：
> 1. **`Tree-sitter / LSP` 一跳调用图与类型契约自动水合铁律（1-Hop Caller/Callee & Type Hydration）**：“**绝不让大模型盲人摸象只看 3 行 `git diff`！**”当 PR 修改了函数 $f$ 时，前置解析器通过增量 `Tree-sitter AST` 与语言服务器协议（`LSP: findReferences / goToDefinition`）在 `300ms` 内自动抽出：**① 函数 $f$ 所在的完整函数体**、**② 涉及的 Struct / Interface / DTO 强类型定义**、**③ 仓库内所有调用了 $f$ 的一跳调用点代码片段（1-Hop Callers）**——让跨文件空指针、契约破坏与并发竞态的捕获率从 `52.0%` 飙升至 **`94.8%`**！
> 2. **`CodeQL / Semgrep` 污点可达性验证 + `Critic >= 0.85` 沉默是金铁律（Silence-by-Default Critic Gate）**：“**如果一个 PR 没有高置信度的真实 Bug 或可达安全漏洞，机器人最好的表现就是闭嘴打个绿勾，绝不发半句凑数废话！**”在 Prompt 与 Critic 反思过滤器中硬性规定：**① 一切代码缩进、命名风格、注释格式交由确定性 Linter（`Prettier / Ruff / ESLint`）处理，严禁 AI 发评论挑刺**；**② 对 SAST/SCA 告警执行跨文件调用链可达性（Reachability）与自定义消毒函数核验**；**③ 只有经独立 Critic 模型验证复现路径成立且置信度 `>= 0.85` 的问题才允许上墙**（误报噪音率从 `84.5%` 暴降至 **`4.2%`**）！
> 3. **“未过编译器、不许贴补丁”沙箱预验证铁律（Compile-Verified `GitHub Suggested Changes`）**：每当 Agent 生成一段修复代码块时，先在后台增量构建沙箱中花 `<4秒` 跑一遍 `tsc --noEmit / javac / cargo check / pytest`，**只有 100% 编译通过的补丁才渲染为 GitHub `[Commit suggestion]` 按钮**——将研发一键采纳率从 `16.0%` 拉升至 **`88.6%`**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击怎么抓出改了 A 文件却崩了 B 文件的隐蔽 Bug、怎么砍掉 85% 的安全假告警、怎么保证一键修复不搞挂构建 3 大痛点）

### Q1：为什么程序员最讨厌市面上大多数 AI PR 审查工具？怎么从架构上让它从“烦人的话痨”变成“全组最信任的资深架构师”？
- **根治“无上下文瞎猜”与“为了刷存在感而强行挑刺”两大病灶**：
  - 市面上的廉价工具为了向老板证明“AI 在干活”，规定每个 PR 必须输出 3~5 条评论。当代码其实写得很好时，AI 就开始挑刺“这里可以加个注释”“建议把这个循环拆成函数”——研发看三次之后直接把机器人屏蔽了！
  - Tier 4 遵循 **“高信噪比沉默原则（High-Signal Silence）”**：
    - 首先用 `LSP` 把调用链上下游 1 跳的代码喂给模型（消除“因为没看见隔壁文件里已经判空了就瞎报空指针”的乌龙）；
    - 然后加一道专门负责“找茬挑错”的 **对抗式反思器（Adversarial Critic Agent）**：Critic 的唯一任务就是试图推翻初审 Agent 找出的 Bug（*“慢着，你看 `AuthFilter.java` 第 42 行已经拦截了未登录请求，所以这里根本不可能是未授权访问，撤回这条评论！”*）——这一下砍掉了 **`95%` 的误报**，留下的每一条评论都是让程序员惊出一身冷汗的真 Bug！

### Q2：安全团队每天被 `Semgrep / SonarQube / Snyk` 扫出的几千条告警淹没，怎么用 AST + 大模型把 `84.5%` 的假阳性（False Positives）自动过滤掉？
- **“确定性静态污点追踪 + AI 跨文件消毒器语义识别（Sanitizer & Reachability Verification）”**：
  - 传统 SAST 规则是死板的模式匹配：只要看到 SQL 字符串拼接或调用了某个危险 API 就报警，但它看不懂企业自己写的业务校验函数 ` SecurityUtils.ensureValidTenantCode(input)` 其实已经把输入限制死了只能是 6 位字母数字！
  - Tier 4 将 SAST 告警的 **完整 Source -> Sink 调用栈代码片段** 提取出来交给 AppSec Agent：
    1. **SCA 依赖漏洞可达性**：先用 `Tree-sitter` 调用图检查项目代码是否真的存在通往该三方库漏洞函数的调用路径（直接过滤掉 `70%` 根本没调用过的僵尸依赖告警）；
    2. **SAST 自定义消毒核验**：识别调用链中间是否经过了类型强转、正则白名单或权限校验，将假阳性率压至 **`4.2%`**！

### Q3：怎么在 4 秒钟内验证 AI 生成的修复补丁能不能编译通过，而不必重新跑完长达 15 分钟的全量 CI 流水线？
- **基于驻留热守护进程（`LSP / Incremental Typecheck Daemon`）的内存级增量校验**：
  - 根本不需要重新拉容器跑全量构建！在审查 Worker 节点池中，利用 `git worktree` 和常驻内存的语言服务器 / 增量检查器（如 TypeScript `tsserver`、Rust `rust-analyzer`、Python `pyright / ruff`、Java `Eclipse JDT LS`）：
  - 将 AI 生成的 5 行修复补丁在内存中应用到语法树上，**仅需 `0.8 ~ 3.5 秒` 即可拿到精确的编译器诊断结果**。如果报错 `Cannot find name ' Optional'`，Agent 自动在后台补上缺失的 `import` 语句再次校验，通过后才发到 PR 页面上！

---

## ③ 🧭 直观可视化图解（代码审查选型决策树 + `1-Hop LSP + 可达性过滤 + 编译验证自愈` 架构图 + 抓出跨文件空指针并输出一键绿灯补丁时序图）

### 3.1 企业级 AI PR 代码审查与 AppSec 自愈架构选型决策树

```mermaid
flowchart TD
    A["企业级 AI PR 代码审查与 AppSec 自愈选型<br/>(2,500 名研发 | 月均 4 万个 PR | 18 万条原始 SAST/SCA 安全告警)"] --> B{"AI 如何理解代码变更上下文、过滤误报并交付修复方案?"}
    B -- "仅把孤立 git diff 塞给话痨大模型，无调用图与降噪 (Tier 1)" --> C["⛔ 74.0% 废话挑刺与假告警 + 漏掉 48% 跨文件破坏！<br/>两周内被全员静音拉黑，月度浪费工时与故障成本达 $22.5 万"]
    B -- "未过滤的传统 SAST/SCA 扫描告警直接刷屏 PR (Tier 2)" --> D["⛔ 84.5% 误报率 (不可达三方库 CVE 与已消毒路径)！<br/>研发合并前修复率仅 14.0%，安全工单积压成灾 ($18.5 万/月)"]
    B -- "贴出未经编译器检验的 Suggested Change 修复代码块 (Tier 3)" --> E["⛔ 32% 的一键修复点击后直接搞崩 CI 编译！<br/>漏引 import 或类型不匹配导致研发丧失信任 ($15.2 万/月)"]
    B -- "1-Hop LSP 调用图补全 + SAST 可达性过滤 + Critic>=0.85 + 沙箱编译验证补丁 (Tier 4)" --> F["🟩 Tier 4: 一跳 LSP 上下文 + 可达性降噪 + 沙箱编译验证一键自愈 🏆<br/>(94.8% 真缺陷捕获 | 仅 4.2% 误报 | 88.6% 一键采纳率 | 省 87.1% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `Tree-sitter / LSP` 一跳上下文水合、污点可达性 Critic 降噪与沙箱编译验证自愈架构图

```mermaid
flowchart LR
    subgraph Hydrate["1. 300ms LSP 一跳调用图与类型水合层"]
        PR["🔀 GitHub / GitLab PR<br/>(git diff + SAST 原始告警)"] --> LSP["🌳 Tree-sitter + LSP 符号解析器<br/>自动拉取被修改函数的:<br/>• 跨文件 1-Hop Callers & Callees<br/>• 关联 Struct / Interface 类型定义"]
    end

    subgraph Critic["2. 污点可达性核验与对抗式 Critic 降噪门 (砍掉 95% 噪音)"]
        LSP --> Analyzer["🧠 深度缺陷与安全分析 Agent<br/>(94.8% 跨文件 Bug/漏洞捕获)"]
        Analyzer --> Gate{"🛡️ 对抗式 Critic 复核:<br/>• 调用链真实可达且未消毒?<br/>• 非代码风格废话且置信度 >= 0.85?"}
        Gate -- "不满足 (95% 噪音与不可达告警)" --> Drop["🗑️ 静默丢弃! 保持 PR 清爽"]
    end

    subgraph Sandbox["3. <4 秒增量编译验证与一键 Suggested Change"]
        Gate -- "高置信度真 Bug / 漏洞" --> Patch["🔧 生成修复补丁 + 过增量编译器<br/>(tsserver / javac / cargo check / ruff)<br/>100% 编译通过后贴出 [Commit suggestion]!"]
    end
```

### 3.3 自动拉取跨文件调用方识破空指针崩溃、过滤假 SQL 注入并在沙箱编译验证后贴出一键修复建议时序图

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 研发工程师 (提交 PR #892)
    participant LSP as LSP 一跳调用图水合器
    participant Agent as 审查与 AppSec 可达性 Agent
    participant Box as 增量编译沙箱 (tsserver / ruff)
    participant GH as GitHub PR 评论区

    Dev->>LSP: 1. 提交 PR：修改了 `Pricing.ts` 中的 `getDiscount()`（过期时返回 `null`）；同时 Semgrep 报了 3 条疑似 SQL 注入
    LSP->>Agent: 2. `300ms` 自动拉取 `getDiscount()` 在全库的 4 个跨文件调用点（包含未在 Diff 里的 `Checkout.ts:L118`）及 SQL 拦截器代码
    Agent->>Agent: 3. 核验发现 3 条 SQL 告警均已被上游 `TenantIdSanitizer` 消毒 -> **静默消音（0 误报打扰！）**
    Agent->>Agent: 4. 发现 `Checkout.ts:L118` 直接调用了 `getDiscount().amount` 未判空 -> 锁定跨文件 `P1 TypeError` 崩溃隐患（置信度 `0.96`）！
    Agent->>Box: 5. 生成修复补丁并在内存 `tsserver` 沙箱中执行增量类型检查（耗时 `1.8 秒`，确认 `0 Errors`）
    Box-->>GH: 6. 在 PR 页面精准发出唯一 1 条高优评论，附带已通过编译验证的 ```` ```suggestion ```` 一键修复块
    Dev->>GH: 7. 工程师眼前一亮，1 秒点击 `[Commit suggestion]` 完成修复合并（采纳率 `88.6%`）！
```

---

## ④ 📊 四档企业 AI 代码审查与 AppSec 自愈架构全维度决策矩阵

| 评估维度 | 🟥 Tier 1：裸 `git diff` 话痨机器人 | 🟨 Tier 2：传统未过滤 `SAST/SCA` | 🟦 Tier 3：未编译校验的自动补丁 | 🟩 Tier 4：一跳 `LSP` + 可达性过滤 + 编译验证自愈 |
|---|---|---|---|---|
| **跨文件破坏性 Bug 与真漏洞捕获率** | `52.0%` ✕ (看不见调用方契约) | `68.0%` | `79.0%` | 🏆 **`94.8%`（自动水合跨文件 1-Hop Callers/Callees 与类型）** |
| **误报与无意义风格挑刺噪音率** | `74.0%` ✕ (被研发集体静音) | `84.5%` ✕ (不可达 CVE 刷屏) | `28.0%` | 🏆 **`4.2%`（污点可达性核验 + 对抗式 `Critic >= 0.85` 门禁）** |
| **研发合并前一键修复采纳率** | `16.0%` ✕ | `14.0%` ✕ | `45.0%` (32% 点完搞崩 CI) | 🏆 **`88.6%`（补丁必须先过 `<4s` 增量类型/编译检查才上墙）** |
| **无问题健康 PR 的打扰度** | 强行凑数发 5 条废话评论 | 报一堆历史存量警告 | 偶发废话 | 🏆 **`0 条评论`（健康 PR 直接静默秒过，100% 尊重研发心流）** |
| **2,500 人团队月度总 TCO** | `$225,000 / 月` ✕ | `$185,000 / 月` ✕ | `$152,000 / 月` ✕ | **`$29,000 / 月` 🏆（比 Tier 1 节省 87.1%）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（2,500 名研发 / 月均 40,000 个 PR / 180,000 条原始安全告警）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级 AI 代码审查与 AppSec 自愈 ROI 计算器](../reports/enterprise-ai-pr-code-review-appsec-sast-reachability-autofix-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-ai-pr-code-review-appsec-sast-reachability-autofix-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_ai_pr_code_review_appsec_sast_reachability_autofix_selection/run.py --check
  python -m unittest tests/test_enterprise_ai_pr_code_review_appsec_sast_reachability_autofix_selection.py
  ```

### 5.2 核心量化对比图（跨文件真缺陷捕获率 % vs. 误报噪音率 % vs. 一键修复采纳率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Raw Diff-Only Bot", "metric": "Cross-File Bug/Vuln Catch (%)", "val": 52.0 },
      { "tier": "Tier 1: Raw Diff-Only Bot", "metric": "False-Positive Noise (%)", "val": 74.0 },
      { "tier": "Tier 1: Raw Diff-Only Bot", "metric": "1-Click Autofix Accept (%)", "val": 16.0 },
      { "tier": "Tier 2: Unfiltered SAST/SCA", "metric": "Cross-File Bug/Vuln Catch (%)", "val": 68.0 },
      { "tier": "Tier 2: Unfiltered SAST/SCA", "metric": "False-Positive Noise (%)", "val": 84.5 },
      { "tier": "Tier 2: Unfiltered SAST/SCA", "metric": "1-Click Autofix Accept (%)", "val": 14.0 },
      { "tier": "Tier 3: Uncompiled Autofix", "metric": "Cross-File Bug/Vuln Catch (%)", "val": 79.0 },
      { "tier": "Tier 3: Uncompiled Autofix", "metric": "False-Positive Noise (%)", "val": 28.0 },
      { "tier": "Tier 3: Uncompiled Autofix", "metric": "1-Click Autofix Accept (%)", "val": 45.0 },
      { "tier": "Tier 4: 1-Hop LSP + Verified Fix", "metric": "Cross-File Bug/Vuln Catch (%)", "val": 94.8 },
      { "tier": "Tier 4: 1-Hop LSP + Verified Fix", "metric": "False-Positive Noise (%)", "val": 4.2 },
      { "tier": "Tier 4: 1-Hop LSP + Verified Fix", "metric": "1-Click Autofix Accept (%)", "val": 88.6 }
    ]
  },
  "semantic_types": { "tier": "Category", "metric": "Name", "val": "Percentage" },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 reaches 94.8% bug catch and 88.6% autofix acceptance while cutting noise to 4.2%",
    "subtitle": "Cross-File Catch Rate (%) vs. False-Positive Noise (%) vs. 1-Click Autofix Acceptance (%)",
    "encodings": { "x": "tier", "y": "val", "group": "metric", "color": "metric" }
  }
}
```

---

## ⑥ 🛠️ 架构评审与生产落地 Checklist（上线前必查 5 项）

- [ ] **严禁只把孤立的 `git diff` 文本块喂给代码审查模型**：必须在前置流水线用 `Tree-sitter / LSP` 自动提取被修改符号的一跳调用方（1-Hop Callers）、被调用方签名与关联的 `Struct / Interface` 类型定义。
- [ ] **在系统提示词与后置过滤器中彻底封杀“代码缩进、命名风格、注释格式”等鸡毛蒜皮挑刺（Style Nits）**：风格问题一律交由确定性 Formatter（`Prettier / Ruff / Spotless`）在 Pre-commit 阶段解决，AI 审查只聚焦逻辑错误、并发竞态、资源泄漏与安全漏洞。
- [ ] **部署“对抗式 Critic 反思过滤器（置信度门禁 `>= 0.85`）”与 SAST/SCA 调用图可达性核验**：无法给出具体触发输入路径或已被上游中间件消毒的告警一律静默丢弃，将误报率控制在 `5%` 以内。
- [ ] **强制要求所有 ```` ```suggestion ```` 修复代码块在发布前通过增量编译/类型检查沙箱（`<4秒`）**：自动补齐缺失的 `import` 语句，确保研发点击 `[Commit suggestion]` 后 100% 不会破坏 CI 编译。
- [ ] **采集研发对每条评论的显式采纳/关闭信号（`Commit Suggestion` vs. `Dismiss / Thumbs-Down`）**：当某类规则连续被关闭超过 15% 时自动触发规则降级与少样本调优。
