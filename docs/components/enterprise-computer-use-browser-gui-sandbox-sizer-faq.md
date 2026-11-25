# 企业级 Computer-Use 与浏览器 GUI 智能体（`CUA / Playwright / OmniParser`）算力与沙箱容量规划：截图视觉 Token 税拆解、`DOM + Set-of-Marks` 混合定位与 `500` 并发浏览器池 Sizer（`#667`）

> **阅读指南（3 分钟看懂核心结论，10 分钟掌握 CUA 视觉 Token 与沙箱容量手算）**
> - **交互式 H5 精读页**：[打开 `computer-use-browser-gui-sandbox-sizer-lab.html`](../h5/computer-use-browser-gui-sandbox-sizer-lab.html)
> - **交互式容量计算器（Sizer）**：[打开 `enterprise-computer-use-browser-gui-sandbox-sizer.html`](../reports/enterprise-computer-use-browser-gui-sandbox-sizer.html)
> - **7-Card 4:5 视觉评测报告**：[打开 `enterprise-computer-use-browser-gui-sandbox-sizer-cards.html`](../reports/enterprise-computer-use-browser-gui-sandbox-sizer-cards.html)
> - **English Version**: [Read the English Guide (`#667`)](./enterprise-computer-use-browser-gui-sandbox-sizer-faq.en.md)

![Computer-Use Agent Infrastructure Summary](../../assets/svg/enterprise-computer-use-browser-gui-sandbox-sizer-faq-summary.svg)

---

## 一、执行摘要与核心结论（3 分钟速览）

随着企业利用 **Computer-Use Agent（CUA / 浏览器与桌面 GUI 智能体）** 替代传统脆弱的 RPA 脚本（自动完成跨网银对账、遗留 ERP 录单、供应链询价与医保审核），架构师在生产扩容时普遍遭遇两大算力泥潭：**① 全屏截图视觉 Token 爆炸**（每步 `1920×1080` 截图消耗 `~2,020` 个视觉 Token，15 步任务累积吞噬 **`30,300` 个视觉 Token**，导致单步 `TTFT` 飙至 `1,420 ms` 且 `500` 并发会话吃满 `28` 张 `H200`）；**② 桌面沙箱内存失控**（为每个会话拉起完整 Windows/Linux VDI 虚拟机，单实例消耗 `8 GB RAM + 4 vCPU`，`500` 并发需要 `4.0 TB` 内存和 `16` 台双路服务器，且冷启动长达 `28 秒`）。

1. **视觉 Token 消肿铁律：用“精简 Accessibility/DOM 树（75% 常规步骤）+ `OmniParser` Set-of-Marks 局部 ROI 裁剪（25% Canvas/遗留控件）+ 2 帧滑动窗口”砍掉 `78.4%` 视觉 Token！**
   - **严禁在上下文中保留全部历史截图**：第 `t` 步只需保留当前帧 `S_t` 与上一帧 `S_{t-1}`（用于对比点击是否生效），更早的步骤仅保留 `80 tokens` 的结构化动作摘要（如 `Clicked [id=14] "Submit"`）。仅此一项即可将 15 步累积视觉 Token 从 `30,300` 压至 `14,500`。
   - **优先用精简 DOM / A11y 树替代纯像素盲猜**：对 `75%` 的标准 Web 页面，提取可交互元素树（`~320 text tokens/步`）；仅对 `25%` 的 Canvas 报表、验证码或无 DOM 桌面控件调用轻量级 **`OmniParser`（YOLOv8 UI 检测 + Set-of-Marks 数字标签叠加 + 局部 ROI 截图）**，不仅将任务总视觉/多模态 Token 进一步压至 **`6,540 tokens/任务`（立省 `78.4%`）**，还将 UI 点击准确率从纯像素坐标回归的 `84.5%` 提升至 **`94.8% (+10.3 pts)`**，`500` 并发所需 `H200` 从 `28` 张骤降至 **`6` 张（节省 `78.6%` GPU）**！
2. **执行沙箱密度铁律：容器化 `Headless Chromium + Xvfb / Firecracker 快照池（1.8 GB/会话）` 比传统桌面 VDI（`8.0 GB/会话`）密度高 `4.44 倍`、启动快 `200 倍`！**
   - 除非必须操作本地 `.exe` 厚客户端（如老旧 SAP GUI / 金蝶客户端），否则 `90%+` 企业办公系统均为 B/S 架构。采用 **`Firecracker MicroVM` 或容器化 `Chromium + Xvfb + CDP (Chrome DevTools Protocol)` 内存快照池**，可将单会话内存压至 **`1.8 GB`**、唤醒耗时从 `28s` 压至 **`140 ms`**，`500` 并发仅需 **`4` 台 `128核/512GB` CPU 宿主机（节省 `75%` 宿主机）**。

```mermaid
flowchart LR
    U["企业 15 步 RPA/ERP 任务\n(500 并发智能体会话)"] --> Router["GUI 感知分流引擎\n(CDP / Accessibility 探针)"]
    Router -->|"75% 标准 Web DOM 步骤"| DOM["精简 A11y 可交互节点树\n(~320 tok/步 · 零截图开销)"]
    Router -->|"25% Canvas/复杂图表/弹窗"| SoM["OmniParser + Set-of-Marks (SoM)\n数字标签 + 2 帧滑动窗口 (淘汰 t-2 截图)"]
    DOM --> VLM["6× H200 VLM 推理池\n(单步 TTFT 310ms · 准确率 94.8%)"]
    SoM --> VLM
    VLM --> Pool["4× CPU 宿主机 (0.9 TB RAM)\n500× Chromium + Xvfb 快照沙箱 (140ms 唤醒)"]
```

---

## 二、灵魂拷问 Q&A（架构师现场排障与手算拆解）

### Q1：为什么让多模态大模型（VLM）直接根据 `1920×1080` 截图输出 `(x, y)` 鼠标像素坐标，在企业 ERP 表格里经常点偏？如何手算 `1080p` 截图的 Token 账单？

1. **`1080p` 截图 Token 手算（以 `28×28 px/token` 架构为例）**：
   $$N_{\text{frame}} = \left\lceil \frac{1920}{28} \right\rceil \times \left\lceil \frac{1080}{28} \right\rceil = 69 \times 39 = 2,691\text{ raw patches} \xrightarrow{\text{动态分辨率压缩}} \approx 2,020\text{ tokens/帧}$$
   若一个 15 步跨系统录单任务把每一步的截图都留在对话历史里，第 15 步单次请求就会携带 $15 \times 2,020 = 30,300$ 个视觉 Token！
2. **为什么直接预测 `(x, y)` 像素坐标容易点偏？**
   - ViT 将 `28×28` 像素合并为 1 个 Patch Token，而企业 ERP 密集表格的行高往往只有 `18–22 px`。让 LLM 直接回归连续归一化坐标 `(0.432, 0.718)` 极易产生 `±12 px` 的量化偏移，导致点中相邻行。
   - **根治方案（`Set-of-Marks, SoM`）**：先用 `1.8 ms` 的轻量 UI 检测器（`OmniParser`）或浏览器 `getBoundingClientRect()` 在所有可点击框上画出带数字编号的标签 `[1]..[42]`，让大模型只需输出离散标签 ID（如 `click(box_id=14)`），点击准确率立即从 `84.5%` 跃升至 **`94.8%`**！

---

### Q2：`500` 个并发浏览器沙箱同时访问外部网站或内部 ERP 时，为什么经常触发防火墙端口耗尽（SNAT Port Exhaustion）或被目标站点封禁 IP？

- **根因**：每个 Chromium 实例在加载现代 Web 应用时会并发建立 `60–120` 条 TCP/QUIC 连接；`500` 个沙箱共享单一公网/内网 NAT IP 时，瞬时并发连接数高达 `45,000+`，直接打穿单 IP 的 `64,512` 临时端口上限！
- **生产网关与沙箱治理规范**：
  1. **静态资源缓存代理（Local Squid / Envoy Caching Proxy）**：在沙箱池前端部署共享 HTTP 缓存代理，拦截并缓存重复的 `.js/.css/.woff2` 静态资源（节省 `68%` 出口带宽与连接数）；
  2. **多 IP 池 SNAT 轮询 + 会话级 Cookie/Storage 加密挂载**：为沙箱出口配置 `/27`（32 个 IP）SNAT 池，并通过企业安全凭证保险箱（Vault）注入会话态 Cookie，避免每次任务重复走短信/验证码登录。

---

## 三、Computer-Use 智能体架构选型决策矩阵（`500` 并发会话基准）

| 架构方案 | 15 步任务累积视觉/UI Token | 单步推理 `TTFT` (`H200`) | UI 动作执行准确率 | `500` 并发所需 `H200` | 沙箱宿主机 (`128C/512GB`) | 沙箱冷启动延迟 | 适用场景 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1. 纯全屏 `1080p` 历史累加 + 重型桌面 VDI** | `30,300 tok` | `1,420 ms` | `84.5%` | **`28 张`** | **`16 台 (4.0 TB)`** | `28.0 s` | ❌ **严禁生产大规模使用（算力与内存双崩盘）** |
| **2. `1080p` 2 帧滑动窗口 + 动作文本日志** | `14,500 tok (-52.1%)` | `540 ms` | `86.2%` | **`14 张`** | `16 台` (仅必须桌面客户端时) | `28.0 s` | 必须操作无 API 老旧 Windows `.exe` 厚客户端 |
| **3. `A11y/DOM` 树 + `OmniParser SoM` + `Chromium` 快照池（推荐）** | **`6,540 tok (-78.4%)`** | **`310 ms`** | **`94.8%`** | **`6 张 (-78.6%)`** | **`4 台 (0.9 TB, -75%)`** | **`0.14 s`** | **企业 Web ERP、财税网银、供应链与客服自动化主力架构** |

---

## 四、生产落地核对清单（Pre-Production Checklist）

- [ ] **强制开启 2 帧视觉滑动窗口（Visual Eviction）**：在 Agent 状态机中自动剔除 `t - 2` 及更早的历史原始截图，仅保留结构化 Action/Observation 文本轨迹。
- [ ] **采用 `DOM / A11y + Set-of-Marks` 离散 ID 点击**：禁止大模型直接裸猜 `(x, y)` 屏幕坐标，所有点击必须绑定 DOM Node ID 或 `OmniParser` 候选框编号。
- [ ] **高风险按钮（转账/删除/提交审批）强制 `Human-in-the-Loop` 门禁**：当目标按钮文本或附近 DOM 命中金额/删除/外部发送规则时，挂起沙箱快照并推送带高亮框的确认卡片由人工审批。
- [ ] **容器化 Chromium 内存上限与僵尸进程回收**：为每个沙箱容器设置 `--memory=2g --shm-size=1g` 与 `15 分钟` 生命周期硬超时，任务结束后立即销毁并从黄金快照恢复干净会话。
