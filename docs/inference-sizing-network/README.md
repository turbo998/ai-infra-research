# AI Infra 推理性能、智算网络与容量测算精学馆

**语言**：简体中文 | [English](README.en.md) · [打开交互 H5 精学馆](../h5-academy/index.html)

面向企业 **Cloud & AI Solution Architect**，把“用户为什么觉得慢”一路推导到“需要多少吞吐、显存、GPU 与网络”。不靠缩写堆砌、不把模拟当实测。每册有双语正文、可调 SVG 沙盘、手算复核、FAQ、自测与上线核对清单。

## 六册学习路径

| 专题 | 你能解决的问题 | 交互入口 |
|---|---|---|
| 用户核心指标 | TTFT、平均 TPOT、尾部 ITL、E2EL、RPS、Goodput 与成本口径 | [01 指标与生命周期](../h5-academy/01-metrics.html) |
| TPM/TPS 与容量 | QPS→输入/输出吞吐→Little's Law 在途并发→实例下界；PTU 归一化 | [02 容量规划](../h5-academy/02-tpm-tps-capacity.html) |
| 三道 GPU 水位线 | 总权重、KV、运行时、HBM/GDDR 带宽和 Prefill 算力的不同约束 | [03 GPU 显存与性能](../h5-academy/03-gpu-sizing.html) |
| PD 分离 | 混部与 chunking 干扰、KV 传输预算、两池成本和适用边界 | [04 PD 与 KV 交接](../h5-academy/04-pd-disaggregation.html) |
| 推理并行与加速 | TP/PP/DP/EP 切分、通信原语、投机、Prefix Caching、FP8 的条件性收益 | [05 并行与加速](../h5-academy/05-parallelism-acceleration.html) |
| 智算网络 | GPUDirect RDMA、RoCEv2、NVLink/NVSwitch、多轨、NCCL algbw/busbw | [06 网络精解](../h5-academy/06-rdma-roce-nvlink.html) |

**30 分钟速读**：01 生命周期 → 02 默认容量算例 → 03 三道水位线。

**晚间精读**：按六册顺序阅读，一次只改一个变量。先将并发从 1 增至 32，再将上下文从 2K 增至 32K，观察显存、MoE 专家并集、Goodput 和 KV 网络负载的瓶颈转移。

## 双语 FAQ 与证据边界

- [中文深度 FAQ、手算、决策矩阵与核对清单](faq.md)
- [English FAQ, calculations, decision matrix and checklist](faq.en.md)
- [原始来源目录与复核边界](../h5-academy/sources.json)，复核日期：2026-11-21。

**不是实测 benchmark 报告**：SLA目标、负载、价格、tradeoff曲线与P99曲线均为原创教学假设，页面中明确标记 `synthetic`。PD分别对200个请求均值取P99 TPOT、对同批6200个间隔取P99 ITL。硬件规格与软件能力分别标记 `specification` / `implementation`，不能推导出性能保证。未独立复现实机吞吐或延迟。

重要纠偏：L40S 使用 GDDR6 且没有 NVLink；B200 192 GB 设计/假设值与 DGX B200 180 GB/卡交付表分开；DeepSeek 671B 主模型之外，额外加载 MTP 需要预算 14B；80–85% 是架构规划余量，不是 vLLM 当前默认值。

## 离线与移动端使用

保留整个 `docs/h5-academy/` 目录，双击 `index.html`。七个页面均为自包含 HTML，CSS、SVG、计算代码、原生 JS 和双语内容内联，无 CDN、外部字体、图片或自动网络请求。每册也可以独立复制打开；跨册导航则需要相邻页面。

语言切换和明暗主题通过页面控件完成，并随跨册导航保留。手机首页支持横向卡片滑动，长表格/图表在自身区域滚动；支持键盘操作、输入错误提示、单 Token 的 N/A、打印/PDF 和减少动画偏好。没有用户数据上传或第三方统计。

## 复现生成与验证

在仓库根目录运行（仅需 Node.js 标准库；无 npm 安装）：

```powershell
node docs\h5-academy\build.mjs
node --test docs\h5-academy\academy.test.mjs
node docs\h5-academy\build.mjs --check
python scripts\validate_research.py
```

如机器已安装 Windows Microsoft Edge，可另外运行无第三方包的真实离线桌面/手机检查：

```powershell
node docs\h5-academy\browser-check.mjs
```

它在临时独立 profile 中加载七册，检查 1440px 中文浅色与 390px 英文深色、字段边界/选项/错误、自测、主题/语言、页面溢出、JS/CSP 和自动资源请求。临时探针只加入测试副本，不修改交付 HTML。

收口检查还覆盖320px英文深色、360px中文浅色、原生键盘操作、重点文本/图形对比度与较短的辅助技术状态播报。每册顶部提供一个“今晚先试”任务，图表坐标可展开成数据表。结果只是这些检查的证据，不是完整WCAG合规认证。

只检查受影响页面可使用：

```powershell
node docs\h5-academy\browser-check.mjs --only 03-gpu-sizing.html
```

源码位于 `h5-academy`：`content.mjs` 维护双语课程，`academy-math.cjs` 为可测试公式内核，`academy-ui.js` 为交互，`academy.css` 为主题，`build.mjs` 内联所有资源并为脚本生成 CSP 哈希。修改源码后必须重新生成 HTML；`--check` 可检测漂移。

生成器会拒绝缺失/重复来源ID、明显私网/带凭据URL、缺失中英片段与错误的实机测量归因，防止来源被静默丢弃。来源与公式仍需人工技术复核。

## 集成导航建议

统一发布时，将精学馆链接加入主仓库的推理/网络学习入口，并在个人主页对应 AI Infra 学习栏目归档七个 HTML 与 FAQ 链接。全部使用相对目录结构。此独立交付目录不修改共享总目录或根 README，避免并行集成冲突。
