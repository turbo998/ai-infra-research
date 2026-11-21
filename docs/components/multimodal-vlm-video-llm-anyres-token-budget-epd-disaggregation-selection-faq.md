# 多模态视觉语言模型（`VLM / Video-LLM`）推理架构选型：动态分辨率（`NaViT / AnyRes`）视觉 Token 手算、三阶段 `E-P-D` 分离（`Encoder-Prefill-Decode`）与视觉哈希缓存指南

> **语言 / Language**: **中文 (当前)** | [English](./multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect 与多模态系统架构师的图片/视频视觉 Token 精确手算公式（$N_{\text{vis}} = \lceil H/28 \rceil \times \lceil W/28 \rceil$）、三阶段 `E-P-D`（`ViT 编码池 -> LLM Prefill 池 -> LLM Decode 池`）异构硬件解耦与 `SHA-256` 视觉嵌入/KV 缓存选型白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业多模态推理平台容量规划与架构设计）

![多模态 VLM 与 Video-LLM 视觉 Token 手算及三阶段 E-P-D 分离总结图](../../assets/svg/multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**多模态大模型（如 `Qwen2.5-VL-72B`、`InternVL-78B`）在线上最容易引发雪崩的根源有两个：① 视觉 Token 爆炸——按 `14×14 Patch + 2×2 Pixel-Shuffle（即每 28×28 像素 = 1 个视觉 Token）` 计算，1 张未限流的 `4K (3840×2160)` 高清图纸会瞬间暴涨出 **`10,764` 个视觉 Token**（占 `1.76 GB` KV 显存，相当于塞进了一整本中篇小说！）；② 异构阶段互相阻塞——`0.6B ViT 视觉编码器`（纯算力、零 KV Cache）、`72B LLM Prefill`（二次方算力）与 `72B LLM Decode`（HBM 带宽受限）混跑在同一张 `H200` 上，会导致 Decode P99 TPOT 劣化至 `68.4ms`！生产最优解是实施“按任务分级 `max_pixels` 预算 + 三阶段 `E-P-D` 异构解耦（`L40S` 跑 ViT 编码池 $\to$ `H100` 跑 Prefill 池 $\to$ `H200` 跑 Decode 池）+ 基于图片 `SHA-256` 的视觉 Embedding 与 Prefix KV 双重缓存”，将首字延迟压缩 `8.0 倍`（`480ms`）、P99 TPOT 压至 `15.2ms`，并降低 `78.2%` 多模态推理成本！**

- **交互式图片/视频视觉 Token 与 `E-P-D` 容量计算器**：[打开交互式计算器 (`multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-sizer.html`)](../reports/multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-selection-cards.html`)](../reports/multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：给定一张分辨率为 $H \times W$ 的图片或一段时长为 $D$ 秒的视频，到底会转化成多少个视觉 Token？占用多少 KV Cache？
1. **单张图片的视觉 Token 手算公式（主流 `NaViT / 2×2 MLP Merger` 架构）**：
   - 底层 ViT 按 $14 \times 14$ 像素切 Patch，随后经过一层 $2 \times 2$ 空间相邻 Token 合并（Pixel-Shuffle / Patch Merger），即 **每 $28 \times 28 = 784$ 个像素生成 1 个最终送入 LLM 的视觉 Token**：
     $$N_{\text{vis\_image}} = \left\lceil \frac{H}{28} \right\rceil \times \left\lceil \frac{W}{28} \right\rceil$$
   - **典型分辨率速查**：
     - `720p (1280×720)`：$\lceil 1280/28 \rceil \times \lceil 720/28 \rceil = 46 \times 26 = \mathbf{1,196\text{ Tokens}}$（FP8 KV $\approx 196\text{ MB}$）；
     - `1080p (1920×1080)`：$69 \times 39 = \mathbf{2,691\text{ Tokens}}$（FP8 KV $\approx 441\text{ MB}$）；
     - `4K (3840×2160)`：$138 \times 78 = \mathbf{10,764\text{ Tokens}}$（单张图 FP8 KV 高达 **`1.76 GB`**！）。
2. **视频理解的视觉 Token 手算公式（含 `2:1` 3D 时空卷积压缩）**：
   - 设视频时长为 $D$ 秒，抽帧率为 $f\text{ FPS}$，每两帧在时间维合并为 1 个 3D Tubelet：
     $$N_{\text{vis\_video}} = \left\lceil \frac{D \times f}{2} \right\rceil \times N_{\text{vis\_frame}}$$
   - 例如 1 分钟（$D=60\text{s}$）、抽帧 $f=2\text{ FPS}$ 的 `720p` 视频：共 $\frac{120}{2} \times 1,196 = \mathbf{71,760\text{ 个视觉 Token}}$，单条视频请求在 `72B GQA` 模型上需占用 **`11.75 GB` KV Cache**！

### Q2：为什么多模态模型比纯文本模型更需要从“两阶段 P/D 分离”升级到“三阶段 `E-P-D`（Encoder-Prefill-Decode）异构分离”？
- **三个阶段的硬件脾气截然不同**：
  1. **E 阶段（`Vision Encoder / ViT`）**：参数量极小（通常仅 `0.4B ~ 6B`），**完全不产生 KV Cache**，是纯粹的密集算力小矩阵乘——如果跑在昂贵的 `8x H200 (1,128 GB HBM3e)` 上，98% 的大显存带宽全被浪费，且会抢占 LLM 的 SM 时间片！将其独立拆到高性价比的 **`L40S (48GB)` 编码池**，硬件单价直降 `65%`；
  2. **P 阶段（`LLM Prefill`）**：接收 `L40S` 算好的紧凑视觉 Embedding 特征，在算力强悍的 **`H100 (TP=4/8)` 池** 上并行完成数千个视觉 Token 的注意力 Prefill；
  3. **D 阶段（`LLM Decode`）**：通过 `400G GPUDirect RDMA` 接收 KV Cache，在大显存高带宽的 **`H200` 池** 上稳定输出文本回答（P99 TPOT 稳在 **`15.2 ms`**）！

---

## 3. 🧭 直观可视化图解（Mermaid 三阶段 E-P-D 异构流水线）

### 3.1 多模态三阶段 `E-P-D`（`L40S ViT 池 -> H100 Prefill 池 -> H200 Decode 池`）架构流
```mermaid
flowchart LR
    Img["输入图片/视频流<br/>(计算 SHA-256 指纹)"] --> Cache{"命中视觉缓存？"}
    Cache -- "未命中 (新图片)" --> E["Stage 1: E 池 (Vision Encoder)<br/>机型: NVIDIA L40S 48GB<br/>• 跑 0.6B~6B ViT + 2x2 Merger<br/>• 零 KV 占用 · 成本极低"]
    E -->|"写入 SHA-256 Embedding 缓存<br/>并流式推送视觉特征 (~15MB)"| P["Stage 2: P 池 (LLM Prefill)<br/>机型: NVIDIA H100 TP=4/8<br/>• 跑 72B 大模型视觉+文本 Prefill"]
    Cache -- "命中历史图片!" --> P
    P -->|"400G GPUDirect RDMA<br/>逐层传输 KV Cache"| D["Stage 3: D 池 (LLM Decode)<br/>机型: NVIDIA H200 141GB<br/>• 零 ViT/Prefill 抢占 · TPOT 15.2ms"]
```

---

## 4. 📊 架构师多方案选型决策矩阵（`72B VLM` 四档多模态部署方案对比）

| 评估维度 | Tier 1：单体混部（`ViT+P+D` 同卡）+ 无限制 4K 原图 | Tier 2：单体混部 + 任务分级 `max_pixels` 封顶 | Tier 3：两阶段 P/D 分离（`ViT` 仍混在 `H200` Prefill 卡上） | Tier 4（推荐）：三阶段 `E-P-D` 异构分离（`L40S + H100 + H200`）+ `SHA-256` 视觉缓存 |
| :--- | :--- | :--- | :--- | :--- |
| **P95 首字延迟 (`TTFT`)** | `3,850 ms` ✕ | `1,240 ms` | `820 ms` | **`480 ms` ✓（提速 `8.0 倍`）** |
| **P99 逐字延迟 (`TPOT`)** | `68.4 ms` ✕（被 ViT/Prefill 卡死） | `36.2 ms` ✕ | `17.8 ms` ✓ | **`15.2 ms` ✓（提速 `4.5 倍`）** |
| **折算每节点达标吞吐 (`Goodput`)** | `2.4 RPS` | `6.8 RPS` | `11.2 RPS` | **`16.5 RPS` ✓（提升 `6.88 倍`）** |
| **每 1,000 次多模态请求成本** | `$14.20 / 1K` | `$6.40 / 1K` | `$4.95 / 1K` | **`$3.10 / 1K` ✓（降低 `78.2%`）** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **网关与引擎层强制按场景设置 `min_pixels / max_pixels` 上限**：普通看图对话封顶 `1,280` 视觉 Token；高密财报/图纸 OCR 封顶 `4,096` 视觉 Token，严禁裸放無限制 4K/8K 原图进模型。
- [ ] **高并发多模态集群实施三阶段 `E-P-D` 解耦**：将无状态的 `ViT` 剥离至 `L40S / A10G` 算力卡，释放 `H200` 宝贵的 HBM3e 显存。
- [ ] **开启基于图片字节 `SHA-256` 的视觉 Embedding 缓存与多轮 Prefix KV 缓存**。
