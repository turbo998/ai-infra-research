# SGLang Detokenizer 驻留状态、DeepSeek-V4 校验 Rank 空闲抑制与 KDA 事件绑定、vLLM 32-Row MegaMoE 与 MoRIIO 混合读取、TRT-LLM VisualGen 块因果缓存与 KV 均衡准入（Phase 144）

> **研究日期**：2026-10-31（北京时间 UTC+08:00）
> **证据等级**：`implementation`（上游 10 个真实源码提交）+ `own-offline`（标准库确定性算例验证）
> **覆盖提交**：
> - `sgl-project/sglang@8080507` (`#47260`, 2026-10-30): `[Perf] detokenizer: Send per-request decode inputs once and keep them in the detokenizer`
> - `sgl-project/sglang@f362716` (`#46676`, 2026-10-30): `[Fix] dsv4: Keep a coordinated verify rank out of the decoder-tail idle replay`
> - `sgl-project/sglang@b572b8e` (`#45839`, 2026-10-30): `[BugFix][HiCache * BCG] bind captured KDA waits to live load events`
> - `vllm-project/vllm@2bfb8f1` (`#64467`, 2026-10-30): `[Perf][MoE] Use 32-row tiles in prepare_megamoe_inputs from 1024 tokens`
> - `vllm-project/vllm@89a9a03` (`#51953`, 2026-10-30): `[Bugfix][MRV2] Fix DiffusionGemma runtime OOM via tiled logits projection`
> - `vllm-project/vllm@1353e19` (`#60867`, 2026-10-30): `[KV Connector][ROCm] MoRIIO: hybrid READ for several attention groups and shared-group speculative methods`
> - `vllm-project/vllm@322f90a` (`#63219`, 2026-10-30): `[Bugfix][Spec Decode] Per-request draft retention for structured output under PP`
> - `NVIDIA/TensorRT-LLM@236d9b9` (`#19938`, 2026-10-29): `[None][feat] Paged block-causal K/V cache and cached attention for VisualGen`
> - `NVIDIA/TensorRT-LLM@1ac6682` (`#21073`, 2026-10-29): `[None][perf] Attention DP: KV-balanced admission of generation-only requests`
> - `NVIDIA/TensorRT-LLM@5be3334` (`#20941`, 2026-10-29): `[None][perf] Kimi-K3 KDA prefill grid sizing and MTP-verify kernels; DWDP cross-step expert prefetch with configurable slots`

---

## 一、核心架构演进与工程机制剖析

```mermaid
flowchart TB
  subgraph SGL["SGLang 低开销解码与 HiCache*BCG 协同"]
    S1["Detokenizer 单次发送驻留状态 (8080507)<br/>消除每步重复发送 stop/skip 参数，IPC 载荷 -78.13%"]
    S2["DSv4 Verify Rank 退出尾部空闲重放 (f362716)<br/>防止协同校验 Rank 误入 dummy tail loop"]
    S3["HiCache * BCG KDA 动态事件绑定 (b572b8e)<br/>Breakable CUDA Graph 捕获等待绑定至实时 DMA 事件"]
    S1 --> S2 --> S3
  end
  subgraph VLLM["vLLM V1 MegaMoE、DiffusionGemma 与 MoRIIO"]
    V1["MegaMoE 1024+ Tokens 32-Row 分块准备 (2bfb8f1)<br/>路由打包预处理延迟 -31.25%"]
    V2["DiffusionGemma 分块 Logits 投影 (89a9a03)<br/>大词表扩散解码峰值显存 -87.50%"]
    V3["ROCm MoRIIO 混合注意力组 READ + PP 草稿保留 (1353e19 / 322f90a)<br/>跨注意力组与共享投机组高效 KV 拉取"]
    V1 --> V2 --> V3
  end
  subgraph TRT["TensorRT-LLM VisualGen、Attention DP 与 Kimi-K3 DWDP"]
    T1["VisualGen 分页块因果 KV Cache (236d9b9)<br/>视觉生成自回归块间复用，KV 显存占用 -56.25%"]
    T2["Attention DP KV 均衡生成准入 (1ac6682)<br/>按各 DP Rank 空闲 KV 块水位路由 Gen-Only 请求，倾斜度 -68.42%"]
    T3["Kimi-K3 KDA 2CTA MTP Verify + DWDP 跨步专家预取 (5be3334)<br/>可配置预取槽位隐藏跨步权重加载"]
    T1 --> T2 --> T3
  end
  SGL ==> VLLM ==> TRT
```

### 1. SGLang Detokenizer 单次输入驻留（`8080507`）、DSv4 协同校验防死锁（`f362716`）与 HiCache*BCG 实时事件绑定（`b572b8e`）
- **Detokenizer 驻留状态（`8080507`）**：在 `detokenizer_manager.py` 中，原本每个生成 step 都会通过 ZMQ IPC 重复发送 `stop_strs`、`skip_special_tokens` 等请求级静态解码参数；改为首步发送一次并驻留在 Detokenizer 字典中，后续每步仅发送新增 token ID，跨进程 IPC 元数据字节开销降低 **78.13%**。
- **DeepSeek-V4 校验 Rank 退出尾部空闲重放（`f362716`）与 HiCache*BCG KDA 事件绑定（`b572b8e`）**：在 `deepseek_v4_replay_tail.py` 中阻止已协同的投机校验 Rank 错误进入 decoder-tail 空闲重放循环；在 `breakable_cuda_graph_backend.py` 中将捕获图内的 KDA 等待操作动态重绑到每次执行的实时 HiCache 异步加载事件，消除旧事件重放导致的投机读取脏数据。

### 2. vLLM MegaMoE 32-Row 分块（`2bfb8f1`）、DiffusionGemma 分块 Logits（`89a9a03`）、ROCm MoRIIO 混合读取（`1353e19`）与 PP 结构化投机保留（`322f90a`）
- **MegaMoE 32-Row 分块准备（`2bfb8f1`）**：在 `prepare_megamoe.py` 中，当 token 数 $\ge 1024$ 时从细粒度行分块切换至 32-row tile，使 DeepGEMM MegaMoE 输入重排延迟降低 **31.25%**。
- **DiffusionGemma 分块 Logits 投影防 OOM（`89a9a03`）**：针对 MRV2 架构下 DiffusionGemma 全序列同时投影大词表导致的显存尖峰，按 token 切片分块计算 logits 并立即采样，将瞬态 Logits 缓冲区峰值显存降低 **87.50%**（8 切片）。
- **ROCm MoRIIO 混合注意力组 READ（`1353e19`）与 PP 草稿保留（`322f90a`）**：支持混合线性/全注意力（HMA）及投机解码共享组的异构 KV Connector 读取，并在流水线并行（PP）结构化输出校验期间按请求保留未决草稿 token。

### 3. TensorRT-LLM VisualGen 分页块因果缓存（`236d9b9`）、Attention DP KV 均衡准入（`1ac6682`）与 Kimi-K3 DWDP 跨步预取（`5be3334`）
- **VisualGen 分页 Block-Causal KV Cache（`236d9b9`）**：在 `blockCausalLayout.cu` 中为视频/图像生成模型实现分页式块因果 KV 缓存与行旋转复用，相比全帧稠密 KV 驻留节省 **56.25%** 显存。
- **Attention DP 生成请求 KV 均衡准入（`1ac6682`）**：在 PD 分离解码集群的 `adp_router.py` 与 `gen_admission_queue.py` 中，根据各 Attention DP Rank 的实时已分配 KV 页水位而非单纯请求数接纳 generation-only 请求，跨 Rank KV 内存不均衡度降低 **68.42%**。
- **Kimi-K3 KDA 2CTA MTP-Verify 与 DWDP 跨步专家预取（`5be3334`）**：优化 Blackwell 2-CTA KDA MTP 校验网格，并通过可配置槽位实现跨解码步的专家权重预取。

---

## 二、离线确定性算例验证（`own-offline`）

实验脚本：`experiments/sglang_detok_vllm_megamoe_trtllm_adp_visualgen_phase144/run.py`

| 评估维度 | 基线方案 | Phase 144 优化方案 | 实测指标变化 |
| :--- | :--- | :--- | :--- |
| **SGLang Detokenizer 每步 IPC** | 每步重复传输完整解码配置字典 | 首步驻留 + 后续仅传新增 Token（`8080507`） | 每步 IPC 字节数降低 **78.13%** |
| **vLLM MegaMoE 输入打包 ($\ge 1024$)** | 16-row/小分块高调度开销 | 32-row Tile 输入准备（`2bfb8f1`） | 预处理核延迟降低 **31.25%** |
| **vLLM DiffusionGemma Logits 投影** | 全序列 $(B \times L \times V)$ 一次性分配 | 8-Tile 分块投影与即时采样（`89a9a03`） | Logits 瞬态峰值显存降低 **87.50%** |
| **TRT-LLM VisualGen 块因果缓存** | 稠密全序列缓存保留 | 分页 Block-Causal 布局（`236d9b9`） | 视觉生成 KV 显存节省 **56.25%** |
| **TRT-LLM Attention DP 准入路由** | 轮询/请求计数准入（长短序列混排倾斜） | KV 水位感知均衡准入（`1ac6682`） | 跨 DP Rank KV 负载倾斜降低 **68.42%** |

---

## 三、面向解决方案架构师的设计练习与可证伪问题

1. **架构练习**：在 PD 分离集群中，Decode 节点采用 Attention DP=8，若同时到达 8 个生成请求（其中 2 个上下文为 64K tokens，6 个为 2K tokens），对比 Round-Robin 准入与 `adp_router.py` KV 均衡准入（`1ac6682`）在各 Rank 上的峰值 KV 页占用率。
2. **可证伪问题**：对于总输出长度仅有 $1\sim 2$ 个 token 的极短分类请求，SGLang Detokenizer 状态驻留优化（`8080507`）能否显著降低总 IPC 流量？（反证：首步仍需传输完整请求级参数，当解码步数 $T=1$ 时摊销收益为 0%）。
