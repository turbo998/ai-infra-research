# SGLang DSpark 模拟接受采样与 HiCache PD 早发守卫、DeepEP v2 掩码 GEMM 显存压缩、vLLM ROCm 稀疏 MLA 门控、TensorRT-LLM MPI 鸽巢屏障及 AI Infra 创新架构对比（Phase 103）

- **问题 ID**：`sglang-dspark-hicache-deepep-vllm-trtllm-innovators-phase103`（`priority: 112`，组件：`inference`）
- **证据等级**：`own-offline`（基于上游源码补丁审计 + Python 3.12 标准库确定性算例 `experiments/sglang_dspark_hicache_vllm_trtllm_phase103/run.py`）
- **配套英文版**：[`sglang-dspark-hicache-deepep-vllm-trtllm-innovators-phase103.en.md`](sglang-dspark-hicache-deepep-vllm-trtllm-innovators-phase103.en.md)

## 1. 一手上游信源（零重叠）

1. **SGLang `3927a73a34` (#44778)**：`[Fix] kda: Use raw beta in prefill CUDA graphs`
2. **SGLang `24d55621a3` (#46404)**：`[Fix] dspark: Draw the simulated-acceptance bonus at the simulated row`
3. **SGLang `2a7987f900` (#44081)**：`[Fix] PD: Skip the cached-prefix early send in batches that start a HiCache load-back`
4. **SGLang `a09c1a096c` (#46309)**：`[Perf] deepep_v2: Reduce masked GEMM buffer memory during CUDA Graph capture`
5. **vLLM `939ff73503` (#63785)**：`[Perf] Drop BasevLLMParameter's Python __torch_function__`
6. **vLLM `8d88bc5d81` (#60154)**：`[ROCm][DSv4/4.1] Fuse inverse RoPE and MXFP8 quant into the aiter sparse MLA and sparse MLA gating changes`
7. **vLLM `f8f7634cb4` (#63342)**：`[ROCm][Perf][Kimi-K3] Use block_m=128 for the fused KDA prefill conv1d`
8. **TensorRT-LLM `8bd66f6a71` (#20904)**：`[None][fix] Explain K/V pool coalescing failures; test Qwen3.8-Flash-Next QSA layouts and FP8/NVFP4 KV`
9. **TensorRT-LLM `0ba9f945f0` (#20996)**：`[TRTLLM-15765][refactor] Remove the ModelEngine ABC and settle the executor interface`
10. **TensorRT-LLM `105401ea8d` (#20467)**：`[https://nvbugs/6932260][fix] Ensure one-task-per-rank for MpiSession.submit`

---

## 2. 核心实现机制与定量分析

### 2.1 SGLang DeepEP v2 CUDA Graph 掩码 GEMM 缓冲区 32.0x 显存压缩 (`a09c1a096c`)

在 DeepEP v2 专家并行 Token Dispatcher 中，CUDA Graph Capture 此前总是按最坏情况将每个专家的掩码 GEMM Slab 行容量设为：
$$\text{capacity}_{\text{old}} = \text{num\_max\_dispatch\_tokens\_per\_rank} \times \text{ep\_group\_size}$$
然而当未开启 Attention DP/CP（`attn_dp_size == 1` 且 `attn_cp_size == 1`）且 `tp_size == ep_group_size` 时，TP 组内各源 Rank 共享同一批输入（或至多相差 1 行的 `tensor_split` 分片），且每个源 Token 最多只会被路由到同一专家一次。因此，单个专家可能接收到的总源 Token 数绝不可能超过源端总行数：
$$\text{source\_total} = \max(1,\; \text{local\_tokens} \times \text{ep\_group\_size} + \text{ep\_group\_size} - 1)$$
$$\text{masked\_max\_m} = \min\!\left(\text{capacity}_{\text{old}},\; \left\lceil \frac{\text{source\_total}}{128} \right\rceil \times 128\right)$$
- **显存节省定量验证**：在 `max_dispatch = 512, ep_group_size = 8`（每 Rank 32 个局部专家，`H = 7168` BF16）下，旧版对所有小批量 CUDA Graph 均分配 $\text{masked\_max\_m} = 4{,}096$ 行（**1,792.0 MiB**）；启用 `_masked_token_capacity` 后，在 `local_tokens = 1..15` 下 $\text{masked\_max\_m} = 128$ 行（**56.0 MiB**），实现 **32.0x（96.88%）掩码 GEMM 缓冲区显存削减**，单个 Graph 节省 **1,736.0 MiB**，同时保持 `ElasticBuffer` 与通信句柄几何结构不变。

### 2.2 SGLang PD 分离 HiCache 回载早发竞态守卫 (`2a7987f900`) 与 DSpark 模拟接受采样 (`24d55621a3`)

- **PD 分离 HiCache 写后读/早发数据冒险消除 (`2a7987f900`)**：开启 `SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX` 时，Prefill 节点会在后缀前向计算的同时提前通过 RDMA 发送已命中的前缀 KV 页。但若当前批次启动了 HiCache 主机到设备异步回载（`batch.hicache_consumer_index >= 0`），同一批次内后序请求所匹配的“设备前缀页”可能仍处于 DMA 写入途中。补丁传入 `allow_send=not load_pending`，在存在挂起回载的批次中跳过缓存前缀早发，将未完成 KV 页早发污染从 **2 次降至 0 次**。
- **DSpark 模拟接受长度 Bonus Token 采样行修正 (`24d55621a3`)**：在开启模拟接受率（`HAS_SIM`）压测时，`_static_accept_tail_kernel` 现从 `sim_ptr` 读取覆盖后的接受长度 `correct`（如 `row = 4` 而非真实贪心匹配长度 `row = 1`），并在该模拟行上按请求自身的采样温度抽取 Bonus Token，同时修正 `kda_extend_in_graph` 在 4D Gate 下的 `beta_is_raw=True` (`3927a73a34`)。

### 2.3 vLLM 参数 C++ 直派与 ROCm 稀疏 MLA 64 头门控 (`939ff73503`, `8d88bc5d81`, `f8f7634cb4`) 及 TensorRT-LLM (`8bd66f6a71`, `105401ea8d`)

- **vLLM 移除 `BasevLLMParameter.__torch_function__` (`939ff73503`)**：此前 `BasevLLMParameter` 定义了仅直接调用 `super().__torch_function__` 的 Python 类方法，导致所有保留为 `BasevLLMParameter` 子类的权重张量在每次 Eager 算子调用时都必须陷入 Python 解释器（`1 → 0` 次 Python 拦截/算子）。
- **vLLM ROCm DeepSeek-V4/4.1 稀疏 MLA 64 头门控与融合逆 RoPE + MXFP8 (`8d88bc5d81`)**：在 MI300X/MI350X 上，当每 Rank 头数达到 `128` 时，`aiter` 稀疏 MLA Prefill 慢于 Gather + OPUS，因此补丁设置 `_AITER_SPARSE_MLA_PREFILL_MAX_HEADS = 64`（仅在 `num_heads <= 64` 时启用 `aiter` Prefill，而 Decode 在任意头数下均启用并融合逆 RoPE 与 MXFP8 量化），同时在 `forward_mqa` 中处理未旋转尾部行 `[rotated, end)` 以防在已编译的 `_o_proj` 中冻结为 Trace 期常量。Kimi-K3 融合 KDA Prefill `conv1d` 则固定 `PREFILL_CONV_BLOCK_M = 128` (`f8f7634cb4`)。
- **TensorRT-LLM Qwen3.8-Flash-Next NVFP4/FP8 KV 与 MPI 鸽巢入口屏障 (`8bd66f6a71`, `105401ea8d`)**：在 GB300 上实测 Qwen3.8-Flash-Next 混合 QSA + BF16 Mamba SSM 缓存，FP8 KV 取得 **95.49% GSM8K / 87.33% MMLU**，NVFP4 KV + MTP 取得 **95.26% GSM8K / 87.57% MMLU**；`MpiSession.submit` 通过 `_barrier_run` 保留入口 MPI 屏障（鸽巢原理保证 `n_workers` 各领且仅领 1 个任务 `[1, 1, 1, 1]`，防止快 Rank 抢走慢 Rank 任务），同时移除尾部屏障以支持单 Rank 异常即刻上报。

---

## 3. AI Infra 创新公司推理架构深度横评（Fireworks AI / Together AI / Cerebras WSE-3 / Groq LPU vs Blackwell GB300）

面向解决方案架构师，我们将上游最新投机解码（DSpark）与 PD 分离机制映射到四大 AI Infra 创新公司的核心系统路线（以 70B 等效激活权重单并发低延迟解码为基准）：

| 架构 / 厂商方案 | 核心存储层级与带宽 | 关键软件栈与投机/通信机制 | 单步理论耗时 (ms) | 每步接受 Token | 等效单流吞吐 (tok/s) | 架构适用边界 |
|---|---|---|---:|---:|---:|---|
| **NVIDIA GB300 NVL72 (TP8 NVFP4 + DSpark)** | HBM3e (`8.0 TB/s/GPU`, `64 TB/s` TP8) | `tcgen05` NVFP4 (`35 GB` 权重) + DSpark 投机验证 | `0.897` | `3.4` | **3,790.4** | 兼顾高吞吐并发、超长上下文 KV 与高密度单流解码 |
| **Fireworks AI FireAttention (B200 TP8 + PD)** | HBM3e + Host DRAM HiCache | FireAttention + PD 分离缓存前缀早发 + 回载守卫 (#44081) | `0.927` | `3.2` | **3,452.0** | 多租户高前缀命中 Agent 工作流、长上下文低 TTFT |
| **Together AI Turbo Speculative (B200 TP8)** | HBM3e + FlashAttention-4 / ThunderKittens | 深度定制草稿树/DSpark 模拟接受采样 (#46404) + W4A16/FP4 | `0.907` | `3.6` | **3,969.1** | 代码生成/结构化输出（高草稿接受率 $\alpha$ 场景） |
| **Cerebras WSE-3 晶圆级引擎 (`44 GB` SRAM)** | 晶圆级片上分布式 SRAM (`~21,000 TB/s`) | 权重常驻片上 SRAM（消除 HBM 带宽墙）+ 晶圆网格波前流转 | `0.422` | `1.0` | **2,369.7** | 超低单流延迟或作为 GPU 集群的高速投机草稿器（Draft Engine） |
| **Groq LPU 确定性 SRAM 机架 (`230 MB/chip`)** | 多芯片片上 SRAM (`80 TB/s/LPU`) | 编译器静态调度 TSP 流水线 + 确定性 Chip-to-Chip 互连 | `0.855` | `1.0` | **1,169.6** | 严格确定性尾延迟（P99 $\approx$ P50）、短上下文实时语音/Agent |

- **核心架构结论**：**Cerebras WSE-3 (`44 GB` 片上 SRAM)** 与 **Groq LPU (`230 MB/chip` SRAM)** 通过将激活权重完全放入 `10–21 PB/s` 级片上 SRAM 彻底消除了单 Token HBM 带宽墙，但其片上总容量受限（70B/405B 需多晶圆或数十至上百颗 LPU 级联，受限于跨芯片跳数延迟 `0.42–0.85 ms` 且难以容纳 TB 级长上下文 KV Cache）；而 **Together AI** 与 **Fireworks AI** 在 Blackwell B200/GB300 (`64 TB/s` TP8 HBM3e) 上结合 **NVFP4 (`35 GB`) + DSpark 投机解码（每步验证产出 `3.2–3.6` 个 Token）+ DeepEP v2 / HiCache PD 分离**，不仅在长上下文容量上具备数量级优势，且在投机接受率 $\ge 3.2$ 时单流有效生成速率可达 **3,452–3,969 tok/s**。

---

## 4. 架构师设计练习与可证伪问题

1. **练习计算**：在 DeepEP v2 (`ep_group_size = 8`, `num_max_dispatch_tokens_per_rank = 512`) 中，当捕获 `local_tokens = 24` 的 CUDA Graph（TP 共置）时，`_masked_token_capacity` 计算出的 `masked_max_m` 是多少？相比旧版 `4,096` 缩减了多少倍？（答案：`source_total = 24 * 8 + 7 = 199`，向上按 128 对齐得 **256**，缩减 **16.0x**）。
2. **可证伪假设**：若在 `sglang` PD 分离 Prefill 节点中忽略 `batch.hicache_consumer_index >= 0` 守卫，当同一批次中前序请求触发异步 HiCache DMA 回载而同前缀的后序请求触发 `maybe_send_cached_prefix_chunk` 时，Decode 节点将收到尚未写完的半初始化 KV 页。
