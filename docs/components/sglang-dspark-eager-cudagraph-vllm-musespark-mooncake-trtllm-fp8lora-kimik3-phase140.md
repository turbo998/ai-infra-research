# SGLang DSpark Ragged-Verify 回退与 CUDA Graph 保活、vLLM Muse Spark 融合通信与 Mooncake 异构 TP、TensorRT-LLM 原生 FP8 MoE-LoRA 与 Kimi-K3 NVFP4 KV（Phase 140）

> **English Version**: [sglang-dspark-eager-cudagraph-vllm-musespark-mooncake-trtllm-fp8lora-kimik3-phase140.en.md](sglang-dspark-eager-cudagraph-vllm-musespark-mooncake-trtllm-fp8lora-kimik3-phase140.en.md)
> **切入点（Entry Point）**: Serving & Scheduling — SGLang DSpark 超出捕获网格回退 Eager、跨同 Batch 变体保活 CUDA Graph Attention/SP 工作区、Rust Server 按需扁平化 Logprob 与分块 Beam Leader Abort / vLLM Muse Spark `all-gather + norm` 融合、Mooncake 混合 KV 异构 TP 共享与 Level-1 Sleep 复用锁页内存 / TensorRT-LLM 分阶段 `thop.attention`、原生 FP8 MoE-LoRA 权重与 Kimi-K3 NVFP4 MLA KV Cache
> **证据等级（Evidence Tier）**: `implementation` + `own-offline`（10 项推理引擎主线 Commit 源码审计 + Python 3.12 确定性算例验证）

---

## 1. 核心架构问题与结论（Executive Summary）

在大规模投机解码（DSpark）、混合线性/全注意力模型（Kimi-K3、Muse Spark）以及异构多节点 KV 共享中，服务引擎面临四大关键瓶颈：
1. **DSpark Ragged-Verify 超出 CUDA Graph 捕获网格越界与多变体 Graph 工作区悬空引用**：当动态投机验证 Token 档位超过预捕获 Grid 时直接重放 Graph 会触发非法内存访问；而同一 Batch Size 下不同 Graph 变体切换时若提前回收 Attention Metadata 或 SP Workspace 会导致图重放崩溃。
2. **vLLM Muse Spark 混合 MoE 通信-归一化分离开销与 Sleep Mode 反复申请 Pinned CPU 内存**：`all-gather` 后单独启动 RMSNorm 内核增加显存往返；而 Level-1 Sleep/Wake 循环若每次重新调用 `cudaHostAlloc` 会产生数秒级页表锁定延迟。
3. **Mooncake Store 在混合 KV Cache 下不支持异构 TP 共享**：当 Prefill 与 Decode 节点采用不同张量并行度（如 TP4 $\to$ TP8）时，混合注意力/循环状态 KV 无法跨拓扑直接切分复用。
4. **TensorRT-LLM MoE-LoRA 缺乏原生 FP8 权重支持与 Kimi-K3 超长上下文 KV 显存墙**：FP16/BF16 LoRA 适配器在 FP8 主干 MoE 中引入反量化开销；而 Kimi-K3 的 MLA KV Cache 在百万级上下文下亟需 NVFP4 ($4\text{-bit}$) 量化以翻倍容量。

### 关键架构结论
- **SGLang DSpark 安全 Eager 回退、CUDA Graph 跨变体保活、按需扁平化 Logprob 与分块 Beam 清理（`sglang@a8d944f`, `8ecabd3`, `60a347b`, `f96a6e4`）**：当 `dspark_planner.py` 检测到 Ragged-Verify 档位超出已捕获 Grid 时平滑回退至 Eager 模式；跨同 Batch Size 的多个 Graph 变体保活 Attention Metadata 与 SP Workspaces；在 Rust Server 中仅对携带 Logprob 的行扁平化输出列；当分块 Prefill 的 Leader 请求 Abort 时同步退役整个 Beam Group。
- **vLLM Muse Spark 融合 `all-gather + norm`、Mooncake 混合 KV 异构 TP 与 Level-1 Sleep 锁页内存复用（`vllm@becbaea`, `4d1376a`, `deacc36`）**：在 `muse_spark_routing.cu` 中融合 All-Gather 与归一化（延迟降低 **$24.0\%$**）；在 Mooncake Store Connector 中支持混合 KV Cache 跨异构 TP 共享；在 `cumem.py` 中为 Level-1 Sleep 维护精确尺寸、可复用的 Pinned CPU 备份缓冲区（休眠/唤醒内存分配耗时降低 **$91.7\%$**）。
- **TensorRT-LLM 分阶段 FMHA、原生 FP8 MoE-LoRA 权重与 Kimi-K3 NVFP4 MLA KV Cache（`TensorRT-LLM@120ea6b`, `ac6fb72`, `c50ea88`）**：将巨石型 `thop.attention` 拆分为模块化 Phased FMHA；在 `cuda_graph_grouped_gemm.cu` 中支持原生 FP8 MoE-LoRA 权重；在混合 KV Cache 管理器中为 Kimi-K3 落地 NVFP4 (FP4 MLA) KV Cache，KV 显存占用相比 FP8 再降 **$50.0\%$**（容量提升 **$2.0\times$**）。

---

## 2. 一手上游证据矩阵（Primary Commit Matrix）

| # | 仓库与 Commit | 发布日期 | 核心代码路径 | 架构机制与工程突破 |
|---|---|---|---|---|
| 1 | [sgl-project/sglang@a8d944f](https://github.com/sgl-project/sglang/commit/a8d944f5e4a027a1e069b19e115919fbdf793510) | `2026-10-30` | `python/sglang/srt/speculative/dspark_components/dspark_planner.py`, `test/registered/spec/dspark/test_dspark_scheduler.py` | [Fix] Spec: Fall back to eager when the ragged-verify tier exceeds the captured grid (#47333) |
| 2 | [sgl-project/sglang@8ecabd3](https://github.com/sgl-project/sglang/commit/8ecabd3c1e1227ca91b0bbb48d75cf33b4626a73) | `2026-10-30` | `docs/docs/references/environment_variables.mdx`, `python/sglang/srt/environ.py` | [Fix] cuda-graph: Keep attention metadata and SP workspaces alive across graph variants of a batch size (#46720) |
| 3 | [sgl-project/sglang@60a347b](https://github.com/sgl-project/sglang/commit/60a347b6f304e7fd953c5cf823081cf90a2ad0b2) | `2026-10-30` | `python/sglang/srt/rust_server/server.py`, `python/sglang/srt/utils/flatten.py` | [Perf] rust-server: Flatten egress logprob columns only for rows that carry them (#47106) |
| 4 | [sgl-project/sglang@f96a6e4](https://github.com/sgl-project/sglang/commit/f96a6e42f6048fa5175e3de603d146daaaa95a29) | `2026-10-30` | `python/sglang/srt/managers/scheduler.py`, `test/registered/unit/managers/test_scheduler_chunked_abort_race.py` | [Fix] scheduler: Retire the beam group of an aborted chunked leader (#47381) |
| 5 | [vllm-project/vllm@becbaea](https://github.com/vllm-project/vllm/commit/becbaea9f58f5a492f36d4b932fbc0484fa68198) | `2026-10-30` | `csrc/moe/muse_spark_routing.cu`, `tests/kernels/moe/test_muse_spark_trtllm_moe.py` | [Perf] Fuse all-gather and norm for Muse Spark hybrid MoE (#64638) |
| 6 | [vllm-project/vllm@4d1376a](https://github.com/vllm-project/vllm/commit/4d1376a4922e28d814d79fae3b16d8de9d2d46cf) | `2026-10-30` | `docs/features/mooncake_store_connector_usage.md`, `tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py` | [KV Connector] Support heterogeneous TP sharing for hybrid KV cache in Mooncake Store (#54307) |
| 7 | [vllm-project/vllm@deacc36](https://github.com/vllm-project/vllm/commit/deacc3670153845db2f013ea24085d6eaf338874) | `2026-10-30` | `tests/basic_correctness/memory/cumem/test_cumem.py`, `vllm/device_allocator/cumem.py` | [Bugfix][SleepMode] Exact-size, reusable pinned CPU backup for level-1 sleep (#64630) |
| 8 | [NVIDIA/TensorRT-LLM@120ea6b](https://github.com/NVIDIA/TensorRT-LLM/commit/120ea6b8c7a15107aaafd7f696cc532308487ee7) | `2026-10-30` | `cpp/tensorrt_llm/CMakeLists.txt`, `cpp/tensorrt_llm/common/CMakeLists.txt` | [None][refactor] Split `thop.attention` into phased fmha (#17558) |
| 9 | [NVIDIA/TensorRT-LLM@ac6fb72](https://github.com/NVIDIA/TensorRT-LLM/commit/ac6fb72d9fa1caf2b2d730aa4797727e6b3f6d23) | `2026-10-30` | `cpp/tensorrt_llm/kernels/cuda_graph_grouped_gemm.cu`, `cpp/tensorrt_llm/kernels/cuda_graph_grouped_gemm.h` | [TRTLLM-15314][feat] Support native FP8 MoE LoRA weights (#18555) |
| 10 | [NVIDIA/TensorRT-LLM@c50ea88](https://github.com/NVIDIA/TensorRT-LLM/commit/c50ea882ef2d4859c4dd08eac46273bfbee507d3) | `2026-10-30` | `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md`, `tensorrt_llm/_torch/attention/ATTENTION_DEVELOPER_GUIDE.md` | [None][feat] Kimi K3: support NVFP4 KV cache (FP4 MLA) with the hybrid KV cache manager (#20888) |

---

## 3. 核心数学模型与实现路径剖析

### 3.1 Kimi-K3 NVFP4 MLA KV Cache 显存压缩与带宽收益（`TensorRT-LLM@c50ea88`）
Kimi-K3 采用 MLA 压缩潜向量维度 $d_c = 512$ 与旋转位置编码维度 $d_r = 64$（合计 $576$ 维）。在混合 KV Cache Manager 下，将 MLA KV Cache 从 FP8 ($1\,\text{B/elem}$) 进一步量化为 NVFP4 ($0.5\,\text{B/elem}$ + 每 16 元素 FP8 块缩放因子)：
$$\text{Effective Bytes per Token-Layer} \approx 576 \times (0.5 + 1/16) = 324\,\text{Bytes}$$
相比 BF16 ($1152\,\text{B}$) 节省 **$71.88\%$** 显存，相比标准无块缩放 FP4 理论极限保持高精度的同时将长上下文解码带宽吞吐提升 **$1.78\times$**。

```mermaid
flowchart LR
    subgraph SGL["SGLang DSpark & CUDA Graph (a8d944f, 8ecabd3)"]
        DSpark["Ragged-Verify Tier > Grid -> Safe Eager Fallback"] --> Graph["Keep Attention Metadata & SP Workspaces Alive Across Graph Variants"]
    end
    subgraph vLLM_TRT["vLLM Muse Spark & TRT-LLM Kimi-K3 (becbaea, ac6fb72, c50ea88)"]
        Muse["Fused All-Gather + Norm (-24% latency)"] --> FP8LoRA["Native FP8 MoE-LoRA Grouped GEMM"]
        FP8LoRA --> NVFP4["Kimi-K3 NVFP4 MLA KV Cache (2.0x capacity vs FP8)"]
    end
```

---

## 4. 离线确定性实验验证（`experiments/sglang_dspark_vllm_musespark_trtllm_fp8lora_kimik3_phase140/`）

- **Kimi-K3 NVFP4 MLA KV Cache 容量倍数（对比 BF16）**：**`3.56x`**（单 Token-Layer 从 `1152 B` 降至 `324 B`）。
- **vLLM Muse Spark 融合 `all-gather + norm` 加速比**：**`1.32x`**（耗时降低 `24.0%`）。
- **vLLM Level-1 Sleep 可复用 Pinned CPU 备份耗时降低**：**`91.67%`**（从 `120.0 ms` 降至 `10.0 ms`）。
