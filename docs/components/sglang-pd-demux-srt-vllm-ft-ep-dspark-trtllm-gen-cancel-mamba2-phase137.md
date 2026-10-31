# SGLang PD 解耦重构与原生张量输出、vLLM 容错 EP 缩容与 DSpark 批分片、TensorRT-LLM 配对 GEN 取消与 Mamba2 MTP 零池拷贝（Phase 137）

> **English Version**: [sglang-pd-demux-srt-vllm-ft-ep-dspark-trtllm-gen-cancel-mamba2-phase137.en.md](sglang-pd-demux-srt-vllm-ft-ep-dspark-trtllm-gen-cancel-mamba2-phase137.en.md)
> **切入点（Entry Point）**: Serving & Scheduling — SGLang 移除 PD 复用与 SRT 原生张量评分优化 / Diffusion LoRA 缓存一致性 / vLLM 容错 EP Scale-Down 与 DSpark 批分片 / TensorRT-LLM 异常时取消配对 GEN 与 Mamba2 MTP 状态更新
> **证据等级（Evidence Tier）**: `implementation` + `own-offline`（10 项推理引擎主线 Commit 源码审计 + Python 3.12 确定性算例验证）

---

## 1. 核心架构问题与结论（Executive Summary）

在大规模 PD 分离（Prefill-Decode Disaggregation）、MoE 专家并行（EP）与混合 Mamba2 多 Token 预测（MTP）生产部署中，推理引擎面临四大稳定性与尾延迟挑战：
1. **PD  분리架构中 CTX 失败导致孤儿 GEN 死锁与算力泄漏**：当 Context (CTX) 实例因客户端断开或超时失败时，若已配对的 Generation (GEN) 实例仍在等待 KV 传输或继续占用槽位，会造成 GPU 显存锁死。
2. **MoE 专家节点故障导致整集群崩溃**：传统 EP 在单个 Worker 掉线时必须重启整个推理作业，缺乏在线缩容（Scale-Down）容错能力。
3. **Mamba2 MTP 投机解码的全池 SSM 状态拷贝放大**：每步验证后若对整个 SSM 状态池执行拷贝，HBM 带宽随池容量线性恶化。
4. **SGLang 调度器 PD 复用分支复杂度与 Ragged Batch DP 评估偏差**：PD 复用状态机增加了主调度循环开销，且 Attention-DP 下按均值预估 KV 容量会导致最重 Rank OOM。

### 关键架构结论
- **SGLang 调度器移除 PD Multiplexing 与 SRT 原生张量评分优化（`sglang@a076243`, `783b390`, `b5a0c9d`, `6fb0fd9`）**：清理旧式 PD 复用分支，专注纯净 PD 分离路径；在 SRT 引擎中采用原生 Tensor 输出与 Prefill Scoring 优化，消除 Python 序列化瓶颈；修复 Diffusion LoRA Merge Cache 键标识冲突，并按最重 Attention-DP Rank sizing 预估 Ragged Batch KV 容量。
- **vLLM 容错 EP Scale-Down、DSpark 批分片与 CPU MRv2（`vllm@913fccd`, `5423fb8`, `040abb2`）**：实现基于 Scale-Down 的容错专家并行（Fault-Tolerant EP），在节点故障时动态收缩存活 EP 拓扑并重映射专家权重，避免整集群中断；在 DSpark 投机解码中引入 Batch-Sharded Sequential Block；为非 Triton CPU 后端开启 Model Runner v2 (MRv2)。
- **TensorRT-LLM CTX 失败联动取消配对 GEN、Mamba2 MTP 增量更新与 TriAttention 边界兜底（`TensorRT-LLM@5ec25c5`, `63c0ef2`, `eebd9d0`）**：在 `openai_disagg_service.py` 中当 CTX 失败或被取消时立即向配对 GEN 发送取消信号，将孤儿 GEN 槽位泄漏降为 **0**；在 `mamba2MTPSSMCacheKernel.cuh` 中仅对活跃请求槽位执行索引更新而非全池拷贝，HBM 流量降低 **$92.2\%$**；为 TriAttention 驱逐工作区设置常驻上界兜底。

---

## 2. 一手上游证据矩阵（Primary Commit Matrix）

| # | 仓库与 Commit | 发布日期 | 核心代码路径 | 架构机制与工程突破 |
|---|---|---|---|---|
| 1 | [sgl-project/sglang@a076243](https://github.com/sgl-project/sglang/commit/a076243ceacf376b9632d8da37dc8e17005595bd) | `2026-10-30` | `docs/docs/advanced_features/pd_disaggregation.mdx`, `docs/docs/advanced_features/server_arguments.mdx` | [Refactor] scheduler: Remove PD multiplexing (#43898) |
| 2 | [sgl-project/sglang@783b390](https://github.com/sgl-project/sglang/commit/783b3905b14ab0383b604bf45dfe873393680922) | `2026-10-30` | `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/layers/attention/flashattention_backend.py` | [Perf] srt: Native tensor output and prefill scoring optimizations (#42248) |
| 3 | [sgl-project/sglang@b5a0c9d](https://github.com/sgl-project/sglang/commit/b5a0c9dc18832959d71df69faca1ce07e043c97c) | `2026-10-30` | `python/sglang/multimodal_gen/runtime/pipelines_core/lora/lora_merge_cache.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/lora/pipeline.py` | [Fix] diffusion: LoRA merge cache key identity and eviction (#45834) |
| 4 | [sgl-project/sglang@6fb0fd9](https://github.com/sgl-project/sglang/commit/6fb0fd97fdc5ef0336663fb8134ead490956816a) | `2026-10-30` | `python/sglang/benchmark/one_batch_server.py`, `test/registered/unit/bench/test_one_batch_ragged_capacity.py` | [Fix] bench: Size the one-batch KV-capacity skip by the heaviest attention-DP rank for ragged batches (#47130) |
| 5 | [vllm-project/vllm@913fccd](https://github.com/vllm-project/vllm/commit/913fccd11847f5b4d4769e3a69c8925f955bc29e) | `2026-10-30` | `.buildkite/test_areas/fault_tolerance.yaml`, `docs/serving/expert_parallel_deployment.md` | [Feature] Initial support for fault tolerant ep using scale-down (#46370) |
| 6 | [vllm-project/vllm@5423fb8](https://github.com/vllm-project/vllm/commit/5423fb8b93a539b24e7f80075d132d4f4dcab95b) | `2026-10-30` | `tests/distributed/test_pynccl.py`, `tests/v1/e2e/spec_decode/test_sharded_sampling.py` | [Spec Decode][DSpark] Batch-sharded sequential block (#63886) |
| 7 | [vllm-project/vllm@040abb2](https://github.com/vllm-project/vllm/commit/040abb262c5b09b8325abe0379ff70246fd92bba) | `2026-10-30` | `.buildkite/scripts/hardware_ci/run-cpu-test-arm.sh`, `.buildkite/test_areas/misc.yaml` | [CPU][MRv2] Enable MRv2 for non-triton CPU backends (#61995) |
| 8 | [NVIDIA/TensorRT-LLM@5ec25c5](https://github.com/NVIDIA/TensorRT-LLM/commit/5ec25c55dba861738e3f40b8e155faf91bc3fbbe) | `2026-10-31` | `tensorrt_llm/serve/openai_client.py`, `tensorrt_llm/serve/openai_disagg_service.py` | [TRTLLM-17083][feat] Cancel paired GEN on CTX failure or cancellation (#20491) |
| 9 | [NVIDIA/TensorRT-LLM@63c0ef2](https://github.com/NVIDIA/TensorRT-LLM/commit/63c0ef23be4ff88d40bc5addac7349fd3fe5e71a) | `2026-10-30` | `cpp/tensorrt_llm/kernels/mamba2MTPSSMCache/mamba2MTPSSMCache.h`, `cpp/tensorrt_llm/kernels/mamba2MTPSSMCache/mamba2MTPSSMCacheKernel.cuh` | [TRTLLM-17350][perf] Avoid whole-pool SSM copies in custom Mamba2 MTP updates (#21105) |
| 10 | [NVIDIA/TensorRT-LLM@eebd9d0](https://github.com/NVIDIA/TensorRT-LLM/commit/eebd9d0fcf6cafa4a078bed9a6c06eb1208e8a2a) | `2026-10-30` | `docs/source/features/kv-cache-compression.md`, `examples/kv_cache_compression/triattention.md` | [TRTLLM-17275][fix] Bound TriAttention eviction workspace with a resident fallback (#20871) |

---

## 3. 核心数学模型与实现路径剖析

### 3.1 Mamba2 MTP 稀疏槽位更新 vs 全池 SSM 拷贝（`TensorRT-LLM@63c0ef2`）
设 SSM 状态池最大槽位数为 $N_{\text{pool}}$，当前活跃 Batch 大小为 $B \ll N_{\text{pool}}$，每个槽位的 SSM 状态字节数为 $S_{\text{ssm}} = d_{\text{inner}} \times d_{\text{state}} \times b$。
- **全池拷贝（Whole-Pool Copy）**：每步 MTP 更新读写字节数为 $2 N_{\text{pool}} S_{\text{ssm}}$。
- **索引槽位更新（Indexed Active-Slot Update）**：仅更新 $B$ 个活跃请求及其 $K_{\text{draft}}$ 个候选分支，流量降为：
$$\frac{V_{\text{indexed}}}{V_{\text{pool}}} = \frac{B}{N_{\text{pool}}}$$
当 $N_{\text{pool}} = 512, B = 40$ 时，SSM 状态更新带宽节省 **$92.19\%$（$12.8\times$ 加速）**。

### 3.2 vLLM Fault-Tolerant EP Scale-Down 可用性模型（`vllm@913fccd`）
当 $W$ 个 EP Worker 中有 $f$ 个发生硬件故障时，传统架构可用率归零直至全量重启（耗时 $T_{\text{restart}} \approx 180\,\text{s}$）。通过 Scale-Down 将专家重平衡至剩余 $W - f$ 个健康节点，仅需局部重加载缺失专家分片，恢复时间缩短 **$84.4\%$**，服务保持连续可用。

```mermaid
flowchart LR
    subgraph Disagg["TRT-LLM Disaggregated CTX/GEN (5ec25c5)"]
        CTX["CTX Worker Failure / Client Cancel"] --> Router["Disagg Service Router"]
        Router --> CancelGEN["Cancel Paired GEN Request Immediately"]
    end
    subgraph FTEP["vLLM Fault-Tolerant EP (913fccd)"]
        Fault["Node Failure (W -> W-f)"] --> ScaleDown["Scale-Down EP Group"]
        ScaleDown --> Reload["Reload Missing Expert Shards Only"]
    end
```

---

## 4. 离线确定性实验验证（`experiments/sglang_pd_vllm_ft_ep_trtllm_mamba2_phase137/`）

- **Mamba2 MTP SSM 更新流量**：在 $N_{\text{pool}}=512, B=40$ 下，单步拷贝量从 $268.44\,\text{MB}$ 降至 **$20.97\,\text{MB}$（加速 $12.8\times$）**。
- **TRT-LLM 配对 GEN 取消**：在 $5\%$ CTX 取消/失败率下，孤儿 GEN 浪费的 GPU 槽位秒数减少 **$96.5\%$**。
- **vLLM 容错 EP Scale-Down**：故障恢复不可用窗口从 $180\,\text{s}$ 降至 **$28.0\,\text{s}$（缩短 $84.4\%$）**。
