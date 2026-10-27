# SGLang MNNVL `custom_all_reduce_v2` VMM 门控移除与 DSA 维度压缩、vLLM ROCm W4A16 `b128` LDS 宽加载与 TRT-LLM Blackwell Low-M FP8 解码 GEMM 深度解析（Phase 104 · Priority 113）

> **English Version**: [sglang-custom-ar-vmm-vllm-b128-trtllm-lowm-fp8-innovators-phase104.en.md](sglang-custom-ar-vmm-vllm-b128-trtllm-lowm-fp8-innovators-phase104.en.md)
> **实验脚本**: [`experiments/sglang_custom_ar_vmm_vllm_b128_trtllm_lowm_fp8_phase104/run.py`](../../experiments/sglang_custom_ar_vmm_vllm_b128_trtllm_lowm_fp8_phase104/run.py) · **定量汇总**: [`summary.json`](../../experiments/sglang_custom_ar_vmm_vllm_b128_trtllm_lowm_fp8_phase104/summary.json)
> **7-Card 视觉报告**: [`docs/reports/ai-infra-phase104-benchmark-cards.html`](../reports/ai-infra-phase104-benchmark-cards.html)

---

## 1. 证据分级与上游源码锚点（Evidence Classification）

| 证据 ID / 来源 | 仓库与 Commit | 关键路径 | 证据分级 |
|---|---|---|---|
| `src-sglang-custom-ar-remove-multinode-vmm-gate-2026` | `sgl-project/sglang@de26bb16a972d93f354218a29dbe177bce5655a4` | `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py` | `implementation` |
| `src-sglang-dsa-token-first-sparse-output-squeeze-2026` | `sgl-project/sglang@8db96fd40202c068137fc56f524b2a53ce9a1026` | `python/sglang/srt/layers/attention/dsa_backend.py` | `implementation` |
| `src-sglang-dsv4-recipe-null-tool-effort-2026` | `sgl-project/sglang@449617bc94a5d4ae23ad62d0da2f842e711ca983` | `rust/sglang-processor/src/render/models/deepseek_v4.rs` | `implementation` |
| `src-sglang-qwen3-coder-mixed-union-preserve-2026` | `sgl-project/sglang@9d45fdda46d5918acb38b16d05df0b0985b286ba` | `python/sglang/srt/function_call/qwen3_coder_detector.py` | `implementation` |
| `src-vllm-rocm-w4a16-skinny-gemm-lds-b128-2026` | `vllm-project/vllm@83909ad84b3fe6e64536591fc26f02347b65fd63` | `csrc/rocm/skinny_gemms_int4.cu` | `implementation` |
| `src-vllm-triton-override-forward-launch-grid-2026` | `vllm-project/vllm@ca8dbccf204abfcdf9dcb178f7e108c06b224b0f` | `vllm/triton_utils/dispatcher.py` | `implementation` |
| `src-vllm-qwen38-flash-next-quant-qkv-bf16-qsa-2026` | `vllm-project/vllm@a91a1d06b72a53d6ba05840164b27a61e3b03d4e` | `vllm/models/qwen4_exp/nvidia/qsa.py` | `implementation` |
| `src-vllm-spec-decode-mhc-tilelang-refactor-2026` | `vllm-project/vllm@e7fa26d5c8d94636a4d0525e1a2462cb661ea331` | `vllm/model_executor/kernels/mhc/tilelang.py` | `implementation` |
| `src-trtllm-blackwell-fp8-block-scale-low-m-gemm-2026` | `NVIDIA/TensorRT-LLM@071298058728f652a416023cf1c94f2fc89368af` | `tensorrt_llm/_torch/kernels/blackwell/fp8_block_scale_low_m_gemm.py` | `implementation` |
| `src-trtllm-responses-namespace-tool-replay-2026` | `NVIDIA/TensorRT-LLM@a1eebc47dde5b8c1e1b4be503f889617d76ba040` | `tensorrt_llm/serve/responses_utils.py` | `implementation` |

---

## 2. 核心实现路径、数据布局与同步机制

### 2.1 SGLang `custom_all_reduce_v2` 多节点 MNNVL VMM 门控移除与 DSA Token-First 压缩 (`de26bb16a9`, `8db96fd402`)
- **MNNVL 对称内存与 VMM 解耦 (`de26bb16a9`)**：在 GB200 NVL72 多节点（`world_size = 16`）场景下，`custom_all_reduce_v2` 原先在检测到跨节点通信组时要求 `uses_vmm == True`（CUDA Virtual Memory Management 分配器），否则直接禁用自定义 all-reduce 并回退到 NCCL Ring。然而多节点 MNNVL 路径实际通过 `torch.ops.symm_mem` 对称内存 workspace 完成跨节点 NVLink 归约，根本不依赖单节点 `FABRIC/POSIX-fd` VMM 映射。移除该多余门控后，标准 `cudaMalloc` 池下的 16 卡 MNNVL one-shot all-reduce 正常激活，将小包归约延迟从 `48.0 us`（NCCL Ring）降至 `14.0 us`（**`3.4286x` 加速**）。
- **DSA 稀疏注意力 `.squeeze(0)` (`8db96fd402`)**：`tilelang_sparse_fwd` 返回四维张量 `(1, T, H, D)`，导致外层 DP-attention padding 沿 `dim=0` 切分时误操作在 batch 维 `1` 上而抛出维度冲突；新增 `.squeeze(0)` 严格恢复 token-first 三维契约 `(T, H, D)`，结合 DeepSeek-V4 Rust 模板 `null` 内容/工具调用修复 (`449617bc94`) 与 `qwen3_coder` 混合 `anyOf/oneOf` 结构化字典保留 (`9d45fdda46`)，实现端到端零崩溃。

### 2.2 vLLM ROCm W4A16 Skinny GEMM `b128` LDS 宽加载与 Qwen3.8-Flash-Next QSA 混合精度 (`83909ad84b`, `a91a1d06b7`)
- **ROCm `ds_read_b128` 激活加载 (`83909ad84b`)**：在 MI300X W4A16 skinny GEMM（Deep-K $N=1$ 解码路径，`A_CHUNK = 32` 个 FP16/BF16 元素 = `64 B`）中，原先从 LDS（Local Data Share）读取激活采用 `64-bit`（`b64` / `8 B`）指令；新实现引入 `bigTypeA`（`sizeof = 16 B`）并将对齐约束设为：
  $$\text{kActLdsAlign} = \min(\text{sizeof}(\text{bigTypeA}), 16 \times \text{sizeof}(\text{scalar\_t})) = 16\text{ B}$$
  从而将 LDS 读取指令宽度翻倍至 `128-bit`（`b128` / `16 B`），**每 wave 的 LDS 加载指令数直接减少 `50.0%`（`64 -> 32` 条）**。
- **Qwen3.8-Flash-Next QSA 混合精度 (`a91a1d06b7`)**：允许 Q/K/V 主干投影加载量化权重（FP8/AWQ），同时保持 QSA（Query-Sparse Attention）indexer 权重为原生 `BF16`，并在 `vllm/triton_utils/dispatcher.py` (`ca8dbccf20`) 中向 Triton kernel override 转发真实 launch grid。

### 2.3 TensorRT-LLM Blackwell Low-M FP8 Block-Scale GEMM 与 Responses 命名空间恢复 (`0712980587`, `a1eebc47dd`)
- **Blackwell Low-M FP8 解码 GEMM (`0712980587`)**：在 $M \in [1, 16]$ 的低批大小解码投影（如 $K=7168, N=4096$）中，常规 UMMA FP8 算子采用 $M_{\text{tile}} = 128$，在 $M=16$ 时产生 **`87.5%`** 的无效行填充浪费。新增 `trtllm::fp8_block_scale_low_m_gemm`（覆盖 SM100/SM103 与 SM107 策略表）将行块缩减至 $M_{\text{tile}} = 32$，填充浪费降至 **`50.0%`**，有效 MMA 利用率提升 **`4.0x`**，单层解码投影延迟从 `12.8 us` 降至 `6.0 us`（**`2.1333x` 加速**）。
- **Responses `namespace::tool` 重放 (`a1eebc47dd`)**：在多轮 Responses API 重放时自动从工具声明表反查并恢复被剥离的 `namespace::` 前缀，保持 `100%` 工具 Schema 命中率。

---

## 3. 创新公司对标与架构师设计练习

- **Fireworks AI / Together AI / Groq / Cerebras 对标**：
  - **Fireworks AI & Together AI**：在 GB200 NVL72 多节点推理集群上运行 DeepSeek-V4 或 Qwen3.8-Flash-Next 时，移除 `custom_all_reduce_v2` 的 VMM 误门控与启用 Blackwell Low-M FP8 GEMM 直接决定了交互式解码（$M \le 16$）的 TPOT 竞争力。
  - **Groq & Cerebras**：Groq 在片上 SRAM 流出激活时始终使用全宽向量总线，对应 GPU/ROCm 上将 LDS 激活读取从 `b64` 拓宽到 `b128` 以消除指令发射瓶颈。
- **架构师设计练习**：
  - 当 decode batch $M=8$ 时，比较 $M_{\text{tile}}=128$ 与 $M_{\text{tile}}=32$ 的有效算力利用率比值：$(8/32) / (8/128) = 4.0\times$。
- **后续可证伪问题**：
  1. 当 MNNVL 跨节点消息大小超过 `4 MiB` 时，one-shot `torch.symm_mem` all-reduce 是否需要切换为 two-shot reduce-scatter + all-gather 以避免 NVLink Switch 带宽放大？
