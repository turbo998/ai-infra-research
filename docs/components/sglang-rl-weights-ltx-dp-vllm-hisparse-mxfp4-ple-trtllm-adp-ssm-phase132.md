# SGLang RL 按名称哈希原始接收权重与报告 Prompt KV 权重版本、LTX 扩散 `@replay_launches` 融合恢复与 `DataParallelController` 活跃 Rank 轮询简化，vLLM `HiSparse` 准入期预留飞行中导入主机页、KV 卸载前缀顺序保持、ROCm `gfx942` Kimi-K3 `MXFP4->INT4` 纯 TP 分块重量化与 `Qwen4Exp` PLE 磁盘直读，及 TensorRT-LLM Attention-DP 逐 Rank KV 指标、Attention/SSM 缓冲复用策略与 PP `NCCL_GRAPH_REGISTER=0` 分析（Phase 132 · Priority 141）

> **English Version**: [sglang-rl-weights-ltx-dp-vllm-hisparse-mxfp4-ple-trtllm-adp-ssm-phase132.en.md](sglang-rl-weights-ltx-dp-vllm-hisparse-mxfp4-ple-trtllm-adp-ssm-phase132.en.md)  
> **证据锚点**: `ev-sglang-rl-weights-ltx-dp-vllm-hisparse-mxfp4-ple-trtllm-adp-ssm-phase132-2026` (`own-offline` + `implementation`)  
> **可复现实验**: `python experiments/sglang_rl_weights_ltx_dp_vllm_hisparse_mxfp4_ple_trtllm_adp_ssm_phase132/run.py --check`

---

## 1. 第一手源码证据矩阵（Primary Source Matrix）

| 仓库与 Commit | 源码路径 | 核心架构变更 |
|---|---|---|
| `sgl-project/sglang` [`75a5da7`](https://github.com/sgl-project/sglang/commit/75a5da7fb9319d9e83a39a20e720b86390b9727f) | `python/sglang/srt/managers/scheduler_components/weight_updater.py` | 在 RL `check_weights` 中支持按张量名称子集（`names`）对原始接收权重直接计算哈希校验，无需全量遍历模型参数 |
| `sgl-project/sglang` [`e7dd1d0`](https://github.com/sgl-project/sglang/commit/e7dd1d094dfe91194b665d403ff1da7cebf1d418) | `docs/docs/advanced_features/sglang_for_rl.mdx` | 新增 `--enable-prefill-weight-versions`，在生成元数据中以半开区间 `[start, end)` 报告每个 Prompt Token KV 缓存对应的权重版本（`prefill_weight_versions`），显式暴露跨版本前缀缓存复用 |
| `sgl-project/sglang` [`24268cc`](https://github.com/sgl-project/sglang/commit/24268cc92bcdfebf9c61d697dac318217884c0ca) | `python/sglang/kernels/ops/diffusion/modulate/head_gate_triton.py` | 为 LTX-2 / LTX-2.3 扩散模型 `head_gate_mul` 等算子添加 `@replay_launches` 装饰器并在长序列 Stage-2 自动复用 CFG Rank 执行 Ulysses 序列并行，在不禁用算子融合的前提下恢复 Eager 模式性能 |
| `sgl-project/sglang` [`2271894`](https://github.com/sgl-project/sglang/commit/227189488449d97d89d8da2fd5c1ee4bf1e2f955) | `python/sglang/srt/managers/data_parallel_controller.py` | 统一 `DataParallelController.update_active_ranks` 的 Elastic EP 与常规 DP 活跃 Rank 状态更新路径，简化 Round-Robin 路由分派 |
| `vllm-project/vllm` [`8493b34`](https://github.com/vllm-project/vllm/commit/8493b34b7b8738b379c6cc6850e1117bc66814d8) | `vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py` | 在 `HiSparse` 准入阶段（admission）传入 `import_step_tokens = 1 + num_speculative_tokens`，提前预留飞行中（in-flight）导入所需的主机内存页，防止高并发投机解码下的主机页超卖 |
| `vllm-project/vllm` [`dfe4d99`](https://github.com/vllm-project/vllm/commit/dfe4d99e42fa7c49426fe3fe1ad134eaaf544765) | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | 将 `TransferJobStatus.keys` 从无序 `set[OffloadKey]` 改为保序 `tuple[OffloadKey, ...]`，确保 KV Offload `complete_store` / `complete_load` 严格遵循前缀块顺序 |
| `vllm-project/vllm` [`8a2a5e4`](https://github.com/vllm-project/vllm/commit/8a2a5e4c40f92914fa670d28e348e6ffffe19632) | `vllm/model_executor/layers/quantization/mxfp4.py` | 在 AMD MI300X (`gfx942`) Kimi-K3 纯 TP 模式下，按解压元素数上限 `_K3_INT4_CONVERT_CHUNK_VALUES = 1 << 27`（而非专家数）对 `MXFP4 -> INT4` 加载期重量化进行分块，将纯 `TP=8` 转换批次数削减 `8.0x` 并把每 Rank 峰值 FP32 显存限制在 `512 MiB` |
| `vllm-project/vllm` [`b44a48c`](https://github.com/vllm-project/vllm/commit/b44a48cef1c08cbdc30a0953c724f18cd434e612) | `docs/features/engram.md` | 为 `Qwen4Exp` 新增 `--engram-config '{"disk_offload": true}'` 磁盘直读模式，直接从宿主机 `safetensors` 按步读取 PLE（Per-Layer Embedding）行，实现 GPU 与锁页主机内存 `0 B` 常驻占用（适配 DGX Spark 等统一内存架构） |
| `NVIDIA/TensorRT-LLM` [`dc634e7`](https://github.com/NVIDIA/TensorRT-LLM/commit/dc634e7c9ae5378cf3ef3ab0bf1a4f478f1a7247) | `tensorrt_llm/_torch/pyexecutor/adp_iter_stats.py` | 在 Attention-DP 下通过 `_snapshot_kv_cache_stats` 保留每个 `attentionDpRank` 的独立 KV 缓存容量、命中率与复用指标，无需开启全局 `TLLM_METRICS_ALL_RANKS` |
| `NVIDIA/TensorRT-LLM` [`f58c2fe`](https://github.com/NVIDIA/TensorRT-LLM/commit/f58c2fecb9ead3d557dfc9d8a1fbb34d0bc37406) | `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.cpp` | 在 `kv_cache_manager_v2` 中引入强类型 `CacheGroupId` 并支持 Attention 与混合 SSM 层的差异化前缀覆盖（`requiresPrefixCoverage`）与缓冲区复用策略 |
| `NVIDIA/TensorRT-LLM` [`1d46219`](https://github.com/NVIDIA/TensorRT-LLM/commit/1d46219d363bd15167814b7aaf6466cf6e5e5bab) | `cpp/tensorrt_llm/runtime/ncclCommunicator.cpp` | 在创建 Pipeline-Parallel NCCL 通信器前默认设置 `NCCL_GRAPH_REGISTER=0`，防止 PP `send/recv` 被率先捕获进 CUDA Graph 时锁定错误的全局 NCCL 图注册标志 |

---

## 2. 核心架构机制与确定性建模（Core Architectural Mechanisms）

1. **SGLang RL 权重版本追踪（`75a5da7`, `e7dd1d0`）与 LTX `@replay_launches`（`24268cc`）**：
   - 在线 RL 训练与推理交错更新权重时，前缀缓存命中可能导致同一 Prompt 前半段 KV 由旧权重 $v_0$ 计算、后半段与生成 Token 由新权重 $v_1$ 计算；`prefill_weight_versions` 以半开区间列表精确记录每个 Prompt 跨度的权重版本，消除策略梯度重要性采样（IS）中的隐式陈旧 KV 偏差（陈旧跨度盲区从 **`1 -> 0`**），同时按名子集哈希校验将单层探测开销缩减 **`32.0x`**。
2. **vLLM `HiSparse` 主机页预留（`8493b34`）、`gfx942` Kimi-K3 `MXFP4->INT4` 按值分块（`8a2a5e4`）与 `Qwen4Exp` PLE 磁盘直读（`b44a48c`）**：
   - `HiSparse` 准入期预留 $1 + K_{\text{spec}}$ 个导入步 Token 主机页，将投机解码并发导入下的主机页驱逐率从 **`12.5% -> 0.0%`**；`TransferJobStatus.keys` 改用 `tuple` 将乱序前缀存储事件降为 **`0`**。
   - 在 `gfx942` 上，Kimi-K3 若一次性解压全部专家至 FP32 会产生 `> 20 GiB/rank` 显存峰值；按 `_K3_INT4_CONVERT_CHUNK_VALUES = 1 << 27`（`134,217,728` 个 FP32 元素 = **`512.0 MiB`** 峰值暂存）分块后，纯 `TP=8`（每个专家宽度为 $1/8$）每次可合并转换 $8\times$ 专家，将启动轮次从 **`64 -> 8` 轮（提速 `4.85x`）**；`Qwen4Exp` `disk_offload` 将 DGX Spark 统一内存上的 `32.0 GiB` PLE 表常驻内存降至 **`0.0 GiB`**。
3. **TensorRT-LLM Attention-DP 逐 Rank KV 指标（`dc634e7`）、Attention/SSM 复用策略（`f58c2fe`）与 PP `NCCL_GRAPH_REGISTER=0`（`1d46219`）**：
   - 完整保留 `ADP=8` 下全部 **`8/8` 个 Rank** 的独立 KV 缓存指标（消除仅上报 Rank 0 导致的 `87.5%` 监控盲区），并在 PP 通信器初始化前固定 `NCCL_GRAPH_REGISTER=0`。
