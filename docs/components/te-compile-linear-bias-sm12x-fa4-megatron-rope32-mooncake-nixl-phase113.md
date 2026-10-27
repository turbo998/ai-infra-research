# TransformerEngine `torch.compile` Linear+Bias 融合与 SM120 确定性 FA4 规避、Megatron-LM 视觉 RoPE FP32 保真与 NCCL 确定性、TorchTitan ActorSystem 及 Mooncake / LMCache / Dynamo / NIXL 分析 (Phase 113 · Priority 122)

> **English Edition**: [te-compile-linear-bias-sm12x-fa4-megatron-rope32-mooncake-nixl-phase113.en.md](te-compile-linear-bias-sm12x-fa4-megatron-rope32-mooncake-nixl-phase113.en.md)  
> **问题 ID**: `te-compile-linear-bias-sm12x-fa4-megatron-rope32-mooncake-nixl-phase113` · **优先级**: `122` · **赛道**: Track C (分布式训练与 KV 传输)  
> **确定性实验**: [`experiments/te_compile_linear_bias_sm12x_fa4_megatron_rope32_mooncake_phase113/run.py`](../../experiments/te_compile_linear_bias_sm12x_fa4_megatron_rope32_mooncake_phase113/run.py) (`--check` 通过)

---

## 1. 第一手源码证据矩阵（First-Hand Source Matrix）

| 序号 | 来源 ID | 仓库与提交 | 发布日期 | 证据层级 | 核心变更与代码路径 |
|---|---|---|---|---|---|
| 1 | `te-pytorch-compile-linear-bias-forward-fusion-2026` | [`NVIDIA/TransformerEngine@45223d764a`](https://github.com/NVIDIA/TransformerEngine/commit/45223d764a02fc4557dc5d1e5715c1028836773f) | `2026-10-08` | `implementation` | `transformer_engine/pytorch/ops/basic/basic_linear.py`：支持在 `torch.compile` 下将 `BasicLinear` 与 `Bias` 前向融合，按 TP 规模特化编译算子并在反量化反向传播中保留量化张量 |
| 2 | `te-skip-fa4-deterministic-backward-sm12x-2026` | [`NVIDIA/TransformerEngine@6d6a500860`](https://github.com/NVIDIA/TransformerEngine/commit/6d6a50086016d6bd648c4ef33b3e571ab0b5b970) | `2026-10-06` | `implementation` | `dot_product_attention.py`, `utils.py`：向注意力后端选择传入显式 `device` 与 `requires_backward = torch.is_grad_enabled() and any(t.requires_grad ...)`，在 SM120 (`sm_120`) 需要确定性反向传播时跳过 FA4（无梯度推理仍可使用 FA4） |
| 3 | `megatron-vision-rope-frequencies-fp32-bf16-conversion-2026` | [`NVIDIA/Megatron-LM@3ca52d95d6`](https://github.com/NVIDIA/Megatron-LM/commit/3ca52d95d694c5a373db7c1a5fd0936947cd59a0) | `2026-10-27` | `implementation` | `megatron/core/models/vision/vit_model.py`：在 `Pixtral2DRotaryEmbedding` 中将 `self.freqs` 移出 `register_buffer`，防止 `Float16Module` 在进入 FP32 角度与 `sin/cos` 计算前将其降级为 BF16/FP16 |
| 4 | `megatron-determinism-accept-empty-nccl-algo-ring-nvls-2026` | [`NVIDIA/Megatron-LM@62c063f2b7`](https://github.com/NVIDIA/Megatron-LM/commit/62c063f2b7a5eb4ca01279ef0f249cc54e109032) | `2026-10-18` | `implementation` | `megatron/training/determinism.py`：在非 Hopper GPU（如 Blackwell SM100）的 `--deterministic-mode` 下允许空 `NCCL_ALGO` 及 `Ring,NVLS`，并新增 `assert_nccl_algo_still_empty` 在首个进程组创建后校验环境变量未被篡改 |
| 5 | `torchtitan-rl-select-actor-framework-by-config-monarch-2026` | [`pytorch/torchtitan@f1bb643a71`](https://github.com/pytorch/torchtitan/commit/f1bb643a714492d92cf45ef944a969bb04011906) | `2026-10-18` | `implementation` | `torchtitan/rl/actors/*`：引入由配置驱动的 `ActorSystem` 抽象接口（默认实现 Monarch），解耦控制器、Rollouter、Batcher 与 Trainer 的进程网格编排 |
| 6 | `mooncake-store-reuse-validated-shard-paths-cached-dfs-2026` | [`kvcache-ai/Mooncake@d782a5998f`](https://github.com/kvcache-ai/Mooncake/commit/d782a5998fd1bf06a7c823ef454033df10a5dd06) | `2026-10-19` | `implementation` | `distributed_storage_backend.cpp`：在 `GetOrOpenShard` 中对与首次打开时完全一致的已校验路径跳过规范化（64 次重复读取仅需 **`1` 次规范化**），而对别名或变更路径重新验证但不重开 fd |
| 7 | `mooncake-store-add-session-ids-kv-events-dynamo-indexer-2026` | [`kvcache-ai/Mooncake@0e626c572b`](https://github.com/kvcache-ai/Mooncake/commit/0e626c572b85f47664546b0862e11cc6f0b95f56) | `2026-10-19` | `implementation` | `mooncake-integration/store/store_py.cpp`：在 `ReplicateConfig` 中支持最长 256B 的逐键 `session_ids` 写入归属标注并在 `MakeIndexedConfig` 中按过滤索引同步切片，供 Dynamo 会话前缀索引器消费 |
| 8 | `lmcache-include-request-tags-mooncake-lookup-keys-2026` | [`LMCache/LMCache@e45d4d5a85`](https://github.com/LMCache/LMCache/commit/e45d4d5a85b832a853c51f60f2b0485cb4846d3a) | `2026-10-22` | `implementation` | `mooncake_lookup_client.py`：在 `MooncakeLookupClient.lookup` 中将 `request_configs` 透传给 `token_database.process_tokens`，将请求 tag 纳入缓存键，消除多租户/多 LoRA 跨标签串读 |
| 9 | `dynamo-frontend-skip-per-token-text-decode-no-logprobs-2026` | [`ai-dynamo/dynamo@cffc4fa75c`](https://github.com/ai-dynamo/dynamo/commit/cffc4fa75cc919a342be97fd90ee275f78e59326) | `2026-10-27` | `implementation` | `lib/llm/src/backend.rs`：引入 `needs_token_text = data.log_probs.is_some()`，在未请求 logprobs 时跳过逐 token 文本解码（512-token 流式 4-token 分块将解码调用从 `640` 压至 **`128` 次，`5.0x` 缩减**） |
| 10 | `nixl-mooncake-fix-request-array-leak-postxfer-error-2026` | [`ai-dynamo/nixl@7bd1d1386e`](https://github.com/ai-dynamo/nixl/commit/7bd1d1386eb07e1f7efdd262135a8820dd84289f) | `2026-10-12` | `implementation` | `src/plugins/mooncake/mooncake_backend.cpp`：将 `nixlMooncakeEngine::postXfer` 中的 `new transfer_request_t[request_count]` 替换为 `std::vector<transfer_request_t>`，消除描述符长度不匹配提前返回 `NIXL_ERR_INVALID_PARAM` 时的内存泄漏 |

---

## 2. 核心实现机理与架构量化分析（Implementation Mechanics & Quantitative Analysis）

### 2.1 Megatron-LM 视觉 RoPE FP32 频率保真与 NIXL Mooncake `postXfer` RAII 防泄漏（`3ca52d95d6` & `7bd1d1386e`）
- **视觉 RoPE 精度退化**：`Pixtral2DRotaryEmbedding` 原本通过 `self.register_buffer("freqs", freqs, persistent=False)` 注册频率张量。当模型被 `Float16Module` 包装时，所有注册 buffer 会被自动转为 BF16/FP16。在 `head_dim=64, idx=30, patch_coord=63` 处，BF16 截断使 $\cos(63 \cdot \omega_{30})$ 产生 **`> 1.5 \times 10^{-4}`** 的余弦误差。将 `self.freqs` 保存为普通 Python 属性并在 `forward` 中按需 `.to(device=device)` 后，误差降至 **`0.0`**。
- **NIXL Mooncake `postXfer` 异常路径泄漏**：当 `request_count = 128` 且第 3 个描述符长度不匹配（`local[3].len != remote[3].len`）时，旧代码直接 `return NIXL_ERR_INVALID_PARAM` 跳过了循环后的 `delete[] request`，泄漏全部 **`128` 个 `transfer_request_t` 结构体**；改用 `std::vector<transfer_request_t>` 后泄漏降至 **`0`**。

---

## 3. 确定性实验结果验证（Deterministic Experiment Results · `own-offline`）

运行 `python experiments/te_compile_linear_bias_sm12x_fa4_megatron_rope32_mooncake_phase113/run.py --check` 的密封快照：

| 指标 | 修复前 / 基线 | 修复后 / Phase 113 | 变化率 / 结论 |
|---|---|---|---|
| TE `BasicLinear` + `Bias` `torch.compile` 前向内核数 | `2` (独立 GEMM + Bias) | **`1` (前向融合 + TP 特化)** | 保存量化张量供反向反量化复用 |
| TE SM120 (`sm_120`) 确定性反向 / 无梯度推理注意力后端 | 误选 FA4 导致非确定性反向 | **反向跳过 FA4 / 推理保留 `FlashAttention4`** | 依据 `requires_backward` 与 `device` 路由 |
| Megatron `Pixtral2DRotaryEmbedding` `patch=63` 余弦误差 | `> 0.0001` (BF16 buffer 截断) | **`0.0` (FP32 普通属性)** | 免疫 `Float16Module` 强制转换 |
| Megatron `--deterministic-mode` `Ring,NVLS` 准入 (SM90 vs SM100) | 全部拒绝 `NVLS` | **`SM90: False` / `SM100: True`** | Blackwell NVLS 硬件确定性放行 |
| Mooncake DFS 已校验分片路径 64 次缓存读取规范化次数 | `64` 次 | **`1` 次 (`64.0x` 减少)** | 加上 LMCache `request_configs` 标签隔离 |
| Dynamo 无 logprobs 流式解码调用 (`512` tokens, chunk `4`) & NIXL 泄漏 | `640` 次解码 / 泄漏 `128` 项 | **`128` 次解码 (`5.0x` 减少) / 泄漏 `0` 项** | `needs_token_text` 门控 + `std::vector` RAII |
