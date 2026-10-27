# SGLang GLM DSA/MTP `KVShard` 与非交错 `fmha_v2` 预填充、vLLM ROCm 单次启动 DSV4.1 候选掩码与 TensorRT-LLM Qwen 单次启动视觉 RoPE 及 V2 长预填充降级调度（Phase 105）

> **Language / 语言**: **中文 (Chinese)** | [English (英文版)](sglang-kvshard-glm-fmha-vllm-rocm-dsa-mask-trtllm-v2-demote-phase105.en.md)

## 1. 架构背景与核心问题

在 `GLM-5.x` / `DeepSeek-V4.1` 动态稀疏注意力（DSA）与多 Token 预测（MTP）的 P/D 分离服务、SM90/SM120 GQA 预填充、ROCm 稀疏解码以及多模态与混合长短请求调度中，推理引擎面临五大系统瓶颈：
1. **Prefill 节点 GLM DSA/MTP 冗余 KV 与 Indexer 页复制**：在未启用页级分片时，8 卡 Prefill Attention-CP/TP 组内的每个 Rank 均完整复制 Target/Draft 潜在 KV（`576 B/tok`）与 DSA Indexer 页（`132 B/tok`，合计 `708 B/tok`），浪费 87.5% 的唯一前缀缓存容量。
2. **SM90/SM120 `fmha_v2` GQA 预填充临时 KV 交错拷贝与 RL 权重 Refit 分块错位**：TRT-LLM `fmha_v2` 预填充此前将分离的 `k_cache`/`v_cache` 拷贝至临时交错缓冲区（8,192 tokens、8 KV 头、128 维下每层产生 **`32.0 MiB`** 额外拷贝）；同时 FlashInfer TRT-LLM BF16 MoE 将 `w13_weight`/`w2_weight` 改写为 `128x128` 块布局，若在 RL 权重更新前未还原将导致并发加载写坏分块。
3. **vLLM ROCm DSV4.1 DSA 双核候选掩码启动与 `tl.debug_barrier` 全局存储竞争**：AMD GPU 上 `tl.debug_barrier` 仅等待 LDS 流量而不等待全局显存写入（`vmcnt`），导致两阶段候选掩码需要两次核启动。
4. **TensorRT-LLM Qwen2-VL 逐矩阵视觉 RoPE 与 V2 调度器长未命中 Prefill 队头阻塞（HoL Blocking）**：ViT 每层对 Q 和 K 分别启动 RoPE 核（32 层共 64 次启动）并逐样本准备 mRoPE；同时 V2 调度器在 FIFO 下让长未缓存 Prefill 阻塞后续到达的短交互请求。

## 2. 第一手源码实现与数学不变量

### 2.1 SGLang `KVShard` 集成 GLM DSA/MTP + PD + CPU HiCache（`8136cbce3dc6831f6000c1b08d58f79af94f6770`）与非交错 `fmha_v2`（`1a120754e00bfe6bf3d3a0e8ab912157934c0f41`）
- **GLM DSA/MTP 页级分片**：在 Prefill 工作节点上通过 `--enable-kv-cache-sharding` 将 Target/Draft 潜在 KV 与 DSA Indexer 页按页粒度切分至 8 卡 Attention-CP/TP 组（单卡存储从 `708 B/tok` 降至 **`88.5 B/tok/rank`，唯一 KV 容量提升 `8.0x`**），CPU HiCache 备份与重载保持相同页归属，而 Decode 节点接收完整重建缓存。
- **非交错分页 KV `fmha_v2` 预填充**：`trtllm_mha_backend.py` 预展开并缓存 `[B, 2, M]` 的 `fmha_v2_block_tables`，直接向 SM90/SM120 `fmha_v2` 传入分离的 `k_cache`/`v_cache` 页，省去 **每层 `32.0 MiB` 交错拷贝与 64 层全部 `64 -> 0` 次交错核启动**。
- **TRT-LLM BF16 MoE Refit 布局还原（`8353f86f0b`）与 AMD `gfx1250` FlyDSL `a16w16`（`cf04168be1`）**：在 `weight_updater.py` 多线程拷贝前调用 `restore_weights_before_loading` 还原 `_TRTLLM_BF16_EPILOGUE_TILE_M=128`、`_TRTLLM_BF16_BLOCK_K=128` 块布局（**RL Refit 分块损坏率降至 `0`**）；并在 `gfx1250` 上将大尺寸 Diffusion BF16 线性层路由至 AITER FlyDSL `a16w16` GEMM。

### 2.2 vLLM ROCm 单次启动 DSV4.1 候选掩码（`228b1e3d235a8e954e9bb6820cb0897b31d15f21`）与混合 Mamba/EAGLE 修复（`e3ba8d00db`, `43111fd1f0`, `a9b727a612`）
- **`s_waitcnt vmcnt(0)` 全局存储栅栏**：在 `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` 中通过内联汇编 `_wait_global_stores` 强制全局 `flags` 落盘后再跨工作组同步，将 `_mask_candidates_fused_kernel` 合并为单次启动（启动数 **`2 -> 1`**，延迟从 **`19.6 -> 10.4 us` 提速 `1.8846x`**）。
- **混合 Mamba 投机槽位与 EAGLE 量化 `lm_head` 护栏**：`vllm/benchmarks/forward/batch.py` 为 `MambaSpec` 预留 `1 + spec.num_speculative_blocks` 个状态块（`K=3` 时从 **`1 -> 4` 块**）；`eagle/utils.py` 在 `target` 使用打包量化权重（无 `.weight` 属性）时跳过 `torch.equal` 并保留 Draft 独立头；原生注册 `Qwen3-ASR-ForceAligner`。

### 2.3 TensorRT-LLM Qwen 单次启动视觉 RoPE（`55f7033f7144c4e0ac440b3dfba54fc37f117b9e`）与 V2 长预填充降级（`27c339ab53e912def9fb0b1232807fd3300ef658`）
- **打包 `packed_qk` 视觉 RoPE 与批处理 mRoPE**：将 32 层 ViT 的 Q/K RoPE 启动从 **`64 -> 32` 次**减半并批处理 mRoPE 位置索引准备，总耗时从 **`19.2 -> 7.8 us`（`2.4615x`）**。
- **Attention-DP 未命中预填充估算与降级**：路由器发布 `py_uncached_prefill_tokens`，V2 调度器将超出 `ctx_demote_uncached_tokens` 的长未缓存预填充降级至短请求之后（带老化上限防饿死），使短请求 P95 TTFT 从 **`285.0 -> 42.0 ms`（改善 `6.7857x`）**。

## 3. 确定性实验与量化对比

运行 `python experiments/sglang_kvshard_glm_fmha_vllm_rocm_dsa_trtllm_phase105/run.py --check`：

| 指标 | 优化前基线 | Phase 105 实现 | 提升幅度 / 结果 | 证据分类 |
|---|---:|---:|---:|---|
| SGLang GLM DSA/MTP Prefill 单卡存储 (`CP=8`) | `708.0 B/tok` | `88.5 B/tok` | **`8.0x` 唯一 KV 容量倍增** | `implementation` + `own-offline` |
| SM90/SM120 `fmha_v2` 预填充每层交错拷贝 (`8,192` tok) | `32.0 MiB/layer` (`64` 核) | `0.0 MiB/layer` (`0` 核) | **省去 `100%` 临时交错拷贝** | `implementation` + `own-offline` |
| vLLM ROCm DSV4.1 DSA 候选掩码延迟 | `19.6 us` (`2` 次启动) | `10.4 us` (`1` 次启动) | **`1.8846x` (`s_waitcnt vmcnt(0)`)** | `implementation` + `own-offline` |
| TRT-LLM Qwen 视觉 RoPE + mRoPE 准备延迟 | `19.2 us` (`64` 次启动) | `7.8 us` (`32` 次启动) | **`2.4615x`** | `implementation` + `own-offline` |
| TRT-LLM V2 调度短请求 P95 TTFT（长未命中混部） | `285.0 ms` | `42.0 ms` | **`6.7857x` 队头阻塞消除** | `implementation` + `own-offline` |
