# SGLang MiniMax-M3 Hopper 稀疏注意力 FP8 Index-K、DFlash DP Attention 布局契约、扩散分离容错与 vLLM ModelOpt 混合精度 Draft 映射深度解析

> **文档状态**：Phase 80 (`priority: 89`) 第一手源码级深度审计  
> **双语对照**：[`sglang-minimax-m3-hopper-sparse-dflash-dp-placement-disagg-isolation-vllm-modelopt.en.md`](sglang-minimax-m3-hopper-sparse-dflash-dp-placement-disagg-isolation-vllm-modelopt.en.md)  
> **可复现实验**：[`experiments/sglang_minimax_m3_sparse_fp8_dflash_dp_attn_draft_layout_vllm_modelopt_mixed/run.py`](../../experiments/sglang_minimax_m3_sparse_fp8_dflash_dp_attn_draft_layout_vllm_modelopt_mixed/run.py)

---

## 1. 上游源码证据矩阵

| 证据编号 | 上游提交 | 核心变更模块 | 证据类别 |
| --- | --- | --- | --- |
| `src-sglang-minimax-m3-hopper-sparse-attn-2026` | `sgl-project/sglang@db7b66c1b6a7` (`#46135`) | MiniMax-M3 Hopper (`sm_90`)：`StoreTrait` BF16->FP8 E4M3 融合写入 `index_k`（减半带宽/容量）+ Triton 64-bit `tile_ptrs` | `implementation` |
| `src-sglang-dflash-cuda-dp-attention-2026` | `sgl-project/sglang@38ff13ea7928` (`#46330`) | DFlash/DFlash2 支持 CUDA `--attn-dp-size > 1`、量化词表 `gather_draft_embedding` 与空闲 DP rank M-RoPE 形状修复 | `implementation` |
| `src-sglang-draft-parallel-layout-spec-2026` | `sgl-project/sglang@6623ea445a28` (`#46290`) | 声明式 `DraftParallelSpec` (`SINGLE_GROUP` vs `TARGET_LAYOUT`) 与 `resolve_dflash_draft_cell_size` 精确 KV 池计价 | `implementation` |
| `src-sglang-diffusion-disagg-fault-isolation-2026` | `sgl-project/sglang@a996b5f2244e` (`#46120`) | 扩散模型分离式编排器（`DiffusionServer`）10 类单请求故障隔离、重复 `request_id` 拒绝与槽位回收 | `implementation` |
| `src-vllm-modelopt-mixed-draft-quant-map-2026` | `vllm-project/vllm@5f63ea461ccd` (`#63001`) | `modelopt_mixed` 归一化、MI300 `e4m3fnuz` 转换与 `map_draft_quant_config` 逆序层号偏移 | `implementation` |
| `src-vllm-weight-transfer-torch-accelerator-2026` | `vllm-project/vllm@57ab2f35ac6e` (`#63457`) | RL 权重传输引擎（IPC/NCCL/RDT/SparseNCCL）全面迁移至 `torch.accelerator` 抽象 | `implementation` |

---

## 2. MiniMax-M3 Hopper 稀疏注意力：`StoreTrait` BF16→FP8 融合转型与 64-bit 块指针

在 `sgl-project/sglang@db7b66c1b6a7` (`#46135`) 中，针对 Hopper (`sm_90`) 上的 MiniMax-M3 稀疏注意力解决了两个核心瓶颈：

1. **FP8 Index-K 融合存储 (`fused_store_kv_index.cuh`)**：
   - 此前在 Hopper 上当 `fp8_attn_gemm=False` 时，即便主 KV cache 已启用 `float8_e4m3fn`，稀疏检索用的 `index_k` 仍以 BF16（`2 B/elem`）存储。
   - 新引入的 `StoreTrait<kHeadBytes, kCastFp8, kUsePDL>` 定义输入字节数 `kInHeadBytes = kCastFp8 ? 2 * kHeadBytes : kHeadBytes`，每个线程负责 16 字节（`kWorkerSize = kInHeadBytes / 16`，当 `kHeadBytes=128` 且 `kCastFp8=true` 时为 `16` 个线程/head），在寄存器内利用 `__nv_cvt_bfloat16raw2_to_fp8x2(..., __NV_SATFINITE, __NV_E4M3)` 直接将 BF16 激活压缩为 FP8 E4M3 写入 `index_k_cache`，使 `index_k` 显存占用与解码读取带宽同时降低 **50%**（`2.0x` 压缩）。
2. **移除 Triton 3.9 废弃的 `tl.make_block_ptr` 并升级 64-bit 索引**：
   - 将 `minimax_sparse_Lightning_attn.py` 中的 `tl.make_block_ptr` 替换为 `tile_ptrs_1d/2d/3d`，并强制使用 `tl.arange(0, BLOCK).to(tl.int64)` 计算偏移。当 KV 池序列索引达 $2\times 10^7$、`stride=128` 时，线性偏移为 $2.56\times 10^9 > 2^{31}-1$，64-bit 索引彻底消除了 32-bit 有符号整数溢出崩溃。

---

## 3. DFlash/DFlash2 CUDA DP Attention 与声明式 `DraftParallelSpec`

1. **CUDA DP Attention 投机解码 (`#46330`)**：
   - 允许在 `--attn-dp-size > 1` 且 `--enable-dp-lm-head` 下运行 `DFLASH` / `DFLASH2`。
   - 新增 `gather_draft_embedding`：在跨 DP rank `all_gather` 之前先在本地 shard 执行 `VocabParallelEmbedding` 查找，从而原生兼容 FP8/INT4 量化词表。
   - **空闲 DP Rank M-RoPE 修复**：当某个 DP rank 处于 idle（`bs == 0`）但被 padding 进 target verify batch 时，显式初始化 `ret.mrope_positions = ret.positions.unsqueeze(0).expand(3, -1)`（形状 `(3, 0)`），防止 M-RoPE 旋转位置编码因 `None` 解引用崩溃。
2. **声明式 `DraftParallelSpec` 与 KV 池精确计价 (`#46290`)**：
   - 将投机草稿模型的 TP 拓扑抽象为 `SINGLE_GROUP`（DFlash、Eagle3、非 MoE DSpark）与 `TARGET_LAYOUT`（DeepSeek-V4 MoE DSpark、Standalone、MultiLayerEagle、FrozenKVMTP）。
   - 在分配 KV 池**之前**解析 `DraftParallelLayout(tp_group, attn_tp_size, disable_cp)`。例如在 `TP=8, DP=2`（`attn_tp_size=4`）下，若误用全局 `tp_size=8` 计价，`resolve_dflash_draft_cell_size` 会将草稿每 token KV 字节数低估 **2.0 倍**（`384 B` vs 真实 `768 B`），修复后彻底消除了草稿 KV 池尺寸错配。

---

## 4. 扩散分离编排容错 (`#46120`) 与 vLLM ModelOpt 逆序层映射 (`#63001`)

- **SGLang 扩散分离式编排容错 (`#46120`)**：在 `DiffusionServer` 路由循环中按消息捕获 `unpack_tensors` 异常，通过 `_extract_request_id` 恢复请求 ID，拒绝重复活跃 `request_id` 覆盖在途状态，并在角色错误时立即释放预留接收槽位与调度容量，通过 10 类故障注入验证路由主循环零崩溃。
- **vLLM `map_draft_quant_config` 逆序重映射 (`#63001`)**：将 ModelOpt 混合精度统一为 `modelopt_mixed`，并在将草稿模型检查点中的 `layers.{i}` 偏移至 `layers.{start_layer_id + i}` 时采用**逆序遍历** `reversed(range(num_hidden_layers))`。若按正序遍历且 `start_layer_id = 1`，`layers.0` 会先变为 `layers.1`，随后被 `i=1` 规则再次匹配变为 `layers.2`，最终级联坍缩为 `layers.3`！逆序遍历保证了 `layers.0 -> layers.1, layers.1 -> layers.2, layers.2 -> layers.3` 的严格双射。
