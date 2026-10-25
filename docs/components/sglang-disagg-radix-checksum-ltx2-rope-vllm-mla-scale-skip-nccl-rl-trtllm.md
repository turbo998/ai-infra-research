# SGLang 分离式 Decode Radix Cache 增量 KV 校验和、LTX-2 向量化融合 RoPE、vLLM MLA 单位缩放跳过与共享 NCCL RL 权重握手及 TensorRT-LLM MiniMax-M3 分段图修复

**English companion**: [sglang-disagg-radix-checksum-ltx2-rope-vllm-mla-scale-skip-nccl-rl-trtllm.en.md](sglang-disagg-radix-checksum-ltx2-rope-vllm-mla-scale-skip-nccl-rl-trtllm.en.md)

## 1. 背景与核心问题

在分离式推理（Disaggregated Prefill/Decode）、多模态扩散生成、MLA（Multi-head Latent Attention）FP8 KV Cache 以及在线 RL（Reinforcement Learning）训推权重同步中，推理引擎面临五类跨层正确性与尾延迟挑战：

1. **分离式 Decode 端开启 Radix Cache 时的 KV Checksum 误报冲突**：此前 SGLang 的 `--disaggregation-enable-kv-checksum` 计算整条提示词 `[0, end_idx)` 的 Adler-32 摘要；但开启 `--disaggregation-decode-enable-radix-cache` 后，Decode 节点已持有前缀 `[0, start_idx)`，Prefill 仅传输后缀页 `[start_idx, end_idx)`，导致每次前缀命中都被误判为数据损坏，迫使系统直接禁用 KV 校验。
2. **扩散模型 LTX-2 Q/K RMSNorm + Split-RoPE 标量访存瓶颈**：标量循环逐对（pair）读写 BF16 数据，未能打满 B200 16-byte 向量加载带宽。
3. **vLLM GLM-5.3 / DeepSeek-V3.2 MLA FP8 Cache 冗余缩放计算**：当 `mla_k_scale == 1.0`（unit scale）时，`concat_and_cache_mla_kernel` 仍在每个元素上从全局内存解引用 `*scale` 并执行浮点乘法，拖慢 TPOT。
4. **vLLM 稠密与稀疏 RL 权重传输引擎的 NCCL Rendezvous 重复与死锁隐患**：`NCCLTrainerWeightTransferEngine` 与 `SparseNCCLTrainerWeightTransferEngine` 各自维护初始化握手机制，若 Trainer（rank 0）与 Worker（`rank_offset = 1`）未并发启动进程组握手，极易在阻塞式 `open_trainer_endpoint` 中死锁。
5. **TensorRT-LLM MiniMax-M3 Piecewise CUDA Graph 在空 ADP Rank 下的缓存越界污染**：分段图（piecewise CUDA graph）按 padding 后的 token 数（如 64）准备输入，若在调用 `metadata._build_msa_fields()` 前未将 `metadata._num_tokens` 还原为真实 `live_tokens`（如 19），`msa_out_cache_loc` 会越界覆写后续 45 个无关 KV Cache 槽位。

---

## 2. 第一手源码实现路径审计

### 2.1 SGLang 分离式 Decode Radix Cache 后缀切片 Adler-32 校验（`#46423`, commit `ee98bb09b028`）

审计 `python/sglang/srt/disaggregation/{checksum,prefill,decode}.py`：
- 移除 `decode.py` 中当 `enable_kv_checksum` 与 `disaggregation_decode_enable_radix_cache` 同时开启时强制置 `enable_kv_checksum = False` 的互斥降级逻辑。
- 在 `checksum.py` 的 `page_indices_for_request(scheduler, req, end_idx, start_idx=0)` 与 `state_indices_for_request` 中引入页对齐前缀起点 `start_idx = req.disagg_decode_prefix_len`：
  $$\text{kv\_indices} = \texttt{req\_to\_token}[\texttt{req\_pool\_idx},\; \texttt{start\_idx}:\texttt{end\_idx}]$$
- Prefill 在 `send_kv_chunk` 时仅对实际发送的后缀页区间 `[start_idx, end_idx)` 计算 Adler-32 摘要；Decode 在收到传输完成后同样对自身行中的 `[start_idx, end_idx)` 计算摘要并比对，实现了 Decode 端 Radix Cache 命中与端到端传输完整性校验的无损共存。

### 2.2 SGLang LTX-2 8-Pair（16-Byte）向量化 Q/K RMSNorm + Split-RoPE 与 GPU 预处理泛化（`#46111`, `#45733`）

- 在 `python/sglang/kernels/kda_kernels/csrc/diffusion/ltx2_qknorm_split_rope.cuh`（`a887ac8fc6ea`）中，新增模板参数 `bool kVec`。当 `kVec = true` 时，每个线程以 `kWidth = 8` 个连续 pair（即 16 字节 `uint4` / 8 个 BF16 元素）为单位进行向量化读写，保持与标量路径完全一致的 FMA 舍入语义，将每 token 的全局内存事务数缩减为原来的 **1/8（8.0x 事务缩减）**。
- 在 `#45733`（`933853a3f263`）中，将精确匹配的 GPU 图像张量预处理推广至 Flux、Qwen-Image 与 Cosmos3 全系列扩散管线。

### 2.3 vLLM `concat_and_cache_mla_kernel` 编译期 `skip_scale` 分支（`#62102`, commit `56e4e42868b4`）

在 `csrc/cache_kernels.cu` 中，为 `concat_and_cache_mla_kernel` 增加编译期布尔模板参数 `bool skip_scale`：
```cpp
if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
  dst[dst_idx] = src[src_idx];
} else if constexpr (skip_scale) {
  dst[dst_idx] = fp8::scaled_convert<cache_t, scalar_t, kv_dt>(src[src_idx], 1.0f);
} else {
  dst[dst_idx] = fp8::scaled_convert<cache_t, scalar_t, kv_dt>(src[src_idx], *scale);
}
```
当 GLM-5.3 或 DeepSeek-V3.2 使用单位缩放（`scale == 1.0`）时，消除了每个 token 在 $512 + 64 = 576$ 维 MLA 潜向量上的冗余全局内存指针解引用与乘法指令，实测带来 **1.2%–3.0%** 的端到端 TPOT 改善。同时 `#63495`（`0765565f103d`）将 `clef_joint_schema.py` 中的 `_indices` 构造从多次小张量 `torch.tensor`/`torch.cat` 替换为单次 CPU `numpy` 向量化拼接，消除高频池化元数据开销；`#63496`（`8e11a8a49e11`）在启用 LoRA 时显式拦截不兼容的 MegaMoE 后端。

### 2.4 vLLM 稠密/稀疏 RL 权重传输共享 NCCL Rendezvous（`#63460`, commit `c9dbc92630bc`）

在 `vllm/distributed/weight_transfer/nccl_common.py` 中抽取出统一的 `build_worker_init_payload` 与 `run_trainer_rendezvous`：
- 固定 Trainer 发送端位于 `rank 0`，所有推理 Worker 位于 `rank_offset = 1`；
- 稠密引擎透传 `packed`、`packed_buffer_size_bytes` 与 `packed_num_buffers`，而稀疏引擎保持非打包默认配置；
- `run_trainer_rendezvous` 使用 `ThreadPoolExecutor` 并发执行 `open_trainer_endpoint` 与遠端 Worker 进程组初始化 RPC，彻底消除单线程顺序调用导致的 NCCL 握手死锁。

### 2.5 TensorRT-LLM MiniMax-M3 Piecewise CUDA Graph `live_tokens` 修复与 Runner 接口收敛（`#19721`, `#20270`）

- 在 `NVIDIA/TensorRT-LLM@f1d8a73f640f`（`#19721`, `nvbugs/6848775`）中，当 MiniMax-M3 运行分段 CUDA Graph（包含空 ADP rank）时，在调用 `metadata._build_msa_fields()` 之前显式设置 `metadata._num_tokens = live_tokens`，确保 `metadata.msa_out_cache_loc` 仅生成 `[128, 128 + live_tokens)`（如 19 个有效位置），防止 padding token 覆写后续 45 个有效缓存槽位。
- 在 `87342f447486`（`#20270`）中，将 `DecoderRunner` / `NoKVCacheRunner` 的内部方法收敛为私有 `_prepare_inputs` 与 `_model_forward`，明确 Qwen2-VL/Qwen3-VL 视觉注意力在初始化阶段从全局 `extra_attrs` 注销自身，避免视觉 FMHA 误读文本 Decoder 的 `attention_metadata`。

---

## 3. 可复现实验与定量验证

配套实验 `experiments/sglang_disagg_radix_checksum_ltx2_rope_vllm_mla_scale_skip_nccl_rl_trtllm/run.py` 検証结果：

| 引擎与模块 | 核心机制 | 修复前 / 基线表现 | 修复后 / 优化表现 | 证据级别 |
|---|---|---|---|---|
| SGLang Disagg KV Checksum (`#46423`) | 后缀范围 `[start_idx, end_idx)` Adler-32 | 全行摘要不匹配（强制关闭校验） | 5 个传输页摘要精确匹配 (`0x6708044d`) | `implementation` + `own-offline` |
| SGLang LTX-2 Split-RoPE (`#46111`) | `kWidth=8` pairs (16B vectorized load/store) | 4,096 次标量访存/token | 512 次向量访存/token（**8.0x** 缩减） | `implementation` + `own-offline` |
| vLLM MLA `skip_scale` (`#62102`) | `skip_scale=True` 编译期常数 `1.0f` | 每 token 576 次 `*scale` 读取与乘法 | 端到端 TPOT 降低 **1.2%–3.0%** | `author-benchmark` + `own-offline` |
| vLLM NCCL RL Rendezvous (`#63460`) | `run_trainer_rendezvous` 并发握手 | 稠密/稀疏重复实现且顺序阻塞易死锁 | 统一 `rank_offset=1` 并发无死锁握手 | `implementation` + `own-offline` |
| TRT-LLM MiniMax-M3 (`#19721`) | `_build_msa_fields` 前置 `_num_tokens = live_tokens` | 越界写入 64 个槽位（污染 45 槽位） | 精确写入 `[128, 147)` 共 19 个有效槽位 | `implementation` + `own-offline` |
