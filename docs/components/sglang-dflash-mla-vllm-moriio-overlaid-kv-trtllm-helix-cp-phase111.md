# Phase 111 Track B：SGLang DFlash MLA 草稿复用与融合异步 NaN 探针、vLLM RDNA/MORI-IO/混合重叠 KV 拷贝，以及 TensorRT-LLM Helix CP 跨 Rank 准入共识与 MiniMax-H3 融合 RoPE

> **公开来源与证据边界**：本文基于 SGLang、vLLM 与 TensorRT-LLM 公开 GitHub 仓库一手提交及本项目合成离线验证（`experiments/sglang_dflash_mla_vllm_moriio_trtllm_helix_cp_phase111/run.py`），严格区分源码合同、作者测试与离线建模。

---

## 1. SGLang：DFlash 复用目标 MLA 后端、单内核异步 NaN 断言、NUMA 物理核缓存与 DiT BCG 调用方分配输出

### 1.1 DFlash 通过目标 MLA 注意力服务 SpecForge MLA 草稿（`sgl-project/sglang@b068417bfcc1`, `#44676`）
- **实现路径**：
  - `python/sglang/srt/arg_groups/speculative_hook.py` 新增 `_publish_dspark_mla_draft_backend` 与 `_keep_mla_draft_off_trtllm_mha`，使 SpecForge MLA 草稿模型直接复用目标 MLA 注意力后端；
  - `python/sglang/srt/mem_cache/memory_pool.py` 新增 `set_mla_kv_buffer_prefix_valid(layer, loc_2d, commit_lens, cache_k_nope, cache_k_rope)`：对 `loc_2d` 中列下标 $\ge \text{commit\_lens}[b]$ 的未提交草稿位置，通过 `torch.where` 将写入目标重定向至保留的物理槽位 `0`（MLA KV 池自动跳过槽位 `0` 的写入，与 CUDA Graph padding 语义一致）。

### 1.2 融合异步 NaN 校验单内核（`sgl-project/sglang@ee5443f53026`, `#43720`）与 NUMA 预算/DiT BCG 修复
- **融合异步 NaN 探针（`ee5443f53026`）**：`python/sglang/srt/utils/async_probe.py` 用单个 Triton 内核 `_assert_no_nan_kernel`（`@triton.jit(debug=True)`，`_NAN_CHECK_BLOCK = 1024`）替代原先 `torch.isnan(x) -> .any() -> torch._assert_async()` 的 3 次独立内核启动，消除中间布尔张量分配并将每次探针的内核启动数从 `3` 降至 `1`。
- **NUMA 空节点过滤导致的 CPU 内存预算高估修复（`302d5e7b8ca9`, `#45516`）**：为 `get_cpu_ids_by_node()` 增加 `@lru_cache(maxsize=1)` 并稳定非空物理核 NUMA 节点映射。
- **DiT 注意力 Breakable CUDA Graph（BCG）调用方预分配输出（`c43abd6d9e38`, `#46659`）**：`UlyssesAttention._new_bcg_output` 在 BCG 断点前预分配输出张量并由断点原位写入，确保后续捕获图段在重放（replay）时读取到刷新的输出。

---

## 2. vLLM：ROCm RDNA 2D 分页注意力启动配置、SM90 GDN 分块、MORI IO 跨步长度与混合模型重叠 KV 视图拷贝

### 2.1 ROCm RDNA `kernel_paged_attention_2d` 与 SM90 GDN FP32 状态分块
- **ROCm RDNA（gfx12）大 Tile 防内存故障（`vllm-project/vllm@9af236a33281`, `#63735`）**：在 `vllm/v1/attention/ops/chunked_prefill_paged_decode.py` 中，当 `on_rdna()` 且 `TRITON_BLOCK_SIZE >= 32` 或 `tile_size >= 4096` 时，強制设置 `num_stages = 1` 并按 `tile_size >= 8192 -> 16 warps`、`tile_size >= 4096 -> 8 warps` 扩充 warp 数，修复 gfx12 上 Triton buffer ops 在 2-stage 流水下触发的 GPU memory fault（`#62697`）。
- **SM90 GDN 打包解码 Tile 调优（`vllm-project/vllm@94f56ce92cc4`, `#63864`）**：在 `fused_recurrent_gated_delta_rule_packed_decode` 中，当 `K == V == 128`、设备为 SM90（H100/H200）且 `initial_state.dtype == torch.float32` 时，将 `BV` 收紧为 `4`（BF16 状态则保持 `BV = 32`）。

### 2.2 AMD MI350 DeepSeek-V4-Pro MORI IO PD 分离与混合重叠 KV 视图完整拷贝
- **MORI IO 跨步感知 `block_len`（`vllm-project/vllm@3d81a510f9eb`, `#48989`）**：启用 DeepSeek-V4-Pro 在 AMD MI350 AINIC 上的 1P1D TP8 / WideEP8 MORI IO KV Connector，并将 `moriio_layout.py` 中的 `block_len` 从紧密乘积修正为跨步跨度公式：
  $$\text{block\_len} = \left(1 + \sum_{d \ge 1} (\text{shape}[d] - 1)\cdot \text{stride}[d]\right) \times \text{element\_size}$$
  对于带行对齐填充的 MLA 块（如 `shape=(B, 1, 64, 576), stride=(38912, 38912, 608, 1)`），精确覆盖 `38,880 B` 而非遗漏行尾填充的 `36,864 B`。
- **混合模型重叠 KV 视图原地拷贝修复（`vllm-project/vllm@8701c9b7513d`, `#55112`）**：在混合 Mamba + Attention 模型中，Mamba 状态视图与注意力 KV 视图共享同一基地址 `data_ptr()`，但 Mamba 视图仅覆盖每个块的前导字节（例如 64B 块中的前 24B）。`copy_kv_cache_blocks_inplace` 将去重键从 `(device, data_ptr)` 扩展为 `(device, data_ptr, tuple(shape), tuple(stride), element_size)`，防止先遇到窄 Mamba 视图后跳过宽 Attention 视图导致的块尾部数据残留损坏。

---

## 3. TensorRT-LLM：Helix CP + Attention-DP 生成侧 KV 传输准入共识与 MiniMax-H3 融合 QK-Norm+RoPE

### 3.1 Helix CP 跨 Rank 准入共识（`NVIDIA/TensorRT-LLM@94348c4de6eb`, `#19575`）
- **死锁根因**：在 `enable_attention_dp=True` 且启用 Helix Context Parallelism（`cp_size > 1`）时，V1 KV 管理器按轮询（round-robin）从 CP rank 0 分配块，且取消/超时按 rank 本地退休请求，导致同一 CP 组的不同 rank 在同一轮可能准入不同的 `gen_init` 请求集合，进而在 `receive_gen_init` 末尾的传输状态集合通信中单边挂起。
- **修复机制**：`DisaggTransferCoordinator.align_gen_admission_across_cp` 在 `pp=1` 循环中对 CP 组求交集——仅保留所有 CP rank 均准入且路由至同一 Attention-DP rank 的请求；未通过交集的请求归还 V2 KV 并在后续迭代重试。同时在 Helix CP 下禁用 `KVCacheAwareADPRouter`（因各 CP rank 本地前缀探测结果可能不一致）。

### 3.2 MiniMax-H3 BF16 融合逐头 QK-Norm 与部分 Split-Half RoPE（`NVIDIA/TensorRT-LLM@af32051e1926`, `#19645`）
- **实现路径**：`tensorrt_llm/_torch/visual_gen/models/minimax_h3/fused_rope.py` 使用单个 Triton 内核（`1 token x 8 heads/program`, `1 warp`, `_CHUNK = 16`）在 FP32 寄存器中一次性完成 Q 和 K 的逐头 RMSNorm 与前导 `rotary_dim` 的 split-half RoPE 及尾部直通，将原本 eager 路径的 **5 次 BF16 舍入压缩为 1 次**（相对 FP32 L2 误差 `3.5e-3`），并将多卡 Ulysses LPIPS 门限统一收口至 `0.30`。

---

## 4. 本项目可复现实验验证（`experiments/sglang_dflash_mla_vllm_moriio_trtllm_helix_cp_phase111`）

运行 `python experiments/sglang_dflash_mla_vllm_moriio_trtllm_helix_cp_phase111/run.py --check` 验证：
- SGLang `set_mla_kv_buffer_prefix_valid` 对 `2x4` 槽位矩阵（`commit_lens=[3, 1]`）精准保留 `4` 个有效槽位并将 `4` 个未提交槽位重定向至槽位 `0`；
- vLLM MORI IO 跨步公式修复 `38,880 B` vs `36,864 B` 的 `2,016 B/block` 传输截断；重叠视图几何去重键消除每块 `40 B`（`64 B - 24 B`）的尾部残留损坏；
- TensorRT-LLM `align_gen_admission_across_cp` 正确过滤单侧准入（`req_3`, `req_4`）与 DP 路由分叉请求（`req_2`），仅放行跨 CP rank 一致准入的 `req_1`。
