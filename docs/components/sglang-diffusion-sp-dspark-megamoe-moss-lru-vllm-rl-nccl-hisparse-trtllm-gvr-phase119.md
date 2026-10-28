# SGLang MOSS-Image SP 全隐变量与融合 QKV、DSV4 免宿主长度、MegaMoE 预算修复、PD 预热队列 LRU、vLLM RL NCCL 按阶段广播、HiSparse 越界防护与 TRT-LLM GVR K=2048（Phase 119 / Priority 128）

> **证据分类**：Upstream `implementation` + `author-benchmark`（10 个上游公开提交，2026-10-28）+ 本仓库 `own-offline` 确定性算术/拓扑验证（`experiments/sglang_diffusion_sp_dspark_megamoe_moss_lru_vllm_rl_nccl_hisparse_trtllm_gvr_phase119/run.py`）。本文不包含未标注的真实 GPU 硬件重测声明。

## 1. 核心架构问题与背景

在多模态扩散生成、DeepSeek-V4 (`dsv4` / `DSpark`) 投机解码、MegaMoE 注意力数据并行（`attn_dp`）、PD 分离调度以及 RL 在线权重同步的工程实践中，近期定位并修复了五类关键系统缺陷与性能瓶颈：

1. **MOSS-Image 序列并行（SP）提前切分 Latent 破坏全图 RoPE 与 QKV 三次独立 GEMM 开销**：若在进入 DiT 前将图像 Latent 按 SP 切分，会导致联合图像/文本 Token 的二维旋转位置编码（Full-Image RoPE）错位；同时联合块（Joint Block）的 `q, k, v` 投影分为 3 次独立 Linear 启动，在 `quality="high"` 模式下未能融合。
2. **MegaMoE 在 `attn_dp` 下的 Prefill 预算二次整除与 PD Bootstrap 队列前缀被驱逐**：`ServerArgs` 已将 `chunked_prefill_size` 按 `attn_dp_size` 整除，但 MegaMoE 再次除以 `attn_dp_size`，导致每 Rank 预填充吞吐腰斩；此外 PD 分离 Prefill 节点的 `_touch_waiting_prefixes` 仅刷新 `waiting_queue` 而漏掉了正在建立 KV 握手连接的 `bootstrap_queue`，导致长前缀在传输前被 Radix Tree LRU 驱逐。
3. **vLLM RL 多流水线阶段（PP）NCCL 权重广播源固定为 0 与 LoRA+EP 非线性放置静默错位**：旧版 `NCCLWeightTransferEngine` 将 Trainer 视为单一 Rank 0，无法支持 PP 多阶段 Trainer 直接由各自 Rank 广播权重；且 `FusedMoEWithLoRA` 在启用 EP 时未拦截 `round_robin` 专家放置策略。
4. **DFlash/DSpark 投机解码运行期 Triton JIT 卡顿与 HiSparse 小 Prefill 越界读取**：`OnlineAcceptanceEstimator` 的 `BLOCK_R` 随活跃请求数动态翻倍触发运行期重编译；而 `trtllm-gen` 稀疏 MLA 内核在处理极小分阶段 Prefill KV 缓冲区（如 1 行）时会向后越界预取约 250 行（每行 `656 B`），触发非法内存访问。
5. **TensorRT-LLM GVR（Guided Vocabulary Restriction）在 $K=2048$ 下的单槽位上限**：旧版 `GvrRegClusKernel` 仅支持 $K \le 1024$，当 $K=2048$ 时无法在 `BLKC=1024` 块大小下完成双槽位（`HINT_SLOTS = 2`）Block64 采样与并行救援。

---

## 2. 上游一手实现剖析（10 个零重叠 Commit）

| # | 仓库与提交 | 日期 | 核心实现路径与微架构机制 |
|---|---|---|---|
| 1 | `sgl-project/sglang@5bb7ecc2c2` (`#46664`) | 2026-10-28 | `python/sglang/multimodal_gen/configs/pipeline_configs/moss_image.py`：实现 `shard_latents_for_sp` 返回 `(latents, False)`（`did_sp_shard_latents = False`），保留完整图像 Latent 进入 DiT 构建全图 RoPE 后再对联合 Token 执行 SP 切分。 |
| 2 | `sgl-project/sglang@357c53885f` (`#46897`) | 2026-10-28 | `python/sglang/kernels/ops/attention/dsv4/attn_metadata_kernels.py`：当 `seq_lens_cpu` 或 `extend_seq_lens_cpu` 为 `None` 时，在设备端直接通过 `torch.cumsum(extend_seq_lens, dim=0) - extend_seq_lens` 计算 `extend_start_loc`，使 DSpark 草稿准备在所有设备上零 Host 同步运行。 |
| 3 | `sgl-project/sglang@15d39bd840` (`#46267`) | 2026-10-28 | `python/sglang/srt/models/megamoe.py`：移除对 `cfg.chunked_prefill_size` 的重复除以 `attn_dp_size`（在 `attn_dp_size=2` 下将每 Rank Prefill 预算从被误砍的 `4096` 恢复至正确的 `8192`，**2.0x** 预算恢复）。 |
| 4 | `sgl-project/sglang@610a74d648` (`#46641`) | 2026-10-28 | `python/sglang/multimodal_gen/runtime/models/dits/moss_image.py`：将联合块 `q, k, v` 合并为单层 `nn.Linear(dim, 3 * dim)` 并为 `_norm_rope_kernel` 增加跨步 `TOKEN_STRIDE` 支持；`quality="high"` 时执行单次融合 QKV GEMM（`3 -> 1`），`exact`/`lossless` 时沿用权重切片视图保持逐位精确，切换无需重载权重。 |
| 5 | `sgl-project/sglang@54be2010d3` (`#46812`) | 2026-10-28 | `python/sglang/srt/managers/scheduler.py`：在 `_touch_waiting_prefixes` 中先逆序遍历刷新 `bootstrap_queue[:512]`（`SGLANG_WAITING_PREFIX_REFRESH_BOOTSTRAP_MAX=512`），再刷新 `waiting_queue`，保护 PD 预填充节点握手期间的缓存前缀。 |
| 6 | `vllm-project/vllm@2b678ac739` (`#62557`) | 2026-10-28 | `vllm/distributed/weight_transfer/nccl_engine.py`：扩展 `StatelessProcessGroup` 支持 `num_broadcasters >= 1`（Trainer 占用 `[0, rank_offset)`，推理 Worker 占用 `[rank_offset, world_size)`），并由 `NCCLWeightTransferUpdateInfo.src` 按张量/流水线阶段指定广播源 Rank。 |
| 7 | `vllm-project/vllm@1c787a25cc` (`#62797`) | 2026-10-28 | `vllm/lora/layers/fused_moe.py`：在 `FusedMoEWithLoRA._ep_check` 中检查 `expert_placement_strategy`，若启用 EP 且策略非 `"linear"`（如 `"round_robin"`）则立即抛出 `ValueError`。 |
| 8 | `vllm-project/vllm@458334f00f` (`#63819`) | 2026-10-28 | `vllm/v1/spec_decode/online_acceptance.py` & `dflash.py`：将 `_refit_kernel` 的 `BLOCK_R` 固定为 `max(256, next_power_of_2(max_num_reqs))` 并新增 `warmup()`，同时将 DFlash `_prepare_dflash_inputs_kernel` 的 `BLOCK_SIZE` 固定为 `128`，消除运行期 Triton JIT 编译毛刺。 |
| 9 | `vllm-project/vllm@ebaa812475` (`#61659`) | 2026-10-28 | `vllm/v1/attention/backends/mla/hisparse.py`：引入 `_MIN_STAGED_PREFILL_KV_ROWS = 256`，分配 `max(num_blocks, cdiv(256, block_size))` 个块并切片返回 `[:num_blocks]`，防止 `trtllm-gen` 稀疏 MLA 越界读取约 250 行（`656 B/row`）导致崩溃。 |
| 10 | `NVIDIA/TensorRT-LLM@9cf4530787` (`#20998`) | 2026-10-28 | `tensorrt_llm/_torch/speculative/gvr.py`：扩展 `GvrRegClusKernel` 模板至 `K=2048`（`BLKC=1024`，`HINT_SLOTS = (K + 1023) // 1024 = 2`），引入 64 线程（每组 32 候选）Block64 采样与并行 Warp 救援。 |

---

## 3. 关键公式、数据移动与离线验证结果

### 3.1 HiSparse 暂存 KV 安全分配界与 GVR $K=2048$ 槽位映射
1. **HiSparse 暂存 KV 最小分配块数**：设页块大小为 $B_{\text{blk}}$，请求块数为 $N_{\text{req}}$，由于 `trtllm-gen` 稀疏 MLA 内核在短序列下仍按固定深度预取约 250 行（每行 `656 B`），安全分配块数与保底字节数为：
   $$N_{\text{alloc}} = \max\!\left(N_{\text{req}}, \left\lceil \frac{256}{B_{\text{blk}}} \right\rceil\right), \quad S_{\min} = 256 \times 656\text{ B} = 167,936\text{ B} \approx 164\text{ KiB}$$
2. **GVR $K=2048$ 双槽位与 Block64 采样**：当候选集 $K = 2048$ 且寄存器聚类块大小 $\text{BLKC} = 1024$ 时，每线程提示槽位数为 $\text{HINT\_SLOTS} = \lfloor (K + 1023) / 1024 \rfloor = 2$，Block64 采样由 64 个活跃线程并行分担 $K / 64 = 32$ 个候选/线程组。

### 3.2 离线模型验证汇总（`run.py --check`）

- `moss_sp_did_shard_latents_before_rope`: `False`（保留全图 Latent 构建 RoPE 后再在 DiT 内切分 SP）
- `dsv4_extend_start_loc` / `dsv4_host_syncs_required`: `[0, 4, 11, 14]` / `0`（纯设备端计算起始偏移，零 Host 同步）
- `megamoe_budget_restoration_ratio`: `2.0`（在 `attn_dp_size=2` 下将每 Rank Prefill 预算从 `4096` 恢复至 `8192`）
- `moss_high_quality_gemms_per_block`: `1`（联合块 QKV 投影从 3 次 GEMM 融合为 1 次，跨步 `TOKEN_STRIDE = 9216`）
- `lru_touch_first_item`: `"boot_3"`（`bootstrap_queue[:512]` 优先于 `waiting_queue` 逆序刷新 LRU）
- `nccl_world_size` / `valid_src_ranks`: `12` / `[0, 1, 2, 3]`（4 个 PP Trainer Broadcaster + 8 个推理 Worker）
- `lora_ep_round_robin_rejected`: `True`（拦截 LoRA + EP 下的非线性专家放置）
- `online_acceptance_block_r` / `dflash_fixed_block_size`: `256` / `128`（消除运行期 Triton JIT 重编译）
- `hisparse_oob_prevented`: `True`（分配 `4` 个 `block_size=64` 块共 `256` 行，吸收 `~250` 行硬件预取越界）
- `gvr_hint_slots` / `gvr_candidates_per_group`: `2` / `32`（GVR $K=2048$ 双槽位与 Block64 采样验证通过）

---

## 4. 面向云解决方案架构师的练习与可证伪问题

1. **计算练习**：在启用 `attn_dp_size = 4` 且全局配置 `--chunked-prefill-size 32768` 的 MegaMoE 部署中，`ServerArgs` 将每 DP Rank 的 `cfg.chunked_prefill_size` 设为 $32768 / 4 = 8192$。修复前每 DP Rank 实际只获得了多少 Token 的 Prefill 预算？占全局目标预算的比例是多少？（答案：修复前再次除以 `attn_dp_size=4`，每 Rank 仅获得 $8192 / 4 = 2048$ tokens，4 个 Rank 合计仅 $8192$ tokens，仅为全局目标预算的 $1 / 4 = 25\%$）。
2. **可证伪假设**：在 PD 分离推理中，为何仅在 `_touch_waiting_prefixes` 中刷新 `waiting_queue` 不足以防止缓存抖动？因为处于 `bootstrap_queue` 中的请求正在等待与 Decode 节点完成 KV 传输握手，尚未转入 `waiting_queue`；在高并发长上下文流量下，若不优先刷新 `bootstrap_queue[:512]`，其已命中的共享前缀会在握手完成前被新到达请求从 Radix Cache 中驱逐，导致进入 `waiting_queue` 后被迫重算整个前缀。
