# FlashInfer Cake RMSNorm 协作持久反向 Ring (`SM100`/`SM103`/`SM107`)、SGLang `gfx1250` W4A8 MXFP4 MoE 哨兵槽 EP 路由与 vLLM Model Runner V2 (`MRV2`) 异步投机调度契约

> **English Version**: [flashinfer-cake-rmsnorm-grad-sglang-gfx1250-mxfp4-ep-vllm-mrv2-spec-async.en.md](flashinfer-cake-rmsnorm-grad-sglang-gfx1250-mxfp4-ep-vllm-mrv2-spec-async.en.md)  
> **证据等级**：`implementation`（上游固定 Commit 源码核验）+ `own-offline`（`experiments/flashinfer_cake_rmsnorm_sglang_ep_vllm_mrv2/run.py --check`）

## 1. 一手源码定位与核心架构机制

1. **FlashInfer Cake RMSNorm `H1536`/`H4096` 协作持久反向 Ring（`flashinfer-ai/flashinfer` commit `0d062af33b6b42786bc5442acb802dd265242355`, `flashinfer/cake_rmsnorm_train_backend/cake_jit.py` & `build.json`）**：
   - 针对 `sm_100a`（B200, 148 SMs）、`sm_103a`（GB300, 152 SMs）和 `sm_107a`（R200, 212 SMs）重构 `cake_rmsnorm_train` 非残差融合反向（`dx`, `dweight`）：
     - 在 `sm_100a`/`sm_103a` 上，当 `H=1536, T > 1024` 或 `H=4096, T > 256` 时启用协作持久算子（`cooperative=True`, `ctas_per_sm=1`, `grid_factor=2`）；
     - 在 `sm_107a`（R200）上，阈值设为 `H=1536, T > 2048` 及 `H=4096, T > 1024`（`kernel_cake_rmsnorm_train_22a97cd819b6683e10fe`）；
   - 通过 `cp.async.bulk` + `mbarrier` 共享内存 Ring 流水化暂存 `dy` 与 `x` 行，同时通过 `partial` 缓冲区与 `counters` 保持确定性固定顺序 FP32 `dweight` 归约。作者实测（`author-benchmark`）在 B200 `H=1536, T=65536` 取得 **`1.1245x`** 反向加速（`113.53 -> 100.96 us`），在 R200 `H=4096, T=65536` 取得 **`1.2161x`** 反向加速（`172.90 -> 142.18 us`）。
2. **SGLang AMD `gfx1250` GPT-OSS W4A8 MXFP4 MoE 专家并行（`sgl-project/sglang` commit `89860fbd2e3cf460c0968435670b7fdef6ab5529`, `python/sglang/srt/layers/moe/fused_moe_triton/aiter_mxfp4_w4a8_moe.py`）**：
   - `build_ep_expert_map(num_experts, num_local_experts, ep_rank)` 构造长度为 `num_experts + 1` 的映射表（非本地专家填 `-1`，`[start : start + num_local_experts]` 填 `0..num_local_experts-1`）；
   - `_ep_routing` 基于全局专家总数 `n_expts_tot` 计算 `block_m = max(16, min(next_power_of_2(max(1, n_gates // n_expts_tot)), 128))`，将非本地专家门控路由至哨兵槽（sentinel bin）；
   - 由于 `gfx1250` 算子内 gather 存在硬件缺陷，由主机侧显式执行 `gather_src = (gather_idx // topk).clamp_(0, hidden_states.shape[0] - 1)` 阻断未写入哨兵槽触发越界读取，并将 `gate_valid` 传入 GEMM2 `moe_gemm_a8w4` 屏蔽远端专家贡献，最后由层间 AllReduce 求和；当 `moe_ep_size > 1` 且 `num_fused_shared_experts > 0` 时显式抛出 `NotImplementedError`。
3. **vLLM Model Runner V2 (`MRV2`) 投机解码异步调度解耦（`vllm-project/vllm` commit `f1dbfef79973d0f8d05be56e8614db0409e3d7d3`, `vllm/config/vllm.py`）**：
   - 将 V1 专属的异步调度限制（非 `EAGLE`/`MTP`/`NGram`/`draft_model`/`dspark`/`dflash` 投机方法、`disable_padded_drafter_batch=True`、`pipeline_parallel_size > 1`）收敛至 `if not self.use_v2_model_runner:` 分支 (`v1_async_sched_incompat`)，避免在启用 `MRV2` 时误触发 V1 降级回退；同时保留 ROCm `deepep_high_throughput` DBO 的全局互斥拦截。

## 2. 确定性离线实验核验（`own-offline`）

运行 `python experiments\flashinfer_cake_rmsnorm_sglang_ep_vllm_mrv2\run.py --check`：
- **Cake RMSNorm 协作反向路由与确定性归约**：覆盖 `11` 组架构/形状案例（`7/11` 命中协作 `cp.async.bulk` Ring），B200 `1.1245x`、R200 `1.2161x`，固定顺序 FP32 `dweight` 跨回放漂移为 **`0.0`**；
- **SGLang `gfx1250` W4A8 EP 哨兵钳制**：16 专家 `EP=4` 下，`expert_map` 长度为 `17`，`gather_src.clamp_` 将越界 gather 从 `72` 降至 **`0`**，跨 Rank AllReduce 相对全局基准最大绝对误差为 **`0.0`**；
- **vLLM `MRV2` 异步调度矩阵**：成功保留全部 **`3/3`** 组 `MRV2` 异步投机场景，同时准确拦截 `3/3` 组 V1 不兼容项与 `1/1` 组 ROCm DeepEP DBO 冲突。

## 3. 架构师设计练习与可证伪问题

- **练习**：计算 `T=65536, H=4096` BF16 RMSNorm 反向（读 `dy, x, w, rstd`、写 `dx, dw`）在 R200（`142.18 us`）下的有效 HBM 带宽。
  - **解答**：主数据面 `2(dy) + 2(x) + 2(dx) = 6 B/elem`，总字节数 $= 65536 \times 4096 \times 6 = 1.6106\text{ GB}$，有效带宽 $\approx 1.6106\text{ GB} / 142.18\text{ }\mu\text{s} \approx 11.33\text{ TB/s}$。
- **可证伪问题**：若在 `gfx1250` W4A8 MoE EP 中按本地专家数 `num_local_experts` 而非全局专家数 `n_expts_tot` 计算 `block_m`，会对 Triton Tile 填充率产生多大放大？
