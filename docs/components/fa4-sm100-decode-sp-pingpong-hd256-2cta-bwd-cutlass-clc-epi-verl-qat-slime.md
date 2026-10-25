# FlashAttention-4 SM100 Decode S/P TMEM 乒乓流水、hd256 2-CTA 反向统一、CUTLASS CLC Epilogue 局部计数与 veRL ModelOpt QAT 子量化器屏蔽深度解析

> **文档状态**：Phase 80 (`priority: 89`) 第一手源码级深度审计  
> **双语对照**：[`fa4-sm100-decode-sp-pingpong-hd256-2cta-bwd-cutlass-clc-epi-verl-qat-slime.en.md`](fa4-sm100-decode-sp-pingpong-hd256-2cta-bwd-cutlass-clc-epi-verl-qat-slime.en.md)  
> **可复现实验**：[`experiments/fa4_sm100_sp_pingpong_hd256_2cta_bwd_cutlass_clc_epi_verl_qat_slime/run.py`](../../experiments/fa4_sm100_sp_pingpong_hd256_2cta_bwd_cutlass_clc_epi_verl_qat_slime/run.py)

---

## 1. 上游源码证据矩阵

| 证据编号 | 上游提交 | 核心变更模块 | 证据类别 |
| --- | --- | --- | --- |
| `src-fa4-sm100-decode-sp-pingpong-splitkv-2026` | `Dao-AILab/flash-attention@e9515d5dee6a` (`#2928`, `#2920`) | SM100 Decode (`q_stage=1`) S/P 双 TMEM 槽乒乓流水（`QK(i+1)` 超前于 `PV(i)`）+ `num_splits_heuristic >= 1` 钳制 | `implementation` |
| `src-fa4-sm100-hd256-sm103-pingpong-2cta-2026` | `Dao-AILab/flash-attention@ab3e5bd987de` (`#2951`) | SM103 (`GB300`) 启用 S 乒乓；`hd256` 2-CTA 覆盖 varlen/decode、禁用 CLC 并在复用 caller metadata 时保持 1-CTA | `implementation` |
| `src-fa4-sm100-hd256-bwd-cta-schedule-tmem-2026` | `Dao-AILab/flash-attention@76d933102819` (`#2964`) | `CtaSchedule` 枚举（`ONE_CTA`/`TWO_CTA_PIPELINED`/`TWO_CTA_SERIAL`）与命名 TMEM 列宽映射 | `implementation` |
| `src-fa4-paged-kv-cudagraph-capture-fix-2026` | `Dao-AILab/flash-attention@73019affed8c` (`#2984`) | `mha_fwd_kvcache` 在 CUDA Graph capture 期间跳过触发阻塞拷贝的 `.max().item()` 越界检查 | `implementation` |
| `src-cutlass-cutedsl-clc-epilogue-tile-counter-2026` | `NVIDIA/cutlass@84a62ec421c2` (`#3636`) | Blackwell 动态持久化 GEMM：epilogue warp 维护局部 `num_tiles_executed`，修复短 K 下 TMA store SMEM 阶段覆盖 | `implementation` |
| `src-cutlass-cutedsl-grouped-gemm-2cta-codegen-2026` | `NVIDIA/cutlass@af3d4367a80c` (`#3639`) | CUDA 12.9 分组 GEMM 按 CTA 模式选择代码生成：2-CTA 用向量化 RMEM + `-O2`，1-CTA 用标量 RMEM + `-O3` | `implementation` |
| `src-verl-modelopt-qat-module-ignore-children-2026` | `volcengine/verl@5d09d065b2a6` (`#8329`) | ModelOpt QAT `ignore_patterns` 追加 `f"{key}.*"` 规则，禁用被忽略模块的 `.input_quantizer` 与 `.weight_quantizer` | `implementation` + `author-benchmark` |
| `src-verl-device-agnostic-linear-ce-tests-2026` | `volcengine/verl@d430157d1551` (`#8284`) | 线性交叉熵内核与测试全面迁移至 `verl.utils.device` 抽象 | `implementation` |
| `src-slime-prefilter-raw-reward-logging-2026` | `THUDM/slime@57ff5d669be6` (`#2475`) | 在动态 rollout 过滤之前计算并上报 `raw_reward`，消除过滤子集对训练奖励监控的偏差 | `implementation` |

---

## 2. FlashAttention-4 SM100 Decode S/P TMEM 乒乓流水与 `num_splits >= 1` 钳制

在 `Dao-AILab/flash-attention@e9515d5dee6a` (`#2928`, `#2920`) 与 `ab3e5bd987de` (`#2951`) 中，针对 Blackwell SM100/SM103 (`B200`/`GB300`) 的 `q_stage = 1`（Decode）注意力主循环进行了深度重构：

1. **为什么 `q_stage = 1` 原本存在 TMEM 闲置与串行气泡？**
   - 在 `q_stage = 1` 时，TMEM 中实际上已经分配了两个 S/P 槽位，但旧版只使用了槽位 `0`（`s_p_o_stage = 1`）。MMA warp 必须等待 Softmax warp 排空 $S_i$、写入 $P_i$ 且 Correction warp 完成 $O_i$ 缩放后，才能发射下一轮 $Q K_{i+1}^T$。
   - 启用 `use_s_ping_pong = True`（当每 split 的 KV 块数满足 `S_PING_PONG_MIN_N_BLOCKS_PER_SPLIT = {64: 16, 128: 64}` 时自动开启）后，`s_p_o_stage = 2`：加载顺序重排为 $K_0, K_1, V_0, K_2, V_1, \dots$，MMA warp 在发射 $P_i V_i$ **之前**就向另一个 TMEM 槽位发射 $Q K_{i+1}^T$，使 $Q K_{i+1}^T$ 与第 $i$ 步的 Softmax 完整重叠。在 8 个 KV tile 的基准流水模型下，主循环周期从 `224` 单位降至 `168` 单位（**1.333x** 理论主循环加速）。
2. **`num_splits_heuristic` 除零修复 (`#2920`)**：当 `total_mblocks > num_SMs`（如 `256 > 148`）时，`num_SMs // total_mblocks == 0`，导致 S 乒乓选择器除以 `num_splits = 0` 崩溃。修复为 `max(1, min(num_SMs // total_mblocks, max_splits, num_n_blocks))`。
3. **SM103 (`GB300`) 适配与 `hd256` 2-CTA 安全护栏 (`#2951`, `#2964`, `#2984`)**：
   - 将架构判断从 `arch in (100, 110)` 改为 `arch // 10 in (10, 11)`，使 GB300 (`sm_103a`) 同样获得 S/P 乒乓加速。
   - 当调用方传入预构建 `scheduler_metadata`（按 1-CTA tile 构建）时强制保持 1-CTA，并为 `hd256` 2-CTA 禁用不支持双 CTA 簇映射的 CLC 调度器。
   - 在 `flash_bwd_sm100.py` (`#2964`) 中用 `CtaSchedule` 枚举（`ONE_CTA`、`TWO_CTA_PIPELINED`、`TWO_CTA_SERIAL`）统一替代散落的 20 处字面量分支，并修正 `tmem_dV_offset = tmem_S_offset + tile_m`。
   - 在 `flash_api.cpp` (`#2984`) 中使用 `!at::cuda::currentStreamCaptureStatusMayInitCtx()` 跳过 CUDA Graph 捕获期间的 `.max().item()` 同步检查。

---

## 3. CUTLASS CuTeDSL：CLC Epilogue 局部计数器修复短 K 竞态 (`#3636`)

在 `NVIDIA/cutlass@84a62ec421c2` (`#3636`) 中，作者定位了 Blackwell 动态持久化 GEMM (`dense_gemm_persistent_dynamic.py`) 在短 $K$ 形状（如 `8256x8448x128`）下的静默数据损坏根因：
- 只有 scheduler warp 会调用 CLC (Cluster Launch Control) 推进 `tile_sched.num_tiles_executed`，而 epilogue warp 自己局部的 `tile_sched.num_tiles_executed` 永远停留在 `0`！
- 结果 epilogue warp 在每个 tile 都使用 `0 % epi_stages = 0` 号 SMEM 缓冲区发起 TMA store，在短 $K$ 下前一个 tile 的 TMA store 尚未完成即被下一个 tile 覆盖。通过在 epilogue 内部维护局部递增的 `num_tiles_executed`（轮转 `0, 1, 2, 3` 阶段），彻底消除了该竞态。

---

## 4. veRL ModelOpt QAT 子量化器屏蔽 (`#8329`) 与 Slime 过滤前奖励统计 (`#2475`)

- **veRL ModelOpt QAT (`#8329`)**：在 `_ignore_patterns_to_quant_cfg` 中，原本仅生成 `{"quantizer_name": key, "enable": False}`。然而当用户配置模块级通配符（如 `*self_attention.linear_proj` 或 `*linear_attn.in_proj_a`）时，ModelOpt 的 `fnmatch` 不会匹配其子量化器 `...linear_proj.input_quantizer` 和 `.weight_quantizer`！追加 `f"{key}.*"` 规则后，在单卡 H100 Qwen3-0.6B 10 步 GRPO 实测中，28 层被忽略输出投影的残留启用量化器从 **56/56 降至 0/56**，首层最大 QDQ 伪量化误差从 **0.0859375 降至 0.0**，同时不会误伤同级模块 `in_proj_ab`。
- **Slime 动态过滤前原始奖励上报 (`#2475`)**：在 `slime/ray/rollout.py` 中，将 `raw_reward` 统计提前至动态 rollout 过滤（剔除全对/全错组）之前，保证 WandB/TensorBoard 记录的 `rollout/raw_reward` 反映真实的策略未过滤表现。
