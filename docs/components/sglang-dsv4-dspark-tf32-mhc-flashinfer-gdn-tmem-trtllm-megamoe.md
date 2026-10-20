# SGLang DeepSeek-V4 DSpark TF32 mHC 阈值、FlashInfer Blackwell GDN TMEM 显式屏障与 TensorRT-LLM MegaMoE CuteDSL TEP 全局输出契约

> **English Version**: [sglang-dsv4-dspark-tf32-mhc-flashinfer-gdn-tmem-trtllm-megamoe.en.md](sglang-dsv4-dspark-tf32-mhc-flashinfer-gdn-tmem-trtllm-megamoe.en.md)  
> **证据等级**: `implementation`（上游一手 commit 源码） + `own-offline`（`experiments/sglang_dsv4_mhc_flashinfer_gdn_trtllm_megamoe/run.py --check`）

## 1. 一手源码定位与核心问题

1. **SGLang DeepSeek-V4 DSpark 投机草稿层 SM100 TF32 mHC 统计量阈值下探与权重加载后缓存刷新**（`sgl-project/sglang` commit [`c2e574246485a055c0e51d60d1eae027e5d629c0`](https://github.com/sgl-project/sglang/commit/c2e574246485a055c0e51d60d1eae027e5d629c0)，`2026-10-20`）：
   - **代码路径**：`python/sglang/srt/models/deepseek_v4_mhc.py`（`_mix_stats_impl`）与 `python/sglang/srt/models/deepseek_v4_dspark.py`（`load_weights` / `post_load_weights`）。
   - **问题机理**：在旧实现中，`_mix_stats_impl` 仅对 `hopper_medium = get_platform().is_sm90 and 32 <= x_flat.shape[0] < 4096` 开启 `[32, 128)` 行区间的融合算子，而在 Blackwell（`is_sm100`）上仍要求 `x_flat.shape[0] >= 128`。DeepSeek-V4 DSpark 投机解码草稿阶段（Draft Stages）的典型 batch 行数恰好落在 `32..96` 之间，导致 SM100 全部退化为 4 个独立 PyTorch 算子；此外 `deepseek_v4_dspark.py` 在 `load_weights` 后未调用 `stage.refresh_mhc_norm_weight_cache()`，导致多阶段草稿层持有陈旧的归一化权重缓存。
   - **修复机制**：将 `_mix_stats_impl` 门控统一简化为 `x_flat.shape[0] >= 32 and (is_sm100 or is_sm90)`，并在 `load_weights` 末尾新增 `self.post_load_weights()` 遍历所有 `self.stages` 执行 `stage.refresh_mhc_norm_weight_cache()`。

2. **FlashInfer Blackwell Gated Delta Net (`gdn`) `cg1` 共享累加器释放前的 TMEM 异步读取屏障**（`flashinfer-ai/flashinfer` commit [`80a44189a0972922a98317612647df4f3012be74`](https://github.com/flashinfer-ai/flashinfer/commit/80a44189a0972922a98317612647df4f3012be74)，`2026-10-20`）：
   - **代码路径**：`flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py` 与 `flashinfer/gdn_kernels/blackwell/gated_delta_net_cp_prefill.py` 中的 `compute_group_1_chunk`。
   - **问题机理**：Blackwell Tensor Memory（TMEM）上的 `tTR_rKS` 与 `tTR_rNv[None, sub, 0].load()` 是异步读取指令。消费者计算组（`compute_group_1_chunk`）在发射 TMEM load 后若直接调用 `ks_handle.release()` 或 `nv_handle.release()`，生产者 UMMA warp 会立即重用该共享累加器槽位并写入下一 chunk 的矩阵乘结果，造成 TMEM Read-before-Write（WAR / WAW）数据竞争。
   - **修复机制**：在 `ks_handle.release()` 与 `nv_handle.release()` 之前显式插入 `cute.arch.fence_view_async_tmem_load()`，确保当前消费者 warp 的所有异步 TMEM 读取对流水线释放操作满足順序可见性。

3. **TensorRT-LLM MegaMoE CuteDSL 在 TEP（Attention TP + MoE EP）下的全局路由输出契约**（`NVIDIA/TensorRT-LLM` commit [`46d9966b54d9d11998b712e732fb20f64d4d35c0`](https://github.com/NVIDIA/TensorRT-LLM/commit/46d9966b54d9d11998b712e732fb20f64d4d35c0)，`2026-10-20`）：
   - **代码路径**：`tensorrt_llm/_torch/moe/fused_moe/mega_moe/mega_moe_cute_dsl.py`、`configurable_moe.py`、`interface.py` 及 `modeling_kimi_linear.py`。
   - **问题机理**：`MegaMoECuteDsl`（`MoESchedulerKind.FUSED_COMM`）在内核内部已完成跨 EP/TP 域的规约并在每个 rank 返回全局合并后的路由专家输出（globally combined routed output）。在 TEP（`parallel_size > 1 and not use_dp`）拓扑下，若外层模型包装器（假设返回的是 rank partial）再执行一次 TP `AllReduce(SUM)`，路由专家输出会被错误放大 $TP$ 倍（如 $TP=4$ 时放大 $4\times$）。
   - **修复机制**：在 `MoE` 接口与 `ConfigurableMoE` 中引入 `accepts_global_routed_output: bool` 与只读属性 `returns_global_routed_output: bool`。支持显式编排的模型（如 `KimiLinear`、`MiniMaxM3`）设置 `accepts_global_routed_output=True` 并跳过外部 `_reduce_routed_output`；未显式声明的通用模型则由 `FusedCommMoEScheduler` 转换为 rank partial，从而对所有模型安全开放 MegaMoE CuteDSL TEP。

## 2. 确定性实验验证（`experiments/sglang_dsv4_mhc_flashinfer_gdn_trtllm_megamoe`）

运行 `python experiments\sglang_dsv4_mhc_flashinfer_gdn_trtllm_megamoe\run.py --check` 的量化结果：

| 子方向 | 关键指标 | 修复前（Legacy） | 修复后（Patched） | 架构收益 |
|---|---|---:|---:|---|
| SGLang DSpark SM100 TF32 mHC (`c2e57424`) | 投机 batch (`16..256`) 命中融合核比例 | `2 / 7` (`>=128`) | `6 / 7` (`>=32`) | 累计延迟从 `181.21 us` 降至 `91.18 us`（**`1.9874x` 加速**），陈旧缓存 stage `4 -> 0` |
| FlashInfer GDN TMEM 屏障 (`80a44189`) | 8 Chunks $\times$ 2 Handles 异步 TMEM 覆写冲突 | `16 / 16` | `0 / 16` | `fence_view_async_tmem_load()` 消除全部 `ks_handle`/`nv_handle` 提前释放危害 |
| TRT-LLM MegaMoE TEP (`46d9966b`) | $TP=4, EP=4$ 路由+共享专家组合输出误差 | `+3.75`（路由项 $4\times$ 膨胀） | `0.0` | `accepts_global_routed_output` 契约消除双重 AllReduce |

## 3. 架构师设计练习与可证伪问题

- **计算练习**：当 $TP=4$、真实路由专家输出为 $y_{\text{routed}}=1.25$、共享专家在 4 个 TP rank 上的分片输出之和为 $\sum_{r=0}^{3} y_{\text{shared}}^{(r)} = 1.00$ 时，证明未开启 `returns_global_routed_output` 契约的外部 `AllReduce(SUM)` 会得到 $4 \times 1.25 + 1.00 = 6.00$（相对真实值 $2.25$ 产生 $+166.7\%$ 幅度畸变）。
- **可证伪问题**：在 Blackwell SM100 的 Gated Delta Net chunked prefill 中，若仅保留寄存器级乘法依赖而不插入 `cute.arch.fence_view_async_tmem_load()`，为何硬件仍允许 `tcgen05.ld` 与后续 `mbarrier.arrive`（`handle.release()`）乱序完成？
