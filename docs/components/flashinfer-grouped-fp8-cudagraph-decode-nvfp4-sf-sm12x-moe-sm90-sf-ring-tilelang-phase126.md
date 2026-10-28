# Phase 126 Track A: FlashInfer Grouped FP8 128x64 KG1, CUDA-Graph Decode Merge Skip, NVFP4 Scale-Factor Tile Guard, SM12x Symbolic MoE, SM90 2-Slot Scale Ring & TileLang Cache Isolation

> **Phase 126 (`priority: 135`) · 证据等级**: 上游源码与提交说明 (`implementation` / `author-benchmark`) + 本项目离线合成标定 (`own-offline`)。未在物理 B200 / GB300 / R200 / RTX PRO 6000 / H100 上执行实机 kernel profiling (`own-hardware` 未声明)。

---

## 1. 核心架构问题与源码级解构

### 1.1 FlashInfer Cake Grouped FP8 GEMM `128 x 64` KG1 单波次路由 (`#7745`, `4e1b0a75`)
- **上游路径**: `csrc/cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_488f15389de98fd92896_kernel.cu`、`flashinfer/gemm/cake_grouped_fp8_gemm.py` (`4e1b0a75058abee792d2fa061408bba83abb308c`)。
- **架构机制**: `cake_grouped_fp8_gemm`（通过 `flashinfer.gemm.prepare_group_gemm_fp8_nt_groupwise_contiguous` 调用）接收 FP8 E4M3 输入并输出按 `m_indices` 分组的 BF16 张量，支持 FP32 per-128 block scales（现共 15 条路由）与 packed UE8M0 block scales（14 条路由）。当 `M > 32` 且 `128 x 64` 输出 tile 总数满足单波次约束：
  $$\left\lceil \frac{M}{128} \right\rceil \times \left\lceil \frac{N}{64} \right\rceil \le N_{\text{SM}}$$
  调度器将原本落入 `128 x 128` scale-group-1 (KG1) 路由的调用重路由至新增的 `128 x 64` vector-output KG1 程序，使 N 维切分翻倍以充分填满单波次 SM，在 `M=64..384, N=2048..4096` 离线标定下获得 **`1.309x` 平均加速**。

### 1.2 CUDA-Graph Decode `fixed_uniform_q_len` 跳过冗余 Merge 与 NVFP4 Scale-Factor 越界写零修复 (`#7020`, `c2f8204b`; `#6024`, `ac38c600`)
- **CUDA-Graph Decode 跳过冗余 Merge (`#7020`, `c2f8204b`)**:
  - **问题根因**: 在 CUDA Graph 模式下，`PrefillBinarySearchKVChunkSize` 只要检测到 `enable_cuda_graph=true` 就无条件置位 `split_kv=true`（防止后续 replay 需要切分 KV）。这导致 Tensor-Core Decode 在每个请求无需切分的大 batch 场景下，依然强制写出 FP32 临时部分和并启动 `MergeStates` kernel。
  - **修复机制**: Tensor-Core Decode 自行构造 `qo_indptr`，每次图重放具有固定的 `uniform_q_len` 与固定 query tile 数。新增 `fixed_uniform_q_len=true` 参数传递至 `BatchPrefillWithKVCachePlan`；当 query tiles 超过 `max_batch_size_if_split` 时直接清除 `split_kv`，直接写出最终 BF16/FP16 结果，消除冗余 HBM 读写与 merge kernel 启动（离线模型显示 **`1.312x` 平均加速**）。
- **NVFP4 KV Prefill Scale-Factor 越界写零修复 (`#6024`, `ac38c600`)**:
  - **问题根因**: 在 `include/flashinfer/attention/prefill.cuh` 的 `page_produce_kv_sf` 与 `produce_kv_sf` 中，`NUM_SF_ITERS` 向上取整。当 `head_dim = 64` 且 query tile 为 128 行（对应 64 行 KV tile）时，V scale tile 总大小 `SF_TOTAL_BYTES = 64 * (64 / 16) = 256 B`，而单 CTA 128 线程一次 32-bit 加载覆盖 `128 * 4 = 512 B`。越界线程虽 predicate 为 `false`，但 V loader 使用了 `kFillZero` 的 `cp.async`，硬件仍向共享内存写入 4 字节零，导致 `256 B` 的零直接覆写紧邻的 `q_smem` 前两行（每行 128 B），造成每个 CTA 的 query row 0/1 产生 **`6%–17%` 相对误差**。
  - **修复机制**: 在发起 `cp.async` 前显式插入编译期门控边界检查：
    ```cpp
    if (SF_TOTAL_BYTES % (THREADS_PER_CTA * 4) != 0 && flat_byte >= SF_TOTAL_BYTES) break;
    ```
    将越界 `q_smem` 覆写字节从 `256 B` 降为 **`0 B`**，相对误差恢复至 `0.0%`。

### 1.3 SM12x `b12x` 符号化静态 MoE 编译、SM120 Direct Micro 与 Frost 权重优先调度 (`#6211`, `c7ec1bca`; `#7502`, `55bf742f`)
- **SM12x `cute.sym_int()` 静态 Kernel 单次编译 (`#6211`, `c7ec1bca`)**:
  - **问题根因**: `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py` 中的 `_get_static_kernel` 原先将精确 token 数 `m` 硬编码进编译期张量形状与缓存键 `static_m{m}_...`。在线推理混合 prefill chunk 与 decode token 时每遇到新 `m` 就触发数秒 JIT 重编译，阻塞整个 batch。
  - **修复与优化**: 改用 `cute.sym_int()` 符号化 token 与 routed-pair 维度，使所有 token 数共享单次编译产物（24 种 batch size 下 JIT 编译总耗时从 `108.0 s` 降至 **`4.5 s`，削减 `95.83%`**）；同时在 RTX PRO 6000 Blackwell Server Edition (`SM120`, 188 SMs) 上将 NVFP4 direct micro kernel 阈值扩展至 `M <= 8` tokens，并引入 `_balanced_grid(tasks, max_ctas)` 均衡波次 CTA 数（离线模型 **`1.448x` 平均加速**）。
- **Frost 低延迟 MoE Route Mask 位枚举与 SM120 权重优先 TMA (`#7502`, `55bf742f`)**:
  - 在 `flashinfer/fused_moe/backends/frost_low_latency/tc/route_members.py` 中直接枚举已有的 warp ballot 置位比特，省去逐行二次 ballot 与 shuffle；并为 SM120 稠密 BF16/FP16 `T8/H2048/I512/top-k10` SwiGLU 新增 4 个 `bf16-grouped-tc-w8-weight-first-v16` 策略，在串行激活收集前先行发射权重 TMA（`1.192x` 加速）。

### 1.4 SM90 NVFP4 独立双槽 Scale Ring、Cake Norms/Softmax 与 TileLang 缓存隔离 (`#7542`, `061962ac`; `#7701`, `1a8a0ed0`; `#7702`, `ea2d6e96`; `#7671`, `25957924`; `tilelang#3696`, `0bbdcbfd`)
- **SM90 NVFP4 CuTe DSL MoE EP 双槽 Scale Ring (`#7542`, `061962ac`)**: `N64/K256` 配置下 NVFP4 原先仅有 3 个 AB pipeline stage（相比紧凑 MXFP4 的 4 个 stage）。通过引入独立的双槽缩放环（`nvfp4_sf_stages=2`、`compact_pull_buffer=True` 与 `_weight_sf_decoupled` 延迟释放策略），成功解锁第 4 个 AB stage。上游实测（`author-benchmark`）显示在 `T=512/rank` 下耗时从 `1,232.6 µs` 降至 **`1,189.6 µs` (`-3.68%`)**，`T=1024/rank` 下从 `1,607.2 µs` 降至 **`1,566.6 µs` (`-2.75%`)**。
- **Rubin `sm_107a` `cake_muse_spark_norm` (`#7701`, `1a8a0ed0`)**: 新增 8-CTA 持久化配置 `c8_t128_x8_pers_sc_eo_wc` 服务 `sm_107a` (NVIDIA R200) 上 `M = 1..128` 的 scalar-alpha + `post_weight` next-RMSNorm（总配置数达 78 个），消除此前 `backend="cake"` 在 `M <= 255` 抛错回退的问题（`1.284x` 加速）。
- **Blackwell `cake_gated_act_mxfp8` (`#7702`, `ea2d6e96`) & `cake_softmax` (`#7671`, `25957924`)**: 在 `sm_100a` / `sm_103a` 上为 SwiGLU + MXFP8 量化（每 32 个 E4M3 共享 1 个 RCEIL E8M0 scale，128x4 swizzled 布局）引入 8 CTAs/SM packed-sigmoid inline band-path 程序（`1.176x`），并微调 `sm_103` 10-CTA 与 `sm_100` persistent softmax 分发阈值。
- **TileLang `without_kernel_cache()` 状态恢复 (`tilelang#3696`, `0bbdcbfd`)**: 统一将 `test_tilelang_issue_2524.py` 等缓存敏感测试收敛至 `tilelang.testing.without_kernel_cache()` 上下文管理器，修复单方面调用 `disable_cache()` 导致同一 pytest worker 后续测试全部退化为无缓存重编译的状态泄漏（泄漏 worker 数 `4 -> 0`）。

---

## 2. 离线合成实验标定 (`own-offline`)

实验脚本：`experiments/flashinfer_grouped_fp8_cudagraph_decode_nvfp4_sf_sm12x_moe_sm90_sf_ring_tilelang_phase126/run.py`

| 子项 / 指标 | 基线 (Before) | Phase 126 架构优化 (After) | 关键收益 / 验证结果 |
|---|---|---|---|
| Cake Grouped FP8 GEMM (`M=64..384`) | `128 x 128` KG1 tile | **`128 x 64` KG1 单波次 vector-output** | **`1.309x`** 平均加速 (15 FP32 routes) |
| CUDA-Graph Tensor-Core Decode | 无条件 `split_kv=True` + FP32 Merge | **`fixed_uniform_q_len=True` 跳过 Merge** | **`1.312x`** 平均加速 |
| NVFP4 FA2 Prefill (`head_dim=64`) | `kFillZero` 覆写 `q_smem` `256 B` (`17%` 误差) | **`flat_byte >= SF_TOTAL_BYTES` 提前终止** | **`0 B`** 覆写 / **`0.0%`** 相对误差 |
| SM12x `b12x` 静态 MoE (24 种 `m`) | 按 `static_m{m}` 逐个重编译 (`108.0 s`) | **`cute.sym_int()` 单次符号化编译 (`4.5 s`)** | **`-95.83%`** JIT 编译耗时；Direct Micro **`1.448x`** |
| SM90 NVFP4 MoE EP (`T=512 / 1024`) | 3 AB stages (`1232.6 / 1607.2 µs`) | **`nvfp4_sf_stages=2` 解锁 4 AB stages** | **`1189.6 µs (-3.68%)`** / **`1566.6 µs (-2.75%)`** |
| TileLang Cache-Sensitive Workers | `4` 个 worker 泄漏 `CacheState.disable()` | **`without_kernel_cache()` 自动恢复** | **`0`** 泄漏 worker |

---

## 3. 架构师设计练习与可证伪问题

1. **单波次 Tile 纵横比选择计算**: 在 `N_{\text{SM}} = 148` 的 GPU 上运行 `M = 256, N = 3072` 的 Grouped FP8 GEMM，`128 x 128` tile 仅产生 $2 \times 24 = 48$ 个 CTA（SM 占用率 `32.4%`），而切换至 `128 x 64` KG1 产生 $2 \times 48 = 96 \le 148$ 个 CTA（单波次内 SM 占用率翻倍至 `64.9%`），解释了为何无需承担跨波次尾效应即可降低调用延迟。
2. **可证伪假设**: 若在 `head_dim = 128`（`SF_COLS = 8`）且 64 行 KV tile 下测试 NVFP4 FA2 prefill，`SF_TOTAL_BYTES = 64 * 8 = 512 B` 恰好整除 `THREADS_PER_CTA * 4 = 512 B`，此时无论是否启用 `ac38c600` 补丁，`q_smem` 覆写字节均应严格为 `0`。
