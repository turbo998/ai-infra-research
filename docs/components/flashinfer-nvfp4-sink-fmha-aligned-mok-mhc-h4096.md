# FlashInfer NVFP4 注意力接收器（Attention Sinks）、Blackwell 对齐分页 GQA 解码、MoK TMEM 交接、Mega-mHC Hidden-4096、非协作式因果一维卷积反向与 TileLang 64 位有符号原子操作深度解析

- **研究日期**：2026-10-24（Phase 59 · Priority 68 · Track A）
- **上游证据链**：
  - `flashinfer-ai/flashinfer@5be1442520` (`#6021`, *feat(experimental): support attention sinks with NVFP4 KV cache in prefill and decode*)
  - `flashinfer-ai/flashinfer@8656005c6a` (`#7060`, *perf(cake_fmha): add aligned hd128 single-query paged decode kernels*)
  - `flashinfer-ai/flashinfer@4a35122d9e` (`#7059`, *perf(cake_mok): optimize fused kernel issue paths and TMEM hand-off*)
  - `flashinfer-ai/flashinfer@4b88078d61` (`#7048`, *perf(cake_deepgemm_dense_mqa): refresh 32h kernels and extend 64h KV-split tier*)
  - `flashinfer-ai/flashinfer@afb249ac04` (`#6833`, *perf(gemm): tighten cute-dsl bf16 low-M dispatch*)
  - `flashinfer-ai/flashinfer@9ecd6e31b9` (`#7024`, *feat(cake_deepgemm_mega_mhc): add hidden-4096 bf16 inference kernel*)
  - `flashinfer-ai/flashinfer@608772bd85` (`#7072`, *perf(cake_causal_conv1d): replace cooperative bwd with two-launch reduction*)
  - `tile-ai/tilelang@2573d47079` (`#3307`, *Fix CUDA int64 atomic max/min and atomic load/store support*)
- **证据等级**：`implementation` + `author-benchmark`（源码级审阅与作者实测表核对）+ `own-offline`（`experiments/flashinfer-nvfp4-sink-fmha-aligned-mok-mhc-h4096/run.py` 确定性验证）

---

## 1. 架构背景与核心问题

在 Hopper/Blackwell/Rubin 推理与训练内核栈中，Phase 59 Track A 聚焦六个高价值算子瓶颈与精度护栏：

1. **NVFP4 KV Cache 下的 Attention Sinks 缺失与 JIT 模块名碰撞（`#6021`）**：长上下文窗口注意力（StreamingLLM / Attention Sinks）此前缺乏对打包 NVFP4（`uint8` 存储 E2M1 nibbles + `fp8_e4m3fn` 块缩放因子）KV Cache 的支持；更严重的是，`gen_batch_attention_sink_module` 固定使用 `"batch_attention_sink"` 作为 JIT URI，在同一进程内先后调用 `head_dim=64` 与 `head_dim=128` 时会命中同一个已缓存的编译模块，引发 `0.6–1.2` 的灾难性相对误差。
2. **Blackwell (`sm_100a`/`sm_103a`) `head_dim=128` 单 Query 分页 GQA 解码延迟过高（`#7060`）**：对于 `q_len_per_req == 1`、`head_dim_qk == head_dim_vo == 128` 且 `page_size % 16 == 0` 的对齐分页解码，通用分页注意力路径缺乏按 `(request, KV head)` 动态选择 `8/4/2/1` CTA 簇大小并分摊 KV 页循环的专用程序。
3. **MoK 融合专家内核发射路径与 TMEM 交接开销（`#7059`）以及 Dense MQA 64-Head KV-Split 阈值受限（`#7048`）**：`cake_mok` 的 16 个融合 BF16/MXFP8 路由专家变体在 `sm_100a`/`sm_103a`/`sm_107a` 上存在指令发射与 Tensor Memory (TMEM) 交接停顿；`cake_deepgemm_dense_mqa` 的 64-head whole-grid KV-split 层此前仅覆盖 `Q > 1024`，使 `1024 <= Q <= 1536` 的长 KV 形状落入次优的 `le1024` 路径。
4. **CuTe-DSL BF16 低 M GEMM 过度路由与 Split-K 候选爆炸（`#6833`）及 GLM-5.3-Flash Hidden-4096 mHC 缺失（`#7024`）**：`mm_bf16(backend="cute-dsl")` 在 `M <= 4` 时将过多中大面积形状路由至 SIMT direct runner 且默认每线程仅预取 1 个 B 向量；同时 GLM-5.3-Flash 的 `hidden=4096, n=4` BF16 多头残差连接（mHC）在 `sm_107a` (R200) 上仍依赖 4 次独立内核启动。
5. **因果一维卷积反向 `cudaLaunchCooperativeKernel` 的跨流饥饿死锁（`#7072`）与 TileLang 有符号 `int64` 原子极值无符号强转 Bug（`#3307`）**：`cake_causal_conv1d_train_backward` 采用全设备协作启动，当并发 CUDA 流上运行哪怕 1 个常驻线程（如 NCCL 通信或旋转侧内核）时，协作网格会被阻塞 `20x–108x`；而 TileLang 的 `AtomicMax`/`AtomicMin` 将 `int64_t` 强转为无符号 `unsigned long long` (`NT1`) 后再传给 `cuda::atomic_ref`，导致负数被视为大于正数。

---

## 2. 核心实现机制与源码走读

### 2.1 NVFP4 KV Cache Attention Sinks 与 JIT URI 消歧（`#6021`）

`flashinfer/experimental/attention_sink/` 新增 `BatchDecodeWithAttentionSinkWrapper` 并扩展 `BatchPrefillWithAttentionSinkWrapper`：

- **JIT 模块 URI 消歧**：在 `flashinfer/experimental/attention_sink/jit.py` 中，将 `dtype_kv`、`head_dim_qk`、`pos_encoding_mode` 与 `use_fp16_qk_reduction` 全部编码进模块 URI（如 `batch_attention_sink_kv_uint8_nvfp4_hd_128_pos_0_fp16qk_0`），彻底根除同进程跨 `head_dim=64/128` 或跨 KV 精度调用时的缓存污染（修复前第 2 个 `head_dim` 的相对误差高达 `0.6–1.2`，修复后 NVFP4 decode/prefill 相对误差降至 `~1.8e-3`）。
- **FA2 `uint8` 缩放因子签名与 `sm_scale` 折叠**：在 `csrc/attention_sink_prefill_jit_binding.cu` 中为 FA2 后端补齐 `maybe_k_cache_sf` 与 `maybe_v_cache_sf` 参数（当 `DTypeKV = uint8_t` 时启用），并在 wrapper `plan()` 阶段将全局标量 `q_scale` 与 `k_scale` 直接折叠进 `sm_scale = (1 / sqrt(d)) * q_scale * k_scale`，省去内核内逐元素乘法；同时对不支持打包 FP4 的 FA3 (`sm_90a`) 显式抛出 `NotImplementedError`。

### 2.2 Cake FMHA 对齐 `hd128` 单 Query 分页 GQA 解码（`#7060`）

`csrc/cake_fmha/fmha_paged_decode_launch.cu` 在 `sm_100a` 与 `sm_103a` 上新增 8 个对齐 `hd128` `q1` 分页解码程序（BF16 与 FP16 各覆盖 cluster size `8 / 4 / 2 / 1`）：

- **自适应 CTA 簇选择**：当 `q_len_per_req == 1`、`head_dim == 128`、`1 <= group_size <= 16`、`page_size >= 16` 且 `page_size % 16 == 0` 时，host 发射器根据总 `(batch * kv_heads)` 与序列页数动态挑选每个 `(request, KV head)` 分配 `8`、`4`、`2` 或 `1` 个 CTA 协同扫描 KV 页表，并编译期强制 `static_assert(sizeof(k_tmap) == 128 && alignof(decltype(k_tmap)) == 64)`。
- **B200 (`sm_100a`) 与 GB300 (`sm_103a`) 实测加速（`H_q=64, d=128, page_size=16`）**：

| GPU | Batch | KV Heads | Seq Len | Dtype | Prev (us) | New (us) | Speedup |
|---|---:|---:|---:|---|---:|---:|---:|
| **B200** | 1 | 8 | 512 | BF16 / FP16 | 24.97 / 19.49 | 4.87 / 5.13 | **5.13x / 3.80x** |
| **B200** | 1 | 8 | 1024 | BF16 | 27.99 | 5.40 | **5.18x** |
| **B200** | 4 | 8 | 4096 | BF16 | 49.71 | 17.73 | **2.80x** |
| **B200** | 32 | 1 | 4096 | BF16 | 52.87 | 17.82 | **2.97x** |
| **GB300** | 1 | 8 | 512 | BF16 / FP16 | 46.56 / 35.97 | 4.64 / 4.89 | **10.03x / 7.35x** |
| **GB300** | 1 | 8 | 1024 | BF16 | 49.12 | 5.14 | **9.55x** |
| **GB300** | 4 | 8 | 4096 | BF16 | 72.03 | 16.96 | **4.25x** |
| **GB300** | 2 | 8 | 1153 | BF16 | 51.65 | 6.53 | **7.91x** |

### 2.3 Cake MoK 发射路径与 TMEM 交接（`#7059`）及 Dense MQA 64h KV-Split 扩展（`#7048`）

1. **Cake MoK 16 个融合专家变体刷新（`#7059`）**：
   - 在 `sm_100a`、`sm_103a`、`sm_107a`（每架构 48 个库：16 个融合路由专家 + 32 个调度/尾声内核，共 144 个编译单元）中优化指令发射与 TMEM 交接：在 B200 EP8 `glm52_ep8_bf16_full_ragged_defaults` 上将训练步耗时从 **`20.447 ms` 降至 `19.869 ms`（`1.029x`，`-2.83%`）**，在 GB300 EP4 ragged 上从 **`17.459 ms` 降至 `17.073 ms`（`1.023x`）**。
2. **Cake DeepGEMM Dense MQA（`#7048`）**：
   - 刷新 32-head FP8 K-major 程序（新增 7 个、移除 4 个旧程序），并将 64-head in-kernel whole-grid KV-split 层从 `Q > 1024` 扩展至 `Q <= 1536`（当 `K > 2048` 时），退役两条 `le1024` 旧路由。在 GB300 上 86 个形状几何平均提速 **`1.0053x`**，其中 `dense_q1024_k33803_p32779_h64` 从 **`221.64 us` 降至 `205.51 us`（`1.0785x`）**，B200 峰值提速 **`1.0577x`**（`259.55 -> 245.37 us`）。

### 2.4 CuTe-DSL BF16 低 M GEMM 重门控（`#6833`）与 Mega-mHC Hidden-4096（`#7024`）

1. **CuTe-DSL BF16 低 M GEMM 重门控与 Split-K 裁剪（`#6833`）**：
   - 将 SIMT direct runner 的 autotune 候选条件从 `M <= 4 or M*N <= 8192` 收紧为 `M <= 3 or M*N <= 4096`；默认启发式 `prefer_direct_bf16_gemm_sm100` 仅在 `1 <= M <= 32, K % 256 == 0` 且 `(M*N <= 2048 or (M == 1 and N <= 8192 and 5120 <= K <= 16384))` 时启用。
   - 在 `M <= 4` 时选择使每线程预取 `>= 2` 个 16B B 向量的最大 `block_size`，并在 `N*K < 2^21` 时固定 `outputs_per_block = 1`；同时要求 warp split-K 满足 `>= 96 KiB` 在途字节、cluster split-K 满足 `>= 9 * 64 * 128 * 2 B`（`144 KiB`）权重读取。B300 上 `(2, 1000, 8192)` 从 **`8.35 us` 降至 `5.70 us`（`1.465x`）**，`(1, 128, 4096)` 从 **`3.75 us` 降至 `2.34 us`（`1.603x`）**。
2. **Cake Mega-mHC Hidden-4096 BF16 融合推理内核（`#7024`）**：
   - 针对 GLM-5.3-Flash (`hidden=4096, n=4` 残差流, BF16) 在 `sm_107a` (R200, 208 SMs) 新增第 15 个 `cake_deepgemm_mega_mhc` 程序，将 `rmsnorm + dynamic/static routes + sinkhorn_knopp + sinkhorn_apply` 4 个内核融合为单次启动：在 `T=1` 下从 **`17.28 us` 降至 `9.25 us`（`1.869x`）**，在 `T=1024` 下从 **`105.23 us` 降至 `18.91 us`（`5.564x`）**，在 `T=8192` 下从 **`970.10 us` 降至 `112.83 us`（`8.598x`）**。

### 2.5 因果一维卷积双阶段非协作反向（`#7072`）与 TileLang 有符号 `int64` 原子操作（`#3307`）

1. **因果一维卷积非协作反向 + PDL 固定顺序归约（`#7072`）**：
   - 移除 `cudaLaunchCooperativeKernel` 与 `grid.sync()`，改为两次普通启动：① 持久化主内核以 `grid = min(n_tiles, 2 * ctas_per_sm * sm_count)` 写出 `dx` 并将每个 CTA 的 `dweight` 局部和写入连续 FP32 缓冲区 `[rows, C, 4]`（其中 `rows = ceil(grid / n_ch_tiles)`），尾声调用 `cudaTriggerProgrammaticLaunchCompletion()`；② PDL 归约内核调用 `cudaGridDependencySynchronize()` 后按固定 row 0..rows-1 顺序累加 `dweight` 与 `dbias`（跨调用逐比特确定性一致，`max_abs = 0`）。
   - **并发侧流存在时的抗饥饿实测（`S=250,340, C=512`）**：
     - **B200**：单线程侧流下 **`4.740 ms -> 0.255 ms`（`18.6x`）**，8-CTA 旋转侧流下 **`20.009 ms -> 0.306 ms`（`65.4x`）**；
     - **GB300**：8-CTA 旋转侧流下 **`18.983 ms -> 0.277 ms`（`68.5x`）**；
     - **R200**：8-CTA 旋转侧流下 **`16.217 ms -> 0.149 ms`（`108.8x`）**。
2. **TileLang 有符号 `int64` `cuda::atomic_ref<T1>` 修复（`#3307`）**：
   - 在 `src/tl_templates/cuda/atomic.h` 中，对于 `sizeof(T1) == 8` 的 `AtomicMax`/`AtomicMin`，直接使用原始有符号类型 `cuda::atomic_ref<T1, cuda::thread_scope_device>`（而非先强转为 `unsigned long long` `NT1`），并在 `AtomicStore` 中保留目标引用类型 `T1`，确保正负跨 `2^40` 量级的 `int64` 原子极值与 `bf16`/`fp16` 原子存取完全正确。

---

## 3. 架构师练习与可证伪问题

1. **定量计算练习**：在 `B200` (`148` SMs, `ctas_per_sm = 2`) 上运行 `cake_causal_conv1d_train_backward`（`C = 2048, width = 4`），计算 `grid`、`n_ch_tiles = C / 64`、工作区行数 `rows = ceil(grid / n_ch_tiles)` 以及 FP32 局部和缓冲区字节数（答案：`grid = 592, n_ch_tiles = 32, rows = 19, bytes = 19 * 2048 * 4 * 4 = 622,592 B`）。
2. **可证伪假设**：若在 `cuda::atomic_ref` 之前将有符号 `int64_t` 强转为 `uint64_t` (`unsigned long long`)，任何负数 `x < 0` 的二补数位模式都会映射至 `[2^63, 2^64 - 1]`，从而在 `fetch_max` 中永远击败正数 `y >= 0`。
