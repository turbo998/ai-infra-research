# FlashInfer CAKE Softmax 异步代理屏障、MM FP4 计划回退、VC-Attention-QK8 上下文注意力、QSA NVFP4 线程安全计划表、MiniMax H3 SM103 共享内存残差暂存、Autotuner Delay 流绑定、有界张量 Top-K 快速路径、Radix Top-K 静态共享内存预算、CUTLASS 块缩放尾声批索引与 TileLang 无缓存编译测试深度解析（Phase 121）

> **模块定位**：面向具有云基础设施背景、希望深入掌握 Blackwell (`sm_100a`) / Blackwell Ultra (`sm_103a`) / GeForce Blackwell (`sm_120`) 算子同步契约、量化注意力与采样微架构的解决方案架构师（Solution Architect）。本文档基于 `flashinfer-ai/flashinfer`、`NVIDIA/cutlass` 与 `tile-ai/tilelang` 在 2026-07-18 至 2026-10-28 合入的 10 个核心提交，系统拆解 TMA 异步代理（Async Proxy）内存序、FP4 GEMM 计划回退、Hadamard 旋转 E4M3 量化注意力与 Top-K 采样快速路径。
> **证据等级**：`implementation`（源码固定 commit 级核验）+ `author-benchmark`（上游 PR 实测数据，非本仓库 GPU 复测）+ `own-offline`（本仓库 Python 标准库可复现建模验证）。

---

## 1. 核心架构问题与 10 项上游证据总览

在 Blackwell (`sm_100a`/`sm_103a`/`sm_120`) 架构中，高吞吐算子依赖以下微架构机制的精确配合：
1. **通用代理（Generic Proxy，如 `st.shared`）与异步代理（Async Proxy，如 TMA `cp.async.bulk`）之间的显式代理屏障（Proxy Fence）**；
2. **Per-Token / 动态形状下 AOT 编译表未命中或 TMA Box 越界时的优雅回退契约**；
3. **大词表（128K–262K）下 Per-Row Tensor `top_k` 采样避免全词表扫描与隐式 Host-Device 同步**。

| 序号 | 仓库与 Commit | 日期 | 核心机制与架构变更 | 证据分类 |
| :--- | :--- | :--- | :--- | :--- |
| A1 | `flashinfer-ai/flashinfer` [`fc4df7e`](https://github.com/flashinfer-ai/flashinfer/commit/fc4df7ee1a1edbd713ff1438c25f8de6b3e5d3af) | 2026-10-28 | `cake_softmax`：常驻行缓存（Row-Caching）集群 Softmax 在重填释放槽位前插入 `fence.proxy.async.shared::cta;` | `implementation` + `author-benchmark` |
| A2 | `flashinfer-ai/flashinfer` [`dd21f60`](https://github.com/flashinfer-ai/flashinfer/commit/dd21f60a61303b4555412f3a3d11c06de5968a0b) | 2026-10-28 | `cake_mm_fp4`：当默认 Per-Token 计划未注册（`unregistered_key`）或 TMA Box 越界（`unfit_key`）时回退至适配的已注册程序 | `implementation` |
| A3 | `flashinfer-ai/flashinfer` [`e91141b`](https://github.com/flashinfer-ai/flashinfer/commit/e91141bddada2d373a879bdea7c19b5fdeff0b77) | 2026-10-28 | `prims-ts`：面向长上下文 FMHA 的 `VC-attention-QK8`（中心化 K + 128 点 Hadamard 旋转 E4M3 Q/K 块缩放） | `implementation` + `author-benchmark` |
| A4 | `flashinfer-ai/flashinfer` [`015fff1`](https://github.com/flashinfer-ai/flashinfer/commit/015fff182331b5a11621e7699dac1055602c1d09) | 2026-10-28 | `cake_qsa_nvfp4`：多线程共享 `workspace_buffer` 时对 `_workspace_plans` 和 `_seen_once` 使用原子 `pop(ident, None)` 与异常容错淘汰 | `implementation` |
| A5 | `flashinfer-ai/flashinfer` [`056670d`](https://github.com/flashinfer-ai/flashinfer/commit/056670d018638b536169c6691f548576bf3c464e) | 2026-10-27 | `cake_minimax_h3`：SM103 (`sm_103a`) NVFP4 Tail-Split Out-Projection 在共享内存中暂存残差（`228,480 B` SMEM，8 波次门槛） | `implementation` + `author-benchmark` |
| A6 | `flashinfer-ai/flashinfer` [`93ab5d9`](https://github.com/flashinfer-ai/flashinfer/commit/93ab5d91038eb79a11c97dec81b7c5e31730105a) | 2026-10-28 | `autotuner`：`delay_kernel()` 通过 `with tvm_ffi.use_torch_stream():` 显式绑定当前 PyTorch CUDA 流 | `implementation` + `author-benchmark` |
| A7 | `flashinfer-ai/flashinfer` [`ba25ff7`](https://github.com/flashinfer-ai/flashinfer/commit/ba25ff79fdfab20af333f132c9c70a96c50f9dfa) | 2026-10-27 | `sampling`：新增 `top_k_max` 参数使 Per-Row Tensor `top_k` 走无主机同步的 Top-K-First 快速路径 | `implementation` + `author-benchmark` |
| A8 | `flashinfer-ai/flashinfer` [`205c51e`](https://github.com/flashinfer-ai/flashinfer/commit/205c51eaa90db9b12c9d386e50ed6b68fe4bfa) | 2026-10-27 | `sampling`：`GetRadixTopKLaunchConfig` 计入内核静态共享内存 `attr.sharedSizeBytes`（修复 SM120 `336 B` 溢出） | `implementation` + `author-benchmark` |
| A9 | `NVIDIA/cutlass` [`ad9bd53`](https://github.com/NVIDIA/cutlass/commit/ad9bd53bdaec27a2e88053d57322ccf74efe525e) | 2026-07-18 | `sm100_block_scale_otto`：在归零 `tile_coord_l` 前保存 `int l_norm_const = tile_coord_l` 修复批处理 `norm_constant` 索引 | `implementation` |
| A10 | `tile-ai/tilelang` [`0408712`](https://github.com/tile-ai/tilelang/commit/0408712ead01b4383e0f63e9e0bf74c35c1d7676) | 2026-10-28 | `tilelang/testing`：新增 `without_kernel_cache()` 上下文管理器并保护调用方缓存状态 | `implementation` |

---

## 2. 深度技术剖析与公式推导

### 2.1 CAKE Softmax 常驻槽位重填的异步代理屏障（`fc4df7e`）

在 `flashinfer/cake/kernels/softmax/program.py` 的集群常驻行缓存（Row-Caching Persistent）程序 `c8_t8` 与 `c16_t8` 中：
- **根因分析**：计算 Warp 使用通用代理（Generic Proxy）的 `st.shared` 将归一化后的输出写回共享内存槽位，随后通过 `cp.async.bulk.shared::cta.global` 发起写出并通过 `mbarrier_arrive` 释放 `freed_addr`。然而，当 Warp 0 在 `mbarrier_wait(freed_addr, 0)` 返回后立即发起下一行的 `cp.async.bulk.global.shared::cta` 重填同一共享内存槽位时，PTX 内存模型要求在通用代理访问与后续异步代理重填之间显式插入代理屏障，否则重填 DMA 可能与前一行残留在通用代理通路上的访问发生乱序竞争。
- **修复契约**：在 `mbarrier_wait(freed_addr, 0)` 之后、发射下一轮 `cp.async.bulk` 重填之前插入：
  ```ptx
  fence.proxy.async.shared::cta;
  ```
  重新生成的 AOT cubin 体积小幅增长（`c8_t8`：`83,817 -> 85,145 B`；`c16_t8 sm_100a`：`162,333 -> 163,887 B`；`c16_t8 sm_103a`：`83,834 -> 85,162 B`）。
- **上游实测（`author-benchmark`）**：在 B200 与 GB300 上对每个常驻程序连续压测 **4,000 次启动（0 错误）**，51 个形状的几何平均性能比为 **0.996x**（几乎零开销修复偶发数值损坏）。

### 2.2 VC-Attention-QK8 长上下文注意力与 SM103 MiniMax H3 共享内存残差暂存（`e91141b`, `056670d`）

1. **VC-Attention-QK8（`e91141b`）**：
   - 针对视频生成（如 Wan2.2 P1，`B=2, H=40, S=75.6k, D=128`）的超长上下文全注意力，先对 Key 去均值（Centered K：$\tilde{K} = K - \mu_K$），再对 Q 和 $\tilde{K}$ 施加 128 点快速 Walsh-Hadamard 变换（FWHT）并按块量化为 FP8 E4M3（`q_block_size` 最大 `256`，K Tile 为 `128` tokens）；
   - 在在线 Softmax 指数计算中，将 Query 块缩放因子 `sfQ`、注意力缩放系数 `scale` 与 $\log_2 e$ 折叠为单一行乘子：
     $$\alpha_{\text{row}} = \text{sf}_Q \cdot \frac{1}{\sqrt{D}} \cdot \log_2 e$$
     并在主循环中预取下一 K Tile 的 `sfK`；
   - **B200 实测性能（`author-benchmark`）**：在 Wan2.2 P1 (`S=75.6k`) 上，BF16 Dense 耗时 `166.0 ms`，`VC-QK16` 降至 `142.0 ms`（means）/ `134.0 ms`（repair），而 **`VC-QK8` 进一步降至 `123.0 ms`（means，1.3496x）与 `115.0 ms`（repair，1.4435x）**。
2. **SM103 (`sm_103a`) MiniMax H3 Out-Projection 共享内存残差暂存（`056670d`）**：
   - GB300 (`sm_103a`) 每个 SM 具备 `228,480 Bytes` 共享内存容量。在 NVFP4 Tail-Split Out-Projection 中，每个 Epilogue Warp 分配两个 `2 KiB` SMEM 槽位，在 MMA 累加完成前通过 `cp.async` 将 BF16 残差预取至共享内存；
   - 仅当总 Tile 对数满足 $\text{waves} \ge 8$（即在 76 个 CTA Pair 上至少 $8 \times 76 = 608$ 个 Tile Pair）时启用该内核，以分摊额外的流水线建立成本；在 GB300 15 个路由形状上取得 **1.010x** 几何平均加速（`P=4, M=8368` 从 `125.2 us` 降至 `123.5 us`，**1.0138x**）。

### 2.3 有界张量 `top_k_max` 采样快速路径与 SM120 Radix Top-K 静态共享内存预算（`ba25ff7`, `205c51e`, `93ab5d9`）

1. **有界张量 `top_k_max` 快速路径（`ba25ff7`）**：
   - 先前 `top_k_top_p_sampling_from_logits` 仅在 `isinstance(top_k, int)` 时启用先选 Top-K 再做 Top-P 的快速路径；当推理引擎（如 vLLM）传入形状为 `(batch_size,)` 的张量 `top_k` 时，若调用 `top_k.max().item()` 会引入隐式 Host-Device 同步，因此被迫回退至全词表路径；
   - 新增 `top_k_max: Optional[int]` 上界参数后，内核先无同步选出每行前 `top_k_max` 个有序候选，再在寄存器/共享内存内按每行自身的 `top_k[i]` 掩码截断；
   - **B200 实测（`author-benchmark`）**：在 `vocab=262144, batch=256, top_k=20, top_p=0.9` 下，中位延迟从 `0.819 ms` 降至 **`0.218 ms`（3.7569x 加速）**；在 `vocab=128256, batch=256` 下从 `0.450 ms` 降至 `0.125 ms`（**3.6000x**）。
2. **SM120 Radix Top-K 静态共享内存扣除（`205c51e`）**：
   - 在 RTX 5090 (`sm_120`) 上，每个 Block 的最大共享内存上限为 `max_smem_per_block = 101,376 Bytes`。对于 FP32 `shape=(1, 248077)`，`num_Pass2_threads = 1024` 时的动态共享内存需求为 `101,312 Bytes`，看似小于 `101,376 Bytes`，但内核本身还占用了 **`336 Bytes` 静态共享内存**（`attr.sharedSizeBytes = 336`），导致总需求：
     $$101,312 + 336 = 101,648\text{ Bytes} > 101,376\text{ Bytes}$$
     从而在 `cudaFuncSetAttribute` 时抛出 `cudaErrorInvalidValue`；
   - 修复后 `GetRadixTopKLaunchConfig` 先通过 `cudaFuncGetAttributes` 读取 `attr.sharedSizeBytes`，要求 $\text{smem}_{\text{dynamic}} + \text{smem}_{\text{static}} \le \text{max\_smem\_per\_block}$，自动将 `num_Pass2_threads` 降为 `512`（总计 $99,264 + 336 = 99,600 \le 101,376$ Bytes），使 558 项测试从 `396/558` 提升至 **558/558 全通过**。
3. **Autotuner `delay_kernel` 流绑定（`93ab5d9`）**：
   - `delay_kernel(stream_delay_micro_secs)` 仅接收标量整型参数，TVM-FFI 无法从 Tensor 参数自动推导当前 PyTorch 流，导致延迟内核被发射到错误的原始流上。包裹 `with tvm_ffi.use_torch_stream():` 后，H100 NVL 上 `5,000 us` 延迟在目标流 Event 窗口内的捕获时长从 `0.019 ms` 恢复为 **`5.147 ms`**（同步残差从 `5.117 ms` 降至 `0.012 ms`）。

---

## 3. 本仓库离线建模验证（`own-offline`）

对应实验目录：`experiments/flashinfer_cake_softmax_mm_fp4_vc_qk8_minimax_h3_sampling_cutlass_tilelang_phase121/`

- **运行命令**：`python experiments/flashinfer_cake_softmax_mm_fp4_vc_qk8_minimax_h3_sampling_cutlass_tilelang_phase121/run.py --check`
- **关键输出校验**：
  - `cake_softmax_fence.stress_errors_after_fix = 0`（`4000` 次启动），`b200_51shape_geomean_vs_main = 0.996`；
  - `vc_qk8_and_minimax_h3.wan22_p1_vc_qk8_repair_speedup = 1.4435`，`minimax_h3_sm103_min_tile_pairs = 608`；
  - `autotuner_and_sampling.max_sampling_speedup = 3.7569`，`threads_1024_total_smem_bytes = 101648`（溢出 SM120 的 `101376`），`threads_512_total_smem_bytes = 99600`（适配）；
  - `cutlass_and_tilelang.batch_indexing_restored = true`。

---

## 4. 架构师决策指南与可证伪问题

1. **架构师计算练习**：在 SM120 (`max_smem_per_block = 101,376 B`) 上启动某 CUDA 内核，若 `cudaFuncGetAttributes` 报告 `sharedSizeBytes = 1,024 B` 且驱动保留 `reservedSharedMemPerBlock = 1,024 B`（已被 `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)` 扣除），求该内核可申请的最大动态共享内存字节数。
   - *解答*：$101,376 - 1,024 = 100,352\text{ Bytes}$。
2. **后续可证伪问题**：在 `top_k_top_p_sampling_from_probs`（输入已归一化概率而非未归一化 Logits）且高 Batch（`batch >= 64`）、大词表（`vocab >= 151,936`）但概率分布接近平坦（高熵）时，`top_k_max` 快速路径是否会因两次扫描开销而慢于直接掩码路径？
