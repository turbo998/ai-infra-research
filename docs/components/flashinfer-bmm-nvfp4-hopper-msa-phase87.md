# FlashInfer Decode BMM Swap-AB、Cake NVFP4 Attention、Hopper MSA 编译去重与 TileLang 屏障隔离（Phase 87）

> **证据类型**：First-hand Implementation Audit (`implementation` + `author-benchmark` + `own-offline` 确定性 Python 3.12 标准库算例 `experiments/flashinfer_bmm_nvfp4_hopper_msa_phase87/run.py`)。

## 1. 核心架构发现与第一手源码路径

| 上游提交 / PR | 关键源码路径 | 架构改动与定量指标 |
|---|---|---|
| `flashinfer-ai/flashinfer#7351` (`563a9021ad`, 2026-10-25) | `csrc/cake_bf16_bmm_swap_ab/*` | 为 SM100/SM103 新增 `tcgen05` Swap-AB `bmm_bf16` 解码程序（权重行映射至 MMA 的 $M$，解码 token 映射至 $N$，支持可选 Cluster Split-K 及单次 FP32 累加器舍入）。在 9 个解码形状（$K=1024, M \in [8,33], \text{batch} \in [1,4]$，BF16/FP16/FP32 三种输出）上，相比原 `mma.sync` Cake 内核在 B200 上提速 **1.01x–1.20x**（如 `batch=2, M=33, N=1032, K=1024` 从 `5.50 µs` 降至 `4.59 µs`），在 GB300 上提速 **1.03x–1.22x**（`5.15 µs -> 4.22 µs`）。 |
| `flashinfer-ai/flashinfer#7362` (`f83439d2a6`, 2026-10-25) | `flashinfer/experimental/nvfp4_attention/*` | 交付 SM100/SM103（4 内核链：统计 + GPU 在线平滑量化 + NVFP4 QK + NVFP4 PV）与 SM120（3 内核链：INT8 QK + NVFP4 PV）的 `prepare_nvfp4_attention(backend="cake")`（BF16 输入/输出，$D=128$，FP32 softmax）。在完整端到端 CUDA Graph Replay 下，对比 FlashAttention-4 BF16 在长序列 `R2–R7` ($S \in \{32760, 75600\}$) 上 B200 提速 **1.01x–1.06x**、GB300 提速 **1.03x–1.08x**；在短序列 `R1` ($S=4800$) 因在线统计与量化开销占优为 `0.72x/0.70x`；在 RTX PRO 6000 Blackwell (`sm_120a`) 对比 PyTorch cuDNN SDPA BF16 全形状 `R1–R7` 提速 **1.41x–1.84x**。 |
| `flashinfer-ai/flashinfer#7332` (`01b18dc98d`, 2026-10-25) | `csrc/cake_hopper_msa/*`, `flashinfer/jit/cake_hopper_msa.py`, `flashinfer/msa_ops/cake_hopper_sm90.py` | 对 SM90 `cake_hopper_msa` 89 个 JIT 程序开展四步等价折叠与去重：(1) 89 个绑定文件合并为 15 个按角色共享的 `-D` 参数化绑定（删减 74 个文件）；(2) 38 个仅常量差异的克隆内核折叠为 20 个共享源码；(3) 公共 preamble 与 13 个 device helper 抽离至 `cake_hopper_msa_device.h`；(4) 移除死代码（含未被引用的 `done` scratch buffer）并将 89 份 `arg_plan` 收敛为 7 个 `_ARG_PLANS` 引用。`csrc/cake_hopper_msa/` 文件数从 **179 降至 68**，净删减 **77,342 行代码**（`+2,313 / -79,655`），且经 G-A..G-D 四道门禁证明 89 个程序的 `cuobjdump -sass` 与符号表完全一致。 |
| `flashinfer-ai/flashinfer#7331` (`85fcf53577`, 2026-10-25) & `#7317` (`506fac6203`, 2026-10-25) | `flashinfer/cudnn/{decode,prefill,utils}.py` | `#7317` 使用 `functools.cache` 缓存 `_cudnn_version_supports_shape_override()`（EPYC 9554 上版本谓词开销从 `0.897 µs` 降至 `0.072 µs`，提速 **12.46x**，含环境变量检查从 `1.890 µs -> 1.023 µs`）；`#7331` 在热路径直接读取底层 raw stream 指针而非构造 Python `torch.cuda.Stream` 对象，B200 上 Ragged Prefill 热入队从 `32.796 -> 31.154 µs`（节省 `1.642 µs` / `-5.0%`），HND Decode 热入队从 `35.717 -> 32.706 µs`（节省 `3.011 µs` / `-8.4%`）。 |
| `flashinfer-ai/flashinfer#7320` (`a02c8c5e6b`, 2026-10-25) & `#7354` (`66a780899a`, 2026-10-25) | `flashinfer/diffusion_ops/cake_minimax_h3_sm120_quant_varlen_attention.py`, `flashinfer/cake_causal_conv1d_train_backend/build.json` | `#7320` 在 `MiniMaxH3VarlenPlan` 构建阶段校验 `_MAX_ELEMENTS = (1 << 31) - 1`，拦截 `padded_tokens * num_heads * 128 > 2^31 - 1`（56 heads 下最大安全对齐 token 数为 **299,520**，`299,521` 即触发 32-bit 有符号偏移溢出）以及统计量内核的 `segments * heads * 128` / `num_tiles * 256 * heads` 溢出；`#7354` 将 `cake_causal_conv1d_train` 中 24 个 `sm_107a` 条目的 `min_cuda` 从 `"13.0"` 提升至 `"13.4"`，避免 CUDA 13.0–13.3 触发 `nvcc fatal: Unsupported gpu architecture 'compute_107a'`。 |
| `tile-ai/tilelang#3643` (`efa1ffc1e3`, 2026-10-25), `#3193` (`35ee295075`, 2026-10-23), `#3589` (`baceed6676`, 2026-10-23) | `src/transform/thread_storage_sync.cc`, `tilelang/jit/adapter/nvrtc/adapter.py`, `examples/ascend/deepgemm/*` | `#3643` 修复 `ThreadPartialSyncRewriter` 在依次处理 `"shared"` 与 `"shared.dyn"` 两个存储域时从头分配命名屏障 ID 的缺陷：此前 DeepSeek-V3.2 稀疏 MLA 流水线内核中 Producer Warpgroup (`[256, 384)`) 与 Consumer Warpgroup 0 (`[0, 128)`) 均被分配 `bar.sync 3`，在 H100 上运行数十至数百次后死锁；修复后通过 `PostOrderVisit` 收集已占用的 `used_barrier_ids_` 并跳过。`#3193` 修复 NVRTC 命中磁盘缓存后 autotune 结果保存缺失；`#3589` 为 Ascend DeepGEMM 增加 FP4xFP4 与越界填充支持。 |

## 2. 数学模型与架构师计算法则

### 2.1 Decode-Row `bmm_bf16` Swap-AB 的 Tensor Memory (TMEM) 映射法则
在标准解码阶段批量矩阵乘 $C_{[B, M, N]} = A_{[B, M, K]} \times W_{[B, K, N]}$ 中，解码行数 $M \in [8, 33]$ 极小，而Blackwell `tcgen05.mma` 的硬件指令行维（MMA $M$）最小粒度为 64 或 128。若直接将解码 token 放入 MMA $M$，会有 $50\%\text{–}87.5\%$ 的 Tensor Core 行通道处于无效填充状态。
通过转置恒等式：
$$C^T_{[B, N, M]} = W^T_{[B, N, K]} \times A^T_{[B, K, M]}$$
将权重输出维度 $N \in [40, 1032]$ 映射至 `tcgen05` 的 MMA $M$ 维，将小批量解码 token $M \in [8, 33]$ 映射至 MMA $N$ 维，并在 Cluster 内沿 $K=1024$ 可选执行 Split-K 规约、仅在 Epilogue 执行一次 FP32 $\to$ BF16/FP16/FP32 舍入，从而在 GB300 上实现最高 **1.22x** 的冷 L2 实测加速。

### 2.2 SM120 打包变长注意力的 32-bit 偏移安全上界
设总 token 数为 $T$，块对齐粒度 $B_N = 128$，注意力头数 $H = 56$，头维 $D = 128$。对齐后 token 数为：
$$T_{\text{pad}} = \max\left(1, \left\lceil \frac{T}{128} \right\rceil\right) \times 128$$
内核使用有符号 32 位整数（上限 $I_{\max} = 2^{31} - 1 = 2,147,483,647$）索引操作数、段统计量 $[S_{\text{seg}}, H, 128]$ 与查询块部分统计量 $[N_{\text{tiles}}, H, 256]$。因此必须同时满足三项不变量：
$$T_{\text{pad}} \cdot H \cdot D \le 2^{31} - 1, \quad S_{\text{seg}} \cdot H \cdot 128 \le 2^{31} - 1, \quad N_{\text{tiles}} \cdot H \cdot 256 \le 2^{31} - 1$$
当 $H=56, D=128$ 时，单 token 跨度为 $56 \times 128 = 7,168$ 元素，最大合法 $T_{\text{pad}} = \lfloor (2^{31}-1) / 7168 / 128 \rfloor \times 128 = \mathbf{299,520}$。一旦 $T = 299,521$，$T_{\text{pad}} = 299,648$，总元素数达 $2,147,876,864 > 2^{31}-1$，在 Plan 构造期即抛出 `ValueError`，零运行时开销地消除静默内存越界。

## 3. 可复现实验验证（`experiments/flashinfer_bmm_nvfp4_hopper_msa_phase87`）

运行 `python experiments/flashinfer_bmm_nvfp4_hopper_msa_phase87/run.py --check` 通过全部确定性断言：
- `bf16_bmm_swap_ab.routed_decode_shapes_per_arch == 9`，B200 加速区间 `[1.01, 1.20]`，GB300 加速区间 `[1.03, 1.22]`；
- `cake_nvfp4_attention.gb300_r2_r7_speedup_range_x == [1.03, 1.08]`，`sm120_r1_r7_speedup_range_x == [1.41, 1.84]`，`R1` ($S=4800$) 在线量化开销比 `0.72x` (B200) / `0.70x` (GB300)；
- `hopper_msa.head_files == 68`（自 `179` 缩减），净删减 `77,342` 行；
- `minimax_h3_sm120_32bit_guard.max_safe_padded_tokens == 299520`；
- `tilelang_barrier_isolation.disjoint_ids == True`（`shared -> 1`, `shared.dyn -> 2`）。
