# FlashInfer SM100+ 掩码 MXFP8 专家量化与融合 SiLU-and-Mul、SM90 FP4 MegaMoE 推送调度与 CuTe-DSL `bmm_fp8`/`mm_mxfp8` 单波次优先级深挖

> **一手上游证据（2026-10-21）**：
> - `flashinfer-ai/flashinfer` PR `#6437` (`140f9fc4c5fea322de9de2e520e88cc1ef976d7a`)：`feat(quantization): add CUDA backend for mxfp8_grouped_quantize and fused silu_and_mul_mxfp8_experts_quantize`
> - `flashinfer-ai/flashinfer` PR `#6517` (`da992ed9919604c5eb6dc255c63f5e89a466452c`)：`[MegaMoE] Fix SM90 FP4 fused kernel, enable H200 push dispatch, and tune tile selection`
> - `flashinfer-ai/flashinfer` PR `#6800` (`bf044ba86e8ec7ff5af11034378c6ff6568e4a8d`)：`perf(gemm): fuse scale tensor multiply into Blackwell bmm_fp8 epilogue`
> - `flashinfer-ai/flashinfer` PR `#6848` (`617845d1a3aeadf2b95fb0e5d6a12383b192a963`)：`fix(gemm): gate mm_mxfp8 warp split-K priority on one-wave output fit`

---

## 1. SM100+ 掩码专家级 MXFP8 量化与融合 `silu_and_mul_mxfp8_experts_quantize`（PR `#6437`）

在 DeepEP 低延迟模式下的 MoE 推理路径中，通信层输出固定容量为 `[E, M, K]` 的专家张量以及设备端 `mask[E]`（记录每个专家的有效 token 行数）。此前 `mxfp8_grouped_quantize` 的 `cutile` 与 `cake` 后端缺乏对 `mask[E]` 的设备端稀疏感知或融合激活支持，不仅会对无效填充行执行冗余量化，还在 GEMM1 与 GEMM2 之间引入额外的 `silu_and_mul` 全量写回。

PR `#6437` 新增了 SM100+（Blackwell/Rubin）原生 CUDA 核函数 `cvt_fp16_to_mxfp8_expert<Type, USE_SILU_AND_MUL>`（位于 `csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh`）：

1. **前缀和全局有效行拉平与寄存器/SMEM 双路径二分查找**：
   - 核函数仅处理每个专家的前 `min(mask[e], numRows)` 有效行，将所有专家的有效行拼接为一个连续虚拟行列表，使高度偏斜（skewed）的专家负载也能均匀铺满整个 GPU Grid。
   - 当 `numExperts <= 32` 时，每个 Warp 直接在寄存器中通过蝶形归约（butterfly reduction）维护 `expertFirstRow` 前缀和并执行二分查找；当 `numExperts > 32` 时，CTA 仅需 `(numExperts + 1) * sizeof(int32_t)` 动态共享内存（例如 `E=64` 仅需 `260 B`）。
2. **双 Lane 协作半块向量加载与 `128x4` UE8M0 缩放因子交织（Swizzle）**：
   - 每个缩放块宽度 `SF_VEC_SIZE = 32`，由两个相邻 Lane 各自通过一次 `32 B` 向量加载（`HalfBlockT = PackedVec<Type, 16>`）读取半个缩放块，并在 Warp 内 shuffle 归约最大值，直接写入符合 `grouped_gemm_nt_masked` 要求的 `[32, 4, padded_M // 128, 4, scale_K // 4, E]` 布局。
   - 当 `USE_SILU_AND_MUL=true` 时，同一线程同时加载 Gate（前 `K` 列）与 Up（后 `K` 列）半块，在寄存器内完成 `silu_and_mul` 后直接量化为 E4M3 + UE8M0，彻底消除中间 BF16 激活张量读写。

### B200 实测性能（`E = 32, CAP = 1024`，BF16 输入，单位 `μs`）

| 算子场景 | 每专家有效行 `mask[e]` | `cutile` (`μs`) | `cake` (`μs`) | `cuda` (PR `#6437`, `μs`) | 相对 `cutile` 加速比 | 相对 `cake` 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| `mxfp8_grouped_quantize` (`K=7168`, GEMM1) | `16` | `59.17` | — | **`5.90`** | **`10.03x`** | — |
| `mxfp8_grouped_quantize` (`K=7168`, GEMM1) | `128` | `94.58` | `38.21` | **`20.18`** | **`4.69x`** | **`1.89x`** |
| `silu_and_mul_mxfp8_experts_quantize` (`2x2048`, GEMM2) | `16` | `91.22` (非融合) | — | **`4.88`** | **`18.69x`** | — |
| `silu_and_mul_mxfp8_experts_quantize` (`2x2048`, GEMM2) | `128` | `99.81` (非融合) | `83.65` (非融合) | **`11.70`** | **`8.53x`** | **`7.15x`** |

---

## 2. SM90 FP4 MegaMoE 脏块清理修复、H200 推送调度与 NVFP4/MXFP4 缩放域重平衡（PR `#6517`）

PR `#6517` 解决了 Hopper（SM90，H20/H200）融合 NVFP4/MXFP4 MegaMoE 核函数的三项关键工程问题：

1. **`single_active_dispatch_warp` 工作区脏块静默污染修复**：
   - 在 `BLOCK_M=16` 或 `BLOCK_M=24` 的拉取调度（`!kPushDispatch`）配置下，`kNumDispatchWarps = 2` 但 `kNumActiveDispatchWarps = 1`（Warp 1 跳过拉取循环以让出寄存器/带宽）。
   - 然而 `cleanup_workspace` 由全部 2 个 Dispatch Warp 协作按步长 2 清理对称内存池块：由于 Warp 1 从未调用 `scheduler.fetch_expert_recv_count()`，其缓存的专家 token 计数全为 `0`，导致热点专家（如 H20 上单专家汇聚 `128` tokens）编号 `>= 32` 的池块（在 `128` 块中共 `112` 个块）遗留脏数据，进而破坏后续迭代的计数器栅栏与输出。
   - **修复**：在 `sm90_nvfp4_mega_moe_fused_body.inl` 的 `cleanup_workspace` 前显式补充 `if (warp_idx >= kNumActiveDispatchWarps) scheduler.fetch_expert_recv_count();`，使未激活拉取的 Warp 也能精确清理奇数池块（脏块数从 `112` 降为 `0`）。
2. **H200 EP8 推送调度（Push Dispatch）与对称缓冲区预热**：
   - 新增 `_H200_COUNTER_BUCKETS`：在 H200（`141` SMs，EP8）上对 `M <= 32`（`DeepSeek-V4-Flash` `H=4096, E=256, topk=6`、`MiMo-V2.5-Pro` `H=6144, E=384, topk=8`、`DeepSeek-V4-Pro` `H=7168, E=384, topk=6`）默认启用推送调度，并对 `DeepSeek-V4-Pro` 的 `M = 33..64` 启用旋转池（rotated pool）推送，每次调用节省约 `10 μs`（核函数提速 **`1.03x–1.16x`**）。
   - 新增 `MegaMoESymmBuffer.prewarm()`（在首次真实运行前完成跨 Rank IPC 页表建联，消除首步 ~`20 ms` 尖峰）与 `commit_staged()`（令 staging copy 在调用方主流上对 CUDA Graph capture 可见）。
3. **FP4 × Scale -> E4M3 解码不饱和不变量**：
   - SM90 MegaMoE 核函数将 FP4 权重（最大幅值 `6.0`）与分组缩放因子相乘后解码至单个 E4M3 字节（最大值 `448.0`）。因此 NVFP4 必须满足 `6 * scale <= 448`（即 `scale <= 74.67`），MXFP4 非零组必须满足 `2^-5 <= scale <= 2^6`（在 `DeepSeek-V4-Flash-NVFP4` 原始权重中约 `51%` 的块超出此上限）。通过将超限因子提取至每专家 2 的幂次 `l1_global_scales` / `l2_global_scales`（无损移位），即可在保持比特级精度的同时彻底杜绝解码饱和。

---

## 3. Blackwell/Rubin (`sm_107`) CuTe-DSL `bmm_fp8` 尾声缩放融合与 `mm_mxfp8` 单波次优先级（PR `#6800` / `#6848`）

1. **`bmm_fp8` 尾声（Epilogue）设备端缩放乘积融合（PR `#6800`）**：
   - 此前 `bmm_fp8(..., backend="cute-dsl")` 在主机端调用 `a_scale * b_scale`（以及非 FP32 时的 `.to(torch.float32)`），每次 BMM 额外触发 1–2 个微小 PyTorch 算子启动（60 层模型每步多出 `60–120` 次 kernel launch）。
   - PR `#6800` 将 `a_scale_tensor` 与 `b_scale_tensor` 两个单元素 `Float32` 张量直接传入 `SM107PersistentDenseGemmKernel`，在核函数内部执行 `output_scale = a_scale_tensor[0] * b_scale_tensor[0]`，实现零额外算子启动且完全兼容 CUDA Graph。
2. **`mm_mxfp8` 单波次输出块（One-Wave Output Tile）启发式（PR `#6848`）**：
   - 此前未调优 `mm_mxfp8(backend="cute-dsl")` 在 `m <= 8 or k <= 2048` 时无条件优先选择 `warp` split-K runner。但 `warp` split-K 采用 `16` 列宽输出块（每块 1 个 CTA），当 `N` 较大（如 `N = 4096` 或 `7168`，超过单波次 `148 * 16 = 2368` 列）时，多波次 CTA 在 Programmatic Dependent Launch（PDL）连续调用下会因 `griddepcontrol.wait` 阻塞后续波次，反而慢于 `cluster` split-K。
   - PR `#6848` 将启发式改为 `b.shape[1] // 16 <= get_device_sm_count(a.device)`（在 148-SM 的 B200/B300 上即 `N <= 2368`）：
     - **B300（150 个测试点）**：PDL 性能回退点从 **`63` 个降至 `0` 个**，几何平均加速 **`1.059x`**（最高 **`1.51x`**）。
     - **B200（120 个测试点）**：PDL 性能回退点从 **`43` 个降至 `0` 个**，几何平均加速 **`1.059x`**（最高 **`1.45x`**）。

---

## 4. 确定性复现实验验证

运行本仓库标准库复现实验：

```bash
python experiments/flashinfer_masked_mxfp8_sm90_megamoe_bmm_fp8/run.py --check
```

全部不变量（`total_valid_rows == 1856` vs 未掩码 `32768`、`18.69x` SiLU+Mul+MXFP8 融合加速、SM90 MegaMoE 脏块 `112 -> 0`、`6 * scale <= 448` 2 的幂次全局缩放重平衡、B200/B300 `N <= 2368` 单波次零回退）均 `100%` 验证通过。
