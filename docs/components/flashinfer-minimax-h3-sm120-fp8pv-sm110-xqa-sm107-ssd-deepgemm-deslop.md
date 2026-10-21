# FlashInfer MiniMax-H3 SM120 FP8-PV 混合变长注意力、SM110 XQA Tree/Decode、SM107a Rubin SSD 与 DeepGEMM 五族精简架构深挖

> **一手源码溯源** (`2026-10-21`)：
> - `flashinfer-ai/flashinfer@ddba2f4b2c2b8f6852a565774bc31c2ace39a5c7` (`#6936`): *feat(diffusion): add opt-in FP8-PV mode for MiniMax-H3 SM120 NVFP4 varlen attention*
> - `flashinfer-ai/flashinfer@da1e9f4bc29629724cc4de618085938743beb65a` (`#6945`): *perf(sm110_xqa): faster tree and decode routes as a cake package*
> - `flashinfer-ai/flashinfer@7601d0bb7e92390aae5276951f0e4b7c8a2b5c14` (`#6943`): *perf(ssd): faster sm_107a look-back chunk-parallel programs*
> - `flashinfer-ai/flashinfer@597e3a67ca36e1d2f3820acc9b0e3ca6817c8d2e` (`#6927`): *feat(gemm): add DeepGEMM FP8 1D1D cake package*
> - `flashinfer-ai/flashinfer@4d79be41007c65a3a416a92cda2ae4395ebfbfbf` (`#6920`): *refactor(gemm): deslop cake_deepgemm round 2*

---

## 1. MiniMax-H3 SM120 NVFP4-QK / FP8-PV 混合变长注意力 (`#6936`)

在视频生成扩散模型 MiniMax-H3（固定参数 `H=24`、`D=128`、1D RoPE 维度 `64`、非因果全双向 Packed-Varlen 注意力）中，纯 NVFP4 (`pv_mode="fp4"`) 路径虽然把 `QK^T` 与 `PV` 两个矩阵乘都压到 `E2M1x2` 4-bit 张量核上，但 `PV` 累加链路上的 4-bit 量化截断使相对 $L_2$ 误差（对比 FP32 精确参考）停留在 `0.186 ~ 0.193`。`#6936` 在 `flashinfer.diffusion_ops.minimax_h3_sm120_varlen_attention_nvfp4` 中新增了可选 `pv_mode="fp8"`（默认仍为 `"fp4"`），通过保持 **`QK^T` 使用 NVFP4 块缩放 (`E2M1` + `UE4M3` 每 16 元素块缩放)**、同时将 **`PV` 升级为 `FP8 E4M3` (`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`)**，把相对 $L_2$ 误差大幅压降至 **`0.136 ~ 0.143`**（中位误差降低 **`26.39%`**），并且在 RTX 5090 与 RTX PRO 6000 Blackwell Server Edition 上依然全面快于纯 FP8 基线 `minimax_h3_sm120_varlen_attention_fp8`。

### 1.1 三阶段 PDL 链式 Kernel 与自适应 Program 路由

`pv_mode="fp8"` 由 `cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention` 包中的 5 个 `sm_120a` Program 驱动，在单条 CUDA Stream 上通过 Programmatic Dependent Launch (`cudaLaunchAttributeProgrammaticStreamSerialization = 1`) 串接为 3 次 Launch：

| 阶段 | Program 名称 | 触发条件 (`minimax_h3_sm120_nvfp4_fp8pv_programs`) | 线程块配置 | 动态 SMEM | 核心算子职责 |
|---|---|---|---|---|---|
| Stage 1 | `KV_STATS` | 始终启用 | `grid=(ceil_div(T,128)+S, H, 1)`, `block=(128,1,1)`, `4 CTAs/SM` | `16,512 B` | 分段归约每段/每头均值 $\mu_K$、中心化 $K-\mu_K$ 的最大绝对值（推导 `k_scale`）及每列 $|V|$ 最大值（推导 `v_scale`） |
| Stage 2a | `QUANTIZE_MB3` | `tokens <= 1024` | `grid=(min(max(1,ceil_div(T,8)),max_ctas), H, 1)`, `block=(128,1,1)`, `min_blocks=3` | `0 B` | 原地半旋 RoPE (`D_rope=64`)，写出 NVFP4 打包 `Q4/K4`、`UE4M3` 块缩放因子 `Q_sf/K_sf`，并将 $V$ 转置量化为 FP8 E4M3 `Vt8 [H, D=128, padded_N]` |
| Stage 2b | `QUANTIZE` | `tokens > 1024` | 同上，`8 CTAs/SM` | `0 B` | 大序列吞吐优先的 NVFP4 $Q/K$ + 转置 FP8 $V^T$ 量化内核 |
| Stage 3a | `ATTENTION_SHORT` | `max_seg_len <= 4096` | `grid=(ceil_div(T,128)+S, H, 1)`, `block=(256,1,1)` (`8 warps`) | `60,416 B` | 双缓冲 `K4 + Vt8` 共享内存流水线（`Q4` 常驻 SMEM，`60,416 B < 99 KiB`），`m16n8k64` NVFP4 `QK^T` + 在线 Softmax + `m16n8k32` FP8 `PV` |
| Stage 3b | `ATTENTION_BASE` | `max_seg_len > 4096` | 同上 | `60,416 B` | 针对长分段 (`>4096`) 优化循环展开的混合精度 FlashAttention 内核 |

### 1.2 Workspace 精确字节契约

`workspace_bytes_nvfp4_fp8pv(tokens, num_heads, num_segments)` 的确定性内存布局公式为：

$$\text{num\_kblocks} = \max\left(1, \left\lceil \frac{T}{128} \right\rceil\right), \quad \text{padded} = 128 \cdot \text{num\_kblocks}, \quad \text{tiles} = \left\lceil \frac{T}{128} \right\rceil + \max(1, S)$$

$$\text{Bytes} = \underbrace{2 \cdot T \cdot H \cdot 64}_{q_4 + k_4} + \underbrace{T \cdot H \cdot 8}_{q_{\text{sf}}} + \underbrace{H \cdot \text{num\_kblocks} \cdot 1024}_{k_{\text{sf}}} + \underbrace{H \cdot 128 \cdot \text{padded}}_{Vt_8} + 4 \times \Big(T H + 258 S H + 257 \cdot \text{tiles} \cdot H\Big)$$

当 `T = 33,472, H = 24, S = 1` 时，Workspace 精确为 **`228,440,928 B` (`217.86 MiB`)**；当 `T = 109,952, H = 24, S = 2` 时，精确为 **`749,614,176 B` (`714.89 MiB`)**。

### 1.3 双 GPU 架构实测性能与精度对比

| 扩散工作负载 (`H=24, D=128`) | RTX 5090 `fp8pv` (ms) | RTX 5090 `fp8` (ms) | RTX 5090 加速比 | RTX PRO 6000 `fp8pv` (ms) | RTX PRO 6000 `fp8` (ms) | RTX PRO 6000 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| `center_4s_p1_m33472` | `55.1978` | `75.2480` | **`1.3632x`** | `40.2044` | `47.7775` | **`1.1884x`** |
| `center_5s_p1_m38592` | `73.0156` | `100.6726` | **`1.3788x`** | `54.0714` | `64.0569` | **`1.1847x`** |
| `center_6s_p1_m48768` | `115.2190` | `159.2777` | **`1.3824x`** | `85.2030` | `100.7248` | **`1.1822x`** |
| `center_8s_p1_m58944` | `167.2685` | `231.9048` | **`1.3864x`** | `123.8312` | `147.2781` | **`1.1893x`** |
| `center_10s_p1_m74240` | `263.8216` | `365.4598` | **`1.3853x`** | `195.3989` | `232.0127` | **`1.1874x`** |
| `center_15s_p1_m109952` | `573.6273` | `796.9718` | **`1.3894x`** | `424.0433` | `505.4736` | **`1.1920x`** |
| **双会话 9 负载几何平均** | — | — | **`1.3814x`** | — | — | **`1.1876x`** |

---

## 2. SM110 (`sm_110a` Thor) XQA Tree & Decode 独立 Cake 包 (`#6945`) 与 SM107a Rubin SSD (`#6943`)

1. **SM110 XQA `cake-build/1` 封装 (`#6945`)**：
   - 将 `flashinfer/experimental/sm110_xqa` 重构为导出 **43 个 `sm_110a` Program** 的自包含 `cake_sm110_xqa` 包（面向 NVIDIA Jetson AGX Thor，20 SMs，CUDA 13.4）。
   - **D512 投机解码 Tree Attention**：退役旧 `"tcgen05"`、`"register_mma"`、`"register_mma_split"`、`"register_mma_auto"` 4 条路径（传入即抛出 `ValueError`），统一收敛至 `tmem` 与 `pair` 两条 `tcgen05` 调度；当 `kernel="auto"` 时，若 KV Cache 为 `e4m3` 且 $\lceil Q_{\text{rows}} / 128 \rceil$ 为偶数则自动选 `pair`，否则选 `tmem`。在 Thor 实测中，`D512` Tree Attention (`160` 树节点、`K_len=1920`) 在 FP16 Contiguous 上从 `19.552 us` 降至 **`17.376 us` (`1.125x`)**，FP8 Page128 从 `17.440 us` 降至 **`16.064 us` (`1.086x`)**。
   - **D128 单 Token Decode**：重写 FP16/BF16 与 FP8 (`E4M3`) 核，在 `B=1, cap=1024` 下延迟从 `17.584 / 17.824 us` 降至 **`11.616 / 11.584 us`**，实现 **`1.386x ~ 1.539x`** 加速。
2. **SM107a (NVIDIA R200 Rubin) Look-Back Chunk-Parallel Mamba2 SSD (`#6943`)**：
   - 升级 `cake_mamba2_ssd_combined` 中 6 个 `sm_107a` 变体（`bf16`/`f16`/`f32` $\times$ `batched`/`varlen`），7 组基准的 **主核几何平均加速达 `1.0839x`**，`cp_b4h8c16` 主核从 `33.3 us` 降至 **`26.2 us` (`1.268x`)**，`perf_batched_1x256` 端到端调用从 `92.1 us` 降至 **`77.0 us` (`1.196x`)**。

---

## 3. DeepGEMM FP8 1D1D Cake 包 (`#6927`) 与五族 Round-2 深度精简 (`#6920`)

1. **FP8 1D1D 独立 Cake 包 (`#6927`)**：为 `SM100/SM103` 提供 `forward`（BF16 输出）与 `wgrad`（FP32 原地累加 `C += A @ B`）两条 `sm100_fp8_gemm_1d1d` 路径，每 4 个 K128 `UE8M0` 缩放因子打包进一个 `uint32` 字并在 `[M, ceil(K/512)]` 上按 16B TMA 对齐。在 `4096x7168x4096` 形状上，GB300 (`sm_103a`) 耗时 **`66.801 us`**，GB200 (`sm_100a`) 耗时 **`87.264 us`**。
2. **五族 46 Programs / 89 Builds 的 Round-2 去冗余重构 (`#6920`)**：
   - 横跨 `fp8_1d1d` (`2`)、`native_fp4` (`8`)、`fp4_kgroup` (`21`)、`mixed_fp8_fp4` (`9`)、`fp8_batched` (`6`) 共 **46 个 Program（89 个 SM100/SM103 架构构建）**，在 **72 个文件上净删减 `12,408` 行代码 (`+2,904 / -15,312`)**：
   - 将 `fp4_kgroup` 的 11 个重复 Host Binding 折叠进 7 个共享 TU，并将 **14 个重复 Kernel TU 折叠为 4 个由 `CAKE_KGROUP_{OUT_MODE,STORE_STAGES,LPT}` 宏参数化的内核体**（直接删除 10 个 `.cu` 文件）；
   - 将 **133 份逐字节重复的内联 `encode_tma_*_desc` 辅助函数**提升至 `cake_deepgemm_tma_encoders.cuh` 中的 **35 个唯一编码器实现**（仅保留 42 处特殊张量映射内联副本），且经 `verify_deepgemm_device_gate.py` 验证 **36/36 个 Device-Gated 构建的 SASS 机器码 100% 字节级一致**。
