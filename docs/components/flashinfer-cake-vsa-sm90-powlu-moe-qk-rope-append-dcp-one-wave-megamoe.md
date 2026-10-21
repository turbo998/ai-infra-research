# FlashInfer SM90 Cake VSA 17-Program 调度、CUTLASS Fused-MoE PowLU 激活、SM90/SM100/SM103 Fused QK-RoPE Append 与 One-Wave DCP Decode 深度解析

> **Phase 52 (`priority: 61`) · Track 1 一手源码深挖**  
> **上游提交**: `flashinfer-ai/flashinfer@ddf080d8a56c` (`#6877`), `366214d4325e` (`#5743`), `51eb019768fc` (`#6888`), `49aa24fbc4b3` (`#6887`), `849e8be49ff2` (`#6122`), `90c82e1268c8` (`#6684`)  
> **确定性复现脚本**: `experiments/flashinfer-cake-vsa-sm90-powlu-moe-qk-rope-append-dcp-one-wave-megamoe/run.py --check`

---

## 1. SM90 Cake VSA 自包含 17-Program 后端 (`#6877`)

`flashinfer-ai/flashinfer@ddf080d8a56c87f60a25ee9df750e373c4bcbe92` 将 `cake_vsa_sm90` 重构为自包含的 Hopper (`sm_90a`) Video Sparse Attention (VSA) 后端包（`flashinfer/vsa_sparse/cake_vsa_sm90/`），在同一模块内编译 **17 个 BF16 算子程序**：

- **2 个 Persistent 程序**：
  - `attention`：静态任务划分（static persistent plan）；
  - `attention_queue`：动态 Tile 队列（dynamic tile queue）。
- **15 个 Small 程序**（覆盖 `s_q <= 16,384` 与小 Batch 场景）：
  - Direct (`4`): `k1, k3, k4, k6`
  - DSM Cluster (`6`): `k2c4, k3c3, k3c6, k4c2, k4c4, k6c2`
  - Split-KV (`2`): `k4s, k6s`
  - Split-Cluster (`3`): `k6sc3, k6sc4, k6sc8`

在 H100 SXM 80GB (CUDA 13.0) 上的 4 个基准套件（共 `137` 行）实测中，`119 / 137` 行快于基线：
- `b20` (`20` 行): Geomean **`1.0205x`** (`1.0031x – 1.0510x`, `20/20` 更快)
- `ext33` (`33` 行): Geomean **`1.0515x`** (`0.9963x – 1.2524x`, `30/33` 更快)
- `real39` (`39` 行): Geomean **`1.0367x`** (`0.9854x – 1.2126x`, `36/39` 更快)
- `long45` (`45` 行): Geomean **`1.0120x`** (`0.9776x – 1.0695x`, `33/45` 更快)

---

## 2. CUTLASS Fused-MoE `PowLU` 激活函数与零 ABI 破坏复用 (`#5743`)

`flashinfer-ai/flashinfer@366214d4325e049199c9e79aa1eb30b0d3a867f1` 为 CUTLASS Fused-MoE 后端引入 **PowLU**（Power Linear Unit）激活函数：

$$\mathrm{PowLU}(g; m) = \begin{cases} g^{\frac{m}{\sqrt{g} + 1}} \cdot \sigma(g), & g > 0 \\ g \cdot \sigma(g), & g \le 0 \end{cases}$$

$$\mathrm{out} = \min(\mathrm{PowLU}(\mathrm{gate}, m), \mathrm{limit}) \cdot \mathrm{clamp}(\mathrm{up}, -\mathrm{limit}, \mathrm{limit})$$

- **数值边界与 FP16 溢出防护**：当 $g \le 0$ 时，$\mathrm{PowLU}(g) \equiv \mathrm{SiLU}(g)$；当 $g > 0$ 时，随着 $g \to \infty$，指数 $\frac{m}{\sqrt{g}+1} \to 0$，函数渐近收敛至 $\sigma(g) \to 1$。在默认 `DEFAULT_POWLU_M = 2.5` 下，全局峰值约为 **`4.02`**（当 $m=3.5$ 时峰值约为 **`7.02`**），远低于 FP16 上界 `65504`，从根本上消除了深层 MoE FFN 的激活爆炸。
- **零 ABI 变更复用**：在不改变 `ActivationParams` 结构体布局的前提下，复用 `swiglu_alpha` 传递标量 `m`、复用 `swiglu_limit` 传递 `limit`；C++ `ActivationType::PowLU = 11`（`InvalidType` 顺延至 `12`），Python `tllm_enums.py` 中 `ActivationType.PowLU = 13`。

---

## 3. 跨架构 Fused QK-RoPE Append (`#6888`, `#6887`)、One-Wave DCP Decode (`#6122`) 与 SM120 MegaMoE 密封去重 (`#6684`)

1. **`cake_fused_qk_rope_fp8_append` (`#6888`) & `cake_fused_qk_rope_append` (`#6887`)**：
   - 覆盖 `sm_90a` (H100/H200)、`sm_100a` (B200) 与 `sm_103a` (B300/GB300) 五大 GPU 平台：
   - **FP8 Append (`44` 行/GPU，共 `220` 行)**：全部 `220 / 220` 行 $\ge 1.00\times$，Geomean 分别为 **`1.0423x`** (H100)、**`1.0442x`** (H200)、**`1.0328x`** (B200)、**`1.0347x`** (B300)、**`1.0341x`** (GB300)。
   - **BF16 Append (`28` 行/GPU，共 `140` 行)**：`139 / 140` 行 $\ge 1.00\times$，Blackwell 家族取得 **`1.1362x`** (B200)、**`1.1424x`** (B300) 与 **`1.1428x`** (GB300) Geomean 加速。
2. **SM100a / SM103a Whole-Tile Static One-Wave DCP Decode (`#6122`)**：
   - 针对 `head_dim=128` BF16 balanced DCP decode 新增静态单波次（one-wave）Program `2..5`，在 GB300 (`b=1, s=4096, q=4`) 上将延迟从 `11.07 us` 降至 **`9.73 us`** (**`1.138x`**)；
   - 针对 `dcp_spec_bf16_fp8_d256_balanced` 新增 Program `2`，在 GB300 (`b=1, ctx=32768, q=4, cp=4`) 上将延迟从 `17.02 us` 降至 **`13.73 us`** (**`1.240x`**)。
3. **SM120 MegaMoE `part002.cuh` 去重与 18 重签名重密封 (`#6684`)**：
   - 删除与 `part001.cuh` 字节级完全相同（`1,090` 行、`1,072` 条预处理指令、`258` 条 `#error`）的 `cake_sm120_megamoe_kernel_part002.cuh`，使 `#if MOE_INSTANCE == 0` 与 `#elif MOE_SHARED_EXPERT == 1` 共享同一头文件，设备源码从 `13` 个减至 `12` 个，18 项 Manifest Seal 全部通过且生成的 8 个 Cubin SASS 位级一致。
