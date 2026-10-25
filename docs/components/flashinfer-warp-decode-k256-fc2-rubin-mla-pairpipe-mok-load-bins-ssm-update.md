# FlashInfer `cake_warp_decode` K256 FC2、Rubin (`sm_107a`) MLA 双流尾声与 `cake_mok` / `cake_selective_state_update` 深度源码审计

> **English Edition**: [flashinfer-warp-decode-k256-fc2-rubin-mla-pairpipe-mok-load-bins-ssm-update.en.md](flashinfer-warp-decode-k256-fc2-rubin-mla-pairpipe-mok-load-bins-ssm-update.en.md) | **关联问题**: `q-flashinfer-warp-decode-k256-fc2-rubin-mla-pairpipe-mok-load-bins-ssm-update` (`priority: 91`, Phase 82) | **可复现实验**: `experiments/flashinfer_warp_decode_k256_fc2_rubin_mla_pairpipe_mok_load_bins_ssm_update/run.py`

---

## 1. 架构背景与核心工程矛盾

在 Blackwell (`sm_100a` B200/GB200)、Blackwell Ultra (`sm_103a` GB300) 与 Rubin (`sm_107a` R200) 三代 GPU 上，MoE 路由解码、MLA (Multi-head Latent Attention) NVFP4 分页解码以及 Mamba2 Selective State Update (SSM) 共同面临三类微架构瓶颈：

1. **高专家数 (`E=512, top-10`) 小 intermediate (`I=256/512`) MoE 解码的 FC2 尾延迟与栅栏开销**：在 `cake_warp_decode` 的 1–32 token NVFP4 (`W4A4`, BF16 输出) 融合 MoE 解码中，当 `H=4096, E=512, K=10` 且 `T >= 7..19` 时，传统受节流工作队列与较小 K-tile 的 FC2 程序在 128-row 输出块上产生过多 MMA 迭代与调度同步。
2. **Rubin (`sm_107a`) MLA NVFP4 分页解码的 PV 尾声 TMA 存储停顿与转换流水线串行**：在 `cake_rubin_mla_nvfp4_decode`（`352 B/token` 分页 NVFP4 KV cache）中，原本每个 correction warp 仅配置 **1 个 4 KB 尾声暂存块 (staging tile)**，导致 8 轮 64 维 PV 输出的每一轮都必须等待上一轮 bulk tensor store 读完该暂存块；同时在 `H96-Q8-B64-KV64K` 等满载形状下，converter warp 逐轮串行等待 `16B` 数据块加载、scale 组装与转换。
3. **MoE 训练调度主机开销 (`cake_mok`) 与 Mamba2 SSM 解码 (`cake_selective_state_update` / `cake_mamba_ssd_combined`) 跨代描述符掩码分化**：`cake_mok` 每次前向/重计算都重复在主机端推导专家布局并清零两块计数区，且反向 dead scratch 直至尾声后才释放；而 SM107 的 `tcgen05` 共享内存描述符起始地址字段从 SM100/SM103 的 **14-bit (`0x3FFF`)** 扩展为 **15-bit (`0x7FFF`)**，此前导致 21 份仅掩码不同的重复 `.cu` 渲染文件。

---

## 2. 第一手源码实现路径审计

### 2.1 `cake_warp_decode` 1-D 线性网格无节流 `K256 FC2` (`53f6bc4d3f47`, `#7339`)

审计 `csrc/fused_moe/warp_decode/cake_warp_decode_contract.cuh`、`generated/cake_warp_decode_generated_manifest.cuh` 与 `generated/common/cake_warp_decode_4f408b642a6c28d2940e_kernel.cu`：

- **合约扩展 (`CakeWarpDecodeLaunchContract`)**：新增 `kLaunchLinearWorkfeedUnthrottled = 3u` 启动拓扑枚举，并在 `cake_warp_decode_launch_grid()` 中将其映射为 1-D 线性 CTA 网格 `dim3(cta_count, 1, 1)`，配合无节流原子工作队列 (`unthrottled workfeed`) 消费路由块。
- **`K256 FC2` 内核 (`kernel_cake_warp_decode_4f408b642a6c28d2940e`)**：配置 `__launch_bounds__(256, 1)` 与 **`179,456 B` (`175.25 KiB`) 动态共享内存**。针对 8-token 路由块的每个 128-row 输出块，单次融合执行 **`K=256` NVFP4 块缩放 MMA**，并与 FC1 + SwiGLU 激活及 top-k finalize 融合在同一内核中，输出与原程序逐比特一致 (`bit-identical`)。
- **精确路由表切换**：
  - 在 `sm_100a` (GB200) 上：路由 `h4096_e512_i256_k10` 的 `num_tokens = 9..32`（24 行）与 `h4096_e512_i512_k10` 的 `num_tokens = 19..32`（14 行），共计 **38 行**。
  - 在 `sm_103a` (GB300) 上：路由 `h4096_e512_i256_k10` 的 `num_tokens = 7..32`（26 行）与 `h4096_e512_i512_k10` 的 `num_tokens = 12..32`（21 行），共计 **47 行**。
- **作者实测性能 (`author-benchmark`，GB200 `sm_100a` 与 GB300 `sm_103a`，CUDA 13.4 冷 L2)**：
  - `h4096_e512_i256_k10` 在 `sm_100a` `T=32` 读刷新冷 L2 下从 `98.18 us` 降至 **`93.46 us`**（比值 `0.952`，提速 **`1.0505x`**），memset 冷 L2 下从 `103.84 us` 降至 **`99.38 us`**（比值 `0.957`）。
  - 在 `sm_103a` `T=31` 读刷新冷 L2 下从 `95.57 us` 降至 **`91.24 us`**（比值 `0.955`，提速 **`1.0475x`**）。
  - `h4096_e512_i512_k10` 在 `sm_100a` `T=32` 读刷新冷 L2 下从 `151.61 us` 降至 **`148.80 us`**（比值 `0.981`，提速 **`1.0189x`**）。

### 2.2 Rubin (`sm_107a`) MLA NVFP4 分页解码双暂存块与 `decode_full_pairpipe` (`5c314c510256`, `#7329`)

审计 `flashinfer/experimental/cake_rubin_mla_nvfp4_decode/{build.json,cake_backend.py,README.md}`：

- **每个 Correction Warp 双 4 KB 尾声暂存块**：在 `decode_mixed` 与 `decode_full` 中，PV 尾声将 512 维输出拆分为 **8 轮 × 64 维**。现在每个 correction warp 在 **2 个 4 KB 共享内存暂存块**之间乒乓交替写入，因此第 $r$ 轮只需等待第 $r-2$ 轮的 bulk tensor store (TMA store) 完成（`cp.async.bulk.wait_group` 深度放宽至两轮前），内核动态共享内存从 **`323,584 B` (`316 KiB`) 增至 `332,800 B` (`325 KiB`，净增 `9,216 B`)**。
- **`decode_full_pairpipe` 转换器软件流水线**：新增 `decode_full_pairpipe` 全解码程序 (`csrc/decode_full_pairpipe.cu`)。在 pair 循环体中，converter warp 先一次性预取当前块全部 8 轮的 `16-byte` 压缩片段与 scale 字，随后统一拼装 scale 并执行 NVFP4 $\to$ BF16 转换，使 8 轮加载与转换重叠。主机端 `cake_backend.py` 在 `num_heads == 96 and q_len == 8 and batch_size == 64 and kv_len in (65536, 98304)`（即 `H96-Q8-B64-KV64K` 及 `96K` 扫描点）时派发 `decode_full_pairpipe_run`。
- **作者实测性能 (`author-benchmark`，NVIDIA R200 `sm_107a`，208 SMs，120 MiB L2，CUDA 13.4)**：
  - 74 个配置几何平均提速：Board A 为 **`1.0060x`**，Board B 为 **`1.0062x`**。
  - 小批量 (`B=1..2`) 尾声排空更快：`H448-Q1-B1-KV64K` 从 `25.22 / 26.22 us` 降至 **`24.19 / 25.04 us`**（提速 **`1.043x / 1.047x`**）；`H192-Q1-B2-KV64K` 从 `25.01 / 26.06 us` 降至 **`23.96 / 24.96 us`**（提速 **`1.044x / 1.044x`**）；`H96-Q8-B1-KV64K` 提速 **`1.039x / 1.037x`**。
  - 路由至 `decode_full_pairpipe` 的 `H96-Q8-B64-KV64K` 从 `1109.24 / 1102.01 us` 降至 **`1094.48 / 1092.17 us`**（提速 **`1.013x / 1.009x`**）。

### 2.3 `cake_mok` 所属 Rank 负载分箱与早期反向 Scratch 释放 (`1e47a692ef2a`, `#7326`)

审计 `flashinfer/experimental/cake_mok_bf16/{_kernels.py,backend.py,README.md}`：

- **计数内核内建 Owner-Rank Load Binning**：修改 `sm_100a`/`sm_103a`/`sm_107a` 上的 9 个 scheduler count 内核与 1 个 expert-layout 内核（每架构 10 个编译单元，其余 38 个融合/尾声库机器码不变），在统计每个专家 token 数的同时按所属 EP rank 对路由负载进行分箱。
- **单次预计算 `MoKSchedule.layout` 与单计数区清零**：在 `build_schedule()` 中一次性分配并由 `_launch_expert_layout` 填充 `layout = torch.empty(3 * local_experts + (schedule_capacity // 256), dtype=torch.int32)`；前向/重计算每次启动仅需对单块计数器 slab 执行 `zero_()`，不再每次重算布局。
- **融合反向 Launch 后立即释放 Dead Scratch**：在 `backward()` 中，一旦融合反向内核入队（在输出尾声与 `dscores` 拷贝分配新张量之前），立即将 `schedule.routed_inter`、`schedule.routed_act`、`schedule.shared_inter`、`schedule.shared_act` 及 `schedule.ctx` 置为 `None`，降低反向峰值显存水位。
- **作者实测性能 (`author-benchmark`，B200 `sm_100a` 与 GB300 `sm_103a`)**：B200 EP8 下 `glm52_ep8_mxfp8_full_ragged_defaults` 单步耗时从 `15.975 ms` 降至 **`15.596 ms`**（`-2.38%`，提速 **`1.024x`**）；GB300 EP4 下 `glm52_ep4_mxfp8_full_ragged_defaults` 从 `13.091 ms` 降至 **`12.917 ms`**（`-1.33%`，提速 **`1.013x`**）。

### 2.4 `cake_selective_state_update` 全 122 形状加速 (`882d4a6d65e2`, `#7319`) 与 `cake_mamba_ssd_combined` 15-bit 掩码折叠 (`5e5835ed42f0`, `#7349`)

- **`cake_selective_state_update` (`#7319`)**：覆盖 Nemotron-H 与 Granite-4.0-H 的 Mamba2 解码（BF16/FP32 状态、canonical FP32 `dt/D` + `int64` 槽位索引或 SGLang raw BF16 `dt/D` + `int32` 槽位索引、`hdim=64/128`、`dstate=128`、单 token 与多 token 投机解码 `vertical`/`horizontal`/`simple`）。在全部 **122 个形状**上实现零回退全面提速：
  - **B200 (`sm_100a`)**：提速 **`1.015x–1.253x`**（`granite_h_small_hd64_sglang_raw_b1` `3.808 -> 3.040 us` **`1.253x`**；`sample_stp_bf16` `44.560 -> 39.440 us` **`1.130x`**）。
  - **GB300 (`sm_103a`)**：提速 **`1.018x–1.248x`**（`sample_stp_softplus` `47.168 -> 37.792 us` **`1.248x`**；`sample_stp_bf16` `47.048 -> 37.760 us` **`1.246x`**）。
- **`cake_mamba_ssd_combined` (`#7349`)**：利用 `#if __CUDA_ARCH__ >= 1070` 将 `CAKE_SSD_SMEM_DESC_ADDR_MASK` 设为 **`0x7FFF`**（SM107 15-bit `tcgen05` SMEM 描述符起始地址字段，区别于 SM100/SM103 的 14-bit `0x3FFF`），折叠了 21 个仅掩码不同的 `sm_107a` 渲染文件（每个文件 7 处 `make_warp_uniform` 掩码站点，共 147 处），使 `generated/device/` 从 **93 个文件缩减至 72 个**（净删 `45,749` 行冗余代码），`_SM107_MODULES` 从 **45 项缩减至 19 项**，并通过 63/63 项预处理文本、SASS 与符号表一致性证明。

---

## 3. 可复现实验与验证 (`own-offline`)

运行 `python experiments/flashinfer_warp_decode_k256_fc2_rubin_mla_pairpipe_mok_load_bins_ssm_update/run.py --check` 得到的确定性指标汇总如下：

| 子系统 / 指标 | 基线 / 变更前 | 新实现 / 变更后 | 加速比 / 变化量 | 证据属性 |
|---|---:|---:|---:|---|
| `cake_warp_decode` `h4096_e512_i256_k10` (`T=32`, GB200 读冷 L2) | `98.18 us` | `93.46 us` | **`1.0505x`** (`38` 路由行) | `author-benchmark` + `own-offline` |
| `cake_warp_decode` `h4096_e512_i256_k10` (`T=31`, GB300 读冷 L2) | `95.57 us` | `91.24 us` | **`1.0475x`** (`47` 路由行) | `author-benchmark` + `own-offline` |
| Rubin MLA NVFP4 `H448-Q1-B1-KV64K` (R200 `sm_107a` Board B) | `26.22 us` (`323,584 B` SMEM) | `25.04 us` (`332,800 B` SMEM) | **`1.047x`** (`+9,216 B` SMEM) | `author-benchmark` + `own-offline` |
| Rubin MLA NVFP4 `H96-Q8-B64-KV64K` (`decode_full_pairpipe`, Board A) | `1109.24 us` | `1094.48 us` | **`1.013x`** | `author-benchmark` + `own-offline` |
| `cake_mok` `glm52_ep8_mxfp8_full_ragged_defaults` (B200 EP8) | `15.975 ms` | `15.596 ms` | **`1.024x`** (`-2.38%`) | `author-benchmark` + `own-offline` |
| `cake_selective_state_update` (122 Mamba2 形状, B200 / GB300) | `3.808 / 47.168 us` | `3.040 / 37.792 us` | **`1.015x–1.253x` / `1.018x–1.248x`** | `author-benchmark` + `own-offline` |
| `cake_mamba_ssd_combined` 设备源码文件数 (`0x3FFF` vs `0x7FFF`) | `93` 文件 (`45` SM107 项) | `72` 文件 (`19` SM107 项) | **`-21` 重复渲染文件** | `implementation` + `own-offline` |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **SMEM 容量与 TMA 尾声流水深度权衡计算**：在 `sm_107a` 上，将每个 correction warp 的尾声暂存块从 1 个 (`4 KB`) 扩充为 2 个 (`8 KB`) 使总 SMEM 从 `323,584 B` 升至 `332,800 B`（净增 `9,216 B`，对应 2 个 correction warp 各增 `4,096 B` 加对齐/屏障开销）。若某自定义变体需部署 4 个 correction warp，请计算其双缓冲尾声所需额外 SMEM 字节数，并验证是否超出 `sm_107a` 单 CTA 最大共享内存上限。
2. **可证伪假设**：当 `h4096_e512_i256_k10` 的解码 token 数 $T \le 6$ 时，1-D 线性网格无节流 `K256 FC2` 程序因 8-token 路由块填充率不足，其冷 L2 延迟不会优于现有小 tile 程序（这解释了为何 `sm_103a` 仅在 $T \ge 7$、`sm_100a` 仅在 $T \ge 9$ 时切入该内核）。
