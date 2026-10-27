# FlashInfer `cake_causal_conv1d` 动态原子计数器、Kimi-K3 AttnRes SM103 384 线程配对、Grouped/Batch FP8 GEMM 与 CUTLASS SM110a / TileLang 越界防护深度解析（Phase 104 · Priority 113）

> **English Version**: [flashinfer-causal-conv1d-kimi-attnres-grouped-fp8-cutlass-tilelang-phase104.en.md](flashinfer-causal-conv1d-kimi-attnres-grouped-fp8-cutlass-tilelang-phase104.en.md)
> **实验脚本**: [`experiments/flashinfer_causal_conv1d_kimi_attnres_grouped_fp8_phase104/run.py`](../../experiments/flashinfer_causal_conv1d_kimi_attnres_grouped_fp8_phase104/run.py) · **定量汇总**: [`summary.json`](../../experiments/flashinfer_causal_conv1d_kimi_attnres_grouped_fp8_phase104/summary.json)
> **7-Card 视觉报告**: [`docs/reports/ai-infra-phase104-benchmark-cards.html`](../reports/ai-infra-phase104-benchmark-cards.html)

---

## 1. 证据分级与上游源码锚点（Evidence Classification）

本组件 strictly 区分 **`implementation`（固定 commit 源码审计）**、**`author-benchmark`（PR 作者在真实 GPU 测得的数据）** 与 **`own-offline`（本项目 Python 3.12 标准库确定性建模）**：

| 证据 ID / 来源 | 仓库与 Commit | 关键路径 | 证据分级 |
|---|---|---|---|
| `src-flashinfer-cake-causal-conv1d-grad-speedup-2026` | `flashinfer-ai/flashinfer@c289936e6b8fd07d751dae48db2a976bb58f9cc8` | `flashinfer/cake_causal_conv1d_train.py` | `implementation` + `author-benchmark` |
| `src-flashinfer-kimi-k3-attn-res-sm103-pair384-2026` | `flashinfer-ai/flashinfer@53c4cd234118c053336d77ff7df572469be77fbe` | `flashinfer/experimental/cake_kimi_k3_attn_res/cake_backend.py` | `implementation` + `author-benchmark` |
| `src-flashinfer-grouped-fp8-ue8m0-swap-rp32-2026` | `flashinfer-ai/flashinfer@216d52e164c881f591481aa3108e5d245168667c` | `csrc/cake_grouped_fp8_gemm/..._kernel.cu` | `implementation` + `author-benchmark` |
| `src-flashinfer-batch-deepgemm-fp8-masked-2cta-2026` | `flashinfer-ai/flashinfer@7e2ee7610470492fa2fe05da9e16661e92e4a3c6` | `csrc/cake_batch_deepgemm_fp8/..._kernel.cu` | `implementation` + `author-benchmark` |
| `src-flashinfer-blackwell-msa-shared-tu-deslop-2026` | `flashinfer-ai/flashinfer@4a439a7e645f2d913635254b283328675d29a8f4` | `csrc/blackwell_msa/..._kernel.cu` | `implementation` |
| `src-cutlass-sm110a-cta-reconfig-activated-2026` | `NVIDIA/cutlass@78e46228a1bfc5c972e2f9a634223af784cd6949` | `include/cutlass/arch/reg_reconfig.h` | `implementation` |
| `src-cutlass-tmem-allocator-constants-compat-2026` | `NVIDIA/cutlass@f614dc40e17fb3ddb1b7b474318b48a8d5d21a2c` | `include/cute/arch/tmem_allocator_sm100.hpp` | `implementation` |
| `src-tilelang-sliced-fill-end-bound-fix-2026` | `tile-ai/tilelang@323b02ac2a9d309dae7a00939da4b7f6463a5ba9` | `src/op/fill.cc` | `implementation` |
| `src-tilelang-ws-neg1-all-threads-fix-2026` | `tile-ai/tilelang@ee4151896de44e29c109e543b0e04c109e543b0e` | `src/ir.cc`, `tilelang/cuda/language/warpgroup.py` | `implementation` |
| `src-tilelang-ascend-per-core-conditionals-2026` | `tile-ai/tilelang@060e1ff31bbb0b65f27286fafc4b1953e109c2b1` | `src/ascend/transform/materialize_schedule_units.cc` | `implementation` |

---

## 2. 核心实现路径、数据布局与同步机制

### 2.1 `cake_causal_conv1d_train` 动态设备原子计数器与零残留 Workspace (`c289936e6b`)
- **问题背景**：在 packed variable-length 序列（如 Mamba-2 / Qwen3-Next 线性注意力层的因果一维卷积）前向/反向传播中，原先同一 channel tile 内的多个 CTA 按固定等分（static equal share）划分文档区间。当单个 batch 内存在长短极不均匀的文档（例如 `[4096, 2048, 1024, 512]` tokens）时，负责长文档的 CTA 成为尾延迟瓶颈，其余 CTA 提前空转。
- **架构演进**：
  1. **动态 chunk 窃取（Dynamic Atomic Chunk Counter）**：同一 channel tile 内的各 CTA 通过设备端 `int32` 原子计数器（`atomicAdd`）以固定粒度（`chunk_tokens = 256`）动态申领连续 token 块，将非均匀 `[4096, 2048, 1024, 512]` 序列的单 CTA 最大负载从 `4096` tokens 压降至 `2048` tokens，结合成对 FP32 算术（paired FP32 math）在确定性模型下取得 **`2.2505x`（`4096 -> 1820` cycles）** 加速。
  2. **非 32 倍数通道（`C = 1160, 1344`）反向切分**：消除尾块对齐回退。
  3. **零残留可重用 Workspace（Zero-Left Reusable Workspace）**：kernel 在退出最后一个 chunk 时将原子计数器重置归零，调用方传入预清零 `workspace: torch.Tensor`（`int32`）即可在 CUDA Graph 重放或连续 step 中直接复用，**每步节省 `1` 次 `cudaMemsetAsync` 启动（额外 memset 次数 = `0`）**。
- **作者实测基准（`author-benchmark`）**：在 `528` 组测试（`8` 种梯度变体 $\times 22$ 种形状 $\times 3$ 款 GPU）中**全部正向加速**：B200 提升 **`1.052x–1.926x`**，GB300 提升 **`1.053x–1.762x`**，R200 提升 **`1.037x–1.741x`**。

### 2.2 Kimi-K3 Attention Residuals SM103 384 线程三 Warpgroup 配对 (`53c4cd2341`)
- **问题背景**：在 Blackwell Ultra SM103（B300 / GB300）上运行 Kimi-K3 Attention Residuals（`M = 192, hidden = 7168, K = 4 / 6`）时，旧版 `kPair` 程序采用 `320` 线程（`10` warps：8 consumer + 2 producer）并将 consumer 寄存器上限锁死在 `168` regs/lane，导致 `K = 4` 时每 lane 产生 `28` 个寄存器溢出（spills to local memory）。
- **公式与寄存器预算验证**：
  $$\text{Total CTA Regs} = N_{\text{cons\_warps}} \times 32 \times R_{\text{cons}} + N_{\text{prod\_warps}} \times 32 \times R_{\text{prod}}$$
  升级为 `384` 线程（`12` warps = `2` 个 consumer warpgroups + `1` 个 producer warpgroup）并通过 `setmaxnreg` 分配 $R_{\text{cons}} = 232$、$R_{\text{prod}} = 40$：
  $$\text{Total CTA Regs} = 8 \times 32 \times 232 + 4 \times 32 \times 40 = 59,392 + 5,120 = 64,512 \le 65,536$$
  这一配比恰好卡在 SM 64K 寄存器文件上限之内，将 `K = 4` 的寄存器溢出从 **`28` 降至 `0`**，在 B300 上取得 **`1.0575x`（PDL on）**、在 GB300 上取得 **`1.0553x`（`K=4`）** 与 **`1.0201x`（`K=6`）** 实测加速。

### 2.3 Grouped FP8 32-Row-Block Swap-AB 与 Batch DeepGEMM Masked 2-CTA (`216d52e164`, `7e2ee76104`, `4a439a7e64`)
- **`cake_grouped_fp8_gemm` (`216d52e164`)**：新增 `bs_ue8m0_swap_rp32`（32-row-block UE8M0 swap-AB），针对 $32/64$-row 对齐的 MoE 专家负载（$M \le 1536, N \bmod 256 = 0$）交换 A/B 矩阵在 UMMA 中的角色，将 B200 上 $M=1152, N=1024, K=1024$ 的耗时从 `7.82 us` 降至 `6.86 us`（**`1.1399x` 加速**）。
- **`cake_batch_deepgemm_fp8` (`7e2ee76104`)**：新增 13 个 FP32-scale masked `BLOCK_N` 构建与 1 个双 CTA 大 $N/K$ 构建，在 $B=1, M=128, N=128, K=512$ 上将耗时从 `4.35 us` 压至 `3.63 us`（**`1.1983x` 加速**）。
- **`cake_blackwell_msa` (`4a439a7e64`)**：将近乎重复的 CUDA 源码折叠为通过 `-D` 宏选择的共享编译单元（shared TUs），降低二进制体积与维护漂移。

### 2.4 CUTLASS SM110a `setmaxnreg` 激活与 TileLang 切片/Warp-Specialized 边界修复 (`78e46228a1`, `323b02ac2a`, `ee4151896d`)
- **CUTLASS SM110a (`78e46228a1`)**：此前 `include/cutlass/arch/reg_reconfig.h` 中的 `CUDA_CTA_RECONFIG_ACTIVATED` 宏遗漏了 `__CUDA_ARCH__ == 1100`（Jetson Thor / SM110a），导致 `warpgroup_reg_alloc` / `warpgroup_reg_dealloc` 退化为空操作（no-op），引发 SM110a warp-specialized kernel 静默寄存器溢出或死锁。
- **TileLang 切片上界检查 (`323b02ac2a`)**：`src/op/fill.cc` 原先只检查 `region[i]->extent <= shape[i]` 而忽略了起始偏移 `min`。对于 `shape = [8]` 的共享内存缓冲区执行 `T.fill(As[6:12], 7)`，旧逻辑因 `extent = 6 <= 8` 误判合法，导致 `4` 个元素越界写坏相邻 SMEM；新逻辑强制校验 `min + extent <= shape`（`12 <= 8` 为假），在编译期拦截越界。
- **TileLang `T.ws(-1)` 全线程区间修复 (`ee4151896d`)**：`src/ir.cc` 将 `T.ws(-1)` 转为 `threadIdx.x` 区间时误用 `[-128, 0)`，导致该区域内 `0` 个线程执行（所有写回被静默丢弃）；修复后当 `ws_idx < 0` 时不注入下界裁剪，全部 `256` 个线程正常执行。

---

## 3. 创新公司对标与架构师设计练习

- **创新公司路径对比**：
  - **Fireworks AI / Together AI**：在多租户长短混合请求批处理中，Mamba/Conv1d 层的文档长度偏斜极易拉高 P99 TTFT；采用 `cake_causal_conv1d` 的设备端原子 chunk 窃取与零残留 workspace 可直接嵌入 CUDA Graph，无需主机端重划分。
  - **Groq / Cerebras**：Groq LPU 与 Cerebras WSE 通过编译器静态确定每个 PE/切片的寄存器与 SRAM 边界，这与 TileLang 在 IR 层严格校验 `min + extent <= shape` 及 CUTLASS 显式管理 `setmaxnreg` 的确定性编译哲学高度一致。
- **架构师设计练习**：
  - 给定 SM103 每 SM `65,536` 个 32-bit 寄存器，若某个算子需要 `2` 个 consumer warpgroups（共 `256` 线程）和 `2` 个 producer warpgroups（共 `256` 线程），且 producer 每线程固定占用 `32` 个寄存器，求 consumer 每线程可分配的最大 8 倍数寄存器数。
  - **解答**：$(65536 - 256 \times 32) / 256 = 224$ regs/thread。
- **后续可证伪问题**：
  1. 当 `cake_causal_conv1d_train` 的 `chunk_tokens` 从 `256` 缩小到 `64` 时，L2 `atomicAdd` 争用开销是否会在极短序列（总 tokens $< 1024$）上抵消负载均衡收益？
  2. 在 SM103 上将 `cake_kimi_k3_attn_res` 扩展到 $K=8$ 时，`232` regs/thread 是否仍能维持零溢出，还是需要切换为两阶段 SMEM 累加？
