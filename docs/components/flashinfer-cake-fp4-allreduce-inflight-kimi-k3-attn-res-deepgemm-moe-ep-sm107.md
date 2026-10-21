# FlashInfer Cake FP4 AllReduce 在途 Scale 与共享内存复用、Kimi-K3 Attention-Residual Round-2、Cake DeepGEMM 与 SM107 MoE-EP 原生编译深挖

> **研究阶段**: Phase 47 (`priority: 56`) · **问题编号**: `q-phase47-flashinfer-cake-fp4-allreduce-inflight-kimi-k3-attn-res-deepgemm-moe-ep-sm107` · **确定性复现脚本**: [`experiments/flashinfer_cake_fp4_allreduce_inflight_kimi_k3_deepgemm_moe_ep_sm107/run.py`](../../experiments/flashinfer_cake_fp4_allreduce_inflight_kimi_k3_deepgemm_moe_ep_sm107/run.py)

---

## 1. 核心架构问题与一手提交总览

在 NVIDIA Blackwell (`sm_100a` B200 / `sm_103a` B300 / GB300) 与新一代 `sm_107a` (GR100) 架构上，FlashInfer 在 2026-10-21 落地了六项关键底层内核与工具链演进：

1. **`cake_fp4_allreduce` TP2 融合 RMSNorm 在途 Scale 重叠与 TP4 Two-Shot 共享内存 Value 复用（`#6875`）**：在 TP2 融合 AllReduce + Residual Add + Gemma RMSNorm (`BF16` 输出) 内核中，原先在进入 Pass-1 数据包扫描循环前会阻塞自旋等待远端 Peer 的每行 Scale Word (`os_rs + m_row`, 哨兵值 `0x80000000u`)；同时在 TP4 Two-Shot 路径中，Pass-2 在写出全局归约结果时会重新从 L2 Lamport 缓冲区读取 Pass-1 已经验证过的 4 路 FP4 Value 数据包。
2. **`cake_kimi_k3_attn_res` Round-2 全量 Vendorable 内核包（`#6904`）**：针对 Kimi-K3 Attention-Residual 算子（`hidden_size = 7168`，历史深度 `K in [0, 8]`），需要将残差相加、可选快照写回、两路输入 RMSNorm、`K+1` 候选点积打分、Softmax、加权混合与可选输出 RMSNorm 融合为单内核，并覆盖 `291 × 2 PDL = 582` 个契约形状。
3. **自包含 Cake DeepGEMM 批量投影 / FP8 1D1D 训练内核、NVFP4 持久注意力与 `sm_107a` (GR100) MoE-EP 合并（`#6928`, `#6927`, `#6902`, `#6621`）**：将原本依赖外部 JIT 构建的 DeepGEMM 批量投影、FP8 1D1D FProp/WGrad 与 NVFP4 Attention 固化为自包含预编译包，并修复 `flashinfer.moe_ep` 在 GR100 (`SM107`) 上因 `sm_100f` 缺少 CUTLASS `Sm107` 架构守卫而触发的 `Mercury Validator: Validation failed for SM 107` 致命错误。

| 仓库与提交 | 日期 | 核心架构变更 | 实测性能 / 正确性收益 |
|---|---|---|---|
| `flashinfer@c7d1eac` (`#6875`) | 2026-10-21 | `cake_fp4_allreduce` TP2 在途 `os_rs` Scale 读取 + `j0==0` Warp 投票；TP4 Pass-2 共享内存 Value 复用 | `sm_100a/103a/107a` NVFP4 几何平均 **`1.018x / 1.018x / 1.023x`**；GB300 `1Kx6144` BF16 最高 **`1.0384x`**（`18.15 -> 17.47 us`），TP4Pass-2 L2 Value 流量降低 **`80%`** |
| `flashinfer@3b65505` (`#6904`) | 2026-10-21 | `cake_kimi_k3_attn_res` Round-2（`133` 程序，`161` 库：`75` `sm_100a` + `86` `sm_103a`）覆盖 `582` 契约行 | GB300 几何平均 **`1.0072x`**（`serving_m5_k5` **`1.0888x`**，`4.448 -> 4.096 us`）；B200 `k_sweep_m1_k6` 达 **`1.1034x`** |
| `flashinfer@595bc57` (`#6928`) | 2026-10-21 | `cake_deepgemm_batched_projection_gemm` 6 变体（`bhr,hdr->bhd`, `K=4096, N=1024`, Swapped-AB `M=16`） | GB300 `T=2` Swapped-AB `M=16` 达 **`9.727 us`**（较 `M=64` `10.663 us` 降低 **`8.78%`**） |
| `flashinfer@597e3a6` (`#6927`) & `edfb3f7` (`#6902`) | 2026-10-21 | `cake_deepgemm_fp8_gemm_1d1d` FProp/WGrad 与 `cake_nvfp4_attention` 3-Stage 持久内核 | GB300 `4096x7168x4096` FProp/WGrad **`66.801 / 81.440 us`**；NVFP4 Attention `B=4,H=8,S=4096` **`114.161 us`** |
| `flashinfer@9d3837c` (`#6621`) | 2026-10-21 | 合并 `flashinfer.moe_ep` 并为 GR100 新增原生 `sm_107a` 编译目标 | 消除 GR100 (`SM 10.7`) 上 `sm_100f` 引发的 `Mercury Validator` 校验失败 |

---

## 2. `cake_fp4_allreduce`：TP2 在途 Scale 重叠与 TP4 Two-Shot 共享内存复用 (`#6875`)

```
  TP2 Pass-1 In-Flight Scale Overlap (cake_fp4_allreduce_tp2_rmsnorm_bf16_bf16_kernel.cu):
  [Before #6875]  Spin-wait os_rs != 0x80000000u  --->  Sweep FP4 value/scale packets (j0=0..J-1)
  [After  #6875]  Single ld.volatile rsw[0] ------+-->  Sweep j0=0 packets & check -0.0 nibble sentinel
                                                  +-->  if (j0==0 && __any_sync(0xFFFFFFFF, rsw[0]==0x80000000u))
                                                            spin-wait os_rs until settled before dequant!
```

1. **FP4 Wire Nibble `-0.0` (`0x8`) 哨兵检测器**：在 Lamport 通信缓冲区中，32 位打包字 `_fp4_wire_values_0[i]` 包含 8 个 E2M1 FP4 Nibble，其中 `-0.0` (`0b1000 = 0x8`) 充当未到达哨兵。内核使用无分支位运算检测是否存在任意等于 `0x8` 的 Nibble：
   $$\text{pending} = \big(w \ \& \ \sim((w \ \& \ \text{0x77777777}) + \text{0x77777777})\big) \ \& \ \text{0x88888888}$$
   当且仅当某 Nibble 高位为 `1` 且低 3 位全为 `0`（即 `0x8`）时，`+ 0x7` 不会向 bit 3 进位，从而被精确捕获。
2. **TP2 Pass-1 在途 Scale 读取**：内核不再在循环外阻塞等待 `os_rs + m_row`，而是先执行一次非阻塞 `ld.volatile.global.u32` 读入 `rsw[0]`，紧接着在 `j0 == 0` 完成了首轮 FP4 Value 与 FP8 Scale (`0xFF` 哨兵) 到达校验后，通过 `__any_sync(0xFFFFFFFF, rsw[0] == 2147483648u)` 检查是否仍有 Lane 未就绪，仅在未就绪时才进入短循环补读。
3. **TP4 Two-Shot Pass-2 共享内存 Value 复用**：Pass-1 已经将 4 个 Rank 的 FP4 Value 数据包（每步 `4 × 16B = 64B`）验证并存入共享内存 `smem_u`。`#6875` 在 Pass-2 中直接从 `smem_u` 复读这 `64B` Value，仅从 L2 Lamport 缓冲重读 `4 × 4B = 16B` Scale Word，将 Pass-2 的 L2 读取字节数削减 **`80.0%`**（`80B -> 16B`）。

---

## 3. `cake_kimi_k3_attn_res` Round-2 全量算子融合 (`#6904`)

对于每一行 $i \in [0, M)$ 与历史深度 $K \in [0, 8]$（隐藏维度 $D = 7168$），算子在单次内核启动中完成：
$$u_i = \text{hidden}_i + \text{residual}_i, \quad v_{K,i} = \text{RMSNorm}(u_i, w_u), \quad v_{k,i} = \text{RMSNorm}(R_{k,i}, w_r) \ (k < K)$$
$$s_{k,i} = \langle q, v_{k,i}\rangle, \quad p_{k,i} = \frac{\exp(s_{k,i} - \max_{j} s_{j,i})}{\sum_{j=0}^{K} \exp(s_{j,i} - \max_{m} s_{m,i})}, \quad h_i = \sum_{k=0}^{K-1} p_{k,i} R_{k,i} + p_{K,i} u_i$$
并按需直接输出 $\text{RMSNorm}(h_i, w_{\text{out}})$。Round-2 共打包 `133` 个程序与 `161` 个预编译库（`75` 个 `sm_100a` + `86` 个 `sm_103a`），在 GB300 上使 `serving_m5_k5` 从 `4.448 us` 降至 `4.096 us`（**`1.0888x`**），在 B200 上使 `k_sweep_m1_k6` 提速 **`1.1034x`**。

---

## 4. 自包含 Cake DeepGEMM / NVFP4 Attention 与 `SM107` MoE-EP (`#6928`, `#6927`, `#6902`, `#6621`)

- **Swapped-AB `M=16` 批量投影 GEMM（`#6928`）**：针对小 Token 步长 `T in {1, 2, 4}` 的多头投影 `einsum('bhr,hdr->bhd')`（`H=128, K=4096, N=1024`），当 `T <= 4` 时通过交换 A/B 操作数启用 `M=16` Tile，使 GB300 `T=2` 延迟从 `10.663 us` 降至 **`9.727 us`**（降低 **`8.78%`**）。
- **GR100 (`SM107`) 原生 `sm_107a` MoE-EP 支持（`#6621`）**：将 `flashinfer.moe_ep` 统一合并为单一模块，并在 JIT 编译器中显式为 `major == 10, minor == 7` (GR100) 生成 `sm_107a` 编译标志，替代原先因缺失 CUTLASS `Sm107` 分支而回退 `sm_100f` 导致的 `Mercury Validator` 校验失败。
