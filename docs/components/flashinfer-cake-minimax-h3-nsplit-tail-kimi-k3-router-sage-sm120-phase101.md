# FlashInfer Cake MiniMax-H3 N-Split 尾波拆分、Kimi-K3 Fused Router Arm M 驻留 CTA 守卫、SM120 Sage 稀疏 Attention 与 TileLang/CuTeDSL 编译不变量分析（Phase 101）

- **日期 / 阶段**：2026-10-27 · Phase 101 (`priority: 110`)
- **组件归属**：`gpu-software-stack`
- **关联问题 ID**：`flashinfer-cake-minimax-h3-nsplit-tail-kimi-k3-router-sage-sm120-phase101`
- **关联证据 ID**：`claim-flashinfer-minimax-h3-tail-kimi-k3-router-sage-sm120-phase101`
- **可复现实验套件**：[`experiments/flashinfer_minimax_h3_tail_kimi_router_phase101/`](../../experiments/flashinfer_minimax_h3_tail_kimi_router_phase101/run.py)

---

## 1. 架构背景与核心问题

在 Blackwell (`sm_100a` / `sm_103a` / `sm_120a`) 混合精度算子栈与 DSL 编译器演进中，工程落地存在四类关键挑战：

1. **持久化块缩放 GEMM 的尾波（Tail Wave）SM 空转开销**：在 MiniMax-H3 注意力输出投影（融合索引门控与残差，$N = 5376 = 21 \times 256$）中，当行分块对与 21 个列分块组成的最后一波仅占驻留波容量的一半以内时，按整块（$256 \times 256$）发射会导致半数以上 CTA 闲置等待一整块周期。
2. **MIG 切片或小规模 GPU 下固定专家归属路由器的静默计划漏写**：Kimi-K3 融合 MoE 路由器（896 专家，Top-16）的 Arm M 为前 128 个计划所有者（plan-owner）CTA 静态分配固定 7 个专家（$7b \dots 7b+6$）。当设备或 MIG 切片的持久化网格上限低于 128 个 CTA 时，高编号专家的计数与偏移量完全未被写入却未抛错。
3. **SM120 块稀疏 Sage Attention 在线重缩放投票（Rescale Vote）流水气泡**：在 `sm_120a`（RTX PRO 6000 / RTX 5090）上，`bsa_attn_sm120_blk64_sage_fwd` 执行 INT8 $QK^T$ 与 FP8 (E4M3) $PV$，若等待重缩放跨线程投票完成后再计算 $P$，会暴露寄存器依赖延迟。
4. **TileLang 与 CuTeDSL 编译期的静默降级陷阱**：TileLang 在无物理 GPU 主机上对无架构 CUDA target 静默回退到 `sm_50`，并对不支持随机舍入（`cvt.rs`）的转换组合（如 `f32 -> E8M0`）静默丢弃 `round="rs"` 注解；CuTeDSL 在 `KEEP=ir-debug` 与 `KEEP=ir` 共存时覆盖了未做 Canonicalize/CSE 前的原始 MLIR。

---

## 2. 一手源码实证分析（10 个零重叠上游提交）

### 2.1 FlashInfer `#7363` (`bf82f4d797`)：MiniMax-H3 BF16 与 MXFP8 Out-Projection N-Split 尾波拆分

`flashinfer.diffusion_ops.minimax_h3_out_proj`（BF16）与 `minimax_h3_out_proj_mxfp8`（MXFP8 W8A8，E4M3 数据 + UE8M0 block-32 缩放因子）在 `sm_100a`（GB200）和 `sm_103a`（GB300）上引入与 NVFP4 变体一致的 **N-split 半宽尾块调度**：

- **触发条件**：
  - **BF16**：行数 $M \ge 16$，总波数在 `sm_103a` 上 $\le 10$（`sm_100a` 上 $\le 9$），且最后一波为不超过半波容量（$\le 38$ 个 2-CTA 分块对）的部分波。
  - **MXFP8**：行数 $M \ge 129$，总波数在 `sm_103a` 上 $\le 26$（`sm_100a` 上 $\le 17$），且满足相同半波条件。
- **尾块拆分机制**：将尾波中的 $256 \times 256$ 输出分块拆分为两个 **$256 \times 128$ 半宽输出列单元**，使最后半波负载均匀铺满全部驻留 CTA 对，尾波耗时降至约半个整块周期，且输出与整块程序 **逐比特完全一致（bitwise-identical）**。
- **实测性能（GB300 `sm_103a` / GB200 `sm_100a`，CUDA 13.4，冷 L2 10 轮中位数）**：
  - **GB300 BF16**（7 个命中形状）：加速 **`1.012x–1.041x`**，几何平均 **`1.0343x`**（报告 `1.034x`），在 $P=8, M=4832$ 从 `227.4 µs` 降至 `218.6 µs`（**`1.041x`**）。
  - **GB300 MXFP8**（17 个命中形状）：加速 **`1.018x–1.057x`**，几何平均 **`1.0325x`**（报告 `1.032x`），在 $P=8, M=4825$ 从 `144.7 µs` 降至 `136.9 µs`（峰值 **`1.057x`**）。

### 2.2 FlashInfer `#7365` (`e18eb1d2ee`)：Kimi-K3 Fused Router Arm M 128 驻留 CTA 边界守卫

在 `flashinfer/experimental/kimi_k3_fused_router/cake_backend.py` 中：
- 专家总数 `NUM_EXPERTS = 896`，每 CTA 线程数 `THREADS = 224`（`NUM_WARPS = 7`），因此计划所有者 CTA 数为：
  $$\text{OWNER\_CTAS} = \frac{896}{7} = 128$$
- **Arm M**（处理 `sm_100a` 上 $129 \dots 256$ tokens、`sm_103a` 上 $129 \dots 2048$ tokens）将专家 $7b \dots 7b+6$ 固定绑定给 CTA $b < 128$，无跨 CTA 步进；而 **Arms L / LP / GW** 按活跃所有者数量跨专家步进（stride over experts），任意网格大小均可完整发布 896 个专家计划。
- **修复前缺陷**：当设备或 MIG 切片的持久化网格容量 $\text{cap} = c_{\text{SM}} \times N_{\text{SM}} < 128$ 时（例如 `cap = 96`），Arm M 不会写入专家 $672 \dots 895$ 的计数、偏移与路由块。
- **修复后守卫**：在 `launch_grid` 中显式校验 `cap >= 128`，从而在 **CC 10.0 低于 43 SMs**（$42 \times 3 = 126 < 128$）或 **CC 10.3 低于 32 SMs**（$31 \times 4 = 124 < 128$）时立即抛出 `RuntimeError("... co-resident owner CTAs")`。

### 2.3 FlashInfer `#7364` (`967a8ed858`)、`#7405` (`4482b6b06e`)、`#7465` (`46d67ef62f`)

- **SM120 块稀疏 Sage Attention 前半 $P$ 提前计算 (`#7364`)**：`bsa_attn_sm120_blk64_sage_fwd`（INT8 Q/K `mma.sync` + FP8 E4M3 V，64-token 分块，5 个元数据标志特化为 **20 个程序**）将 $P$ 前半部分的计算提前到重缩放投票（rescale vote）之前发射，在 RTX PRO 6000 Blackwell (`sm_120a`, 188 SMs) 全部 8 个形状上实现 **`1.0031x–1.0100x`** 加速且逐比特一致。
- **SM120 BF16 MoE 非融合 Combine 位级可重复性 (`#7405`)**：融合 MoE epilogue 使用原子加法（atomicAdd），即使在 CUDA Graph 重放下到达顺序仍会改变输出末位比特；PR `#7405` 完善了预规划非融合 combine 在污染输出/scratch/路由元数据后的 Eager 与 Graph 重放逐比特一致性验证。
- **CPU-Only 测试重定位共享框架 (`#7465`)**：提供无 GPU 环境下的算子元数据与主机逻辑验证基础设施。

### 2.4 CUTLASS `#3747` / `#3610` / `#3702` 与 TileLang `#3350` / `#2860`

- **CUTLASS 4.8.1 & CuTeDSL `MLIR_RAW` 源码位置保留 (`#3747`, `#3610`, `#3702`)**：`#3702` 在 `JitFunctionArtifacts` 中新增 `MLIR_RAW` 保存 Canonicalize/CSE 前的原始 IR，并在克隆模块生成 clean IR 时保留源码位置信息（source locations）。
- **TileLang `cvt.rs` 转换三元组校验 (`#3350`, `e84ce3de8a`)**：在 `CodeGenTileLangCUDA::VisitExpr_(const CastNode*)` 分支派发前强制校验 `(source dtype, destination dtype, round)`，仅允许 `f32` 源 + 32-bit `rbits` + 合法打包目标使用 `round="rs"`，拒绝 `f32 -> E8M0`（向 $+\infty$ 舍入）等静默忽略舍入注解的非法组合。
- **TileLang 拒绝无架构裸 `cuda` Target (`#2860`, `f1a34efea6`)**：在 `_cuda_target_from_arch` 统一拦截缺失架构的 `"cuda"` target，防止 TVM 在无 GPU 机器上静默填充 `arch=sm_50` 导致错误指令集编译。

---

## 3. 确定性实验结果总结

运行 `python experiments/flashinfer_minimax_h3_tail_kimi_router_phase101/run.py --check` 得到以下密封实验指标：

| 评估维度 | 核心指标 | 实测 / 仿真结果 |
|---|---|---:|
| MiniMax-H3 BF16 (GB300 `sm_103a`) | 7 个尾波拆分形状几何平均加速 / 峰值加速 | **`1.0343x`** / **`1.041x`** |
| MiniMax-H3 MXFP8 (GB300 `sm_103a`) | 17 个尾波拆分形状几何平均加速 / 峰值加速 | **`1.0325x`** / **`1.057x`** |
| Kimi-K3 Fused Router Arm M | 必需驻留 Owner CTA 数 / 最小 SM 数 (`sm_100a` / `sm_103a`) | **`128 CTAs`** / **`43 SMs` / `32 SMs`** |
| Kimi-K3 Arm M 未守卫 96-CTA 切片 | 漏写专家区间 / 未写入专家数 | **`672..895`** (`224` 个专家漏写) |
| SM120 Sage 块稀疏 Attention | 特化程序数 / 8 形状冷 L2 加速区间 | **`20` 个程序** / **`1.0031x–1.0100x`** |
| TileLang `cvt.rs` 三元组守卫 | 合法 `cvt.rs` 组合 / 拦截静默降级非法组合 | **`3` 放行 / `3` 拦截** |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **MIG 切片容量核算练习**：当在 B200 (`sm_100a`, 3 CTAs/SM) 或 GB300 (`sm_103a`, 4 CTAs/SM) 上将 GPU 切分为 32-SM MIG 实例时，计算 `persistent_grid_cap`，并说明为何 Kimi-K3 路由器在 $M=256$ 时必须在 `sm_100a` 上回退到跨专家步进的 Arm GW，而在 `sm_103a` 上刚好满足 `32 * 4 = 128` 驻留阈值。
2. **尾波拆分盈亏边界问题**：为何 MiniMax-H3 MXFP8 的尾波拆分上限（`sm_103a` 上 $\le 26$ 波）显著高于 BF16（$\le 10$ 波）？（提示：对比 W8A8 单波计算耗时与半宽 $256 \times 128$ 分块调度开销的摊销比例。）
