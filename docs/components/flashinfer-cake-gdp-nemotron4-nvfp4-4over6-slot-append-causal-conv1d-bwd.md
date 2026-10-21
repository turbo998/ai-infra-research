# FlashInfer Cake GDP Nemotron 4 解码/MTP 算子、NVFP4 4/6 槽位映射追加与 Cake 因果 Conv1d 反向路由剖析

> **关联上游提交（2026-10-21）**：
> - `flashinfer-ai/flashinfer` PR `#6883` (`fc1790f4ec33a4df84ae9e6d1dcc9231fde7fd10`): `feat: Add Cake Gated DeltaProduct (GDP) decode & MTP kernels for Nemotron 4 (#6883)`
> - `flashinfer-ai/flashinfer` PR `#6013` (`52fbbd090ca9c85b69c3a94420e360d27779b366`): `feat: support nvfp4_4over6 in slot-mapping paged KV cache append (#6013)`
> - `flashinfer-ai/flashinfer` PR `#6204` (`3821440929ee342fe9f939dbf5186c3e667a851c`): `perf(page): remove .cpu() stream sync from nvfp4 slot-mapping kv append (#6204)`
> - `flashinfer-ai/flashinfer` PR `#6930` (`7c866e85d3bb8095ebb247cdca42a56c2a4cab19`): `perf(causal_conv1d): update Cake backward channel routing for sm103a/sm107a (#6930)`

---

## 1. 核心架构问题与背景

在 **Nemotron 4** 等混合线性注意力架构（Gated DeltaProduct, GDP）与 **Blackwell / Rubin (`sm_100a` / `sm_103a` / `sm_107a`)** 推理/训练链路中，算子层面临三项紧迫瓶颈：

1. **多阶 Householder 状态更新解码与 MTP 验证访存放大**：Gated DeltaProduct 在每个时间步 $t$ 对状态矩阵 $S \in \mathbb{R}^{V \times K}$（$V=64, K=128$，FP32 存储）执行 $n \in \{1, 2, 3, 4\}$ 次顺序秩-1 Householder 增量修正。若回退至未融合递归算子或通用 GEMV，每次 Householder 子步均触发高额 SMEM/HBM 往返；而在小 Batch 单步解码（$T=1, B \times H_V \le 160$）与大 Batch 多 Token MTP 验证（$T \in [4, 16]$）下，单一 CTA 切分策略无法同时兼顾 SM 占用率与寄存器局部性。
2. **NVFP4 分页 KV Cache 追加中的 `[4.0, 6.0]` 量化空洞与 `.cpu()` 同步阻塞**：标准 NVFP4 E2M1 幅值格点为 $\{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$，将块最大值 $\text{amax}$ 对齐到 $6.0$（$sf_6 = \text{fp8\_e4m3}(\text{amax} / (6 \cdot g))$）会在 $[4.0, 6.0]$ 与 $[2.0, 3.0]$ 留出最宽量化步长；同时，非 CUDA Graph 预填充路径在每次调用 `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping` 时通过 `.cpu()` 检查 GPU 标量 `k_scale`/`v_scale`，导致每层一次主机-设备流同步。
3. **GB300 (`sm_103a`) 与 Rubin R200 (`sm_107a`) 上因果 Conv1d 反向通道分档失配**：`cake_causal_conv1d_train` 反向算子在 $C=512$ 中等通道段及 $C \ge 2048$ 宽通道段缺乏针对 `sm_103a` / `sm_107a` 共享内存带宽特性的 `fold_overlap`、`fast_roll` 与 `roll_prefetch` 组合分派。

---

## 2. 数学模型与核心设计机制

```
+--------------------------------------------------------------------------------------------------+
|  FlashInfer Cake GDP Decode/MTP (PR #6883) + NVFP4 4/6 KV Append (#6013/#6204) + Conv1d (#6930)  |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|  [1] Cake GDP Multi-Householder Recurrence (K=128, V=64, FP32 State Pool [pool, HV, 64, 128])    |
|      alpha_t = exp(-exp(A_log) * softplus(a_t + dt_bias)),  S <- alpha_t * S                     |
|      for j = 0 .. n-1:                                                                           |
|        k_hat = l2norm(k_{t,j}),  u = S @ k_hat,  S += (sigmoid(b_{t,j}) * (v_{t,j} - u)) (x) k_hat|
|      out_t = scale * (S @ l2norm(q_t))                                                           |
|      * T=1 & B*HV <= 160 -> splitv8_tile8 (8 CTAs/head, 8 V-rows/CTA)                            |
|      * T>=4 or B*HV > 160 -> lanerow (l8r4 / l8r2 / l4r4, 1 CTA/head, register-resident rows)    |
|                                                                                                  |
|  [2] NVFP4 4/6 Block Scale Selection (PR #6013) & Sync-Free Scale Check (PR #6204)               |
|      sf_6 = fp8_e4m3(amax / (6*g)),  sf_4 = fp8_e4m3(sf_6 * 1.5)  (maps amax onto E2M1 level 4.0) |
|      Select sf_4 iff MSE(sf_4) < MSE(sf_6) over 16 elements; compiled in page_nvfp4_4over6       |
|      Remove host .cpu() on CUDA k_scale/v_scale -> rely on in-kernel `trap;` when rows > 0       |
+--------------------------------------------------------------------------------------------------+
```

### 2.1 Cake GDP 多阶 Householder 状态递推与双核体路由（PR `#6883`）

对于每个批次槽位与价值头 $h \in [0, H_V)$（要求 $H = H_V$，无 GQA/GVA 广播），状态池张量形状为 `[pool_size, HV, 64, 128]`（FP32，16 字节对齐，步长对齐 `STATE_STRIDE_ELEMENTS`）。在时间步 $t \in [0, T)$ 上：

$$\alpha_t = \exp\!\big(-\exp(A_{\log}) \cdot \operatorname{softplus}(a_t + \text{dt\_bias})\big), \quad \beta_{t,j} = \sigma(b_{t,j})$$

首先对状态施加指数衰减 $S \leftarrow \alpha_t S$，随后顺序执行 $n \in \{1, 2, 3, 4\}$ 次 Householder 增量修正：

$$\hat{k}_{t,j} = \operatorname{l2norm}(k_{t,j}), \quad u_{t,j} = S \hat{k}_{t,j}, \quad S \leftarrow S + \big(\beta_{t,j}(v_{t,j} - u_{t,j})\big) \otimes \hat{k}_{t,j}$$

最后输出 $\text{out}_t = \text{scale} \cdot S \operatorname{l2norm}(q_t)$。当 `disable_state_update=True`（例如只读投机验证回滚探针）时，算子在寄存器中完成全部递推并写出精确的 $\text{out}_t$，但跳过向 `initial_state` 池的最终回写。

为了覆盖从单请求解码到大吞吐 MTP 验证的全尺寸范围，`cake_jit.py` 导出了两套核体（Body）：
- **行切分核体（`splitv` / `splitv8_tile8`）**：当 $T = 1$ 且 $B \times H_V \le 160$ 时启用，每个状态头分配 **8 个 CTA**（每个 CTA 负责 $64 / 8 = 8$ 行 $V$），将单步小 Batch 网格扩展 8 倍以铺满 Blackwell SM。
- **Lane 驻留行核体（`lanerow` / `l8r4`, `l8r2`, `l4r4`）**：当 $T \ge 4$ 或 $B \times H_V > 160$ 时启用，每个状态头由 **1 个 CTA** 独占，整个 $64 \times 128$ 状态块在 $T$ 个时间步及 $n$ 次 Householder 子步间常驻寄存器；对 $n=3, T \in \{1, 4, 8\}$ 提供静态编译循环展开程序，其余组合路由至支持 $1 \le T \le 16$ 的运行时程序。

### 2.2 NVFP4 `4/6` 块缩放选择与零主机同步追加（PR `#6013` / PR `#6204`）

在 16 元素 NVFP4 块量化中，PR `#6013` 同时评估两套 FP8 E4M3 缩放因子：

$$sf_6 = \operatorname{fp8\_e4m3}\!\left(\frac{\text{amax}}{6 \cdot g}\right), \quad sf_4 = \operatorname{fp8\_e4m3}(sf_6 \cdot 1.5)$$

由于 $1.5 = 6 / 4$，$sf_4$ 将原本落入 $[4.0, 6.0]$ 粗粒度区间的数值映射到 $\{2.25, 3.0, 4.5, 6.0\}$ 等更贴合次大值分布的重构格点。算子在寄存器内计算 16 个元素的重构均方误差 $\operatorname{MSE}(sf) = \frac{1}{16}\sum_{i=0}^{15}(\operatorname{mag}_i \cdot sf \cdot g - |x_i|)^2$，仅当 $\operatorname{MSE}(sf_4) < \operatorname{MSE}(sf_6)$ 时选用 $sf_4$。此外，通过将 `use_4over6=true` 拆分到独立 JIT 模块 `page_nvfp4_4over6`，默认 `page` 模块中的 **51 个内核保持 SASS 零变化**。

PR `#6204` 进一步消除了 `_validate_Positive_scale` 中对 CUDA 张量调用的 `float(scale.cpu())`：在 Eager 预填充模式下，每一层 KV 追加若调用 `.cpu()` 会强制 CPU 等待当前 CUDA 流排空；移除该主机同步并完全依赖内核首指令校验（当 `num_rows > 0` 且 `scale <= 0` 或非有限值时触发硬件 `trap;`），彻底消除了每层一次的流同步气泡。

### 2.3 Cake 因果 Conv1d 反向最高阈值路由（PR `#6930`）

`cake_causal_conv1d_train` 按 `channels_at_least` 降序列表匹配首个满足 `C >= channels_at_least` 的反向内核配置：
- **`sm_103a` (GB300)**：新增 `channels_at_least=512`（启用 `fold_overlap=True, fast_roll=True`，`SMEM_TOTAL=66560 B`）与 `channels_at_least=768`（恢复默认配置），精准加速 $512 \le C < 768$ 频段而不干扰 $C \ge 768$。
- **`sm_107a` (Rubin R200)**：为 $512 \le C < 1024$ 启用 `fold_overlap=True, fast_roll=True`（`66560 B`），为 $C \ge 2048$ 启用 `fast_roll=True, roll_prefetch=True`（`SMEM_TOTAL=82944 B`）。

---

## 3. 确定性实验与量化基准验证

运行确定性复现实验 `python experiments\flashinfer_cake_gdp_nvfp4_4over6_conv1d_bwd\run.py --check` 得到如下基准数据：

| 评估维度 | 工作负载 / 硬件平台 | 基线表现 | 优化后表现 | 关键量化收益 |
|---|---|---|---|---|
| **Cake GDP Decode / MTP (`#6883`)** | Nemotron 4 48 组形状 (`HV∈{80,20,10,24}, B∈{1..256}, T∈{1,4,8}`) @ **GB200 (`sm_100a`)** | `301.4 us` (`HV=10, B=256, T=8`) | **`75.5 us`** (`l8r2` 静态 $T=8$) | **48/48 全胜 (`1.01x–3.99x`)** |
| **Cake GDP Decode / MTP (`#6883`)** | Nemotron 4 48 组形状 @ **GB300 (`sm_103a`)** | `296.6 us` (`HV=10, B=256, T=8`) | **`75.1 us`** (`l8r2` 静态 $T=8$) | **47/48 胜出 (`0.98x–3.95x`)** |
| **NVFP4 `4/6` 块量化 (`#6013`)** | 16 元素次峰值聚集块 (`±4.5 / ±2.25` 密集分布) | `MSE = 0.141919` (`sf6 = 1.0`) | **`MSE = 0.000494`** (`sf4 = 1.5`) | **块重构 MSE 降低 `99.65%`** |
| **NVFP4 零同步追加 (`#6204`)** | SM80 vLLM `Qwen3.6-35B-A3B-NVFP4` (20 层, 4096 Tokens Eager Prefill) | `499.4 ms` (`495.5–516.0 ms`) | **`451.7 ms`** (`447.5–464.5 ms`) | **`-47.7 ms` (`-9.55%`, `1.1056x`)** |
| **Cake Conv1d 反向路由 (`#6930`)** | **GB300 (`sm_103a`)** 8 组 `C=512` 形状 / **R200 (`sm_107a`)** 11 组路由形状 | `101.3 us` (`R200 B=1,S=250340,C=512`) | **`96.4 us`** (`fold_overlap + fast_roll`) | **GB300 `1.0164x` / R200 `1.0331x` (峰值 `1.0513x`)** |

---

## 4. 生产落地指南与适用边界

- **适用场景**：Blackwell (`sm_100a` / `sm_103a`) 上的 Nemotron 4 混合 Gated DeltaProduct 解码与投机 MTP 验证、对精度敏感的长上下文 NVFP4 分页 KV Cache 追加，以及 GB300 / R200 混合 SSM-Attention 模型的 `causal_conv1d` 反向训练。
- **不适用 / 边界条件**：`cake_gated_delta_product_mtp` 当前固定要求 $K=128, V=64, H=H_V, 1 \le T \le 16, n \in \{1..4\}$ 且 `initial_state` 为 16 字节对齐的 FP32 池；若需要变长预填充（$T > 16$）或 GQA/GVA 头广播，应走分块（Chunked）预填充路径。
