# FlashInfer KCP 反向 Carry Plane 复用、QSA NVFP4 稀疏解码注意力、MegaMHC Launch Epochs 护栏与 TileLang 向量化/原子安全剖析

> **双语导航 / Bilingual Navigation**: [中文版](flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang.md) | [English Version](flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang.en.md)
> **证据锚点 / Evidence IDs**: `src-flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang-2026` · `ev-flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang-2026` (`priority: 73`, Phase 64 Track A)
> **可复现实验 / Reproducible Experiment**: `python experiments/flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang/run.py --check`

---

## 1. 架构师问题定义与工程背景

在长序列线性注意力（KDA Context-Parallel 训练）、NVFP4 稀疏分页解码注意力（QSA）、以及基于持久化计数器的 Fused MegaMHC 算子中，系统架构师面临四类极具代表性的底层性能与正确性权衡：

1. **KDA Context-Parallel (KCP) `boundary_mode="saved"` 的反向冗余重走（Re-walk）开销**：当单个 Rank 上的长序列片段被切分为 $P > 1$ 个分片（Segments）以跑满 SM Wave 时，前向传播已经计算出各分片的 FP32 Carry 状态平面。若反向传播丢弃该平面，则必须在反向阶段重新启动 K1 `_offset` 分片遍历核与 Carry 归并核。
2. **NVFP4 稀疏分页 GQA 解码（Cake QSA）的寄存器解量化与 L2 驻留瓶颈**：在 H100 (`sm_90a`, $H_Q=24, H_{KV}=2, D=256, \text{top-}k=2051$) 上，读取 E2M1 KV Cache 与 E4M3 块缩放因子时，V 提前加载与多指令解量化会加剧寄存器压力与 L2 抖动。
3. **MegaMHC `launch_epochs` 跨 Plan 共享时的 CTA Grid 错位与视图污染**：MegaMHC 使用按活跃 CTA 槽位递增的 `uint64[sms]` `launch_epochs` 缓冲区实现免全局屏障同步；若两个具有不同启动网格（如 64 CTAs vs 148 CTAs）的 Plan 共享同一 `launch_epochs` 存储，或传入非零偏移的切片视图，将直接破坏 CTA 纪元单调性。
4. **TileLang 编译器 `VectorizeLoop` 非单位步长静默错编与 CUDA `AtomicMax`/`AtomicStore` 符号截断崩溃**：TIR 循环向量化器仅接收 `(loop_var, extent, body)`，若不拦截 `step != 1` 会把稀疏步长循环错编为密集连续访问；同时 CUDA 原子模板将有符号 `int64_t` 强转为无符号类型会导致负数比较翻转。

---

## 2. 核心源码变更与数学建模

### 2.1 FlashInfer Cake KDA Context-Parallel 反向 Carry Plane 复用 (`#7117`, `ddc9524`)

在 `flashinfer/experimental/cake_kda_train/cake_kcp.py` 与 `cake_kcp_plan.py` 中，当 Rank 的分段数 $P > 1$ 且处于 `boundary_mode="saved"`（默认 `SAVED_CARRY_DEFAULT = 1`）时：
- **前向保存集扩展**：`KcpSaved` 新增 `carry` 平面，保存前向计算的每分片 FP32 Carry 状态；
- **反向只读绑定**：反向传播直接只读绑定 `KcpSaved.carry` 来重放 `fwdh_tape` / `dhu_tape`，**彻底跳过 K1 `_offset` 每分片重走核（Re-walk）与 Carry 合并核启动**，且与重走路径（`_kcp_test_knobs={"saved_carry": 0}`）保持逐位一致（bitwise identical）。
- **显存增量公式**：设每个 Rank 的片段数为 $n_{\text{frag}}$，每片段分片数为 $P$，跨 Rank 传入/传出状态链数为 $n_{\text{in}}, n_{\text{out}}$，注意力头数 $H = 64$，单头 FP32 状态瓦片字节数 $\text{STATE\_TILE\_BYTES} = 128 \times 128 \times 4 = 65536\text{ B}$：
  $$\text{plane\_slots} = n_{\text{frag}} \cdot (P - 1) + \max(n_{\text{in}}, n_{\text{out}})$$
  $$\Delta M_{\text{carry}} = \begin{cases} \text{plane\_slots} \times H \times 65536\text{ B}, & P > 1 \\ 0, & P = 1 \end{cases}$$
  在 GB300 (`sm_103a`, $H=64$) 评测矩阵（Q1–Q6, CP2/4/8/16）中，$\Delta M_{\text{carry}}$ 为 **12.0 MiB – 80.0 MiB / Rank**，且仅抬升前向稳态显存，不增加反向峰值显存。
- **GB300 实测性能（作者基准 `author-benchmark`）**：在 144 组配置（2 个独立节点会话 s1/s2 交叉验证）中，B3（前向 + 重放 + 反向）加速比在 s1 达到 **`0.9990x – 1.0372x`**、s2 达到 **`0.9950x – 1.0387x`**；其中 **34 / 144** 组在双会话均提速 $>2\%$，**101 / 144** 组保持在 `0.98–1.02x` 之间，**0 组**在双会话退化超过 $2\%$。代表性 8 组 B3 配置几何平均加速比为 **`1.0285x` (s1) / `1.0284x` (s2)**。

### 2.2 FlashInfer Cake QSA NVFP4 稀疏分页 GQA 解码优化 (`#7103`, `34d28b3`)

在 `flashinfer/experimental/cake_qsa_nvfp4`（对应 `flashinfer.sparse.qsa_sparse_paged_attention(..., backend="cake")`，覆盖 `NHD`/`HND` $\times$ Fused/Split 4 个变体）中：
1. **Load K First & Restore `evict_first`**：优先发起 K 页加载以尽早计算 $Q \cdot K^T$ 稀疏注意力分数，并为流式页读取恢复 L2 `evict_first` 缓存提示，防止 2051 个 Top-K 稀疏页冲刷高频 Q/状态行；
2. **`1-prmt` 单指令字节置乱解量化**：用单条 PTX `prmt.b32` 替代多步掩码移位，加速 E2M1 + E4M3 块缩放向 FP32 累加器的解量化；
3. **向量化 `int2` 输出写出**：将末尾 `st.global.v2.b32` 内联汇编统一规整为对齐的 `reinterpret_cast<int2*>(out + out_offset)[0]`。
- **H100 SXM (`sm_90a`) 实测结果（`author-benchmark`）**：在 $H_Q=24, H_{KV}=2, D=256, \text{top-}k=2051, \text{page\_size}=16$ 的 16 组形状中，全矩阵几何平均提速 **`1.0402x`**；在中等批大小 `rows = 16 / 32 / 64` 上几何平均提速 **`1.0852x`**，其中 `rows = 32` 在 NHD（`36.42 us -> 32.32 us`）与 HND（`38.83 us -> 34.45 us`）均达到 **`1.127x`** 峰值加速。

### 2.3 FlashInfer Cake MegaMHC SM107 残差流缓存策略与 `_claim_launch_epochs` 护栏 (`#7113`, `85e886c`)

1. **SM107 残差流动态 L2 Evict-First 策略与 2D TMA**：在 `csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_f24c2f3f98bfe3356398_kernel.cu` 中移除冗余 `tma_3d_gmem2smem`，统一采用 `tma_2d_gmem2smem`，并根据 Token 瓦片数动态切换 L2 `io_policy`：
   $$\text{io\_policy} = \begin{cases} \text{0x12f0000000000000}\ (1364590687093260288), & \lceil N_{\text{tok}} / 64 \rceil \le 64 \\ \text{0x1000000000000000}\ (1152921504606846976), & \lceil N_{\text{tok}} / 64 \rceil > 64 \end{cases}$$
   同时为 SM107 隐藏层 $H=4096$ BF16 程序引入原子尾标（atomic tail flag）消除尾块排空等待。
2. **`_claim_launch_epochs` 弱引用网格护栏**：在 `flashinfer/experimental/deepgemm_mega_mhc/mega_mhc.py` 中，通过 `_EPOCH_GRIDS`（以 `(storage.data_ptr(), nbytes, device.index)` 为键）追踪所有存活 Plan（`weakref.ref`）：
   - 强制校验 `launch_epochs.storage_offset() == 0` 且 `storage.nbytes() == nbytes`，拒绝任何切片或偏移视图；
   - 强制同一存储上的所有存活 Plan 必须具有完全相同的活跃 CTA 数 `grid`，否则立即抛出 `ValueError("... launching {grid} CTAs; this plan launches {sms}")`。

### 2.4 TileLang `VectorizeLoop` 非单位步长拦截 (`#3345`, `cb26ee0`) 与 CUDA 原子类型保留 (`#3307`, `2573d47`)

1. **`VectorizeLoop` 步长校验**：`src/transform/vectorize_loop.cc` 在调用 `TLVectorizer::Vectorize(op->loop_var, op->extent, op->body)` 前新增：
   ```cpp
   ICHECK(!op->step.has_value() || is_one(op->step.value()))
       << "VectorizeLoop does not support a vectorized loop with a non-unit step, got step="
       << op->step.value();
   ```
   防止 `step > 1` 在向量化降级时被静默丢弃而访问错误的连续内存。
2. **CUDA `AtomicMax`/`AtomicMin`/`AtomicStore` 精确类型保留**：`src/tl_templates/cuda/atomic.h` 修复了 `CUDART_VERSION >= 11080` 下将 `int64_t` 强转为无符号 `NT1`（导致 `AtomicMax(-5, 3)` 错误返回 `-5`）以及 `AtomicStore` 在 `int64`/`bf16`/`fp16` 上因目标类型不匹配崩溃的问题，直接使用 `cuda::atomic_ref<T1, cuda::thread_scope_device>`。

---

## 3. 确定性实验与验证结果 (`own-offline` + `author-benchmark`)

运行 `python experiments/flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang/run.py --check` 生成并校验如下关键指标：

| 子系统 / 指标 | 修复前 / 基线 | 修复后 / 当前实现 | 架构收益 / 验证结论 |
|---|---|---|---|
| **KCP B3 反向延迟** (`Q4, CP16, contiguous`, GB300) | `9.794 / 9.788 ms` | **`9.443 / 9.423 ms`** | **`1.0372x / 1.0387x`** 峰值加速（跳过 `_offset` 重走） |
| **KCP Carry Plane 显存开销** ($H=64$, GB300) | `0 MiB` (每次反向重算) | **`12.0 – 80.0 MiB / Rank`** ($P>1$), `0 B` ($P=1$) | 用可控前向保存集换取 144 组 B3 零退化提速 |
| **QSA NVFP4 Decode** (`rows=32`, NHD / HND, H100) | `36.42 us` / `38.83 us` | **`32.32 us` / `34.45 us`** | **`1.127x`** 峰值加速；`rows=16..64` 几何平均 **`1.0852x`** |
| **MegaMHC `_claim_launch_epochs`** | 允许跨 Grid 混用或切片视图 | **拒绝异构 Grid (`64 vs 148`) 与非零偏移视图** | 消除多 Plan 共享纪元时的 CTA 计数器静默踩踏 |
| **TileLang `VectorizeLoop` & `AtomicMax`** | `step=2` 静默丢弃；`AtomicMax(-5, 3) -> -5` | **编译期拒绝 `step=2`；`AtomicMax(-5, 3) -> 3`** | 阻断稀疏步长错编与有符号 64 位原子翻转 |

---

## 4. 架构师设计练习与可证伪问题

1. **显存-算力权衡计算**：在 GB300 NVL72 上训练 $L=60$ 层 KDA 模型，若每层 KCP 在 $P=3, n_{\text{frag}}=2, \max(n_{\text{in}}, n_{\text{out}})=1$ 下开启 `saved_carry=1` 且不跨越激活检查点边界，单 Rank 需额外常驻多少 MiB 显存？（答案：$\text{plane\_slots} = 2 \times 2 + 1 = 5$，单层 $5 \times 4\text{ MiB} = 20\text{ MiB}$；若激活检查点仅在单层内生效（如 B3 模式），则反向完成后立即释放，全模型峰值仅增加 **20 MiB**）。
2. **可证伪假设**：当 QSA NVFP4 Decode 的 `rows >= 128` 时，由于每个 KV 头对应的 Query 行数足够密集，SM 内部 Math Warp 已完全掩盖 Global Load 延迟，因此 `load K first + evict_first` 的加速比会收敛至 `0.995x–0.997x`（已在 H100 `rows=128` 实测数据中验证）。
