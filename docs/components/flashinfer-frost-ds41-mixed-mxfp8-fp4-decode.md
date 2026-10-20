# FlashInfer Frost DeepSeek-V4.1 SM100 混合 MXFP8/FP4 稀疏解码 (`BlackwellV41MixedCacheDecode`)、非交错分页 Scale 布局与 CUDA Green Context SM 分区 (`green_ctx.py`) 机制

> **English Version**: [flashinfer-frost-ds41-mixed-mxfp8-fp4-decode.en.md](flashinfer-frost-ds41-mixed-mxfp8-fp4-decode.en.md) | **关联实验**: [`experiments/flashinfer_ds41_mixed_decode_sim/README.md`](../../experiments/flashinfer_ds41_mixed_decode_sim/README.md)

### 一句话结论

FlashInfer 在 SM100（NVIDIA Blackwell）上引入的 **Frost DeepSeek-V4.1 混合缓存解码算子（`deepseek_v41_decode` / `BlackwellV41MixedCacheDecode`）**，将 128-token 局部滑动窗口 KV 缓存（`swa_mxfp8`，`[pages, 64, 1, 528]`，E4M3 数据 + group32/E8M0 2 的幂次 Scale）与 $K \in \{64, \dots, 512\}$ 压缩稀疏全局 KV 缓存（`main_kv_fp4`，`[pages, 64, 1, 288]`，E2M1x2 数据 + group16/E4M3 Scale）融合在单个 CuTe DSL 内核中；通过**页内全数据行在前、全 Scale 行在后的非交错（Non-Interleaved）64-token 页布局**、针对 E2M1 半值边界精度的 **`tl.div_rn` 最近偶数舍入除法**、小 Batch Split-KV 到大 Batch 16-Warp Warp-Specialization（`use_ws`）+ 波次均衡持久化 CTA（`persistent_ctas = ceil(B / ceil(B / sm_count))`）自适应调度，在不牺牲局部窗口注意力的前提下将 640-token 有效上下文每层 KV 读取字节从 BF16 的 `655,360 B` 降至 `215,040 B`（**3.0476x 压缩 / 节省 67.19% HBM 带宽**），同时配合 `flashinfer/green_ctx.py` 的 `(8, 8)` SM 硬件分区流实现并发微批次与通信内核的无抢占共址。

### 组件责任与上游主线位置

| 模块与文件 | 上游仓库与 Commit / Blob SHA | 核心工程职责 |
| :--- | :--- | :--- |
| `flashinfer/deepseek_v41.py` | `flashinfer-ai/flashinfer` commit `8dc1163fbfbec7098a2418276fd04bc1fa1aeaa5` / blob `52ec7df5577024e57993cb95e4832dc653a7b3e7` | 暴露 `deepseek_v41_decode`、`deepseek_v41_window_decode` 与 `deepseek_v41_quantize_cache` 公开契约，定义 Q/SWA/Global/Sink 张量形状、CUDA Graph 固定地址重放与 Sink-Exclusive 自然对数 LSE 语义。 |
| `flashinfer/experimental/deepseek_v41/decode.py` | `flashinfer-ai/flashinfer` commit `8dc1163fbfbec7098a2418276fd04bc1fa1aeaa5` / blob `e0bdf9529778bbef5eb8c6aa82811bd136c0051e` | 实现 `_Plan` 生命周期、16B 对齐与非重叠校验、小 Batch Split-KV（`desired = {10, 8, 4, 2, 1}`）与大 Batch 16-Warp WS + 波次均衡 `persistent_ctas` 自适应编译调度。 |
| `flashinfer/experimental/deepseek_v41/cache.py` | `flashinfer-ai/flashinfer` commit `8dc1163fbfbec7098a2418276fd04bc1fa1aeaa5` / blob `3181b6636b22f6efef02808703b8d04392075ca0` | 实现 `_quantize_cache` Triton 量化器：分别生成 `swa_mxfp8`（528 B/token）与 `main_kv_fp4`（288 B/token）非交错 64-token 物理页，使用 `tl.div_rn` 与内联 PTX `cvt.rn.satfinite.e2m1x2.f32`。 |
| `flashinfer/green_ctx.py` | `flashinfer-ai/flashinfer` commit `8dc1163fbfbec7098a2418276fd04bc1fa1aeaa5` / blob `66b5970dae4d8e6881217b675368d072a0033494` | 封装 CUDA Driver API `cuDevSmResourceSplitByCount`、`cuGreenCtxCreate` 与 `cuGreenCtxStreamCreate`，按计算能力约束（SM90/SM100 为 `(8, 8)`）将单个 GPU 划分为物理隔离的 Green Context 流。 |

### 核心实现路径、数据布局与同步契约

1. **非交错（Non-Interleaved）64-Token 页面布局（`cache.py:25-70`）**：
   - 输入 MLA 压缩潜向量（已融合 RoPE）形状为连续 `BF16/FP32 [N, 512]`，页面大小固定为 `page_size = 64`。
   - **并非每个 Token 的数据与 Scale 交错存放**，而是在每个 64-token 页面内先连续存放全部 64 行的量化数据，再连续存放全部 64 行的分组 Scale（确保 TMA 128B/256B 对齐加载时数据区与 Scale 区各自连续）：
     - `swa_mxfp8`（形状 `[pages, 64, 1, 528]`，`uint8`）：`WIDTH = 512`（512 个 FP8 E4M3 元素，512 B），`GROUP = 32`，`SF = 512 // 32 = 16`（16 个 E8M0 指数 Scale，16 B）。页内前 `64 * 512 = 32,768 B` 为数据区，后 `64 * 16 = 1,024 B` 为 Scale 区，每页共 `33,792 B`。
     - `main_kv_fp4`（形状 `[pages, 64, 1, 288]`，`uint8`）：`WIDTH = 256`（512 个 FP4 E2M1 元素两两打包为 256 B），`GROUP = 16`，`SF = 512 // 16 = 32`（32 个 FP8 E4M3 Scale，32 B）。页内前 `64 * 256 = 16,384 B` 为数据区，后 `64 * 32 = 2,048 B` 为 Scale 区，每页共 `18,432 B`。
2. **为什么 FP4 量化必须使用 `tl.div_rn` 而非倒数乘法（`cache.py:42-53`）**：
   - 在 `main_kv_fp4` 中，每 16 个元素的最大绝对值 `amax` 通过 `scale8 = (tl.maximum(amax, 6 * 2.0**-9) / 6).to(tl.float8e4nv)` 编码为 E4M3 Scale。
   - E2M1 的正可表示离散点仅有 `{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`，其相邻半值（Halfway）边界为 `{0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}`。若编译器将 `x / scale` 优化为 `x * (1.0 / scale)`，当 `scale` 非 2 的整幂次（例如 E4M3 值 `0.9375`）时，`1.0 / 0.9375` 在 IEEE-754 FP32 下产生尾数截断误差，使原本精确落在半值边界（应向偶数阶梯舍入）的商发生单 ULP 偏移（如 `1.25 -> 1.25000012`），导致 `cvt.rn.satfinite.e2m1x2.f32` 错误跳变到相邻阶梯（`1.0 -> 1.5`，误差高达 50%）。因此内核显式调用 `tl.div_rn(ga, scale[:, :, None])` 并设置 `enable_fp_fusion=False`。
3. **自适应 Split-KV、Warp-Specialization 与波次均衡 Persistent CTA 调度（`decode.py:130-195`）**：
   - 当 `batch <= 64` 时，SM 无法被单请求 CTA 填满，调度器启用 Split-KV：`desired = 10 (B<=4), 8 (B<=16), 4 (B<=32), 2 (B<=64), 1 (B>64)`，实际 `split = min(desired, (128 + ck) // 64)`，并分配 `batch * 64 * split * 513 * 4` 字节的 `int8` 归约工作区。
   - 当 `ck == 512` 且 `batch >= 128` 时，`split = 1`，切换至 **16 个反量化 Warp（`dequant_warps = 16, phase_chunks = 2, use_ws = True`）** 的 Warp-Specialized QK/PV + 融合 Softmax/Correction 流水线。
   - 若 `batch > sm_count`（如 B200 `sm_count = 148`，`batch = 160`），为避免最后一个短尾波次（Tail Wave）仅启动 12 个 CTA 造成 91.9% SM 空转，调度器计算 `waves = ceil(batch / sm_count)` 与 `persistent_ctas = ceil(batch / waves)`（例如 `B=160` 时 `waves=2, persistent_ctas=80`），同时在缓存池小于 $2^{35}$ 字节（32 GiB）时开启 16B 单位的 `compact_metadata`（Int32 偏移）。
4. **Sink-Exclusive 自然对数 LSE 契约與 CUDA Green Context 分区（`hca_v41.py:66-76`, `green_ctx.py:30-115`）**：
   - Split-K 中间结果使用 `log2` LSE 合并，但公开返回的 `lse`（`FP32 [B, 64, 1]`）严格为**不含 Sink 的自然对数 LSE**；`sink`（`FP32 [64]`）仅在最终输出归一化时参与一次分母计算，从而保证即使 Sink Logit 极大也不会污染跨块/跨层 LSE 合并精度。
   - `green_ctx.py` 在 SM90/SM100（`major >= 9`）上强制执行 `min_count >= 8` 且按 `8` SMs 粒度对齐（`get_sm_count_constraint -> (8, 8)`），通过迭代调用 `cuDevSmResourceSplitByCount` 与 `cuGreenCtxGetDevResource` 刷新剩余 SM 池。

### 关键公式、单位与工程假设

1. **非交错 64-Token 页面字节偏移公式**：
   $$\text{base}(\text{slot}) = \left\lfloor \frac{\text{slot}}{64} \right\rfloor \times 64 \times (\text{WIDTH} + \text{SF})$$
   $$\text{data\_base}(\text{slot}) = \text{base}(\text{slot}) + (\text{slot} \bmod 64) \times \text{WIDTH}$$
   $$\text{scale\_base}(\text{slot}) = \text{base}(\text{slot}) + 64 \times \text{WIDTH} + (\text{slot} \bmod 64) \times \text{SF}$$
   其中 `swa_mxfp8` 取 $(\text{WIDTH}, \text{SF}) = (512, 16)$，`main_kv_fp4` 取 $(\text{WIDTH}, \text{SF}) = (256, 32)$，单位均为字节（Bytes）。
2. **波次均衡 Persistent CTA 与 Split-KV 工作区大小**：
   $$W = \left\lceil \frac{B}{\text{SM}_{\text{count}}} \right\rceil, \quad \text{CTA}_{\text{persistent}} = \left\lceil \frac{B}{W} \right\rceil \quad (B > \text{SM}_{\text{count}}, \text{dequant\_warps}=16)$$
   $$\text{Bytes}_{\text{workspace}} = \begin{cases} B \times 64 \times \text{split} \times 513 \times 4, & \text{split} > 1 \\ 0, & \text{split} = 1 \end{cases}$$
3. **Sink-Exclusive LSE 与最终注意力质量归一化**：
   $$\text{LSE}_{\text{real}} = \ln \left( \sum_{j \in \mathcal{S}_{\text{swa}} \cup \mathcal{S}_{\text{global}}} e^{s_j} \right), \quad \mathbf{o}_{\text{final}} = \frac{\sum_{j} e^{s_j} \mathbf{v}_j}{e^{\text{LSE}_{\text{real}}} + e^{s_{\text{sink}}}}$$

### 确定性离线仿真算例与量化对比

通过运行标准库确定性仿真器 [`experiments/flashinfer_ds41_mixed_decode_sim/run.py`](../../experiments/flashinfer_ds41_mixed_decode_sim/run.py)（结果固化于 [`experiments/flashinfer_ds41_mixed_decode_sim/results.json`](../../experiments/flashinfer_ds41_mixed_decode_sim/results.json)），得到以下核验数据：

| 评估维度 | 指标 / 场景 | 基线 (BF16 / 倒数乘法 / 全 SM 网格) | Frost DS4.1 (`swa_mxfp8` + `main_kv_fp4`) | 量化收益 / 工程结论 |
| :--- | :--- | :--- | :--- | :--- |
| **每请求 640-Token KV 读取字节** | 128 SWA + 512 Global | `655,360 B` (全 BF16 `1024 B/tok`) | `215,040 B` (`128*528 + 512*288`) | **3.0476x 压缩**（节省 **67.19%** HBM 带宽） |
| **FP4 E2M1 半值边界量化精度** | 217 组 E4M3 Scale 半值测试 | `x * (1.0 / scale)` 发生 **9 次 (4.15%)** 阶梯跳变 | `tl.div_rn(x, scale)` **0 次误跳变** | 消除 E2M1 半值处最高 **+50%** 的单步量化误差 |
| **大 Batch 尾波均衡 (`B=160, SM=148`)** | 2 波次 CTA 调度效率 | 朴素网格尾波仅 12 CTA（网格利用率 **54.05%**） | `persistent_ctas = 80`（2 波次均衡 **100.0%**） | 消除大 Batch 跨波次尾部空转，且省去 Split-K 工作区 |
| **CUDA Green Context 分区 (`SM=148`)** | 请求 `[20, 36, 64]` SMs | 无法保证 8-SM 簇对齐 | 实际分配 `[24, 40, 64]` SMs，剩余 `20` SMs | 严格满足 SM90/SM100 `(8, 8)` 硬件对齐粒度 |

### 对质量、SLO、拓扑、成本与运维的架构影响

1. **质量（Quality）**：滑动窗口内最近 128 个 Token 保持 MXFP8（E4M3 + per-32 E8M0 Scale），保护局部语法与近期指代精度；历史稀疏检索出的 $K \le 512$ 个远距离 Token 采用 FP4（E2M1 + per-16 E4M3 Scale）配合 `tl.div_rn` 精确舍入，实现长上下文吞吐与精度的帕累托最优。
2. **延迟与吞吐 SLO**：在 Decode 阶段（HBM 带宽受限），67.19% 的 KV 流量削减直接转化为 TPOT（Time Per Output Token）下降；当 `batch >= 128` 时自动切换至 `use_ws=True`（16 个反量化 Warp + 无 Split-K 归约工作区），消除二次归约内核启动开销。
3. **CUDA Graph 与显存运维契约**：`deepseek_v41_decode` 要求在 CUDA Graph 捕获前完成一次 `_Plan` 预热，且重放期间输入张量地址固定、不得与 `_Plan` 内部拥有的 `out`/`lse`/`workspace` 内存重叠（`_overlaps` 检查），同一 `_Plan` 实例严禁跨多 CUDA Stream 并发执行。

### 架构师解释、计算与设计练习

1. **解释练习**：为什么 `cache.py` 将 64 个 Token 的数据行全部排在前面（`data_base = base + (slot % 64) * WIDTH`），而把 64 个 Token 的 Scale 行全部排在后面（`scale_base = base + 64 * WIDTH + (slot % 64) * SF`），而不是采用 `[Data_0, Scale_0, Data_1, Scale_1, ...]` 的交错结构？
   - *参考解答*：Blackwell TMA（Tensor Memory Accelerator）与向量化全局内存加载要求基地址按 16B/128B 对齐。若每个 Token 的 512B 数据与 16B Scale 交错存放（步长 528B），虽然 528 是 16 的倍数，但无法将 64 个 Token 的数据区或 Scale 区作为连续紧凑的二维 Tile 一次性通过 TMA 批量搬运；非交错布局使数据区（32 KiB / 16 KiB）和 Scale 区（1 KiB / 2 KiB）各自成为高对齐连续张量块。
2. **计算练习**：当 `batch = 32`、`compressed_k = 512` 时，`decode.py` 选择的 `split` 是多少？分配的 `workspace` 占多少 MiB？
   - *参考解答*：`batch = 32` 对应 `desired = 4`，`(128 + 512) // 64 = 10`，故 `split = min(4, 10) = 4`。`workspace_bytes = 32 * 64 * 4 * 513 * 4 = 16,809,984 B` $\approx 16.03\text{ MiB}$。
3. **设计练习**：在单张 B200（148 SMs）上同时运行 Prefill/Decode 混合负载或通信重叠时，如何使用 `flashinfer.green_ctx.split_device_green_ctx` 划分 SM 资源？
   - *参考解答*：根据 `major >= 9` 的 `(8, 8)` 约束，将通信/小算子流划分为 16 或 24 SMs（8 的整数倍），将主计算流划分为剩余的 128 或 120 SMs，分别绑定独立 `torch.Stream`，消除 NCCL/NVSHMEM 内核与主 Decode 内核的 SM 调度排队抖动。

### 可证伪后续问题与核验边界

1. **核验边界**：当前分析基于 `flashinfer-ai/flashinfer` commit `8dc1163fbfbec7098a2418276fd04bc1fa1aeaa5` 的一手源码与本仓库 CPU 确定性模型（`own-offline`），未在真实 SM100 B200 GPU 硬件上采集 Nsight Compute 指令级 stall 计数。
2. **可证伪后续问题**：当 `deepseek_v41_decode` 后续扩展支持 MTP（Multi-Token Prediction，`seq_len_q > 1`）与 SM103（B300）时，`dequant_warps = 16` 的寄存器预算与 `mma_qk_tiler_mn = (64, 64)` 是否需要调整为更大的 M 维 Tile？
