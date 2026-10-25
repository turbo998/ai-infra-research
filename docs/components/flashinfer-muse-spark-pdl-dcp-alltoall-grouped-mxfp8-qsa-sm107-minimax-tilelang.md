# FlashInfer Muse-Spark Norm PDL、DCP All-to-All LL128 FIFO、Grouped MXFP8 形状分发、Rubin SM107 NVFP4 QSA Decode 与 TileLang 流水线屏障修复

**English companion**: [flashinfer-muse-spark-pdl-dcp-alltoall-grouped-mxfp8-qsa-sm107-minimax-tilelang.en.md](flashinfer-muse-spark-pdl-dcp-alltoall-grouped-mxfp8-qsa-sm107-minimax-tilelang.en.md)

## 1. 背景与核心问题

在 NVIDIA Blackwell（`sm_100a` B200/GB200、`sm_103a` B300/GB300、工作站级 `sm_120a` RTX 5090/PRO 6000）以及下一代 Rubin（`sm_107a` R200，212 SMs）上，短序列 Decode 与混合专家（MoE）量化算子的尾延迟越来越取决于四个微架构细节：

1. **逐层 Norm/Gate 启动开销与二分准入索引**：Muse-Spark 架构中的 `rmsnorm_sigmoid_gate` 与 `gated_residual_rmsnorm` 在 $T=1\sim 8$ 时单次执行仅 $0.86\sim 2.27\;\mu\text{s}$。若未启用 Programmatic Dependent Launch（PDL），或者在 Python 侧对数十条窗口规则做线性遍历，主机启动与调度间隙会吞噬算子收益。
2. **Decode Context Parallel（DCP）NVLink LL128 FIFO 槽位释放延迟**：在 `cake_dcp_alltoall`（CP2/CP4）中，若接收端在当次 kernel 尾部才释放最后一个 LL128 FIFO 槽位，或者经由共享内存周转写回，跨卡同步与写回路径会增加额外等待。
3. **Grouped MXFP8 量化在稀疏专家掩码下的无效 CTA 启动**：`cake_grouped_mxfp8_quantize` 接收 `x[B, M, K]` 与设备侧 `mask[B]`（每组有效行数）。若主机读取 `mask[B]` 会触发昂贵的 D2H 同步；若始终按完整 `[B, M]` 启动 `row-2D` 网格，当 $B\cdot M \ge 16,384$ 且专家负载稀疏时，大量 CTA 仅读取完掩码即退出。
4. **Rubin `SM107`（R200）的 `tcgen05` NVFP4 稀疏分页 GQA 与 TileLang 编译器流水线不变量**：R200 具备 212 个 SM（compute capability `10.7`），其 `cake_qsa_nvfp4` 需要适配 `sm_107a` 共享内存描述符与 Split-KV 拆分公式；同时 TileLang 在 Hopper/Blackwell Warp-Specialized（WS）纯 TMA 生产者以及兄弟 `T.Pipelined` 共享内存复用中存在两处关键编译器正确性缺陷。

---

## 2. 第一手源码实现路径审计

### 2.1 FlashInfer `cake_muse_spark_norm` PDL 路由与 $O(\log W)$ 准入索引（`#7341`, commit `31d0a811aebe`）

在 `flashinfer/cake_muse_spark_norm_backend/__init__.py` 中，新增了 `bf16_pdl` 路由，支持四种按 token 窗口特化的启动形态：
- `pdl`：完整启用 Programmatic Dependent Launch；
- `late_pdl`：延迟触发前导依赖信号以保护高带宽访存段；
- `plain`：在 PDL 收益为负的窗口内直接借用非 PDL 内核且不携带 PDL launch attribute；
- `gap`：委托给 CuTe DSL + PDL 实现。

为消除 `backend="auto"` 在每次调用时遍历窗口列表的 $O(W)$ 开销，导入阶段将每条路由在各架构/头数下的准入区间折叠为不相交有序跨度列表 $\text{above} < \text{rows} \le \text{at\_most}$，并通过 `bisect` 在 $O(\log W)$ 内完成判定：
- **B200（`sm_100a`）实测**：`rmsnorm_sigmoid_gate`（`h32, T=1`）从 $1.248\;\mu\text{s}$ 降至 $0.866\;\mu\text{s}$（**1.441x**）；`gated_residual_rmsnorm`（`mode1, M=8`）从 $2.272\;\mu\text{s}$ 降至 $1.506\;\mu\text{s}$（**1.509x**）。
- **GB300（`sm_103a`）实测**：`rmsnorm_sigmoid_gate`（`h32, T=1`）从 $1.235\;\mu\text{s}$ 降至 $0.862\;\mu\text{s}$（**1.433x**）；`gated_residual_rmsnorm`（`mode1, M=1`）从 $2.007\;\mu\text{s}$ 降至 $1.369\;\mu\text{s}$（**1.466x**）。

### 2.2 `cake_dcp_alltoall` 延迟释放最后 LL128 FIFO 槽位与寄存器直写（`#7340`, commit `53feaae34c0a`）

在 `csrc/generated/dcp_alltoall/cake_dcp_alltoall_{ff3dfded867e5738be23,b8b8ebc8b91fd813f375}_kernel.cu` 中，针对 GB200（`sm_100a`）与 GB300（`sm_103a`）的 CP2/CP4 NVLink LL128 FIFO 交换做出两项关键改动：
1. **延迟槽位释放（Deferred Last FIFO Slot Release）**：接收端不再在当次 kernel 尾部同步等待并清理最后一个 FIFO 槽位，而是推迟到下一次 `cake_dcp_alltoall` 启动开头释放；
2. **寄存器 Plain Store**：接收到的 `partial_o` 与 `softmax_stats` 直接从寄存器通过普通存储指令写回输出张量。
- **实测加速**：全形状实现 **1.15x–1.33x** 提速。GB300 上 CP2 $T=1$ warm 从 $5.872\;\mu\text{s}$ 降至 $4.784\;\mu\text{s}$（**1.23x**），CP4 $T=1$ 从 $5.440\;\mu\text{s}$ 降至 $4.480\;\mu\text{s}$（**1.21x**）；GB200 上 CP2 $T=1$ warm 从 $5.456\;\mu\text{s}$ 降至 $4.288\;\mu\text{s}$（**1.27x**），cold 从 $5.584\;\mu\text{s}$ 降至 $4.192\;\mu\text{s}$（**1.33x**）。

### 2.3 `cake_grouped_mxfp8_quantize` 零主机同步三态形状分发（`#7322`, commit `5396dc3b7a95`）

在 `flashinfer/quantization/fp8_quantization.py` 中，完全根据输入张量形状 `[B, M, K]` 在主机侧零同步选择三类 Cake 程序（输出 `128x4` grouped-GEMM 布局的 FP8 E4M3 + UE8M0 缩放因子）：

$$\text{Program}(B, M, K) = \begin{cases} \texttt{bulk-copy}, & \text{if } B \cdot M \cdot K \cdot \text{sizeof(dtype)} \le 4\;\text{MiB} \\ \texttt{row-tiled}, & \text{else if } B \cdot M \ge 16,384 \\ \texttt{row-2D}, & \text{otherwise} \end{cases}$$

- **`bulk-copy`**：在读取 group 的 `mask[B]` 字的同时将整行分块预取至共享内存，使掩码读取与数据搬运重叠，小尺寸满掩码（如 `[3, 256, 2048]`）在 GB300/GB200 上提速 **1.07x–1.11x**。
- **`row-tiled`**：每个线程负责量化本组的 2 行，当每行块数 $B_{\text{row}} = \lceil K / 32 \rceil < 128$ 时，单个 CTA 横向并排容纳 $R_{\text{side}} = \lfloor 128 / B_{\text{row}} \rfloor$ 行，即每个 CTA 处理 $2 \cdot R_{\text{side}}$ 行。在 `[32, 1024, 7168]` 稀疏掩码（如 `512 in group 0`）下，GB300 耗时从 $34.724\;\mu\text{s}$ 降至 $17.955\;\mu\text{s}$（**1.9340x**），GB200 从 $36.392\;\mu\text{s}$ 降至 $18.802\;\mu\text{s}$（**1.9357x**）。

### 2.4 Rubin `SM107`（R200）`cake_qsa_nvfp4` 与 `SM120` MiniMax-H3 优化（`#7284`, `#7334`, `#7333`）

- **Rubin `SM107`（`cc8552a3a015`）**：为 NVIDIA R200（`sm_107a`，212 SMs）编译 `tcgen05` decode 与 split-KV merge 内核（`CTAS_PER_SM["sm_107a"] = 1`）。由于 R200 上 fused cooperative 形式在任何行数下均不快于 split 形式，`10.7` 固定采用 split 形式。在 `rows = [1, 2, 4, 8, 16, 32, 64, 128]`、`num_kv_heads = 2`、`topk = 2051`（$\text{tiles} = \lceil 2051/64 \rceil = 33$）下，212 SMs 的 `num_splits` 序列为 `[33, 33, 26, 13, 6, 3, 1, 1]`，相对 FlashInfer 参考实现获得 **1.33x–2.76x** 端到端加速（`rows=128` NHD 从 $151.44\;\mu\text{s}$ 降至 $55.52\;\mu\text{s}$，**2.73x**）。
- **SM120 MiniMax-H3（`b8ed186df31b`, `53ecbde224f4`）**：FP8 变长注意力通过提前移交 PV 累加与下一计算单元预取，在 RTX 5090 上提速 **1.014x–1.055x**（RTX PRO 6000 上 **1.030x–1.041x**）；持久化 FP8 out-proj 采用每 swath 单 M-tile 布局提速 **1.006x–1.011x**。

### 2.5 TileLang Warp-Specialized TMA 生产者屏障与兄弟流水线 Buffer 调和（`#3503`, `#2342`）

1. **多 Warp TMA-Only 生产者挂起修复（`fe6d319c78c0`）**：在 `src/cuda/transform/producer_consumer_ws.cc` 中，原先只要没有 SIMT/`cp.async` 拷贝，前向 `mbarrier` 的 `arrive_count` 就设为 `1`（仅 elected thread 到达）。当生产者包含多个 warp（如 4 个 warp = 128 线程）且存在前缀绑定或条件分支时，producer warps 1–3 仅等待 back-pressure 屏障而从不在前向屏障上同步，可能落后 2 个 phase 导致死锁（`#3470`）。修复后严格限定：
   $$\texttt{producer\_is\_single\_warp} = \neg\texttt{has\_async\_producer} \land \neg\texttt{has\_producer\_prefix} \land \neg\texttt{loop\_body\_condition.defined()}$$
   仅当该谓词为真时 `fwd_arrive_count = 1`，否则设为完整 `producer_extent`（如 `128`）。
2. **兄弟 `T.Pipelined` 共享内存双重展开修复（`fc0fbf0e50de`）**：当两个并列 `T.Pipelined` 循环复用同一 `alloc_shared` 缓冲区但 `num_stages` 不同（如 `2` 与 `4`）时，第二个流水线会将已展开的 3D `(2, M, K)` 再次展开为 4D `(4, 2, M, K)` 并触发 `LayoutInference` 崩溃（`#2309`）。`PipelineBufferReconciler`（`src/transform/inject_pipeline.cc`）通过比对底层 data `Var` 还原 2D 形状重新按最大阶段数展开为 `(4, M, K)`，并统一重写整个语法树中的 `BufferLoad`/`BufferStore`/`BufferRegion`。

---

## 3. 可复现实验与定量汇总

本组件配套标准库可复现实验 `experiments/flashinfer_muse_spark_pdl_dcp_alltoall_grouped_mxfp8_qsa_sm107_minimax_tilelang/run.py`：

| 子方向 | 关键配置 / 形状 | 基线延迟 ($\mu\text{s}$) | 优化后延迟 ($\mu\text{s}$) | 加速比 / 検証结论 | 证据类型 |
|---|---|---:|---:|---|---|
| `cake_muse_spark_norm` PDL | B200 `gated_residual_rmsnorm` `M=8` | 2.272 | 1.506 | **1.509x**（GB300 最高 **1.466x**） | `author-benchmark` + `own-offline` |
| `cake_dcp_alltoall` LL128 | GB200 CP2 `T=1` cold / warm | 5.584 / 5.456 | 4.192 / 4.288 | **1.33x** / **1.27x** | `author-benchmark` + `own-offline` |
| `cake_grouped_mxfp8_quantize` | `[32, 1024, 7168]` 稀疏 (`512 in group 0`) | 34.724 (GB300) | 17.955 (GB300) | **1.934x**（GB200 **1.936x**） | `author-benchmark` + `own-offline` |
| `cake_qsa_nvfp4` Rubin SM107 | R200 (212 SMs) NHD `rows=128, topk=2051` | 151.44 | 55.52 | **2.73x**（read flush **2.76x**） | `author-benchmark` + `own-offline` |
| TileLang WS & Sibling Pipeline | `num_stages=(2, 4)` 共享 `(64, 32)` | 4D `[4,2,64,32]` 崩溃 | 3D `[4,64,32]` 通过 | 消除 `mbarrier` 2-phase 挂起与 `ICHECK` | `implementation` + `own-offline` |

---

## 4. 面向解决方案架构师的设计练习与可证伪问题

1. **计算题**：在 R200（`sm_107a`，212 SMs，`CTAS_PER_SM=1`）上运行 `cake_qsa_nvfp4`，若 `rows=8`、`num_kv_heads=2`、`topk=2051`（每 tile 64 tokens），求 `num_splits`。
   - **解答**：$\text{tiles} = \lceil 2051 / 64 \rceil = 33$，容量上限 $\lfloor (1 \times 212) / (8 \times 2) \rfloor = \lfloor 212 / 16 \rfloor = 13$，故 $\text{num\_splits} = \min(33, 13, 64) = 13$。
2. **可证伪假设**：在 `[32, 1024, 7168]` 满掩码（32,768 有效行）下，`row-tiled` 是否仍显著快于 `row-2D`？
   - **证伪**：否。此时算子完全受 HBM 带宽约束，GB300 上 `row-tiled` 为 $104.558\;\mu\text{s}$ vs `row-2D` $103.497\;\mu\text{s}$（`0.9899x`），用 1%–2.8% 的满掩码回退换取了稀疏 MoE 场景最高 **1.94x** 的收益与零主机同步。
