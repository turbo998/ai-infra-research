# FlashInfer CuTe-DSL MegaMOE 持久化 TopK / 128 对齐、SM103a All-Gather Matmul `_cg2` CTA-Pair、NVFP4 SVDQuant Tail-Split 寄存器偏置、KDA Weakref 激活重算与 Hopper MXFP4×FP8 K128 候选调优深度解析

- **研究日期**：2026-10-24（Phase 57, Priority 66）
- **证据等级**：`implementation` + `author-benchmark`（源码审计）+ `own-offline`（确定性标准库算例验证）
- **上游基准提交**：
  - `flashinfer-ai/flashinfer@3f398977d75b4017cbd4c00824a5108017aade06`（PR `#5666`，2026-04-20）
  - `flashinfer-ai/flashinfer@89009ad44586710d5cb94f15113929c6a8728af9`（PR `#7083`，2026-04-20）
  - `flashinfer-ai/flashinfer@1eee894ea0e142f18faaedb11f37114a2c4e1670`（PR `#7091`，2026-04-20）
  - `flashinfer-ai/flashinfer@4f88101e601728e1c701939d78a46471c7bb976e`（PR `#7069`，2026-04-20）
  - `flashinfer-ai/flashinfer@b17eaa011241866e45a3ea06797c5fecaf6c7bc7`（PR `#6465`，2026-04-20）
  - `flashinfer-ai/flashinfer@1de520d5b54feb4ae6d97a89b06cbc33b1bbd775`（PR `#6534`，2026-04-20）
- **本地验证脚本**：`experiments/flashinfer-megamoe-topk-sm103-ag-svdquant-kda-weakref-mxfp4-k128/run.py`

---

## 1. 架构背景与核心问题

在 Blackwell (`sm_100a` GB200 / `sm_103a` GB300) 与 Hopper (`sm_90a` H200) 的融合算子栈中，2026-04-20 合入的 6 个 FlashInfer 核心提交解决了以下五个生产级瓶颈：

1. **CuTe-DSL MegaMOE 融合路由与非 128 对齐 Hidden 维度（PR `#5666`）**：GPT-OSS-120B 等模型的隐藏维 `H = 2880`（`2880 % 128 = 64 != 0`）无法满足 Blackwell NVFP4/FP8 TMA descriptor 的 128 元素对齐约束；同时部分多阶段归约流水线需要跳过内置 finalize atomic-add 或在 FC2 之前提前乘路由权重。
2. **B300 (`sm_103a`) All-Gather + GEMM 单 CTA 调度次优（PR `#7083`）**：`cake_all_gather_matmul` 在 `TP=4` Llama-3.1-70B QKV 投影（每 rank `512 × 2560, K = 8192`，K-major）下沿用单 CTA (`_cg1`) 模板，未能利用 B300 2-CTA Cluster TMA 多播带宽。
3. **GB200 (`sm_100a`) NVFP4 SVDQuant Epilogue 尾部共享内存读放大（PR `#7091`）**：`cake_nvfp4_svdquant` 在大 `M` 低秩残差回加（`rank=32/96, bias=1`）时，主循环与尾部共享同一 epilogue 结构，导致不必要的 shared memory 偏置往返。
4. **Cake KDA 非重入 Activation Checkpointing 下的显存暴涨与冗余清零（PR `#7069`）**：`ChunkKDAFunction` 在 PyTorch autograd `ctx` 上强引用 `workspace`，导致 261K 序列长度、48 层重算时每层滞留 `4.48 GiB ~ 17.93 GiB` 工作区；且每次新建 `KdaWorkspace` 默认 `torch.zeros` 带来约 `28%` 的无谓耗散。
5. **Hopper MXFP4 × FP8 MegaMoE 在 `I = 640` 下的 Tactic 荒漠（PR `#6465`）与外部算子 Autotuner 委派（PR `#6534`）**：Qwen3.8-Flash-Next 的 `intermediate_size = 640` 无法整除 `K = 256`（`640 % 256 = 128`），过滤掉了全部 `K256` 候选，仅剩 3 个候选。

---

## 2. 关键实现路径与数学/硬件机制

### 2.1 CuTe-DSL MegaMOE：持久化 TopK、Skip-Finalize 与 128 元素隐藏维对齐 (`#5666`)

在 `flashinfer/cute_dsl/megamoe/`（`moe_Persistent_7168.py`, `launch.py`, `ops.py`）中：
- **持久化 TopK + Dispatch 融合**：新增 `topk_Persistent.py`，将门控 `logits -> top_k` 选择直接与 all-to-all dispatch barrier 融合，省去独立 TopK kernel 启动与中间索引写回。
- **跳过 Finalize 与 FC2 前置加权**：
  - `skip_finalize: bool = False`：当设为 `True` 时，算子跳过末尾跨专家归约（`finalize`），直接暴露 per-(token, expert) FC2 输出缓冲区，供调用方做自定义确定性归约。
  - `apply_routing_weights_before_fc2: bool = False`：将路由权重乘法从 FC2 epilogue 提前到 FC1 激活输出后（并在量化为 FC2 输入前先转为 BF16）：
    $$\tilde{\mathbf{h}}_1 = \text{BF16}\!\left(\sigma(\mathbf{x}\mathbf{W}_1) \cdot w_{t,e}\right), \quad \mathbf{y}_{t,e} = \mathcal{Q}(\tilde{\mathbf{h}}_1)\mathbf{W}_2$$
- **128 元素对齐公式**：对任意原始 `hidden_size` $H$，要求调用方按
  $$H_{\text{aligned}} = \left\lceil \frac{H}{128} \right\rceil \times 128$$
  做零填充（例如 GPT-OSS-120B 的 $H = 2880 \to H_{\text{aligned}} = 2944$），从而保持 `128B/256B` TMA box 对齐。

### 2.2 SM103a `cake_all_gather_matmul` `_cg2` CTA-Pair 双生内核 (`#7083`)

在 `csrc/cake/all_gather_matmul/all_gather_matmul.cu` 与 `programs_bf16_sm103.cuh` / `programs_fp16_sm103.cuh` 中：
- 新增 6 个 `sm_103a` 专属 `_cg2`（`cluster_shape = 2x1x1` CTA-pair）内核，覆盖 `world_size in {2, 4, 8}` 与 `BF16/FP16` K-major。
- 在 `select_all_gather_matmul_program()` 中，仅当硬件为 `sm_103a`（B300）、`world_size == 4`、K-major 布局且单 rank 填充后输出形状为 `M_padded == 512, N == 2560`（对应 `global_M = 2048, K = 8192` 的 Llama-3.1-70B TP=4 QKV 投影）时路由至 `_cg2`；`sm_100a`（GB200）完全保持原 `_cg1` 路径不变。

| 精度 | 节点 | 原 `_cg1` 延迟 ($\mu\text{s}$) | 新 `_cg2` 延迟 ($\mu\text{s}$) | 加速比（`author-benchmark` on B300） |
|---|---:|---:|---:|---:|
| BF16 | Node 1 | `115.4` | `106.2` | **`1.0855x`** |
| BF16 | Node 2 | `114.4` | `105.3` | **`1.0902x`** |
| FP16 | Node 1 | `114.8` | `104.0` | **`1.1143x`** |
| FP16 | Node 2 | `115.3` | `107.3` | **`1.0767x`** |

### 2.3 SM100a NVFP4 SVDQuant Tail-Split 寄存器偏置 (`#7091`)

在 `csrc/cake/nvfp4_svdquant/nvfp4_svdquant.cu` 与 `sm100_2cta_persistent_tailsplit_kernel.cuh` 中：
- 针对 2-CTA persistent 模板新增 `TailSplit + RegisterBias` 变体：在 epilogue 阶段直接从寄存器加载 per-column bias 并拆分尾块流水，消除 `smem` 偏置重读。
- 经过对 GB200 上全部 `28` 个基准形状的实测筛选，仅对证明有净收益的两个形状启用新模板，其余 `26` 个形状及整个 `sm_103a`（GB300）保持原二进制不变：
  - `(M=6912, N=3072, K=3072, rank=96, bias=1)`：**`30.96 us -> 29.92 us`（`1.035x`）**
  - `(M=16384, N=3072, K=3072, rank=32, bias=1)`：**`64.69 us -> 63.90 us`（`1.012x`）**

### 2.4 Cake KDA Weakref Checkpointing 与 `zero_init=False` (`#7069`)

在 `flashinfer/cake/kda.py` 中：
1. **Autograd `ctx` 弱引用解耦**：`ChunkKDAFunction.forward` 不再执行 `ctx.workspace = workspace`，而是保存 `ctx._workspace_ref = weakref.ref(workspace) if workspace is not None else None`。
   - 若调用方未显式传入 `workspace`（或传入局部临时 `workspace`），前向结束即被 GC 回收，激活重算期间每层滞留显存从 **`4,482.1 MiB` (`H=4`) / `17,926.0 MiB` (`H=16`) 降至 `0.0 MiB`**（在 `S=261,393` tokens、`4` 文档、GB300 上实测）。
   - 若调用方在外部持有层间共享的 `KdaWorkspace` 强引用，`weakref` 在 backward 时仍能命中同一缓冲区。
2. **默认 `zero_init=False` (`torch.empty`)**：所有 KDA C++ scratch buffer 在消费者读取前均被完整覆写（源码通过向 arena 注入全 `NaN` 验证前反向输出与 `zero_init=True` 逐比特一致）。去除 `torch.zeros` 后，GB300 隐式工作区前反向总耗时：
   - `H=4`：**`6.48 ms -> 4.66 ms`（`1.391x` 加速，`-28.09%`）**
   - `H=16`：**`26.10 ms -> 18.83 ms`（`1.386x` 加速，`-27.85%`）**
3. **Lean Arena Helper**：新增 `kda_train_arena(lean=True, tail_passes=4)`，仅分配 `bwd_lean` / `bwd_tape_lean` 所需缓冲区，省去 `akk_int` 等重放缓冲。

### 2.5 Hopper MXFP4 × FP8 MegaMoE Folded `K128/N64` 候选 (`#6465`) 与 Delegated Autotuner (`#6534`)

- **为什么 `I = 640` 在旧版退化**：Hopper WGMMA MXFP4×FP8 MegaMoE 旧版候选以 `K = 256` 为主，而 Qwen3.8-Flash-Next 的 `I = 640` 满足 $640 \bmod 256 = 128 \neq 0$，导致 `K256` 全部因对齐检查被剔除，搜索空间仅剩 `3` 个基础 tactic。
- **新增 5 个 `M256/K128` 实测候选**：
  1. `1` 个非折叠 `(256, 16, 128)` `CGA 1x1x1`（`group=330`，2 dispatch warps，`fc1_store_offload=True`）；
  2. `3` 个折叠生产者 warp (`fold_producer_warps=True`) `(256, 64, 128)` `CGA 1x1x1`（`group in {330, 264, 132}`，1 dispatch warp，`fc1_early_done_publish=True`）；
  3. `1` 个折叠 `(256, 64, 128)` `CGA 1x2x1`（`group=132`，`tail_split_pairs=True`）。
- 基础 tactic 表从 `17 -> 22`（全集上限 `38 -> 43`），`I = 640` 可用候选从 `3 -> 8`。在 `2x H200`（`EP=2, E=512, top_k=10, H=2560, I=640`）上：

| Tokens / rank | 原 Compute ($\mu\text{s}$) | 新 Compute ($\mu\text{s}$) | Compute 降幅 | 原 E2E ($\mu\text{s}$) | 新 E2E ($\mu\text{s}$) | E2E 降幅 |
|---:|---:|---:|---:|---:|---:|---:|
| `128` | `339.216` | `316.272` | **`-6.76%`** | `501.264` | `475.856` | **`-5.07%`** |
| `1024` | `593.264` | `488.128` | **`-17.72%`** | `756.176` | `643.296` | **`-14.93%`** |
| `4096` | `1405.728` | `1315.200` | **`-6.44%`** | `1573.472` | `1475.216` | **`-6.24%`** |
| `8192` | `2670.336` | `2343.024` | **`-12.26%`** | `2891.616` | `2560.176` | **`-11.46%`** |
| `16384` | `5223.024` | `4442.880` | **`-14.94%`** | `5628.496` | `4873.776` | **`-13.41%`** |

- **Delegated Autotuner (`#6534`)**：`AutoTuner.choose_with_protocol()` 允许 CuTe-DSL MegaMOE 等无 `CustomOp` 包装的底层内核通过协议回调直接复用 FlashInfer 的 CUDA Event 计时、L2 flush、namespace 分区的 v2 JSON 磁盘缓存以及 `capacity` 越界最近邻回退。

---

## 3. 架构师设计练习与可证伪问题

1. **计算题**：一个 48 层 KDA 模型在 `S=261,393, H=16` 下开启非重入 `torch.utils.checkpoint`，若每层调用 `chunk_kda` 时未复用外部 workspace，修复前 48 层前向累计滞留多少 GiB 显存？
   - **解答**：$48 \times 17926.0\text{ MiB} = 860,448\text{ MiB} \approx 840.28\text{ GiB}$，远超单卡 GB300 显存上限；改为 `weakref.ref` 后若每层临时分配则前向滞留为 `0.0 MiB`（峰值仅为单层 `17.51 GiB`）。
2. **可证伪假设**：在 Hopper MXFP4×FP8 MegaMoE 中，对于 $I \bmod 256 = 0$ 的形状（如 $I = 1024$），新增的 `K128/N64` 折叠候选是否仍能击败 `K256` 候选？需在大 token 批（`>= 4096`）下对比 WGMMA 指令发射密度验证。
