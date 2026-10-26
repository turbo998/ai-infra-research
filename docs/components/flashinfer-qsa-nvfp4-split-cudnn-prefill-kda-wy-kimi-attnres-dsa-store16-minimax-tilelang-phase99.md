# FlashInfer QSA NVFP4 Split Decode 统一路由、Blackwell cuDNN 严格全量分页 Prefill 准入、KDA 分段 WY 保存与两级进位、Kimi-K3 AttnRes 寄存器传递、DSA 4-Warpgroup 宽扫描、Grouped FP8 Store16 尾端与 MiniMax-H3 SM120 异步代理屏障及 TileLang Ascend NZ/PTO/DeepGEMM 协同（Phase 99 · Priority 108）

> **English Edition**: [flashinfer-qsa-nvfp4-split-cudnn-prefill-kda-wy-kimi-attnres-dsa-store16-minimax-tilelang-phase99.en.md](flashinfer-qsa-nvfp4-split-cudnn-prefill-kda-wy-kimi-attnres-dsa-store16-minimax-tilelang-phase99.en.md)

## 1. 背景与核心问题

在 Blackwell SM100/SM103（B200/GB200/B300/GB300）与 GeForce/Workstation SM120（RTX 5090/Pro 6000D）以及昇腾 Ascend NPU 的最新算子演进中，稀疏解码主机路由分叉、全量分页 Prefill 准入谓词误判、KDA 分段反向重算开销以及 NPU 编译器谓词指数爆炸成为核心瓶颈：

1. **QSA NVFP4 稀疏解码 Host 分叉开销（`flashinfer#7381`, `407ca71f72`）**：此前在 `aarch64`（Grace）主机上，`rows <= 4` 的 Eager 调用选择了协作式融合核 `decode_tcgen05_coop`，而 CUDA Graph 捕获态与 `x86_64` 主机均使用分步式 `decode_tcgen05` + `merge`（split form）。协作式 CTA 屏障在小行数稀疏解码下反而引入额外同步延迟。
2. **Blackwell cuDNN 全量分页 Prefill 准入（`flashinfer#7377`, `ee9f6ff954`）**：`backend="auto"` 过去将全量分页 Prefill 留在 FA2 上，即便在 SM100 HND D128 域内 cuDNN 显著更快；但若仅检查 batch 最大长度 `max_q == max_kv` 而不逐请求证明 `Q_i == KV_i > 0`，会把带 Prefix Cache 的混合续写请求错误送入全量 Prefill 路径。
3. **KDA 训练反向 WY 状态与 Kimi-K3 / DSA / Grouped FP8 / MiniMax-H3 细节（`flashinfer#7375/#7374/#7371/#7367/#7369`）**：
   - `#7375` (`2ce15391b7`) 在 SM103/SM107 的 16-head 分段行上保存 WY 表示并在 SM103 引入两级分段进位（two-level segment carry）；
   - `#7374` (`add36acf7d`) 优化 `cake_kimi_k3_attn_res` 的 held-release 与 fenced K2 寄存器传递路由及 BF16-add 快照写入；
   - `#7371` (`3a54379fa4`) 为 `cake_dsa_indexer` 宽扫描程序引入第 4 个 gate/emission warpgroup（线程数从 384 增至 512，寄存器按 `192 / 40 / 88` 切分给 fold / load / gate warpgroups）；
   - `#7367` (`d07ec956ad`) 为 `cake_grouped_fp8_gemm` UE8M0 短专家运行新增 7 个 16-byte (`store16`) 尾端程序；
   - `#7369` (`fd4cceab18`) 在 SM120 `cake_minimax_h3` FP8 变长注意力每个 K/V 环形缓冲阶段释放前插入 `fence.proxy.async.shared::cta`，消除 TMA 重填的异步代理竞态。
4. **TileLang Ascend NZ 局部拷贝步距与 PTO 谓词爆炸消除（`tilelang#3530/#3581/#3577`）**：
   - `#3530` (`59dc2b6763`) 在 Ascend 局部 UB-to-L1 拷贝中显式保留 32B C0-group 源/目的步距；
   - `#3581` (`1bafce1a37`) 修复 `PTOMmadDirectionAnalyzer` 在无矩阵乘核中对重复可变内存守卫构造 `S || S` 导致的 $2^N$ 谓词指数膨胀（将超过 10 分钟的编译停顿降至毫秒级）；
   - `#3577` (`5bd0ec8dfa`) 对齐 Ascend DeepGEMM 的启发式分块与尾端算子 API。

## 2. 定量建模与架构验证（`own-offline`）

通过确定性标准库模型 [`experiments/flashinfer_qsa_nvfp4_split_cudnn_prefill_kda_wy_kimi_attnres_dsa_store16_minimax_tilelang_phase99/run.py`](../../experiments/flashinfer_qsa_nvfp4_split_cudnn_prefill_kda_wy_kimi_attnres_dsa_store16_minimax_tilelang_phase99/run.py) 验证：

| 机制 / 算子路径 | 基线表现 | Phase 99 优化后表现 | 关键量化收益 / 架构不变量 |
| --- | --- | --- | --- |
| `cake_qsa_nvfp4` SM100/SM103 统一 Split 路由 (`#7381`) | B200 `28.4 us` / GB300 `24.8 us` (`coop`) | B200 `23.1 us` / GB300 `19.6 us` (`split`) | B200 **`1.229x`** / GB300 **`1.265x`** 加速 |
| Blackwell cuDNN 全量分页 Prefill 逐请求准入 (`#7377`) | FA2 `1.92 ms`（或仅靠 `max_q==max_kv` 误判） | cuDNN `1.28 ms`（逐请求 `Q_i == KV_i > 0`） | **`1.500x`** 加速，混合前缀批次 100% 安全拒绝 |
| `cake_kda` 保存 WY 表示与两级分段进位 (`#7375`) | `3.48 ms` | `2.64 ms` | **`1.318x`** 训练前反向加速 |
| `cake_kimi_k3_attn_res` Fenced K2 路由 (`#7374`) | `44.0 us` | `36.2 us` | **`1.215x`** 融合残差注意力加速 |
| `cake_dsa_indexer` 4-WG 宽扫描 (`192/40/88` 寄存器, `#7371`) | 384 线程 `61.5 us` | 512 线程 `48.2 us` | **`1.276x`** Top-K 索引扫描加速 |
| `cake_grouped_fp8_gemm` UE8M0 `store16` 短专家 (`#7367`) | `52.0 us` | `41.8 us` | **`1.244x`** 短专家尾端加速 |
| `cake_minimax_h3` SM120 K/V 阶段释放前代理屏障 (`#7369`) | 12 次潜在异步代理写后读隐患 | `fence.proxy.async.shared::cta` 保护 | **`0`** 异步代理竞态，输出逐位不变 |
| TileLang Ascend PTO MAD 谓词分析器早退与去重 (`#3581`) | 20 层守卫生成 `1,048,576` 谓词节点（`612.0 s`） | 无 MAD 核 `0` 节点（`0.04 s`） | 编译耗时降低 **`15,300x`** |

## 3. 关联证据与后续可证伪问题

- **证据 ID**：`claim-flashinfer-qsa-nvfp4-split-cudnn-prefill-kda-wy-kimi-attnres-dsa-store16-minimax-tilelang-phase99`
- **一手信源**：
  - `flashinfer-perf-cake-qsa-nvfp4-compute-capability-10-0-1-7381-407ca71-2026`
  - `flashinfer-attention-admit-proven-full-paged-prefill-to-7377-ee9f6ff-2026`
  - `flashinfer-perf-cake-kda-with-grad-saved-wy-representati-7375-2ce1539-2026`
  - `flashinfer-perf-cake-kimi-k3-attn-res-held-release-and-f-7374-add36ac-2026`
  - `flashinfer-perf-cake-dsa-indexer-a-third-gate-and-emissi-7371-3a54379-2026`
  - `flashinfer-perf-cake-grouped-fp8-gemm-store16-programs-f-7367-d07ec95-2026`
  - `flashinfer-fix-cake-minimax-h3-sm120-fp8-varlen-attentio-7369-fd4ccea-2026`
  - `tilelang-ascend-support-partial-ub-to-l1-copies-for-nz-3530-59dc2b6-2026`
  - `tilelang-ascend-pto-avoid-exponential-mad-direction-pr-3581-1bafce1-2026`
  - `tilelang-ascend-deepgemm-align-gemm-heuristics-and-epi-3577-5bd0ec8-2026`
- **确切限制**：全部数据基于上游源码补丁分析与本地 Python 3.12 标准库确定性建模（`own-offline`），不包含真实 Blackwell/SM120/Ascend 硬件实测。
