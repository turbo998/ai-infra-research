# GPU 软件栈 Phase 167：FlashInfer SM120 Paged FP8 MQA Logits 原生指令路由、Cake SM100a 分块 LM-Head Loss 屏障时序与 FlashAttention SM100 Softmax 寄存器溢出消除及 MLA Learnable Sink

## 1. 背景与核心问题

在 NVIDIA Blackwell 消费级/工作站架构（`SM120a`，如 RTX 5090）与数据中心架构（`SM100a`）共存的异构算子栈中，GPU 软件栈在深层解码与超长词表训练/预填充上面临三项关键挑战：
1. **SM120 Paged FP8 MQA Logits 的 Speculative `next_n=5/6` 路由退化**：在多步投机解码（MTP / Speculative Decoding，`next_n` 为 5 或 6）且 `head_dim = 64` 时，若按默认启发式回退至通用分块/标量累加内核，无法发挥 RTX 5090 `SM120a` 原生 Tensor Core Atom 的吞吐。
2. **Cake 分块 LM-Head + Cross-Entropy Loss 在 `nvcc` 12.9/13.0 下的 `mbarrier` 竞态**：FlashInfer 的 `cake_lm_head_loss` 将大词表 BF16 投影、FP32 累加、在线 Log-Sum-Exp（LSE）与交叉熵损失融合以避免物化完整 `[B*T, V]` Logits 张量。但在 `sm_100a` 与 CUDA 12.9/13.0 编译器下，若生产者线程未严格保证 `expect-tx` 先于消费者等待到达（arrive-first），会导致异步 TMA 事务计数错位与内核死锁。
3. **CuTe SM100 非因果 `hdim=128` Softmax 寄存器溢出与 FlashMLA Learnable Sink 缺失**：SM100 前向非因果注意力在 `hdim=128` 时 Softmax warp 因局部累加器生命周期过长产生局部内存溢出（Local Memory Spill）；同时 SM100 FlashMLA 前向与变长（varlen）反向急需统一的 Learnable Attention Sink 缩放支持。

---

## 2. 上游源码机制深挖（11 条零重叠提交）

### 2.1 FlashInfer：SM120 原生 Atom 路由、Cake KDA/DSA 与 LM-Head Loss 修复
- **`e5eb9a5019` (`2026-10-31`, `#8136`)**：更新 `flashinfer/experimental/deepgemm_sm120_paged_mqa_logits/sm120_paged_mqa_catalog.json`，将 RTX 5090 上 9 个 `h=64, next_n ∈ {5, 6}` 的 Paged FP8 MQA Logits 形状显式路由至 `native-atom` 程序，消除调度回退开销。
- **`b5be99dd30` (`2026-10-29`, `#7858`)**：在 `flashinfer/experimental/cake_lm_head_loss` 中修复 `sm_100a` 下 `mbarrier` 的 `expect-tx` 到达顺序（确保在 `nvcc` 12.9 与 13.0 下 `expect-tx` 严格先于等待阶段完成），并将 `sm_100a` 编译工具链下限固定为 CUDA `12.9`。
- **`9f5c44ea10` (`2026-10-29`, `#7840`) & `2d1a97b50f` (`2026-10-29`, `#7829`)**：为 `cake_kda` 的 KCP 测试引入树内 FP64 精度参考辅助函数，并在 `cake_dsa_indexer_sampling` 基准测试中改用每次调用自身的 `max_seqlen_k` 精确计时，消除全局最大长度带来的带宽虚高失真。

### 2.2 FlashAttention & FlashMLA：SM100 Softmax 零溢出、MLA Sink 与多卡 `CUDAGuard`
- **`117d189e87` (`2026-09-10`, `#2869`)**：在 `flash_attn/cute/flash_fwd_sm100.py` 中重构非因果 `hdim=128` 的 Softmax warp 寄存器分配，将每线程寄存器占用从触发溢出的 `255`（溢出 `64 B/thread`）压降至 `224`，彻底消除 Local Memory Spill。
- **`62892fe4a7` (`2026-09-10`, `#2768`)**：在 `flash_attn/cute/flash_fwd_mla_sm100.py` 与 `flash_bwd_postprocess.py` 中抽取共享的 `softmax.load_learnable_sink` / `apply_learnable_sink`，为 SM100 FlashMLA 前向与变长（varlen）反向补齐 Learnable Attention Sink。
- **`47c35a7123` (`2026-02-06`, `#160`) & `c741387bcb` (`2026-01-19`)**：在 `deepseek-ai/FlashMLA` 的 `csrc/sm100/prefill/dense/fmha_cutlass_{fwd,bwd}_sm100.cuh` 中补齐 `at::cuda::CUDAGuard` 与显式设备 ID 绑定，以及 `<span>` 头文件依赖，防止多 GPU 进程下跨卡上下文污染。

### 2.3 CUTLASS & Triton：IKET 示例与 LLVM `18f9e623` 升级
- **`7ac18190ca` (`2026-07-28`, `#3407`)**：CUTLASS 新增 IKET 示例与配套文档。
- **`1e259627c3` (`2026-10-21`, `#12236`) & `64ccfff9e9` (`2026-10-21`, `#12361`)**：Triton 升级底层 LLVM 至 `18f9e623`，并在 `testing.perf_report` 保存结果前自动创建父目录 `save_path`。

---

## 3. 确定性离线建模与验证结果

通过 `experiments/flashinfer_sm120_mqa_cake_lmhead_fa_sm100_sink_phase167/run.py` 对上述核心机制进行合成建模验证：

| 评估维度 | 基线表现 | Phase 167 优化后表现 | 核心收益 / 架构影响 |
|---|---|---|---|
| SM120 Paged FP8 MQA Logits (`9` shapes, `h=64, next_n=5/6`) | 平均延迟 `48.456 μs` | 平均延迟 `32.433 μs` (`native-atom`) | **`1.4929x` 加速**，消除多步投机验证瓶颈 |
| Cake SM100a 分块 LM-Head + Loss (`T=4096, V=131072, 8 chunks`) | 物化 FP32 Logits 流量 `2.0 GiB` | 仅写出分块 LSE/Loss `256 KiB` | **`8192x` HBM 写流量削减**，`expect-tx` 竞态归零 |
| CuTe SM100 非因果 `hdim=128` Softmax | `64 B/thread` 寄存器溢出 | **`0 B/thread` 溢出**（`224 regs/thread`） | **`+18.6%` TFLOPS**；MLA Sink varlen 反向最大误差 `1.2e-6` |

---

## 4. 解决方案架构师设计建议

1. **SM120a 投机解码形状目录对齐**：在 RTX 5090 / RTX PRO 6000 Blackwell 工作站部署 MTP（`next_n=5/6`）时，确保启用包含 `#8136` 目录路由表的 FlashInfer 版本。
2. **SM100a 编译工具链下限锁死**：编译包含异步 TMA `mbarrier` `expect-tx` 的分块融合算子（如 `cake_lm_head_loss`）时，必须强制 `nvcc >= 12.9` 并校验 arrive-first 语义。
