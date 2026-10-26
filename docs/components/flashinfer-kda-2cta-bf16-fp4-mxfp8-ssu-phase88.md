# FlashInfer KDA 2-CTA CP 状态递推、BF16×FP4 TMEM 提前释放与分块缩放预载、分组 MXFP8 整数 UE8M0 量化、Mamba2 SSU P-Split/TMA-Ring 与 TileLang SM120 寄存器-A GEMM 展开（Phase 88）

> **Language / 语言**: **中文 (Chinese)** | [English (`flashinfer-kda-2cta-bf16-fp4-mxfp8-ssu-phase88.en.md`)](flashinfer-kda-2cta-bf16-fp4-mxfp8-ssu-phase88.en.md)

- **关联队列题（Queue ID）**：`flashinfer-kda-2cta-bf16-fp4-mxfp8-ssu-phase88`（`priority: 97`）
- **关联证据 ID（Evidence ID）**：`ev-flashinfer-kda-2cta-bf16-fp4-mxfp8-ssu-phase88-2026`
- **验证状态（Verification Status）**：`own-offline`（上游一手源码核验 + Python 3.12 标准库离线复现 [`experiments/flashinfer_kda_2cta_bf16_fp4_mxfp8_ssu_phase88/run.py`](../../experiments/flashinfer_kda_2cta_bf16_fp4_mxfp8_ssu_phase88/run.py)；GPU 硬件耗時は上游作者在 B200/GB300/SM107/SM120 实测之 `author-benchmark`）

---

## 1. 核心架构问题与结论速览

在 NVIDIA Blackwell（`sm_100a` B200、`sm_103a` GB300、`sm_107a`、以及工作站级 `sm_120` RTX PRO 6000 Blackwell）上，混合线性注意力（Kimi Delta Attention, KDA / Mamba2 Selective State Update, SSU）与超低精度（NVFP4 W4A4、BF16×FP4 W4A16、MXFP8 E4M3+UE8M0）算子面临五大微架构瓶颈：
1. **Context-Parallel（CP）KDA 状态递推的 SM 驻留度受限**：每个 `(segment, head)` 的 $128 \times 128$ FP32 递推状态占 $65,536\text{ B}$，全宽单 CTA 方案每个 SM 只能驻留 1 个 CTA；当单卡仅分担 1~2 条长文档链时，大量 SM 空闲。
2. **Blackwell `mm_bf16_fp4` 双 CTA `tcgen05` 尾声同步与小 $M$ 缩放重复加载**：Cluster 级 TMEM 释放若等待最末尾 cluster barrier，会拖长尾声流水线；Split-K 小 $M$ 寄存器流式核在主循环内反复读取并转换 E4M3 块缩放因子。
3. **分组 MXFP8 量化的 FP32 除法指令开销**：在 $K=4096/2048/7168$ 下按每 32 元素计算 UE8M0 缩放字节时，若使用 FP32 除法求幂指数，算术指令吞吐成为瓶颈。
4. **CTA-Pair（`cta_group::2`）All-Gather Matmul 的异步到达竞态（PTX §2.2.2）**：Leader CTA 的 MMA warp 通过多播 `tcgen05.commit` 向双 CTA 的 `mma_done` mbarrier 发送到达信号，若 Peer CTA 的 load warp 在最后一次 refill 后直接退出线程块，仍在飞行中的多播 `commit` 将打在已退出的 CTA 共享内存上，触发未定义行为。
5. **SM120 Register-A 块缩放 GEMM 的局部内存溢出（Local Memory Spill）**：当寄存器片段 $A$ 跨越多个 64 宽度的 $K$ 原子且使用运行期循环索引时，编译器被迫将整个寄存器片段溢出到 Local Memory，且每次 MMA 重复从共享内存发起 3 次缩放加载。

### 一手源码对标矩阵

| 上游仓库与提交 | 关键算子 / 路径 | 微架构机制 | 实测 / 离线核验收益 |
|---|---|---|---|
| `flashinfer-ai/flashinfer` `#7415` (`182159eb7a`, 2026-10-26) | `csrc/fused_moe/warp_decode/cake_warp_decode_contract.cuh` | `sm_103a` 上 100 行切换至融合 route-pack FC1 + K512 MMA-U2 FC2；20 行切换至固定 top-10 向量 finalize；24 行切换至线性网格 K256 FC2 | 消除独立 route-pack 启动（每层 MoE 启动从 `3` 降至 `2`，减少 `1.50x`），输出逐位一致 |
| `flashinfer-ai/flashinfer` `#7436` (`4f14b25dc6`, 2026-10-26) | `flashinfer/experimental/cake_kda_train/cake_kcp.py` | CP 状态递推新增双 CTA 记录（每 CTA 负责 64 列，$32,768\text{ B}$ FP32 状态，每 SM 驻留 2 CTA），按链长自动路由 | GB300 144 组测试中最高 `1.0754x`（B1 前向）与 `1.0512x`（B3 前向+重放+反向），0 组退化 >2% |
| `flashinfer-ai/flashinfer` `#7419` (`328e862b34`, 2026-10-26) | `csrc/blackwell_bf16_fp4/` | 双 CTA $M=128$ `tcgen05` 在最后存储飞行中提前释放对端 TMEM；小 $M$ Split-K 在主循环前一次性加载转换 E4M3 缩放 | $768/2112/2048$ 达 `1.099x`（GB300）/ `1.085x`（B200）；$16/1024/1024$ 达 `1.166x`（GB300）/ `1.205x`（B200） |
| `flashinfer-ai/flashinfer` `#7431` (`bdaa2401ad`, 2026-10-26) | `csrc/cake_grouped_mxfp8_quantize/` | 新增 $K=4096$（256 线程 2 行/CTA）与 $K=2048$（128 线程/CTA）程序，并在 $K=7168$ 改用整数位运算提取 UE8M0 缩放字节 | $[32,1024,4096]$ 16 行/组稀疏掩码下达 `1.3552x`（GB300）/ `1.3632x`（GB200） |
| `flashinfer-ai/flashinfer` `#7407` (`fdd121f46b`, 2026-10-26) | `csrc/cake_selective_state_update/` | Mamba2 SSU 单 token 解码沿头维 2/4 路 P-Split（CTA 数扩大 `2x~4x`）、1~2 token 投机解码 TMA-Ring、Z-gated 持久双头流水 | 填满 B200（148 SM）与 GB300（152 SM）尾波，投机解码绕过冗余 proxy fence |
| `flashinfer-ai/flashinfer` `#7417` (`a8d23b20bf`) & `#7434` (`9e656abf02`) | `cake_rmsnorm_train_backend/` & `cake_minimax_h3_nvfp4_pre_attention/` | RMSNorm 反向 `dweight` 分阶段 slab 归约（`sm_107a` 每波 8/16 趟）；MiniMax-H3 NVFP4 预注意力融合 Norm+AdaLN+Quant+QKV GEMM | RMSNorm 反向在 `sm_107a` 达 `1.18x`（`sm_103a` `1.07x`）；MiniMax-H3 GB300 达 `1.1198x` |
| `flashinfer-ai/flashinfer` `#6885` (`0af9594727`, 2026-10-26) | `csrc/cake_all_gather_matmul/` | 7 个 CTA-pair 主程序在 tile 循环结束后追加 `producer_tail`，等待最后 `ring_depth` 级 `mma_done` 屏障释放后再退出 CTA | 彻底消除 PTX §2.2.2 跨 CTA 共享内存悬空到达竞态（GB300 375 项测试 0 hazard） |
| `tile-ai/tilelang` `#3286` (`32f56122e2`) & `#3649` (`89b5b5b9ad`) | `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py` & `tilelang/jit/adapter/` | SM120 寄存器-A 块缩放 GEMM 显式展开 $K$-atom 循环并按原子打包加载 SFA/SFB 缩放；JIT 适配器在调用方 `stream` 上分配 `out_idx` | 消除 $K>64$ 局部内存溢出，每 lane 每原子共享内存缩放加载从 `12` 次降至 `2` 次（减少 `6.0x`）；跨流覆盖风险清零 |

---

## 2. 关键实现路径、数据布局与同步机制深挖

### 2.1 `cake_kda` Context-Parallel 双 CTA 状态递推（PR `#7436`）

在 Kimi-K3 等混合架构的打包序列（packed document sequence）训练中，KDA 的每个注意力头维护一个维度为 $D_k \times D_v = 128 \times 128$ 的 FP32 递推状态矩阵 $H \in \mathbb{R}^{128 \times 128}$（单头字节数 $128 \times 128 \times 4 = 65,536\text{ B}$）。
- **单 CTA 全宽记录（Full-Width Record）**：每个 `(segment, head)` 启动 1 个 CTA 负责全部 128 列值维度。由于寄存器与共享内存占用，每个 SM 只能驻留 1 个 CTA。当 Context Parallelism（如 `CP=8` 或 `CP=16`）切分长文档后，若单 rank 仅包含 1~2 条长分段链且头数较少，总 CTA 数 `n_walk * num_heads` 无法填满 GB300 的 152 个 SM。
- **双 CTA 列切分记录（Two-CTA Record）**：由于递推更新沿值维度（Value Columns）完全正交独立：
  $$H_t[:, c] = \text{diag}(\alpha_t) H_{t-1}[:, c] + k_t \cdot v_{t, c}^{\top}, \quad c \in \{0..63\} \cup \{64..127\}$$
  PR `#7436` 将每个 `(segment, head)` 拆分为 2 个 CTA，分别处理左半 64 列（$32,768\text{ B}$）与右半 64 列（$32,768\text{ B}$），使每个 SM 可同时驻留 2 个 CTA。
- **按启动自适应选择（`cake_kcp_jit.walk_form`）**：两个写每块状态带的 `tape` 递推（`kcp_fwdh_tape`、`kcp_bwd_tape`）固定启用双 CTA 记录；两个仅传递段间状态的 `offset` 递推（`kcp_fwdh_offset`、`kcp_bwd_offset`）根据当前批次各链长度、头数与 SM 数量进行列表调度（list-scheduling）拟合判定：当拆分为双 CTA 不会延长最长链所在波次时自动切换为双 CTA，在 GB300 16 卡 NVL72（`CP=16`）上将模块前向（B1）提速至 **1.0754x**（`2.667 ms -> 2.480 ms`），前向+重放+反向（B3）提速至 **1.0512x**（`9.634 ms -> 9.165 ms`）。

### 2.2 Blackwell `mm_bf16_fp4` TMEM 提前释放与分块缩放预转换（PR `#7419`）

1. **双 CTA `tcgen05` TMEM 提前释放**：在 `cta_group::2` 的 $M=128$ `tcgen05` 程序（如形状 $768 \times 2112 \times 2048$）中，累加器驻留在 Tensor Memory（TMEM）。旧版代码在尾声将 TMEM 搬入寄存器并发起 TMA 全局存储后，等待一次 cluster-wide barrier 才调用 `tcgen05.dealloc`。PR `#7419` 在最后一批 TMEM-to-register 读取完成、TMA 存储仍在飞行（in-flight）时立即执行 TMEM 释放，缩短了集群尾部占用，在 GB300 / B200 上获得 **1.099x / 1.085x** 加速。
2. **小 $M$ 寄存器流式 Split-K 缩放预载**：对于 $M=16, N=1024, K=1024$ 等小批量解码形状，`blackwell-tiled` 后端采用 4-CTA mailbox、8-CTA reduce-scatter 或 6-way Split-K。由于每个 CTA 负责的 $(N, K)$ 子块固定，PR `#7419` 将 E4M3 块缩放因子的全局加载与格式转换全部提至主循环之前一次性完成，并在 PDL（Programmatic Dependent Launch）等待结束后立即读取 `alpha`，在 GB300 / B200 上实现 **1.166x / 1.205x** 加速。

### 2.3 分组 MXFP8 量化的整数位运算 UE8M0 提取（PR `#7431`）

MXFP8 为每 32 个连续元素计算一个无符号 8 位纯指数缩放字节（`UE8M0`，表示 $2^{e-127}$）。给定 32 元素块的最大绝对值 $m = \max_i |x_i|$，理想缩放因子需满足 $m / 2^{e-127} \le 448.0$（FP8 E4M3 最大有限值）。
- 旧版通过浮点数除法 `m / 448.0f` 再向上取整指数，受制于 SFU/FP32 除法吞吐。
- PR `#7431` 将 BF16/FP16 元素按双元素打包处理，直接利用整数位运算提取 IEEE-754 阶码与尾数阈值来计算 `UE8M0` 字节，并为 $K=4096$ 定制 256 线程（每 CTA 处理 2 行）、为 $K=2048$ 定制 128 线程程序。在 $[B=32, M=1024, K=4096]$ 且每专家组仅 16 个有效行的稀疏 MoE 场景下，GB300 / GB200 分别提速 **1.3552x / 1.3632x**。

### 2.4 CTA-Pair All-Gather Matmul 的 `producer_tail` 同步修复（PR `#6885`）

```
Leader CTA (MMA Warp)                  Peer CTA (Load Warp)
---------------------                  --------------------
tcgen05.mma (...)
tcgen05.commit.cta_group::2 ─────────► mbarrier (mma_done in Peer SMEM)
                                       [Bug before #6885]:
                                       Peer Load Warp exited loop after last refill!
                                       Peer CTA exits while multicast commit is in flight
                                       -> PTX ISA §2.2.2 undefined behavior!

                                       [Fix in #6885 (producer_tail)]:
                                       After tile loop, wait on mma_done for last
                                       ring_depth stages before CTA exit.
```

根据 PTX ISA §9.7.18.6，不同 `tcgen05.commit` 指令之间不保证先后顺序，因此尾声等待累加器 `commit` 并不能掩盖前面环路槽位的 `mma_done` 多播到达。追加 `producer_tail` 排空最后 `ring_depth` 个阶段后，`compute-sanitizer --tool racecheck` 与 `synccheck` 在 GB300 双卡 375 个测试上实现 **0 hazard / 0 error**。

### 2.5 TileLang SM120 Register-A 块缩放 GEMM 与流安全输出分配（PR `#3286` & `#3649`）

- **SM120 Register-A 循环展开与缩放打包（PR `#3286`）**：在 `sm_120` 上，当左矩阵 $A$ 保存在寄存器片段（`is_gemm_rs`）且 $K > 64$（跨越多个 64 宽度的 $K$ 原子）时，运行期循环变量 indexing 会迫使 NVCC 将寄存器数组落入 Local Memory。PR `#3286` 在 TIR 中对 $K$-atom 循环插入 `T.unroll(explicit=True)`，并将共享内存缩放读取从“每对 MMA 执行 3 次共享内存加载”改为“每 lane 每 $K$ 原子加载一次紧凑选择器包”：
  $$\text{Loads}_{\text{lane, atom}} = \left\lceil \frac{\text{warp\_rows}}{2} \right\rceil_{\text{SFA}} + \left\lceil \frac{n_{8\text{-blocks}}}{4} \right\rceil_{\text{SFB}}$$
  在 $\text{warp\_rows}=2, n_{8\text{-blocks}}=4$（每原子 4 对 MMA）时，每 lane 每原子的共享内存缩放读取从 $4 \times 3 = 12$ 次降至 $1 + 1 = 2$ 次（降低 **6.0x**）。
- **JIT 输出张量流绑定（PR `#3649`）**：修复 `cython`、`nvrtc`、`cutedsl` 适配器在使用非默认 `stream=` 启动时仍在 `torch.cuda.current_stream()` 上调用 `torch.empty` 分配 `out_idx` 输出张量的缺陷，防止 PyTorch 缓存分配器跨流提前复用仍在计算中的显存块。

---

## 3. AI Infra 创新架构横向对比（Fireworks / Together / Cerebras / Groq）

| 维度 | FlashInfer Cake + TileLang（Blackwell B200/GB300/SM120） | Fireworks AI（FireAttention / FP4-FP8 托管推理栈） | Together AI（Turbo / ThunderKittens / FlashAttention-4） | Groq LPU（确定性全片上 SRAM 流式芯片） | Cerebras WSE-3（晶圆级全片上数据流引擎） |
|---|---|---|---|---|---|
| **低精度矩阵与块缩放路径** | 原生 `tcgen05` NVFP4 W4A4、BF16×FP4 W4A16（TMEM 提前释放）与整数 UE8M0 分组 MXFP8 | 针对 MoE Decode 定制 FP8/FP4 block-scaled 融合路由 GEMM，减少 kernel launch 间隙 | 深度共研 FA4 CuTe-DSL 与寄存器切片原语，优化动态 batch 下高吞吐量化 GEMM | 硬件原生支持混合 INT4/INT8/FP8/FP16 流式矩阵乘法单元（MXM），编译期静态绑定缩放流 | 晶圆级稀疏/稠密张量核心，权重固定在片上 SRAM，激活以细粒度数据流小包跨核传输 |
| **混合线性注意力（KDA / Mamba2 SSU）状态管理** | CP 双 CTA 64 列切分递推（2 CTA/SM）+ P-Split/TMA-Ring 投机解码状态流 | 将 SSM/线性注意力状态与 Paged KV 统一纳入分层缓存池，降低长上下文首字延迟 | 针对混合 Mamba-Transformer（如 Nemotron-H / Qwen3-Next）采用分块重放与前缀对齐 | $230\text{ MB}$ 片上 SRAM 直接驻留固定并发槽位的循环状态，单 token 状态更新零外部 DRAM 访存 | $44\text{ GB}$ 片上 SRAM 容纳完整万亿模型状态与权重，无需跨节点 CP 状态传递或 HBM 换入换出 |
| **多流/多核同步机制** | 硬件 `mbarrier`、`tcgen05.commit` 多播、`producer_tail` 排空与 PDL 依赖启动 | 细粒度 CUDA Graph 捕获消除 Python/Host 调度开销，固化多流事件拓扑 | 依托 CuTe 显式屏障与静态描述符，避免运行期寄存器溢出与同步气泡 | 编译器按时钟周期精确编排核间数据流（零硬件缓存缺失、零运行期仲裁屏障） | 片上二维网格硬件路由（Swarm fabric）通过颜色通道自同步，避免全局内存屏障 |

---

## 4. 架构师练习与可证伪问题

1. **计算练习（KDA CP 双 CTA 驻留度与波次计算）**：设模型具有 $H = 32$ 个 KDA 头，在 `CP=16` 下单 rank 负责 $S_{\text{seg}} = 2$ 个文档分段链，GPU 具有 $152$ 个 SM（GB300）。
   - 单 CTA 全宽方案启动 $2 \times 32 = 64$ 个 CTA（每 SM 最多 1 个 CTA），SM 占用率为 $64 / 152 = 42.1\%$，另 $57.9\%$ 的 SM 空闲。
   - 双 CTA 64 列方案启动 $2 \times 2 \times 32 = 128$ 个 CTA，单 CTA 仍独占 1 个 SM 时可覆盖 $128 / 152 = 84.2\%$ 的 SM；当多文档分段达到 $S_{\text{seg}} = 4$（$256$ 个双 CTA）且每 SM 驻留 2 个 CTA（容量 $304$ CTA）时，全部在 **1 个 Wave** 内完成，而单 CTA 全宽方案（$128$ CTA vs $152$ SM，或 $S_{\text{seg}}=6 \Rightarrow 192$ CTA）在 $S_{\text{seg}}=6$ 时将跨入第 2 个 Wave（$384 \le 304$ 不成立时才分波）。
2. **后续可证伪问题**：
   - 在 `cake_grouped_mxfp8_quantize` 中，当专家路由极度倾斜（例如 32 个专家中仅 2 个专家有 $>512$ 行、其余 30 个专家为 0 行）时，按固定两行/CTA 的 $K=4096$ 行分块程序是否会因空行掩码跳过产生 CTA 负载不均？何时应切换回紧凑前缀和索引（compacted row index）路径？
   - 在 SM120 寄存器-A GEMM 中，当 $K$ 进一步增大至 $512$（8 个 64 宽度原子）时，完全展开（`T.unroll(explicit=True)`）是否会触及指令缓存（I-Cache）容量拐点？
