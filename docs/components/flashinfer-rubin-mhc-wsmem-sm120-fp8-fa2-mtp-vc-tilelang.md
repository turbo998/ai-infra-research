# FlashInfer Rubin (`sm_107a`) `mega_mhc` 权重暂存 Tailpipe、`SM120` E4M3-Query FA2 Tensor Core 解码 (`USE_VO_SPLIT`)、Rubin MLA MTP 自动路由、PrimsTS `VC-Attention-QK16` 与 TileLang 编译器护栏深度解析

> **语言 / Language**: [中文](flashinfer-rubin-mhc-wsmem-sm120-fp8-fa2-mtp-vc-tilelang.md) | [English](flashinfer-rubin-mhc-wsmem-sm120-fp8-fa2-mtp-vc-tilelang.en.md)  
> **Phase**: 71 (`priority: 80`, Track A — GPU Kernel & Quantization)  
> **核心证据等级**: `implementation` (固定 commit 源码审计) + `author-benchmark` (作者 R200 `sm_107a` 208 SMs / B200 / B300 实测报告) + `own-offline` (本项目确定性标准库仿真 `experiments/flashinfer-rubin-mhc-wsmem-sm120-fp8-fa2-mtp-vc-tilelang/run.py`)

---

## 1. 架构背景与核心工程问题

在 NVIDIA Rubin (`sm_107a` / `SM107`)、Blackwell 工作站/边缘架构 (`SM120`)、Blackwell 数据中心架构 (`B200`/`B300`) 以及 TileLang CUDA/CuTeDSL/Ascend 编译器栈的最新演进中，注意力与 MoE 算子面临五个核心架构挑战：

1. **Rubin (`sm_107a`) `mega_mhc` Hidden-4096 BF16 多波次网格下的 RMSNorm 权重重复全局内存读取 (`flashinfer-ai/flashinfer@b75cff30e9ce`, `#7123`)**：在流形约束超连接（Manifold-Constrained Hyper-Connection, mHC）融合算子中，当 Token 数 $T$ 增大使得每个 CTA 需要在网格上循环执行 5 个或更多任务波次（waves）时，原有 `tailpipe` 程序在每个 Tile 尾部都会重新从 HBM 读取 8 KiB 的 `rmsnorm_weight[4096]`。如何在不改变算子签名且保持 **Bitwise 完全一致** 的前提下，引入第三套 `tailpipe_wsmem` 程序消除重复权重流量？
2. **`SM120` 上 `head_dim=256` 的 E4M3-Query FA2 Tensor Core 解码与寄存器风暴 (`flashinfer-ai/flashinfer@5295d50ac9ff`, `#6574`)**：在 `SM120` 上执行 `q_len=1`、`head_dim=256` 的 GQA 解码时，若将 E4M3 KV Cache 反量化为 BF16 再跑 `m16n8k16` MMA，既浪费指令吞吐又导致每个线程需要维护全宽 256 维输出累加器（`128` 个 FP32 寄存器/线程）及跨 Warp 共享内存合并。如何利用原生 `m16n8k32` E4M3 MMA 配合 `USE_VO_SPLIT` 将输出累加寄存器压降 **`4x`**（至 `32` 个寄存器/线程）？
3. **Rubin `SM107` 吸收式 FP8 MLA 分页注意力的 MTP 投机解码内核调度与 `Int32` LSE 偏移溢出 (`flashinfer-ai/flashinfer@368aae5e288f`, `#6177`) 及 `rubin_megamoe` 延迟 Top-K 路由权重双重缩放修复 (`@fb644f5e0a4d`, `#6414`)**：在 Multi-Token Prediction (`Q in {2, 4}`) 下，何时应从 `cute-dsl-monolithic` 切换至 `cute-dsl-rubin-mtp`？当 `total_q * max_num_splits * num_heads * sizeof(float)` 超过 `2 GiB` 时为何会发生 `Int32` 指针回绕？
4. **PrimsTS `VC-Attention-QK16` 值中心化（Value-Centered）FP8 E4M3 残差注意力与高残差 Repair Tiles (`flashinfer-ai/flashinfer@0d5dfbaf5424`, `#6296`)**：直接将 $V$ 量化为 FP8 E4M3 会因为跨通道/Token 均值偏置损失精度。如何将 $V$ 存储为围绕 128-token Tile 均值 $\mu_t$ 的 E4M3 残差，并通过 `K=16` BF16 UMMA 均值补偿或 `repair_tiles` 在 B200/B300 上实现 **`1.15x–1.33x`** 加速？
5. **TileLang 步进空循环展开越界 (`tile-ai/tilelang@dccd4b919ba7`, `#2887`)、Warp-Specialized 共享内存 `T.cumsum` 线程偏移缺失 (`@b69e0c2c0c45` `#3576` / `@cb6455d680cc` `#3536`)、Keep-Dim 归约形状推导 (`@c362c962d085`, `#2815`) 与 Ascend DeepGEMM L1 流水级钳制 (`@5bd0ec8dfa0a`, `#3577`)**：为何 `extent=0, step=4` 的空循环在 C++ 整除 `1 + (0 - 1) / 4` 下会被错误执行 1 次？为何 Hopper/Blackwell 消费者 Warp (`threadIdx.x >= 128`) 调用 `CumSum1D/2D` 会静默失效？

---

## 2. 核心实现路径与源码级机制拆解

### 2.1 Rubin (`sm_107a`) `mega_mhc` Hidden-4096 三阶段 BF16 路由与 `tailpipe_wsmem` (`#7123`)

在 `flashinfer/experimental/deepgemm_mega_mhc/mega_mhc.py` 与 `csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_41024b807c02040677b3_kernel.cu` 中，针对 `sm_107a` 增补了权重暂存程序 `cake_mega_mhc_41024b807c02040677b3` (`regime="tailpipe_wsmem"`)：

- **三阶段宿主端精确路由 (`_expected_h4096_program`)**：
  设 $M = \lceil T / 64 \rceil$ 为 64-token 块数，`splits = mhc_infer_num_splits(T, num_sms)`，`grid = mhc_infer_num_ctas(T, num_sms, splits)`，每个 CTA 承担的任务波次数为：
  $$
  \text{waves} = \left\lceil \frac{M \times \text{splits}}{\text{grid}} \right\rceil
  $$
  1. **`bf16` (`cake_mega_mhc_f24c2f3f98bfe3356398`)**：当 $\text{splits} \in \{32, 64\}$（在 208-SM R200 上对应 $T \le 384$）时启用；
  2. **`tailpipe` (`cake_mega_mhc_c549b064530c3765f9ba`)**：当 $\text{splits} \le 22$ 但未达到 5 波次阈值（在 208-SM GPU 上对应 $385 \le T \le 3328$）时启用，利用尾管线（tailpipe）将第 1 轮加载与归约折叠重叠；
  3. **`tailpipe_wsmem` (`cake_mega_mhc_41024b807c02040677b3`)**：当满足以下双条件时切换：
     $$
     \text{splits} \le 16 \quad \land \quad \text{waves} \ge 5
     $$
     各 SM 数量对应的精确切换阈值 `H4096_WSMEM_SWITCH` 为：`{148: 2369, 152: 2433, 208: 3329, 212: 3393, 216: 3457}`。
- **共享内存权重暂存与流量节省**：
  在 `tailpipe_wsmem` 中，每个 CTA 在进入任务循环前将 `rmsnorm_weight[4096]`（`8,192 B`）一次性加载至共享内存（`wsmem`），后续该 CTA 执行的 $\text{waves} \ge 5$ 个任务直接从 SMEM 读取权重。在 R200 (`208` SMs, `splits=16`) 上：
  - $T=4096$ ($M=64$, $\text{tasks}=1024$, $\text{waves}=5$)：全局权重读取从 `8,388,608 B` 降至 `1,703,936 B`（节省 **`79.69%`**），CUPTI 实测内核耗时从 `54.064 us` 降至 **`53.064 us` (`1.019x`)**；
  - $T=8192$ ($M=128$, $\text{tasks}=2048$, $\text{waves}=10$)：全局权重读取从 `16,777,216 B` 降至 `1,703,936 B`（节省 **`89.84%`**），实测耗时从 `110.680 us` 降至 **`108.648 us` (`1.019x`)**；
  - 三套程序对 `new_residual` (BF16)、`y_bf16` (BF16)、`new_post_mix` (FP32) 与 `new_comb_res_mix` (FP32) 保持 **Bitwise 逐位全等**。

### 2.2 `SM120` E4M3-Query FA2 Tensor Core 解码与 `USE_VO_SPLIT` (`#6574`)

在 `csrc/flash_infer/attention/` 与 `flashinfer/decode.py` 中，为 `SM120` 启用了 `q_dtype=fp8_e4m3fn, kv_dtype=fp8_e4m3fn, out_dtype=bfloat16` 的 Tensor Core FA2 解码 (`head_dim=256`, `q_len=1`, GQA group size $\in \{1, 2, 4, 8\}$)：

- **全链路 `m16n8k32` E4M3 MMA**：$Q \cdot K^T$ 与 $P \cdot V$ 均直接调用 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`，无需将 K 或 V 元素上转为 BF16；相比 BF16 `m16n8k16`，每个 64-token KV Tile 的 $Q \cdot K^T$ MMA 指令数从 `128` 条减半至 **`64` 条**。
- **`USE_VO_SPLIT` 消除跨 Warp 归约与寄存器膨胀**：
  - 在 `head_dim_vo = 256`、`NUM_WARPS_KV = 4` 下，传统切分让每个 KV Warp 持有完整 256 维输出累加器（每线程 `128` 个 FP32 寄存器），并在尾部通过共享内存跨 4 个 KV Warp 合并输出。
  - 启用 `USE_VO_SPLIT` 后，每个 KV Warp 将在线 Softmax 后的概率 $P \times 256$ 以 E4M3 字写入共享内存（`P_smem`），随后第 $w \in \{0,1,2,3\}$ 个 KV Warp 仅负责加载并累加输出维度的 `64` 维切片 $[64w : 64(w+1)]$。
  - 每个线程的 $O$ 累加器寄存器从 `128` 骤降至 **`32` 个 FP32 寄存器**（压降 **`4x`**），且彻底消除了尾部跨 Warp 输出归约阶段；实测相对 FP32 参考的最大绝对误差仅 `0.0138`（LSE 最大误差 `7.9e-6`）。

### 2.3 Rubin `SM107` 吸收式 FP8 MLA `cute-dsl-rubin-mtp` 自动调度 (`#6177`) 与 MegaMoE 延迟路由修复 (`#6414`)

1. **`BatchMLAPagedAttentionWrapper` Rubin MTP 后端 (`flashinfer@368aae5e288f`, `#6177`)**：
   - 针对 DeepSeek 吸收式 MLA (`H=128, D_ckv=512, D_kpe=64`, FP8 KV Cache, BF16 Q/Out) 在 Multi-Token Prediction (`q_len_per_req` $\in \{2, 4\}$, `page_size` $\in \{64, 128\}$) 下的投机解码场景，集成 `cute-dsl-rubin-mtp` 内核。
   - **Auto-Selection 策略**：当满足 (a) CUDA Graph 模式且 $\text{total\_q} = B \times Q \ge 256$、KV 容量 $\le 2048$，或 (b) Eager 模式且均匀长上下文 $K \in [8192, 32768]$ 时，自动优先选择 `cute-dsl-rubin-mtp` 替代 `cute-dsl-monolithic`。
   - **Rubin 实测加速 (`author-benchmark`)**：
     - Graph `B=64, Q=4, page=64, KV=129` ($\text{total\_q}=256$)：从 `25.149 us` 降至 **`17.252 us` (`1.458x`)**；
     - Graph `B=128, Q=2, page=128, KV=1024` ($\text{total\_q}=256$)：从 `36.074 us` 降至 **`28.159 us` (`1.281x`)**；
     - Eager `B=32, Q=4, page=128, KV=32768`：从 `277.684 us` 降至 **`249.732 us` (`1.112x`)**。
   - **`Int64` LSE 偏移防溢出**：将工作区部分 LSE 缓冲区的指针算术显式拓宽为 `int64_t`（例如当 $\text{total\_q}=32768, \text{splits}=128, H=128$ 时，字节偏移达 $32768 \times 128 \times 128 \times 4 = 2,147,483,648\text{ B} > 2^{31}-1$，旧 `int32_t` 乘法直接溢出为负数）。
2. **`rubin_megamoe` 延迟 Top-K NVFP4 路由权重一致性 (`@fb644f5e0a4d`, `#6414`)**：
   - 修复了 `defer_topk_reduce=True` 时 `workspace._unfinalized_weights` 误查原始入参 `k.apply_topk_in_fc1` 而非前端解析后的 `workspace._frontend.config.apply_topk_in_fc1` 的缺陷（防止路由权重被漏乘或重复乘两次）；同时修复了 `is_default_profile` 缓存键比对，要求同时检查 `max_tokens_per_rank` 与 `live_tokens == tuning_live_tokens`。

### 2.4 PrimsTS `VC-Attention-QK16` 值中心化 FP8 E4M3 注意力 (`#6296`)

在 `flashinfer/primsts/` 中实现了 `VC-Attention-QK16`（Value-Centered Attention，`head_dim=128`，Blackwell 2-CTA UMMA）：
- **数学分解**：将每个 128-token KV Tile $t$ 的 Value 矩阵拆解为 Tile 均值向量 $\mu_t \in \mathbb{R}^{1 \times D}$ 与零均值残差 $\Delta V_t = V_t - \mathbf{1}\mu_t$：
  $$
  P V = \sum_t P_t \Delta V_t^{\text{(E4M3)}} + \sum_t \left(\sum_{j \in t} P_{t,j}\right) \mu_t^{\text{(BF16)}}
  $$
  主循环使用 `expcast_e4m3_quad_relu` 将 $P_t$ 直接量化为 E4M3 并执行 $K=32$ 的 E4M3 UMMA（步骤数比 BF16 $K=16$ 减半）；每 16 个 Tile 构成的 Group 仅需补充 **1 步** $K=16$ BF16 UMMA 即可恢复均值项 $\mu_t$。
- **Repair Tiles 替代模式**：将残差范数最高的 Top-$R$ 个 128-token 块作为 BF16 `repair_tiles` 附在序列末尾（不参与 Softmax 分母累加，仅修正分子输出）。在 B200 上取得 **`1.29x–1.33x`** 加速；在 B300 Wan2.2 P1 (`B=2, H=40, S=75.6k`) 上耗时 **`131.3 ms`**（优于标准 BF16 Dense 的 `132.4 ms` 与纯均值模式的 `136.2 ms`）。

### 2.5 TileLang 编译器循环展开、Warp-Specialized 扫描与 Ascend DeepGEMM 修复 (`#2887`, `#3576`, `#3536`, `#2815`, `#3577`)

1. **步进空循环展开越界修复 (`tile-ai/tilelang@dccd4b919ba7`, `#2887`)**：在 `loop_vectorize` / unroll pass 中，当循环 `extent == 0` 且 `step > 1`（如 `step = 4`）时，旧代码直接计算 `1 + (extent - 1) / step`。在 C++ 向零截断整除下，`1 + (-1) / 4 = 1`，导致本应执行 0 次的空循环被错误展开并执行 1 次越界访存！修复后对 `extent <= 0` 显式保留空循环或直接消去。
2. **Warp-Specialized 共享内存 `T.cumsum` 线程偏移修复 (`@b69e0c2c0c45` `#3576`, `@cb6455d680cc` `#3536`)**：在 Hopper/Blackwell Warp Specialization 下，前 128 个线程（Warp 0–3）为 TMA Producer，Consumer 线程从 `thread_offset = 128` 开始。旧 `CumSum1D/2D` 与 `CumMax1D/2D` 直接使用物理 `threadIdx.x`，导致 `CumSum1D` 中 `threadIdx.x < 32` 恒为假（**0 个活跃线程**，输出全未初始化），`CumSum2D` 中 `item = threadIdx.x // 32` 从 `4` 开始越界。修复后统一减去 `thread_offset`（`tidx = threadIdx.x - thread_offset`）。
3. **Keep-Dim `ReduceOp` 布局推导 (`@c362c962d085`, `#2815`) 与 Ascend DeepGEMM 优化 (`@5bd0ec8dfa0a`, `#3577`)**：支持 `T.reduce_sum(..., keepdims=True)` 保持 1 维轴；在 Ascend DeepGEMM 中将 L1 流水级数钳制为 `min(configured_stages, num_k_iters)`，并使用 `vst2` 512B 双源交错向量存储指令替代逐通道标量写回。

---

## 3. 本项目确定性离线实验与验证 (`own-offline`)

实验脚本：`experiments/flashinfer-rubin-mhc-wsmem-sm120-fp8-fa2-mtp-vc-tilelang/run.py`（结果固化于 `summary.json`，由 `tests/test_phase71_experiments.py` 校验）：

| 验证子项 | 关键输入 / 参数 | 实测 / 仿真输出 | 架构结论 |
|---|---|---|---|
| Rubin `mega_mhc` `tailpipe_wsmem` | R200 (`208` SMs), `splits=16`, $T=4096 / 8192$ | $T=4096 \to 5$ waves (`53.064 us`, `1.019x`); $T=8192 \to 10$ waves (`108.648 us`, `1.019x`, 省 `89.84%` 权重流量) | 多波次下 SMEM 暂存 RMSNorm 权重消除重复 HBM 读取且 Bitwise 全等 |
| `SM120` E4M3 FA2 Decode (`USE_VO_SPLIT`) | `head_dim=256`, `GQA=8`, `num_warps_kv=4` | $O$ 累加寄存器从 `128` 降至 `32` (`4x` 压降)，QK MMA 指令减半至 `64` | 避免 BF16 KV 反量化与跨 Warp 输出合并，MaxAbs 误差 `0.0138` |
| Rubin MLA `cute-dsl-rubin-mtp` & LSE `Int64` | Graph `(64,4,64,129)` / Eager `(32,4,128,32768)` | 自动命中 `cute-dsl-rubin-mtp` (`1.458x` / `1.112x`)；`2 GiB` LSE 偏移需 `Int64` | 加速 MTP 投机解码并消除大容量工作区 `Int32` 负偏移崩溃 |
| PrimsTS `VC-Attention-QK16` | `S=8192`, `tile=128`, `group=16`, `repair=2` | PV UMMA 步数从 `512` 降至 `256` (E4M3) + `4` (均值) / `16` (Repair) | B200 提速 `1.29x–1.33x`，B300 Wan2.2 P1 达 `131.3 ms` |
| TileLang 步进空循环 & Warp-Spec `cumsum` | `extent=0, step=4`; `thread_offset=128, seg=32` | 空循环迭代从 `1` 修正为 `0`；Consumer 活跃线程从 `0` 恢复为 `32` | 消除空循环 OOB 与 Warp Specialization 下扫描算子静默失效 |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **计算练习**：在 148-SM 的 B200 GPU 上运行 `mega_mhc` Hidden-4096 BF16 路径，当 `splits=16` 时，求满足 $\text{waves} = \lceil 16 \times \lceil T/64 \rceil / 148 \rceil \ge 5$ 的最小 Token 数 $T$，并与 `H4096_WSMEM_SWITCH[148]` 对照。
   - *解答*：要使 $\lceil 16 M / 148 \rceil \ge 5$，需 $16 M > 4 \times 148 = 592 \implies M \ge \lfloor 592/16 \rfloor + 1 = 38$ 个 64-token 块。对应最小 Token 数为 $T = (38 - 1) \times 64 + 1 = 2369$，与源码表项 `H4096_WSMEM_SWITCH[148] = 2369` 完全一致。
2. **可证伪假设**：在 `SM120` 上对 `head_dim=256` 执行 FP8 解码时，保留跨 Warp 全宽 256 维累加比启用 `USE_VO_SPLIT` 具有更高数值精度且无性能代价？
   - *证伪条件*：不启用 `USE_VO_SPLIT` 时每线程需独占 `128` 个 FP32 输出寄存器，直接触发寄存器溢出（Register Spilling）或将 CTA 并发率压半，且尾部需额外分配共享内存同步合并 4 个 KV Warp 的 256 维向量；`USE_VO_SPLIT` 仅需 `32` 个寄存器/线程且将输出误差控制在 `0.0138` 以内。
