# FlashInfer SM120 MegaMoE 寄存器溢出消除、TRT-LLM-Gen 多波次 KV 切分、Kimi-K3 AttnRes Round-2 与 NVFP4 Attention 独立包及 TRT-LLM 零拷贝视频编码

> **一手上游来源**:
> - `flashinfer-ai/flashinfer@cd6c1885a897c95a3728aae6eb7a2d14ca88aa16` (`fix(cake_mega_moe): [SM120] MegaMoE register spill with CUDA 13.3 headers; compile-check every EP (#6770)`, 2026-10-21)
> - `flashinfer-ai/flashinfer@a3f2525e39dffc98dcad4ce50706641a580358f0` (`perf(trtllm-gen): opt-in multi-wave KV split for decode when one wave gives <= 2 splits (#6331)`, 2026-10-21)
> - `flashinfer-ai/flashinfer@3b65505f40e34d1bd08d9364811057242e3342fd` (`perf(cake_kimi_k3_attn_res): round-2 Kimi-K3 AttnRes routes and programs for SM100 / SM103 as a vendorable cake-build package (#6904)`, 2026-10-21)
> - `flashinfer-ai/flashinfer@edfb3f7f883950308f74ce6e8d41759eb21afa12` (`refactor(cake_nvfp4_attention): ship the experimental NVFP4 attention as a self-contained vendorable package (#6902)`, 2026-10-21)
> - `NVIDIA/TensorRT-LLM@c018848da6f1250dcf7abb7eaf3b113a774f9aa0` (`[None][fix] Encode VisualGen video without a GIL-held frame copy (#20575)`, 2026-10-20)

---

## 1. SM120 Cake MegaMoE 在 CUDA 13.2–13.4 下的寄存器溢出根因与三重消解 (`flashinfer@cd6c1885a8`)

### 1.1 根因：libcu++ `cuda::atomic_ref` 自动存储检测与多 GIN 后端内联膨胀
在 `sm120_mxfp4_mxfp8_bf16_cake` 持久化算子中，内核紧贴 **168 寄存器上限（`SM120_REG_CAP = 168`）** 与 **8 字节局部栈上限（`LOCAL_BYTES_LIMIT = 8`）** 运行。一旦发生哪怕 1 个寄存器溢出（`1 spilled register = 16 bytes local memory`），`_build_workspace` 就会拒绝加载模块。
上游定位出三大导致 CUDA 13.1–13.4 下 `34/352`（13.3）乃至 `80/352`（13.4）实例溢出的根因：
1. **CCCL `isspacep.local` 分支膨胀（+10 regs）**：从 CUDA 13.2 起，`libcu++` 对每个通用指针上的 `cuda::atomic_ref` 操作插入自动存储地址空间检测（`isspacep.local`）与局部内存回退分支。而 NCCL Device API（`nccl_device.h`）的 GIN 代理队列、信号量与计数器仅访问 NCCL 管理的设备/锁页主机内存，绝不会指向线程局部自动存储。
2. **三路 GIN 后端运行时 `switch` 内联膨胀（+10~20 regs & 8B stack）**：此前单份内核同时编译进 `PROXY`、`GDAKI` 与 `GPI` 三个 NCCL GIN 后端并在运行时分支切换，且 `GPI` 栅栏引入了 8 字节栈帧。
3. **`m128` 4 步 K 展开与 >16 Rank 寄存器表膨胀**：`m128` 变体在主循环做 4 步 MMA K 展开，且当 `EP > 16`（如 `EP=32`）时任务构建器将 per-source 路由表放在寄存器数组中。

### 1.2 修复方案与 15-bit 有符号 ABI 边界 (`ACTIVE_ROWS_FIELD_MAX = 0x7FFF`)
- **Part A (`_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE`)**：在 `nccl_device.h` 之前及 `nvcc` 命令行定义 `_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE`，完全剔除 `isspacep.local` 检查，且在 CUDA 13.0.88 下生成的 cubin 与原版 64/64 实例字节级完全一致。
- **Part B（按 GIN 后端独立编译 + `m128` 单步 K 迭代 + `tb_pub` 工作区）**：
  - 按 `NCCL_GIN_{PROXY,GDAKI,GPI}_ENABLE` 拆分编译独立模块，使 `PROXY` 与 `GDAKI` 栈帧降至 `0 B`（`GPI` 为 `8 B`）；
  - `m128` 改为主循环每次迭代发射 1 步 MMA K，且 `EP > 16` 时将 per-source 表移入 `tb_pub` 共享工作区；
  - **15-bit ABI 修正**：内核参数 `active_rows` 是有符号 32 位整数，低 16 位存当前调用行数，高 16 位存单 Rank 最大行数并通过**算术右移（arithmetic shift）**解码。若设为 `0xFFFF`（`65408`），当行数 `>= 32768 (0x8000)` 时符号位为 `1`，算术右移会解码出负数（如 `-32768`）。因此将 `ACTIVE_ROWS_FIELD_MAX` 从 `0xFFFF` 紧缩为 `0x7FFF`，对应 `max_rows <= 32640`（`128` 对齐）。
- **验证矩阵**：覆盖 **11 个 CUDA 13 版本（13.0.48 至 13.4.92）× 3 个 GIN 后端 × 480 个实例（EP 4/8/16/32）= 每版本 1440 个 JIT 实例**，实现 **0 溢出（130–168 寄存器，0 B local memory）**，且新增支持 `DeepSeek-V4.1-Flash`（`E384 / H5120 / I2304 / top-k 6`）。

---

## 2. TRT-LLM-Gen GQA Decode 多波次 KV 切分 (`flashinfer@a3f2525e39`)

### 2.1 单波次切分在中等 Batch 与偏斜长上下文下的尾延迟瓶颈
在 `TllmGenFmhaKernel` 中，默认单波次规则按 `splits = SMs // (batch * headsKv)` 向下取整计算每个序列的 KV 切分数。在 NVIDIA B200（`148 SMs`）运行 32 Q / 2 KV 头（`base_ctas = 2 * batch`）时：
- 当 `B = 25`（`base_ctas = 50`）或 `B = 37`（`base_ctas = 74`）时，单波次仅给出 `2` 个 split；
- 当 `B = 38..148`（`base_ctas = 76..296`）时，单波次仅给出 `1` 个 split！
在长上下文或对数正态分布（`lognormal, sigma=0.6`）的混合长度 Decode 批次中，单个超长 KV 请求会由极少数 CTA 串行扫完整个 KV Cache，主导整个 Kernel 的尾延迟。

### 2.2 `FLASHINFER_TRTLLM_GEN_MULTI_WAVE_KV_SPLIT=1` 最多 4 波次扩容与安全回退
提交 `a3f2525e39` 引入可选的多波次 KV 切分机制：
- **触发条件**：仅针对最终选定的 GQA Decode 内核（不含 MLA，不影响 tile 搜索），当单波次计算的 `splits <= 2` 时，允许按最多 `kMaxNumWavesForKvSplit = 4` 波次计算切分数：`candidate_splits = (4 * SMs) // base_ctas`（最多产生 11 个 splits，在 `B=25` 时 `2 -> 11`，`B=38` 时 `1 -> 7`，`B=74` 时 `1 -> 4`）。
- **三大硬件与工作区安全不变量**：
  1. **规约内核网格上限**：`static_assert(kMaxNumWavesForKvSplit <= 4)` 且要求 `base_ctas <= 2 * SMs`，防止独立规约内核 `fmhaReduction.cu` 计算出 0 大小网格；
  2. **动态 Partial Stats 缓冲区定容**：`partialStatsBufferSize = max(mMultiProcessorCount, totalNumCtas) * mStepQ`，按实际启动的 CTA 总数（可超过 SM 数）而非固定 SM 数分配；
  3. **工作区精确容量回退**：启动器传入剩余工作区字节数 `mMultiCtasKvScratchBytes`；若在 CGA/GMEM 重选后剩余空间不足（哪怕仅差 1 字节），立即无损回退至默认单波次切分。
- **实测收益（B200 148 SMs, CUDA Graph `max_seq_len=1M`）**：
  - 在 `64k lognormal` 下改动点几何平均提速 **1.86x**（`B=38`: `311 -> 177 us`, **1.75x**；`B=74`: `721 -> 386 us`, **1.87x**）；
  - 在 `288k lognormal` 下 `B=38` 从 `1566 us` 降至 `775 us`（**2.02x**）；在 SM107 上 `B=54..64` 混合长度 Decode 延迟最高降低 **3.0x**。

---

## 3. Cake Kimi-K3 AttnRes Round-2、NVFP4 Attention 独立包与 TRT-LLM 零拷贝视频编码

| 模块与上游提交 | 核心架构机制 | 定量基准与验证契约 |
| :--- | :--- | :--- |
| **Cake Kimi-K3 AttnRes Round-2** (`flashinfer@3b65505f40`) | 为 `H=7168, K in [0, 8]` BF16 算子融合残差相加、可选快照写出（`block_write_idx == K`）、`K+1` 候选打分、Softmax、混合与输出 RMSNorm；采用 `cake-build/1` 清单封装 `133` 个程序（`161` 个库：`75` `sm_100a` + `86` `sm_103a`）。 | 覆盖 `291` 个契约形状 × 2 PDL 模式 = `582` 行测试：GB300 几何平均 **1.0072x**（`serving_m5_k5` **1.0888x**，`4.448 -> 4.096 us`），B300 **1.0064x**，B200 `k_sweep_m1_k6` PDL-on **1.1034x**（`5.120 -> 4.640 us`）。 |
| **Cake NVFP4 Attention 独立包** (`flashinfer@edfb3f7f88`) | 将 SM103a 实验性 NVFP4 Q/K/V + BF16 输出持久化 Tile Attention（`HEAD_DIM=128`, `SF_VEC=16`, `TILE_ROWS=512`, `E2M1_BOUNDARIES=(0.25..5.0)`）重构为自包含 `cake-build/1` 包，支持 AOT 预编译 `.so` 与 JIT 自动回退。 | GB300 4 个基准形状几何平均性能比 **0.9975x**（`B=4, H=8, S=4096` 为 `114.161 us` vs `114.177 us`，**1.000x**）。 |
| **TRT-LLM VisualGen 零拷贝编码** (`TensorRT-LLM@c018848da6`) | 在 `tensorrt_llm/media/encoding.py` 的 `encode_video` 中，用 `memoryview(video_np.reshape(-1))` 替代 `video_np.tobytes()` 喂入 `ffmpeg` 标准输入管道。 | 对 120 帧 1080p RGB 视频消除 **711.91 MiB**（`746,496,000 B`）的主机端额外内存分配与持有 GIL 的全帧内存拷贝。 |

---

## 4. 本地确定性复现实验 (`experiments/flashinfer-sm120-megamoe-spill-trtllm-multiwave-kimi-k3-nvfp4/run.py`)

运行以下命令可完整复现 SM120 15-bit ABI 溢出校验、B200 多波次 KV 切分表与 1 字节差回退、NVFP4 E2M1 边界量化及 VisualGen 零拷贝 `memoryview` 验证：

```bash
python experiments/flashinfer-sm120-megamoe-spill-trtllm-multiwave-kimi-k3-nvfp4/run.py --check
```
