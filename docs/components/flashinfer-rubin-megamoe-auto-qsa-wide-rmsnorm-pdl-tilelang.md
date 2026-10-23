# FlashInfer Rubin (`sm_107a`) MegaMoE `knobs="auto"` 跨 Rank 集体工作区协调、`cake_qsa_nvfp4` (`sm_80`) 宽流全 Tile 解码、`cake_rmsnorm_train` PDL 非协作反向发射与 TileLang 16-Bit 逆三角/`nextafter` 次正规数学降级深度解析

> **语言 / Language**: [中文](flashinfer-rubin-megamoe-auto-qsa-wide-rmsnorm-pdl-tilelang.md) | [English](flashinfer-rubin-megamoe-auto-qsa-wide-rmsnorm-pdl-tilelang.en.md)  
> **Phase**: 70 (`priority: 79`, Track A — GPU Kernel & Quantization)  
> **核心证据等级**: `implementation` (固定 commit 源码审计) + `author-benchmark` (作者 A100-SXM4-80GB / A100-SXM4-40GB / GB300 / B200 实测报告) + `own-offline` (本项目确定性标准库仿真 `experiments/flashinfer-rubin-megamoe-auto-qsa-wide-rmsnorm-pdl-tilelang/run.py`)

---

## 1. 架构背景与核心工程问题

在 NVIDIA Rubin (`sm_107a`)、Blackwell (`sm_100a`/`sm_103a`)、Ampere (`sm_80`) 与 TileLang CUDA/Ascend 编译器的最新提交中，算子自动调优、多流并发调度与低精度数学语义暴露出五个关键工程问题：

1. **Rubin (`sm_107a`) MegaMoE `knobs="auto"` 跨 EP Rank 集体工作区生命周期协调 (`flashinfer-ai/flashinfer@3a4e382bc46e`, `#6535`)**：在 Expert Parallelism (EP) 下，`RubinMegaMoEWorkspace.create(..., device_buffer_bytes=None)` 通过 NVSHMEM/对称堆进行集体分配与释放 (`destroy()`)。若单 Rank 因形状桶或 `autotune_v2()` 缓存失效独自重建 `_AutoWorkspaceHolder` 或仅部分 Rank 进入 `autotune_v2()`，为何会立刻引发跨 Rank 集体通信死锁或破坏已捕获的 CUDA Graph？
2. **`cake_qsa_nvfp4` 在 `sm_80` (A100) 上的宽行长流（Wide, Long-Stream）全 Tile 解码调度 (`flashinfer-ai/flashinfer@179af3f6e08c`, `#7133`)**：当 `rows * num_kv_heads * splits` 超过 `sm_80` 单波驻留上限（$2 \times \text{sm\_count} = 216$ CTAs）且每个 CTA 需流式处理 $\ge 16$ 个 64-token 块时，为何 K-first `decode` 内核的每 Tile K/V 共享内存屏障成为瓶颈？`decode_whole_tile` 如何消除该屏障并在 A100-80GB/40GB 上获得 `1.060x–1.085x` 加速？
3. **`cake_rmsnorm_train` 反向算子移除全设备 `cudaLaunchAttributeCooperative` 改用 PDL 流序列化 (`flashinfer-ai/flashinfer@28b2c6d93f33`, `#7131`)**：为什么使用 `cudaLaunchAttributeCooperative` 实现单内核跨 CTA `dweight` 归约会在侧流（Side Stream）仅驻留 1 个单线程小内核时导致主训练流延迟暴涨 **`10x–280x`**？改用“动态申领分块主内核 + PDL (`cudaLaunchAttributeProgrammaticStreamSerialization` / `griddepcontrol.wait`) 归约内核”双发射后，如何将侧流干扰压降至 `1.13x` 以内且保持单流持平或更快？
4. **`trtllm_fused_moe_workspace_size` 未混洗（Unshuffled）FP8 路由模式初始化顺序与显式 `tactic` 估算 (`flashinfer-ai/flashinfer@75d4c6e61ddc`, `#7025`)**：为何在 `use_shuffled_weight=False` 且 `quant_mode="ds_fp8"` 或 `"fp8_qdq"` 时工作区测算会错误返回 `0` 字节？
5. **TileLang CUDA 16-Bit 逆三角/双元数学函数降级与 `nextafter` 次正规数位步进 (`tile-ai/tilelang@43f13c7ad1d1` `#3172`, `@f1a34efea68c` `#2860`, `@d85a70a97218` `#3582`)**：为什么在 CUDA `--use_fast_math` (FTZ) 下直接用浮点比较实现 FP16/BF16 `nextafter` 会把次正规数（Subnormals）冲刷为零？为什么无架构（arch-less）CUDA 目标必须显式报错拒绝而非回退至 TVM 默认的 `sm_50`？

---

## 2. 核心实现路径与源码级机制拆解

### 2.1 Rubin (`sm_107a`) MegaMoE `knobs="auto"` 与跨 EP Rank 集体工作区协调 (`#6535`)

在 `flashinfer/fused_moe/rubin_megamoe.py` 中，`rubin_megamoe(..., knobs="auto")` 通过 `_RubinMegaMoERuntimeRunner`（继承自 `TunableRunner`）接入 FlashInfer `autotune_v2()`：

- **两级 `DynamicKnobSpec` 候选过滤**：
  1. `knobs[0]` (`num_tokens_per_cta` $\in \{64, 128, 256\}$)：枚举当前形状合法的 CTA token 分块；
  2. `knobs[1]` (`option_idx`)：根据选定的 `num_tokens_per_cta` 过滤 `_candidate_runtime_knobs()` 中共享同一 `num_tokens_per_cta` 的运行时流水线参数组合（`prefetch_ahead`, `combine_slots`, `grad_prefetch_ahead`），避免笛卡尔积爆炸。
- **版本化调优缓存契约**：定义了 `BACKEND_REVISION = "sm107-block-scaled-1667b47a-runtime-options-v1"`，写入 `TuningContext.get_Instance().set_value(...)` 并参与 `ConfigRecord` 校验，一旦底层 PTX/CUBIN 修订号更新即自动淘汰陈旧调优记录。
- **`coordinate_workspace_replacement` 跨 EP 集体同步**：
  - 在 `_AutoWorkspaceHolder` 中，当单 Rank 发现缓存的 `RubinMegaMoEWorkspace` 为空或不再兼容当前 `(knobs, problem)` 时，不能独自调用集体 `workspace.destroy()` 或 `RubinMegaMoEWorkspace.create()`。
  - `coordinate_workspace_replacement(ep_group, replace=not reusable)` 执行 `dist.all_gather_object(decisions, (context.tuning_enabled, replace), group=ep_group)`：
    - 若仅部分 Rank 启用了 `autotune_v2()`（`any(tuning_ranks) and not all(tuning_ranks)`），立即抛出 `RuntimeError("rubin_megamoe(..., knobs='auto') requires all ranks in ep_group to enter autotune_v2() together...")`，阻止部分 Rank 循环跑 `tuning_configs` 而其余 Rank 只跑 1 次导致的集体通信死锁；
    - 只要任一 Rank 需要替换工作区（`any(rank_replace for _, rank_replace in decisions)`），全体 EP Rank 统一同步重建工作区。
- **CUDA Graph 捕获工作区保活**：
  - 当 `torch.cuda.current_stream(device).is_capturing()` 为真时，强制禁用在线调优（直接走缓存或启发式 `select_rubin_megamoe_knobs`）；
  - 当捕获期间分配的工作区后续在 Graph 外被新工作区替换时，旧工作区被移入 `_captured_workspaces` 列表保活而非立即 `destroy()`，确保已录制的 CUDA Graph 重放时其绑定的对称堆地址依然有效。

### 2.2 `cake_qsa_nvfp4` (`sm_80`) 宽流全 Tile 解码调度 (`#7133`)

在 `flashinfer/experimental/cake_qsa_nvfp4/lib.py` 中，针对 Ampere `sm_80` (A100, `108` SMs) 增补了 `decode_whole_tile` 编译单元与 `select_wide` 启发式规则：

- **根因诊断**：原有的 `sm_80` K-first `decode` 内核在每个 64-token Tile 内先加载 K 并计算注意力分数，经过一次 CTA 共享内存屏障（`__syncthreads`），再复用同一块共享内存加载 V 并累加输出。当 `rows=64`（`splits=4`，总计 `2048` CTAs，每 CTA 仅 `9` 个 Tile）时，较小的共享内存占用与 K-first 流水线占优；但当 `rows=128`（`splits=2`，总计 `512` CTAs，每 CTA 需长流式处理 `17` 个 Tile）时，每 Tile 一次的 K/V 共享内存复用屏障累计开销显著。
- **`select_wide` 判定公式**：
  $$
  \operatorname{select\_wide} = (\text{form} == \text{"split"}) \land (\text{rows} \times H_{KV} \times \text{splits} > 2 \times N_{\text{SM}}) \land \left(\left\lceil \frac{\text{tiles}}{\text{splits}} \right\rceil \ge 16\right)
  $$
  其中 $\text{tiles} = \lceil N_{\text{selected}} / 64 \rceil$。在 A100 ($N_{\text{SM}}=108 \implies 2 N_{\text{SM}} = 216$) 上，`rows=128, H_KV=8, selected_tokens=2051` ($\text{tiles}=33, \text{splits}=2$) 产生 $512 > 216$ 个 CTA 且每 CTA 分担 $\lceil 33/2 \rceil = 17 \ge 16$ 个 Tile，触发 `decode_whole_tile`（将单 Tile 的 K 与 V 同时驻留在独立 SMEM 区域，每 Tile 省去一次 CTA 屏障）。
- **作者实测性能 (`author-benchmark`, `rows=128, num_kv_heads=8, selected_tokens=2051`)**：
  - **A100-SXM4-80GB**：NHD 从 `198.50 us` 降至 **`182.91 us` (`1.085x`)**；HND 从 `198.53 us` 降至 **`185.63 us` (`1.069x`)**；
  - **A100-SXM4-40GB**：NHD 从 `218.48 us` 降至 **`201.83 us` (`1.083x`)**；HND 从 `220.37 us` 降至 **`207.99 us` (`1.060x`)**；
  - 同时 `rows=64`（$\lceil 33/4 \rceil = 9 < 16$）保持在原 `decode` 内核，避免负优化。

### 2.3 `cake_rmsnorm_train` 非协作 PDL 反向双发射消除侧流 280x 阻塞 (`#7131`)

在 `flashinfer/cake_rmsnorm_train.py` 与 `flashinfer/cake_rmsnorm_train_backend/` 中，重构了 `sm_100a`/`sm_103a`/`sm_107a` 上 `H in {512, 1536, 2048, 4096, 6144}` 的 RMSNorm 反向算子：

- **全设备 Cooperative Launch 的多流队头阻塞隐患**：
  - 旧实现通过 `cudaLaunchAttributeCooperative` 启动占满全 GPU 的单内核，在内核尾部执行 `cooperative_groups::this_grid().sync()` 后归约 `partial[n_chunks, H] -> dweight[H]`。
  - CUDA 硬件调度器要求 Cooperative Kernel 的所有 CTA 必须在同一时刻全部驻留。一旦另一个非阻塞 CUDA Stream 上有哪怕 **1 个单线程驻留内核**占用了 1 个 SM 的 1 个 Warp 槽位，请求全芯片满配驻留的 Cooperative Launch 就会被硬件挂起，直到侧流内核完全退出，在训练重叠通信/监控流场景下造成 **`10x–280x`** 的灾难性停顿。
- **非协作主内核 + PDL 流序列化归约内核**：
  - **内核 1（Chunk-Claiming 主内核）**：普通非协作发射（不带 `cudaLaunchAttributeCooperative`），CTAs 通过原子计数器 `counters` 动态申领行分块（`rows_per_chunk`），计算 `dx` 并将分块部分和写入 `partial[chunk, H]`；若某 SM 被侧流占用，已驻留的 CTA 自动多申领分块，无需等待全芯片空出。
  - **内核 2（`dweight` 归约内核）**：携带 `cudaLaunchAttributeProgrammaticStreamSerialization` (`programmaticStreamSerializationAllowed = 1`) 发射，并在入口执行 PTX `griddepcontrol.wait`，由硬件保证主内核全局内存写入对归约内核可见，省去传统双内核主机端/流间冷启动气泡。
- **实测对比 (`author-benchmark`, GB300 `152` SMs / B200 `148` SMs)**：
  - **并发侧流抗干扰**：当侧流驻留 1-thread 内核时，旧 Cooperative 实现停顿 **`10x–280x`**，新 PDL 双发射在全形状下最大减速仅 **`1.13x`**（在 $T=65536$ 时）；
  - **单流延迟对比**：GB300 上 $(T=4096, H=512)$ 从 `16.0 us` 降至 **`13.7 us` (`1.168x`)**，$(T=16384, H=2048)$ 从 `27.6 us` 降至 **`24.4 us` (`1.131x`)**；B200 上 $(T=4096, H=512)$ 从 `18.8 us` 降至 **`15.9 us` (`1.182x`)**，$(T=16384, H=2048)$ 从 `31.1 us` 降至 **`27.9 us` (`1.115x`)**。

### 2.4 `trtllm_fused_moe_workspace_size` 未混洗 FP8 修复与显式 `tactic` 估算 (`#7025`)

在 `csrc/trtllm_fused_moe_kernel_launcher.cu` 与 `flashinfer/fused_moe/core.py` 中：
- **Bug 根因**：在测量工作区大小时，旧代码要么在 `scratch_.begin_measure()` 之前调用了 `prepare_routing()`（漏计路由缓冲区），要么在未调用 `prepare_routing()` 的情况下直接调用 `prepare_moe_runner(tactic)`。对于 `use_shuffled_weight=False` 的 DeepSeek FP8 (`ds_fp8`) / `fp8_qdq`，`prepare_moe_runner` 依赖 `prepare_routing()` 写入的路由模式标志（如 `RoutingMethodType`），缺失该标志会导致 `get_Configs()` 认为没有任何合法配置，从而使 `trtllm_fused_moe_workspace_size` 返回 **`0` 字节**。
- **修复与 `tactic=(tile_tokens_dim, config_index)` 扩展**：统一在 `scratch_.begin_measure()` 之后、遍历/测算 runner 之前执行 `prepare_routing()`；同时公开可选参数 `tactic=(tile_tokens_dim, config_index)`，支持在离线调优固定 tactic 后仅按该特定 tile 与 config 分配最小精确工作区。

### 2.5 TileLang 16-Bit 逆三角/双元数学降级与 `nextafter` 次正规数位步进 (`#3172`, `#2860`, `#3582`)

1. **16-Bit 数学内建函数与 `hnextafter` 位级步进 (`tile-ai/tilelang@43f13c7ad1d1`, `#3172`)**：
   - 为 `asin, acos, asinh, acosh, atanh, log1p, atan2, hypot, copysign, ldexp, nextafter` 注册 CUDA lowering 规则；当 `CCCL_VERSION >= 3000000` 且 `_LIBCUDACXX_HAS_NVFP16() && _LIBCUDACXX_HAS_NVBF16()` 时直接复用 `cuda::std::*` 16-bit 重载，否则走 FP32 提升桥接。
   - 针对 `hnextafter(x, y)`，若在设备端使用浮点比较，NVCC `--use_fast_math` (FTZ) 会将 FP16/BF16 次正规数（`0x0001..0x007F`）刷成 `0`。TileLang 改为纯 16 位无符号整数位模式运算：提取幅值 `x_mag = x_bits & 0x7FFF` 与 `y_mag = y_bits & 0x7FFF`，若 `x_mag == 0` 则直接返回带 `y` 符号位的最小次正规数 `(y_bits & 0x8000u) | 0x0001u`，否则按符号与幅值方向对 `x_bits` 执行 `+1` 或 `-1`，精确保持次正规数单步递进。
2. **拒绝无架构 CUDA Target 静默回退 `sm_50` (`@f1a34efea68c`, `#2860`) 与 Ascend PTO 有符号 Warp 归约 (`@d85a70a97218`, `#3582`)**：
   - 在无物理 GPU 的构建机上调用 `tilelang.compile(func, target="cuda")` 时，TVM 会静默填充 `arch=sm_50`。TileLang 在 `_cuda_target_from_arch` 统一拦截无 `-arch=sm_xx` 的 CUDA target 并抛出明确错误；同时修复了 Ascend PTO 后端 `T.warp_reduce_max/min` 对有符号整数（`int32` 等）发射无符号比较指令的问题。

---

## 3. 本项目确定性离线实验与验证 (`own-offline`)

实验脚本：`experiments/flashinfer-rubin-megamoe-auto-qsa-wide-rmsnorm-pdl-tilelang/run.py`（结果固化于同目录 `summary.json`，由 `tests/test_phase70_experiments.py` 守护）：

| 验证子项 | 关键输入 / 参数 | 实测 / 仿真输出 | 架构结论 |
|---|---|---|---|
| Rubin MegaMoE `coordinate_workspace_replacement` | 4 EP ranks, Rank 1 缓存失效 (`replace=True`) | `collective_replace=True`, `preserved_in_captured_list=True` | 消除单 Rank 销毁对称堆导致的跨 Rank 死锁并保护已捕获 CUDA Graph |
| Rubin MegaMoE 混合调优护栏 | Rank 0/2 `tuning=True`, Rank 1/3 `False` | `mixed_autotune_error=True` | 强制全 EP 组同步进入 `autotune_v2()` |
| `cake_qsa_nvfp4` (`sm_80`) `select_wide` | A100 (`108` SMs), `selected_tokens=2051` (`33` tiles) | `rows=128 -> decode_whole_tile` (`17` tiles/CTA, `1.085x`); `rows=64 -> decode` (`9` tiles/CTA) | 仅在超单波驻留且每 CTA $\ge 16$ tiles 时切换全 Tile 解码 |
| `cake_rmsnorm_train` PDL 双发射 | GB300 (`152` SMs) / B200 (`148` SMs), $T \in [4096, 65536]$ | 侧流停顿从 `10x–280x` 降至 `<= 1.13x`，GB300 $(4096, 512)$ `1.168x` | 彻底根除全设备 Cooperative Launch 在多流训练下的饥饿阻塞 |
| `trtllm_fused_moe_workspace_size` | `quant_mode="ds_fp8"`, `use_shuffled_weight=False` | 修复前 `0 B`，修复后全 tile `24,096,768 B`、`tactic=(16,0)` `15,184,896 B` | 修复未混洗 FP8 路由标志缺失导致的 0 字节工作区 |
| TileLang `hnextafter` 位级步进 | `+0.0 -> -1.0`, `0x007F -> 1.0`, `0x0080 -> 0.0` | `0x8001` (`-min_subnormal`), `0x0080` (`min_normal`), `0x007F` (`max_subnormal`) | 在 `--use_fast_math` 下完整保留 BF16 次正规数与符号零过渡 |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **计算练习**：在 NVIDIA H100 (`sm_90` 之外若回退至 `sm_80` 兼容路径，`132` SMs) 上运行 `cake_qsa_nvfp4`，若 `rows=64, num_kv_heads=8, splits=2, selected_tokens=2048`，请计算 `ctas`、`2 * sm_count` 与 `ceil(tiles / splits)`，判断 `select_wide` 是否触发？
   - *解答*：$\text{tiles} = \lceil 2048/64 \rceil = 32$，$\text{ctas} = 64 \times 8 \times 2 = 1024 > 2 \times 132 = 264$，且 $\lceil 32 / 2 \rceil = 16 \ge 16$，因此 `select_wide = True`，路由至 `decode_whole_tile`。
2. **可证伪假设**：在 Megatron/TorchTitan 训练中，若通信/监控侧流仅运行耗时 `< 5 us` 的事件查询，将 RMSNorm backward 保持为 `cudaLaunchAttributeCooperative` 是否仍可接受？
   - *证伪条件*：只要侧流存在持久驻留内核（如 NCCL/NVSHMEM 轮询内核或后台抢占式 telemetry 内核），Cooperative Launch 就会产生最高 `280x` 的尾延迟尖峰；改为非协作 chunk-claiming + PDL 归约可在零精度损失下完全消除全芯片同步等待。
