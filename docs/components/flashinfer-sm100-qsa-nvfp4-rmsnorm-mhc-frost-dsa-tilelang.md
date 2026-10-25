# FlashInfer SM100/SM103 `tcgen05` QSA NVFP4 稀疏解码、RMSNorm/mHC 训练反向内核、FROST Grouped Tensor Core 与 TileLang 编译器护栏深度解析

> **English Version**: [flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang.en.md](flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang.en.md)  
> **Phase**: 72 (`priority: 81` · Track A) · **核验日期**: `2026-10-25` (UTC+08:00)  
> **实验脚本与密封输出**: [`experiments/flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang/run.py`](../../experiments/flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang/run.py) · [`summary.json`](../../experiments/flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang/summary.json)

---

## 1. 背景与核心架构问题

在 NVIDIA Blackwell (`sm_100a` B200 / `sm_103a` GB300 / `sm_120` RTX PRO 6000 Blackwell) 与 Vera Rubin (`sm_107a` R200) 架构上，稀疏注意力解码（QSA NVFP4 / DSA Indexer）、训练态归一化与多流残差连接（`cake_rmsnorm_train` / `cake_mhc_train`）以及低延迟 MoE 路由（`frost_low_latency`）面临以下四个底层硬件—编译器耦合问题：

1. **Blackwell `tcgen05` 张量内存 (TMEM) 占用与 Split-KV 并发预算失配**：在 `sm_80`/`sm_90a` 上，`cake_qsa_nvfp4` 基于 `mma.sync` 并在每个 SM 上驻留 `2` 个 CTA；而 `sm_100a`/`sm_103a` 引入 `tcgen05` 指令与 SM 级 TMEM 分配，每个 SM 只能驻留 `1` 个 `tcgen05` 消费 CTA。若直接复用 Hopper 的 split 计算公式，会导致 wave 尾效应或小 batch 时 kernel 间全局屏障开销主导延迟。
2. **NVFP4 E2M1 解量化热循环中的冗余浮点乘法**：`cake_qsa_nvfp4` 在 `sm_80`/`sm_90a` 上原本对每个解量化 token 块执行一次 `float(scale) * rebias` 乘法，并把 E2M1 常量表放在全局符号中，增加了指令数与流水线压力。
3. **训练反向算子 (`cake_rmsnorm_train` / `cake_mhc_train`) 的寄存器压力、尾块划分与 Rubin `min_cuda` 门槛**：Blackwell/Rubin 上 RMSNorm 反向与 4 流 mHC 反向在 `H ∈ [512, 6144]` 区间受限于非均匀寄存器地址计算、BF16 解包指令开销与 `sm_107a` 对 CUDA 13.4 编译器的硬性依赖。
4. **SM90/SM120 超小 Batch MoE 路由与 TileLang 向量化/原子语义漏洞**：当 decode token 数仅为 `T ∈ {4, 8}` 时，标量 SIMT MoE 内核无法利用 Tensor Core；而在 SM120 CUDA Graph 捕获下，W4A16 路由出现高度倾斜（每个被选行落入不同专家）时，按 `packed_slots` 分配的 scratch 缓冲区会发生越界截断。同时 TileLang 编译器在 `VectorizeLoop` 步长检查、有符号 `int64` 原子比较及 NVRTC 缓存加载路径上存在正确性隐患。

---

## 2. 一手源码审计与实现机制

### 2.1 FlashInfer `e73481617190` (`#7205`) & `cb93a30db500` (`#7175`)：`tcgen05` QSA NVFP4 稀疏 GQA 解码与 E4M3 Rebias 折叠

在 [`flashinfer/comm/cake_qsa_nvfp4.py`](https://github.com/flashinfer-ai/flashinfer/commit/e73481617190444ab56ac258f6be5b4f673bc8b6) 与 `csrc/cake_qsa_nvfp4/` 中，上游为 `sm_100a` 和 `sm_103a` 补齐了原生 `tcgen05` QSA NVFP4 稀疏分页 GQA 解码内核（面向 `num_qo_heads=24, num_kv_heads=2, head_dim=256, top_k=2051`，同时支持 `NHD` 与 `HND` 布局）：

- **架构感知 Split-KV 启发式公式**：
  $$\text{tiles} = \left\lceil \frac{\text{selection\_width}}{64} \right\rceil, \qquad \text{splits} = \max\left(1, \min\left(\text{tiles}, \left\lfloor \frac{\text{CTAS\_PER\_SM}[\text{arch}] \times \text{sm\_count}}{\text{rows} \times H_{kv}} \right\rfloor\right)\right)$$
  其中 `CTAS_PER_SM = {"sm_80": 2, "sm_90a": 2, "sm_100a": 1, "sm_103a": 1}`。以 `selection_width = 2051`（`tiles = 33`）、`H_kv = 2` 为例，在 B200 (`sm_100a`, `148` SMs) 上，`rows = [1, 2, 4, 8, 16, 32, 64, 128]` 对应的 `splits` 精确为 `[33, 33, 18, 9, 4, 2, 1, 1]`；而在 GB300 (`sm_103a`, `160` SMs) 上对应 `[33, 33, 20, 10, 5, 2, 1, 1]`。
- **双路径执行（Split + Merge vs Fused Co-Resident）**：
  - 通用路径采用 `qsa_nvfp4_tcgen05_split` + `qsa_nvfp4_merge`（当 `splits == 1` 时直接由主内核写出最终 `out`，跳过 merge）。
  - 在 Grace-Blackwell (`platform.machine() == "aarch64"`) 主机且处于非 CUDA Graph 捕获态（`not is_current_stream_capturing`）、`rows <= 4` 且 `splits > 1` 时，自动切换至单内核 `qsa_nvfp4_tcgen05_fused`：将 `rows * 2 * splits` 个 decode CTA 与 `rows * 2` 个 merge CTA 共驻（co-resident）在同一网格中（`4 * 2 * (20 + 1) = 168 <= 320` 槽位），省去第二次 kernel launch 与全局内存 barrier。
- **E4M3 块缩放 Rebias 折叠 (`cb93a30db500`, `#7175`)**：
  在 `sm_80`/`sm_90a` 上，原本每次读取 E4M3 块缩放因子都要乘以 $\text{rebias} = 2^{120 - \text{bias}}$。`#7175` 利用乘法结合律，在 tile 入口处将 `rebias` 直接乘入 8 项 E2M1 幅值查找表 $[0, 0.5, 1, 1.5, 2, 3, 4, 6]$，并将两个打包表字常驻寄存器，同时用单条 `prmt.b32` 字节置换指令将两个 token 对的 E4M3 缩放字节组装为 PV 缩放字。

| 平台与模式 | 布局与规模 (`top_k=2051`) | 参考实现 `qsa_nvfp4` ($\mu\text{s}$) | `cake_qsa_nvfp4` ($\mu\text{s}$) | 实测加速比 | 证据性质 |
|---|---|---:|---:|---:|---|
| GB300 (`sm_103a`, Eager memset flush) | `NHD`, `rows=1` | `46.50` | `25.81` | **1.80x** | `author-benchmark` (`#7205`) |
| GB300 (`sm_103a`, Eager memset flush) | `NHD` / `HND`, `rows=128` | `191.09` / `192.02` | `71.94` / `71.87` | **2.66x / 2.67x** | `author-benchmark` (`#7205`) |
| GB300 (`sm_103a`, CUDA Graph replay) | `NHD`, `rows=1` / `rows=128` | `17.60` / `178.94` | `9.92` / `67.15` | **1.77x / 2.66x** | `author-benchmark` (`#7205`) |
| B200 (`sm_100a`, Eager memset flush) | `NHD` / `HND`, `rows=128` | `224.61` / `224.64` | `77.26` / `76.94` | **2.91x / 2.92x** | `author-benchmark` (`#7205`) |
| H100 (`sm_90a`, E4M3 rebias fold) | `NHD` / `HND`, `rows=4` | `19.65` / `19.52` | `18.98` / `18.98` | **1.035x / 1.028x** | `author-benchmark` (`#7175`) |

---

### 2.2 FlashInfer `03f82fddadae` (`#7200`), `0fac207b466d` (`#7199`) & `26d0d5dc906c` (`#7198`)：Blackwell/Rubin 训练态 RMSNorm 与 4 流 mHC 内核优化

1. **`cake_rmsnorm_train` 反向内核重构 (`#7200`)**：
   - **均匀寄存器描述符 (`shfl.sync.idx.b32`)**：将 CTA 内所有线程共享的指针与偏移通过 warp lane 0 广播固定到均匀寄存器（uniform registers），配合拆分为无分支主循环（pure interior loop）与尾块循环，大幅降低地址算术指令开销。
   - **尾块粒度重平衡**：将 CTA 内部行分块步长从 `max(1, (share + 15) // 16)` 调整为 `max(1, (share + 11) // 12)`。以 B200 (`148` SMs) 处理 `T = 65536` 为例，每个 CTA 负责 `share = 443` 行，新公式将每次迭代块大小从 `28` 行提升至 `37` 行，迭代次数从 `16` 次降至 `12` 次。
   - **PDL (Programmatic Dependent Launch) 存储侧触发**：当 `1024 < T <= 4096` 时，前一层 `rmsnorm_bwd` 在完成 `dx` 存储后立即触发 `cudaTriggerProgrammaticLaunchCompletion()`，允许紧随其后的 `dw` 规约内核与当前 CTA 尾声重叠执行。
   - **实测收益 (`author-benchmark`)**：GB300 上 `H=512` 提速 **1.18x–1.27x**，`H=4096` 提速 **1.04x–1.11x**；B200 上 `H=512, T=65536` 提速 **1.332x** (`118.8 us -> 89.2 us`)；Rubin R200 (`sm_107a`) 新增 bulk-copy 反向内核，在 `H=6144, T=16384` 上实现 **1.331x** (`461.4 us -> 346.6 us`) 加速。
2. **Rubin `sm_107a` CUDA 13.4 最低版本契约 (`#7199`)**：
   - 将 `cake_muse_spark_norm_backend` 的 35 个 `sm_107a` 内核条目与 `cake_tp_moe_finalize_rmsnorm_backend` 的 6 个 `sm_107a` 条目的 `min_cuda` 统一从 `"13.0"` 提升至 `"13.4"`，防止旧版 `ptxas` 在编译 Rubin PTX 时静默退化或报错。
3. **4 流 `cake_mhc_train` (`H=4096`) 位移解包与专用反向规约 (`#7198`)**：
   - 引入 `cake_bf16x2_to_f32x2(uint32_t w)`：直接通过 `__uint_as_float(w << 16)` 和 `__uint_as_float(w & 0xffff0000u)` 将打包的 `__nv_bfloat162` 转换为两个 IEEE-754 FP32 寄存器，省去逐元素转换指令。
   - 针对不同流/列规约规模提供 `bwd_reduce__pt32`、`bwd_reduce__pt64` 与 `bwd_reduce__s27_wcvt_fadd_fshfl` 专用规约内核，将保存统计量严格控制在 `256 B/token`。在 GB300 `T=4096` 上反向加速 **1.17x** (`192.9 -> 164.6 us`)，重计算 checkpoint 路径加速 **1.15x** (`243.4 -> 212.5 us`)；在 R200 上前向加速 **1.16x–1.17x**、反向加速 **1.12x–1.13x**。

---

### 2.3 FlashInfer `94b5e266a966` (`#7194`), `5e4555094c32` (`#7195`) & `54459f0872e5` (`#7187`)：FROST 小 Batch Tensor Core MoE 与 DSA Indexer 流水线

1. **FROST `T ∈ {4, 8}` Grouped BF16 Tensor Core 路径 (`#7194`)**：
   - 针对 `H=2048, I=512, top_k=10, T ∈ {4, 8}`，新增 `route_members` 阶段：利用 warp 级 `ballot` 统计同一专家命中的 token 掩码，将同一专家的多个 token 聚合为 Tensor Core tile（Hopper SM90 使用 `wgmma`，Blackwell SM120 使用 `warp_mma`）。
   - 在 H100 上 `T=4` warm 延迟从 `36.70 us` 降至 `28.81 us` (**1.274x**)，cold 延迟从 `41.15 us` 降至 `32.79 us` (**1.255x**)；在 SM120 (RTX PRO 6000 Blackwell) 上 `T=4` warm 加速 **1.48x**、`T=8` warm 加速 **1.37x**。
2. **SM120 W4A16 CUDA Graph Scratch Slot 精确扩容 (`#7195`)**：
   - 在 SM120 W4A16 路径中，若在 CUDA Graph 捕获下出现极端离散路由（如 `T=4, top_k=10` 共 40 个路由行命中 40 个不同专家，每个专家需对齐 `block_size=16`），原实现仅分配 `packed_slots = 80` 个中间行，导致超出专家被静默丢弃。
   - `#7195` 引入独立辅助函数：
     $$\text{scratch\_slots} = \max\left(\text{packed\_slots}, \min(\text{routed\_rows}, 8 \times \text{topk}) \times \text{block\_size}\right)$$
     仅扩大 `gate_up_out` 和 `act_out` 缓冲区（`T=4` 时从 `80` 扩至 `640` 行），而保持 `route_slots = 80` 不变，既保证 CUDA Graph 捕获对任意专家分布 100% 精确，又避免非必要的路由表扫描膨胀。
3. **`cake_dsa_indexer` 采样候选池翻倍与 `mbarrier_test_wait` 预探测 (`#7187`)**：
   - 在 `sm_100a`/`sm_103a` 上，当启用 tile 采样（`sample_tiles_max > 0`）时，将候选池容量下限 `cand_cap_floor` 从 `8192` 提升至 `16384`（无采样时保持 `cand_cap_floor_unsampled = 8192`），避免深层上下文下候选池提前溢出重筛。
   - 在 wide scan 内核中，发出 `mbarrier_arrive(umma_empty_addr)` 后立即对下一 tile 执行非阻塞 `mbarrier_test_wait`，使 GB200 `Z2cp`/`Z3cp` 提速 **1.12x / 1.10x**，GB300 `Z3cp`/`Z3` 提速 **1.16x / 1.07x**。

---

### 2.4 TileLang 编译器正确性护栏 (`#3345`, `#3307`, `#3193`)

1. **`VectorizeLoop` 拒绝非单位步长 (`cb26ee0fafe5`, `#3345`)**：
   - 此前 `VectorizeLoop` 在将 `For` 循环向量化为 `Ramp(base, 1, extent)` 时忽略了 `op->step`，导致 `step=2` 的跨步访存被错误改写为连续向量读写。现显式加入 `ICHECK(!op->step.has_value() || is_one(op->step.value()))` 强约束。
2. **`cuda::atomic_ref<T1>` 保留有符号 64 位与半精度语义 (`2573d470798a`, `#3307`)**：
   - 此前 `AtomicMax`/`AtomicMin`/`AtomicStore` 将 `int64_t` 强转为 `unsigned long long*`。对于负数输入（如 `atomic_max(-5, 2)`），无符号补码下 `-5` (`0xFFFFFFFFFFFFFFFB`) 大于 `2`，导致返回错误结果 `-5`。修复后直接按原始有符号/半精度类型实例化 `cuda::atomic_ref<T1>(address_ref)`，恢复 `-5 vs 2 -> 2` 的正确语义。
3. **NVRTC `from_database` 库路径绑定恢复 (`35ee29507557`, `#3193`)**：
   - 修复 `NVRTCKernelAdapter.from_database` 加载预编译 `.cubin` 时遗漏 `adapter.libpath = adapter.lib_generator.libpath` 的缺陷，避免缓存命中后 `adapter.libpath` 为 `None` 引发下游工具链崩溃。

---

## 3. 合成 CPU/Offline 验证实验与架构师练习

本节对应的可复现实验位于 [`experiments/flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang/run.py`](../../experiments/flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang/run.py)，完全基于 Python 3.12 标准库验证上述公式与边界行为：

```bash
python experiments/flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang/run.py --check
```

### 核心验证结果摘要 (`own-offline` + `author-benchmark`)

1. **QSA NVFP4 Split-KV 与 Co-Resident 判定**：
   - `selection_width = 2051` $\Rightarrow$ `33` 个 64-token tile。
   - B200 (`148` SMs, `CTAS_PER_SM=1`) 在 `rows = [1, 2, 4, 8, 16, 32, 64, 128]` 下 `splits = [33, 33, 18, 9, 4, 2, 1, 1]`。
   - Grace-Blackwell (`aarch64` + `sm_103a`) 在 eager `rows=4` 时启用 fused co-resident (`True`)，而在 CUDA Graph 捕获或 `x86_64` 主机上自动回退为 split+merge (`False`)。
   - E4M3 `rebias` 折叠至 8 项 E2M1 幅值表后的最大浮点误差为 `0.0`。
2. **RMSNorm 尾块与 mHC BF16x2 位移解包**：
   - B200 (`T=65536`, `share=443`) 下，尾块步长从 `28` 提升至 `37`，循环迭代从 `16` 次降至 `12` 次；PDL store trigger 在 `T=2048` 激活 (`True`)、在 `T=8192` 关闭 (`False`)。
   - `cake_bf16x2_to_f32x2` 将打包字 `0x3FE0C020` 精确解包为 `(-2.5, +1.75)`。
3. **FROST Scratch Slot 与 TileLang 有符号原子验证**：
   - `T=4, top_k=10, block_size=16` 时，`route_slots` 保持 `80`，而 `scratch_slots` 扩容至 `640`（`T=16` 时封顶 `1280`）。
   - `int64` `atomic_max(-5, 2)`：旧版无符号强转错误返回 `-5`，新版 `cuda::atomic_ref<int64_t>` 正确返回 `2`。

### 架构师设计练习与可证伪问题

1. **计算题**：若在 Rubin R200 (`sm_107a`) 上部署 `num_kv_heads = 4`、`selection_width = 4096` 的 QSA NVFP4 解码，假设 `sm_count = 192` 且每个 SM 仅能驻留 `1` 个 `tcgen05` CTA，求 `rows = 8` 时的最优 `splits` 数。
   - *解答*：$\text{tiles} = \lceil 4096 / 64 \rceil = 64$；$\text{budget} = \lfloor (1 \times 192) / (8 \times 4) \rfloor = 6$；故 $\text{splits} = \min(64, 6) = 6$。
2. **可证伪假设**：在 `x86_64` PCIe B200 主机上对 `rows <= 4` 强行开启 `qsa_nvfp4_tcgen05_fused` 是否能复现 Grace `aarch64` 的正收益？
   - *证伪路径*：上游 `#7205` 在 `x86_64` B200 上实测发现小 batch PCIe 启动与 SM 调度抖动使 co-resident 路径在 `rows <= 4` 不优于 split+merge，因此显式限制在 `platform.machine() == "aarch64"` 下启用。
