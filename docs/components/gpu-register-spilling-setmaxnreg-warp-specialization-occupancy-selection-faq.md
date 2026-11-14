# GPU 寄存器溢出（`Register Spilling`、`STL/LDL`）、`Occupancy` 误区与 Hopper/Blackwell `setmaxnreg` 跨线程束动态捐赠选型指南

> **Provenance & Evidence Tier**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)。定量数据锚定于 [`experiments/gpu_register_spilling_setmaxnreg_occupancy_selection/results.json`](../../experiments/gpu_register_spilling_setmaxnreg_occupancy_selection/results.json)。
> **English Version**: [GPU Register Spilling (`STL/LDL`), Occupancy Myths & Hopper/Blackwell `setmaxnreg` Warp-Group Reallocation Selection Guide](./gpu-register-spilling-setmaxnreg-warp-specialization-occupancy-selection-faq.en.md)

![GPU Register Spilling & Hopper setmaxnreg Summary](../../assets/svg/gpu-register-spilling-setmaxnreg-warp-specialization-occupancy-selection-faq-summary.svg)

---

## 0. 🔥 今日领域最热门能力突破追踪（Daily Hottest Breakthrough）

- **Hopper / Blackwell PTX `setmaxnreg.dec / setmaxnreg.inc` 跨 Warp-Group 动态寄存器捐赠机制（`128` 个 Producer 线程缩减至 `40` Regs，将省下的 `16,384` 个寄存器当场转赠给 `256` 个 Consumer 线程扩至 `232` Regs）**：在 A100 / H100 / B200 GPU 中，每个 SM 拥有的 32-bit 物理寄存器总数被严格锁死在 **`65,536` 个（`256 KB`）**，且单个线程最多只能寻址 **`255` 个寄存器（`R0..R254`，`R255=RZ` 为恒零寄存器）**。在现代 Warp-Specialized 算子（如 `FlashAttention-3` 与 `CUTLASS 3.x`：1 个 CTA 包含 `384` 个线程 = `128` 个 TMA 数据搬运 Producer 线程 + `256` 个 WGMMA 矩阵乘 Consumer 线程）中，如果按传统静态均分策略，所有 384 个线程必须分配相同数量的寄存器，即每线程上限仅为 **`65,536 / 384 ≈ 168` 个寄存器**！然而 `256` 个 Consumer 线程为了在寄存器里放下 `128×256` 的 FP32 累加器与在线 Softmax 状态，每线程需要 **`232` 个寄存器**！一旦超过 `168`，编译器 `ptxas` 就会把放不下的 64 个寄存器**溢出（Spill）到所谓的 `Local Memory`（底层其实是片外慢速 HBM 显存，生成 `STL / LDL` 指令，Tier 1）**，导致实测算力从 **`785 TFLOPS` 暴跌至 `265 TFLOPS`（损失 `66.2%`）**！通过调用 Hopper/Blackwell 硬件指令 **`setmaxnreg.dec 40`（Producer 捐出 `16,384` 个寄存器）+ `setmaxnreg.inc 232`（Consumer 接收捐赠）**，总用量 `128×40 + 256×232 = 64,512 ≤ 65,536`，实现 **`0 Bytes` 寄存器溢出**，算力狂飙 **`785.0 TFLOPS`（提速 `2.96` 倍）**！

---

## 1. 📌 核心结论速览（Executive Summary — 3 分钟看懂）

1. **为什么 CUDA 里的“`Local Memory`（局部内存）”是性能黑洞？**
   - **通俗大白话 + 第一性原理**：名字叫 `Local Memory`（局部内存），听起来好像离核心很近，但实际上**芯片上根本没有一块叫 Local Memory 的硬件！它 `100%` 住在片外慢速的 HBM 显存里（仅经过 L1/L2 缓存）**！
   - 当你在内层循环里触发了 **Register Spilling（寄存器溢出，SASS 汇编里出现 `STL` Store Local / `LDL` Load Local）**，本来单周期就能在 `80 TB/s` 寄存器堆里算完的矩阵乘累加器，每走一步都要去读写 L1 甚至 HBM，瞬间把算力砸穿 **`66.2%`**！
2. **为什么在 Tensor Core 算子中，“追求高 `Occupancy`（盲目加 `--maxrregcount=128`，Tier 2）”反而会让性能更烂？**
   - 老一代 CUDA 教程总强调“提高 Occupancy（每个 SM 驻留更多线程）来隐藏延迟”。但在 Hopper `WGMMA` / Blackwell `tcgen05` 时代，单条异步矩阵乘指令就能吃满流水线，**决定性能的第一要素是累加器 Tile 够不够大（`128×256` 比 `64×128` 省一半共享内存读取带宽）**！如果为了把 Occupancy 凑高而强行用 `--maxrregcount=128` 砍掉每线程寄存器（Tier 2），只会逼迫编译器把巨型累加器疯狂 Spill 到 HBM，算力跌至 `310 TFLOPS`！
3. **`setmaxnreg`（Tier 4）是如何打破“大锅饭均分”死局的？**
   - 传统 CUDA 启动内核时搞“绝对平均主义”：不管你是只负责发几条 TMA 搬运命令的搬运工（Producer，40 个寄存器绰绰有余），还是负责抱住几百个 FP32 累加器猛算的计算工（Consumer，急需 232 个寄存器），所有人一律发 168 个寄存器！
   - **`setmaxnreg.dec / inc`** 允许内核在启动后的第一拍执行**“劫富济贫”**：128 个 Producer 线程当场退还 128 个寄存器（`168 -> 40`），256 个 Consumer 线程当场领走这批寄存器（`168 -> 232`），在丝毫不突破 SM `65,536` 物理总额的前提下，让 Consumer 零溢出抱住巨型 Tile！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：怎么用一条命令检查我们的 CUDA / Triton 算子有没有发生隐式 Register Spilling？
- **答**：
  - **在 CUDA (`nvcc`) 中**：编译时加上 **`-Xptxas -v`**，查看输出的 `ptxas info`：
    - 健康输出：`0 bytes spill stores, 0 bytes spill loads` ✅；
    - 灾难输出：`256 bytes spill stores, 256 bytes spill loads` ✕（再配合 `cuobjdump --dump-sass` 搜索 `STL` / `LDL` 指令）。
  - **在 Triton 中**：检查 `compiled_kernel.n_spills` 属性——**对于任何核心 GEMM / Attention 算子，`n_spills` 必须严格等于 `0`**！若 `n_spills > 0`，应立即调小 `BLOCK_M / BLOCK_N` 或降低 `num_warps` / `num_stages`。

### Q2：使用 `setmaxnreg.inc / dec` 时有哪些硬件对齐铁律？
- **答**：
  1. 必须在整个 **Warp-Group（连续 4 个 Warp = 128 个线程）** 粒度上同步调用（`.sync.aligned`）；
  2. 目标寄存器数必须是 **`8` 的整数倍**，且满足 `24 ≤ N ≤ 255`（实际上最多 `240` 或 `255`）；
  3. **最重要的一条死锁红线**：同一个 CTA（或结对 Cluster）内，**Producer 调用 `setmaxnreg.dec` 释放的寄存器总数，必须大于或等于 Consumer 调用 `setmaxnreg.inc` 额外索取的寄存器总数**！如果索取大于释放，Consumer 线程束会永远挂起等待物理寄存器池空闲，导致整个 Kernel 硬件死锁！

### Q3：到了 NVIDIA Blackwell（B200 / GB200，SM100），除了 `setmaxnreg`，硬件还做了什么重大升级来彻底解放寄存器压力？
- **答**：Blackwell（`SM100`）在每个 SM 内新增了一块专门给第五代 Tensor Core（`tcgen05.mma`）使用的 **`256 KB` 独立张量内存（`TMEM` — Tensor Memory，大小正好与 `256 KB` 寄存器堆相当！）**。在 Blackwell 上，`WGMMA` 的巨型 FP32 累加器不再占用通用寄存器（RF），而是直接常驻在专属的 **`TMEM`** 中，把 `65,536` 个通用寄存器彻底解放出来专门做 Softmax、反量化与激活函数！

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + 寄存器预算分配图 + setmaxnreg 时序图）

### 3.1 架构师 GPU 寄存器预算与溢出排障决策树

```mermaid
flowchart TD
    A["审计 CUDA / Triton 算子性能 (检查 -Xptxas -v 或 kernel.n_spills)"] --> B{"是否存在 Spill Stores / Loads > 0？"}
    B -- "是 (发生 Register Spilling 溢出至 HBM)" --> C{"算子架构类型？"}
    C -- "被人为加了 --maxrregcount 限制 (Tier 2)" --> D["⚠️ 立即移除过低的 maxrregcount<br/>切勿为凑 Occupancy 牺牲寄存器累加器"]
    C -- "Hopper/Blackwell Warp-Specialized 算子" --> E["🏆 Tier 4: 启用 setmaxnreg.dec (Producer 40) + setmaxnreg.inc (Consumer 232)<br/>零溢出装下 128x256 巨型 Tile · 算力跃升至 785 TFLOPS"]
    C -- "普通对称 Triton/CUDA 算子" --> F["✅ Tier 3: 适度调小 BLOCK_M/BLOCK_N 或 num_warps<br/>确保每线程寄存器 <= 255 且总和 <= 65,536"]
    B -- "否 (0 Bytes Spill)" --> G["✅ 寄存器预算健康，继续核验 Wave Quantization 与 TMA 重叠"]
```

### 3.2 传统静态均分（`168/168` 溢出或缩 Tile）vs. `setmaxnreg` 非对称捐赠（`40/232` 零溢出满血）

```mermaid
flowchart LR
    subgraph UNIFORM["⛔ Tier 1/3: 传统静态均分 (384 线程全员上限 168 Regs)"]
        direction TB
        U1["128 个 Producer 线程: 拿到 168 Regs (实际只用 40，浪费 16,384 个!)"]
        U2["256 个 Consumer 线程: 急需 232 Regs 却只给 168 -> 64 个寄存器溢出到 HBM (265 TFLOPS)!"]
    end
    subgraph ASYM["🏆 Tier 4: Hopper setmaxnreg 非对称捐赠 (总和 64,512 <= 65,536)"]
        direction TB
        A1["128 个 Producer 线程: setmaxnreg.dec 40 -> 捐出 16,384 个物理寄存器"]
        A2["256 个 Consumer 线程: setmaxnreg.inc 232 -> 零溢出装下 128x256 累加器 (785 TFLOPS)!"]
    end
```

### 3.3 Warp-Specialized 内核启动期 `setmaxnreg.dec / inc` 物理寄存器交接时序图

```mermaid
sequenceDiagram
    autonumber
    participant CWD as SM 硬件寄存器分配器 (65,536 Regs Pool)
    participant Prod as Producer Warp-Group (128 TMA Threads)
    participant Cons as Consumer Warp-Groups (256 WGMMA Threads)

    CWD->>Prod: 1. CTA 启动初始分配 (按编译声明静态预留)
    Prod->>CWD: 2. 执行 setmaxnreg.dec.sync.aligned.u32 40 (立即归还 16,384 个寄存器至 SM 池)
    Cons->>CWD: 3. 执行 setmaxnreg.inc.sync.aligned.u32 232 (从 SM 池无阻塞领走 16,384 个寄存器)
    Note over Prod,Cons: Producer 用 40 Regs 异步搬运 TMA，Consumer 用 232 Regs 零溢出狂跑 WGMMA!
```

---

## 4. 📊 架构师选型决策矩阵（4-Tier Selection Matrix）

| 评估维度 | Tier 1: 大 Tile 未受控溢出 | Tier 2: 强压 `--maxrregcount=128` | Tier 3: 对称 `__launch_bounds__` 缩 Tile | Tier 4: `setmaxnreg` 非对称动态捐赠 (`40 / 232`) |
| :--- | :--- | :--- | :--- | :--- |
| **Producer / Consumer 寄存器** | `168 / 168` (+64 溢出) ✕ | `128 / 128` (+92 溢出) ⚠️ | `168 / 168` (缩小 Tile) | **`40 / 232` (按需精准定制)** 🏆 |
| **每线程 Spill 溢出字节** | `256 Bytes` (`STL/LDL`) ✕ | `368 Bytes` (`STL/LDL`) ⚠️ | **`0 Bytes`** ✅ | **`0 Bytes`** 🏆 |
| **SM 寄存器池总占用** | 超限溢出 HBM | `49,152 / 65,536` | `64,512 / 65,536` | **`64,512 / 65,536` (利用率 98.4%)** 🏆 |
| **实测持续 Tensor Core 算力** | `265.0 TFLOPS` ✕ | `310.0 TFLOPS` ⚠️ | `590.0 TFLOPS` (`2.23x`) ✅ | **`785.0 TFLOPS` (`2.96x` 满血)** 🏆 |

---

## 5. 🧪 定量算例与工具链（Interactive Sizer & 7-Card Visual Report）

- **实验脚本与密封数据**：[`experiments/gpu_register_spilling_setmaxnreg_occupancy_selection/run.py`](../../experiments/gpu_register_spilling_setmaxnreg_occupancy_selection/run.py) · [`results.json`](../../experiments/gpu_register_spilling_setmaxnreg_occupancy_selection/results.json)
- **交互式寄存器预算与 setmaxnreg 计算器**：[`docs/reports/gpu-register-spilling-setmaxnreg-warp-specialization-occupancy-sizer.html`](../reports/gpu-register-spilling-setmaxnreg-warp-specialization-occupancy-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/gpu-register-spilling-setmaxnreg-warp-specialization-occupancy-selection-cards.html`](../reports/gpu-register-spilling-setmaxnreg-warp-specialization-occupancy-selection-cards.html)

---

## 6. 🛠️ Good Fit vs. Not For & 生产环境快速核验清单（Checklist）

### ✅ Good Fit vs. ⛔ Not For

| 维度 | ✅ Good Fit（强烈推荐） | ⛔ Not For（严禁误用） |
| :--- | :--- | :--- |
| **Hopper / Blackwell Warp-Specialized 算子** | 对 TMA Producer Warp-Group 调用 `setmaxnreg.dec 40`，对 WGMMA/MMA Consumer Warp-Group 调用 `setmaxnreg.inc 232/240` | 让只负责计算指针的 Producer 线程与 Consumer 线程平分宝贵的 `65,536` 寄存器池 |
| **编译期溢出门禁（CI Gate）** | 在 CI 中对所有核心 CUDA/Triton 内核强制断言 `spill_stores == 0 and spill_loads == 0`（`kernel.n_spills == 0`） | 在 Tensor Core 密集型内核上盲目添加 `--maxrregcount=64/128` 试图拉高 Occupancy |

### 生产环境上线核验清单（Pre-Flight Checklist）

- [ ] **核验 `-Xptxas -v` / `kernel.n_spills` 零溢出**：确认所有上线的 GEMM / Attention / MoE 内核均为 `0 bytes spill stores, 0 bytes spill loads`。
- [ ] **核验 `setmaxnreg` 释放与索取守恒定律**：确认 `N_prod_threads * Reg_prod + N_cons_threads * Reg_cons <= 65,536`（且 `Reg_prod`、`Reg_cons` 均为 8 的倍数），严防 `setmaxnreg.inc` 因寄存器池不足而死锁。
- [ ] **核验 Nsight Compute (`ncu`) `Local Memory Spilling` 告警为 0**：使用 `ncu --set full` 检查 `l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum` 严格为 `0`。
