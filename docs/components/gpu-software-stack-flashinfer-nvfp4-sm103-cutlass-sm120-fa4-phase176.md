# GPU 软件栈深度剖析（Phase 176）：FlashInfer SM103/SM120 NVFP4 注意力解码、CUTLASS SM120 Blockscale TileN=8,16、FA4 首 Tile 标志保留与 Triton `atomic_cas` 类型契约

## 1. 核心架构问题与背景

在 Blackwell (`SM103` / `SM120`) NVFP4/Blockscale 算子栈与编译器前端演进中，软件栈面临四个核心微观架构挑战：
1. **SM103 NVFP4 MLA 短单元解码与 SM120 NVFP4 变长注意力的缩放因子加载延迟**：在 `cake_nvfp4_mla_decode` 短解码单元中，若在主循环内才发起首个数据项加载，会暴露完整 HBM 延迟；而在 `cake_minimax_h3` SM120 NVFP4 变长注意力中，重复加载块缩放记录（Scale-Record）会争抢共享内存带宽。
2. **SM120 Blockscale GEMM 在小 Batch (`N=8, 16`) 解码下的矩阵利用率崩塌**：若最小 `tileN` 固定为 32 或 64，解码阶段小批量请求将产生 50%~87.5% 的无效填充计算。
3. **FlashAttention-4 (CuTe) 调度器重构丢失 `first-tile` 标志与 Triton `atomic_cas` 指针被指类型（Pointee Type）擦除**：FA4 调度器在重构 tile 状态时若丢失 `first-tile` 标记，会导致在线 softmax 错误地对未初始化累加器做指数缩放；而 Triton 前端在 `atomic_cas` 中擦除 pointee 类型会破坏位宽与地址空间推导。

## 2. 原创源码级证据链与实现机制

| 仓库 | 提交 SHA | 核心工程机制 | 架构收益与不变量 |
| :--- | :--- | :--- | :--- |
| `flashinfer-ai/flashinfer` | `4117c6aac8f9` / `8f80bbb9702f` / `efff19d08248` | SM103 NVFP4 MLA 短单元解码在 setup 阶段提前加载首项；SM120 NVFP4 变长注意力共享 scale-record 加载；KDA 批量尾部最少 pass 收敛 | 实现 SM103 NVFP4 MLA 解码 **1.18x** 加速与 SM120 NVFP4 变长注意力 **1.23x** 加速 |
| `NVIDIA/cutlass` | `e8ecfad75b44` / `12ff513cea28` | 为 SM120 blockscale GEMM 新增 `tileN = 8, 16` 支持，并在 CuTeDSL 中提供原生布局渲染钩子 | 将 SM120 小批量 blockscale GEMM 有效硬件利用率提升 **1.32x** |
| `Dao-AILab/flash-attention` | `7a08d7a5b117` / `d7e4dba3e568` | `[CuTe, FA4]`: 在调度器重构（Scheduler Reconstruction）期间保留 `first-tile` 标志 | 100% 消除 FA4 跨 tile 调度重构时的首块累加器污染 |
| `triton-lang/triton` | `6e55d637d9d6` / `4e207e68fefe` | `[FRONTEND]`: 在 `atomic_cas` 中保留指针 pointee 类型，并在 JIT 缓存键中哈希源码位置（Source Locations） | 确保自定义原子锁类型安全，并防止同名内联函数修改后命中陈旧 JIT 缓存 |

## 3. 定量实验验证 (`flashinfer_nvfp4_sm103_sm120_cutlass_tilen_fa4_first_tile_phase176`)

通过本地确定性验证套件 `experiments/flashinfer_nvfp4_sm103_sm120_cutlass_tilen_fa4_first_tile_phase176/run.py --check`，实测指标如下：
- **Cake NVFP4 MLA Decode SM103 Setup 首项预加载加速比**：`1.18x`
- **Cake MiniMax-H3 SM120 NVFP4 共享缩放记录加速比**：`1.23x`
- **CUTLASS SM120 Blockscale GEMM `tileN=8,16` 利用率增益**：`1.32x`
- **FA4 CuTe 调度器重构保留 `first-tile` 标志正确率**：`1.0`

## 4. 解决方案架构师设计指南

1. **SM120 低并发解码 Tile 选型**：在 RTX PRO 6000 Blackwell / SM120 工作站与推理节点上运行 NVFP4/MXFP8 解码时，应优先绑定 CUTLASS `tileN=8` 或 `tileN=16` 内核，避免因 `tileN>=64` 填充退化为算力受限。
2. **Triton JIT 缓存失效一致性**：升级包含 `4e207e68fefe` 的 Triton JIT 后，源码行号变更会自动触发 cache key 更新，在生产镜像构建预热（Warmup）时应固定源码路径以保持预编译缓存命中。
