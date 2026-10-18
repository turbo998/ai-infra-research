# CUDA Graph Piecewise & Breakable Capture Experiment (`own-offline`)

本实验量化 `vLLM v1` (`PiecewiseBackend` / `CUDAGraphWrapper` / `BreakableCUDAGraphCapture`) 与 `SGLang` (`CudaGraphConfig` `FULL` vs `BREAKABLE`) 在 H100 FP8 与 B200 NVFP4 上的三大核心取舍：
1. **Host CPU 逐算子发射墙（CPU Launch Wall）**：`488` 个 Eager 算子（`~2.001 ms/step`）如何吞噬 B200 NVFP4 低并发 Decode 延迟；
2. **共享图内存池（`global_graph_pool` + `weak_ref_tensors`）**：32 个捕获桶在独立分配 vs 全局池复用下的 HBM 占用（`76.02 GiB` vs `0.55 GiB`，压缩 `138.22x`）；
3. **非对齐 Batch 向上桶对齐 Padding 惩罚**：`Piecewise/Breakable` 仅对 MLP/Norm/MoE 静态子图做 Token Padding，而 Eager Attention 仍按真实 `unpadded_bs` 读取 KV Cache，因此在高并发非对齐点（如 `BS=257 -> 288`）反而快于把 KV 读取也一并 Padding 的 `Full CUDA Graph`。

## 运行与校验

```powershell
python experiments\cuda_graph_piecewise_capture\run.py
python experiments\cuda_graph_piecewise_capture\run.py --check
```
