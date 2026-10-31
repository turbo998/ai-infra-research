# FlashInfer CAKE DeepGEMM Dense MQA / mHC Autograd、FlashAttention SM100 Sparse MLA、CUTLASS 4.8 与 Triton Proton Trace 隔离深度解析（Phase 161）

> **证据分类**：上游源码级提交分析（`implementation`：`fe08dc597e95`、`9f6a84c6ae80`、`deacae9249c2`、`dd8325c1f547`、`843bf0b86bda`、`616b0e8abab1`、`784aee1878bd`、`0b55a2f691d6`、`098de2a652cf`、`cdcf8d86daa9`、`2bdead3a37e9`、`f53372790560`）+ 本项目确定性标准库模型验证（`own-offline`：`experiments/flashinfer_cake_mqa_mhc_fa_sm100_cutlass_48_triton_proton_phase161/run.py`）。非本机 GPU 硬件实测。

## 1. 核心架构问题与代码实现路径

1. **FlashInfer CAKE DeepGEMM Dense MQA 单次 Launch 链长扩展（`fe08dc597e95`）**：
   - 在解码/投机验证路径中，`FP8 Q = 1 / Q = 128` 与 `FP4 Q = 1` 的单次启动（single-launch）KV Block 链上限原先为 `7` 个块，当上下文落入第 `8` 个 KV Block 时会退化为二次 kernel 启动与 LSE 规约。`fe08dc597e95` 将单次启动链扩展 `+1` 个 KV Block（由 `7 -> 8`），在我们 `own-offline` 模型中将第 8 块边界的 launch 开销从 `6.80 us` 降至 `3.40 us`（降低 `50.0%`）。
2. **FlashInfer CAKE mHC 可选非确定性 Autograd 后端（`deacae9249c2`）、LM Head 单线程 Logprob 校验（`9f6a84c6ae80`）与 KCP Causal Conv1D 命名确定化（`dd8325c1f547`）**：
   - `deacae9249c2` 为多头超连接（Multi-Head Hyper-Connections, mHC）提供 opt-in 非确定性反向传播算子，允许高吞吐训练显式放宽原子累加顺序约束（`own-offline` 反向延迟 `148.0 us -> 112.48 us`，提速 `24.0%`）；`9f6a84c6ae80` 在单 CPU 线程内比对 compacted logprob 统计量以消除多线程浮点归约抖动；`dd8325c1f547` 统一了 4 个 Context-Parallel Causal Conv1D 入口的 KCP 确定性符号命名。
3. **FlashAttention SM100 Sparse MLA Exact Max（`784aee1878bd`）/ TVM-FFI 0.1.11（`843bf0b86bda`）/ CUDA 13 + PyTorch 2.15 Windows 构建（`616b0e8abab1`）、CUTLASS 4.8（`0b55a2f691d6`、`098de2a652cf`、`cdcf8d86daa9`）与 Triton Proton Trace 隔离（`2bdead3a37e9`、`f53372790560`）**：
   - 保持 SM100 Sparse MLA 精确最大值（Exact Max `0.0 ULP` 漂移），并在 Triton Proton `ChromeTraceWriter` 并发测试中实现每用例独立输出文件（冲突率 `0.0%`），同时将 MLIR 构造迁移出已弃用的 `Op::create` 接口。

## 2. 确定性实验验证与架构师练习

```bash
python experiments/flashinfer_cake_mqa_mhc_fa_sm100_cutlass_48_triton_proton_phase161/run.py --check
```

- **架构师练习**：计算 `B=64` 解码请求在跨越 `7->8` KV Block 阈值时，单次 Launch 链扩展每年为千卡推理集群节省的 Host CUDA Driver CPU 周期。
