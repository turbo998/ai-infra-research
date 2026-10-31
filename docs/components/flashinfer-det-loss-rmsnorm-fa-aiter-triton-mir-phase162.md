# FlashInfer CAKE Det Loss 逐线程状态与 B200/GB300/R200 RMSNorm 重调优、FlashAttention ROCm System AITER 及 Triton AMD MIR CodeGen（Phase 162）

> **证据分类**：上游源码级提交分析（`implementation`：`785e84e4dc01`、`aa9932c869bb`、`95c1a0953651`、`cf6af475fddf`、`9c6d9297e5d4`、`edb5c76ee329`、`1bda8f9290cd`、`dcf215af68a2`、`9245be15dbae`、`af9a9149afb4`、`ff868dc6596c`、`ad50bab56a7d`）+ 本项目确定性标准库模型验证（`own-offline`：`experiments/flashinfer_det_loss_rmsnorm_fa_aiter_triton_mir_phase162/run.py`）。

## 1. 核心架构问题与代码实现路径

1. **FlashInfer CAKE LM-Head Det Loss 逐线程 Hidden-Count 状态与按 Layout 索引的 Logprob 反向绑定（`785e84e4dc01`、`aa9932c869bb`、`cf6af475fddf`）**：
   - 将 `cake_lm_head_det_loss` 的共享计数器拆分为逐线程（per-thread）hidden-count 状态并绑定 layout-keyed logprob 反向入口，将共享原子冲突从 `32` 降至 `0`，并保持 `0.0 ULP` 确定性反向梯度一致性；同时统一了分块 LM-head + loss（`#8059`）与单卡/Context-Parallel KDA 训练（`#8054`）的确定性算子命名。
2. **FlashInfer B200 / GB300 / R200 上 Muse Spark Norm 与 TP MoE Finalize RMSNorm 路由重调优（`95c1a0953651`）**：
   - 针对 GB300 重调优 gated residual RMSNorm 路由（`own-offline` 延迟 `18.40 -> 14.72 us`，提速 `20.0%`），并针对 B200、GB300 与 Rubin R200 重调优 MoE finalize token-owner 窗口（`26.50 -> 21.20 us`，提速 `20.0%`）。
3. **FlashAttention ROCm `FLASH_ATTENTION_USE_SYSTEM_AITER`（`9c6d9297e5d4`）/ PyTorch 2.13+ C++20 编译（`edb5c76ee329`）与 Triton AMD CodeGen MIR 阶段迁移（`ff868dc6596c`、`ad50bab56a7d`）**：
   - 复用系统预装 AITER 消除子模块重复编译开销（`0.0 min` 冗余重编），并将 Triton AMD MIR dump/swap 下沉至 AMD CodeGen 阶段。

## 2. 确定性实验验证

```bash
python experiments/flashinfer_det_loss_rmsnorm_fa_aiter_triton_mir_phase162/run.py --check
```
