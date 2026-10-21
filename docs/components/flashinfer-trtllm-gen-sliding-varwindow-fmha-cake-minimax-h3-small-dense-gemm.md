# FlashInfer TRTLLM-Gen 滑动/逐 Token 可变窗口 FMHA 与 Cake MiniMax-H3 / 小型及稠密投影 GEMM 归一化

> **关联上游提交（一手源码）**：
> - [`flashinfer-ai/flashinfer@d438333`](https://github.com/flashinfer-ai/flashinfer/commit/d4383335bae57d307edfcb58dce07bc78af2dece) (`#4905`, 2026-10-21) — *feat: integrate TRTLLM-Gen sliding window and variable window FMHA*
> - [`flashinfer-ai/flashinfer@c785284`](https://github.com/flashinfer-ai/flashinfer/commit/c78528451fcd327e6f81fb5586445106cd11abe3) (`#6947`, 2026-10-21) — *[Cake] MiniMax-H3 SM100/SM103 round-2 consolidation*
> - [`flashinfer-ai/flashinfer@a509a27`](https://github.com/flashinfer-ai/flashinfer/commit/a509a27181cbbc22887384c8780cbd5e6bc7b7ca) (`#6946`, 2026-10-21) — *[Cake] Consolidate small GEMMs (`router_gemm`, `bf16_bmm`, `quant_lora_gemm`)*
> - [`flashinfer-ai/flashinfer@2852797`](https://github.com/flashinfer-ai/flashinfer/commit/28527974293499a21687feccc2c6ebf1009d1) (`#6938`, 2026-10-21) — *[Cake] Consolidate `dense_projection_gemm` across SM100/SM107*

---

## 1. 背景与核心痛点

在 Blackwell（SM100a / SM103a / SM107a）推理架构上，长上下文注意力与 Cake AoT 内核库面临两大核心工程挑战：
1. **滑动窗口与逐 Query Token 可变注意力掩码（Variable Window FMHA）**：Gemma-3、MiniMax-H3 及稀疏块注意力需要对每个 Query Token $q_i$ 独立指定闭区间 $[s_i, e_i]$，或支持双向滑动窗口 $(W_{\text{left}}, W_{\text{right}})$；同时在 Decode 阶段若序列长度尚未超过窗口上限（$\text{max\_kv\_len} \le W_{\text{left}} + 1$），绝不能因进入通用滑动窗口分支而退化因果快路径（Causal Fast Path）延迟。
2. **Cake AoT 内核库的组合膨胀**：在 `minimax_h3`、`cake_router_gemm`、`cake_bf16_bmm`、`cake_quant_lora_gemm` 与 `dense_projection_gemm` 中，大量按 Tile / Cluster / MMA 形态复制的 `.cu` 源文件与 Host 绑定包装导致构建树膨胀超过 4.5 万行重复代码，且存在不可达的 `#if __CUDA_ARCH__ == 1000` 死分支。

---

## 2. TRTLLM-Gen Sliding Window 与 VariableWindow FMHA 机制剖析（`#4905`）

### 2.1 六态掩码枚举与 CUBIN 矩阵升级
`flashinfer@d438333` 将 `TrtllmGenAttentionMaskType` 扩展为 6 种掩码模式：
- `Dense = 0`、`Causal = 1`、`SlidingOrChunkedCausal = 2`、`Custom = 3`、`SlidingWindowCustom = 4`（双向滑动窗口 `window_right > 0`）、`VariableWindow = 5`（逐 Query Token 独立 `[start, end]`）。
- 升级 FMHA CUBIN 归档包至 `35cb99413a3ebce2e87a03fc04196e528e22c8d5`（SHA-256 `d7cfe9b3...`），总计 **`36,964` 个 FMHA CUBIN**，其中包含 **`1,424` 个 `VariableWindow` 专用 CUBIN**。

### 2.2 非绑定滑动窗口因果快路径与潜伏 Bug 修复
- **`fixedWindowBinds` 快路径保持**：当 `window_left >= 0` 但 `max_kv_len <= window_left + 1` 时，`fixedWindowBinds = False`，自动保持 `Causal` 掩码快路径。在 **B300**（`B=8, KV=128, Hq=16, Hkv=4, D=128, P=16`）上，无界 Decode 与非绑定 `window_left=127` 延迟同为 **`4.2624 us`**（而 `KV=4096, window_left=127` 绑定窗口为 **`4.4736 us`**）。
- **VariableWindow Prefill 开销**：在 `B=8, Q=512, KV=4096` 下，Fixed Window 对比 VariableWindow 在 **B300** 上为 **`29.438 -> 30.667 us`（`+4.17%`）**，在 **B200** 上为 **`35.520 -> 36.831 us`（`+3.69%`）**。
- **两处历史缺陷修复**：
  1. 修复 `trtllm_fmha_runner.cu` 中在计算完 `mChunkedAttentionSizeLog2` 后被末尾冗余 `runnerParams.mChunkedAttentionSizeLog2 = 0;` 覆盖清零的 Bug；
  2. 在 `BatchPrefillWithPagedKVCacheWrapper.plan()` 中显式拦截 `window_left >= 0 and not causal`，消除底层静默退化为全注意力的正确性隐患。

---

## 3. Cake 内核三大合并重构（净删减 `39,298` 行，100% SASS 级等价）

| PR / 提交 | 目标模块 | 合并策略与核心改动 | 代码净变化 | SASS / 精度验证 |
|---|---|---|---|---|
| **`#6947`** (`c785284`) | `minimax_h3` (SM100/SM103) | 删除 30 个重复 `cake_device_helpers` 头文件（`904` 行）与 112 处未使用宏（`224` 行）；抽取 3 个共享 `norm+AdaLN` 头文件（`bf16/mxfp8/nvfp4`，消除 `733` 行重复） | `+829 / -2,598`（净减 **`1,769`** 行，39 文件） | **`65/65`** CUBIN SHA-256 与 SASS 完全一致；B200/GB300 **`77/77`** 算子 Bit-Exact |
| **`#6946`** (`a509a27`) | `router_gemm` / `bf16_bmm` / `quant_lora_gemm` | 32 个 `router_gemm` 归并为 1 个主体；34 个 `bf16_bmm` 中 32 个归并为 6 个主体；9 个 `quant_lora_gemm` 共享 3 套 Host 绑定并折叠 2 个 Hint 内核 | `+2,759 / -19,641`（净减 **`16,882`** 行，89 文件） | **`143/143`** + **`21/21`** Device 行 SASS 100% 字节一致 |
| **`#6938`** (`2852797`) | `dense_projection_gemm` (SM100/SM107) | 81 个重复 Host 绑定折叠为 6 类 Launcher；10 个克隆内核折叠为 7 个主体；移除 27 个 SM107 专属内核中不可达的 `SM100` 分支；推导 `44 + 54 - 14 = 84` 模板全集 | `+2,650 / -23,297`（净减 **`20,647`** 行，135 文件） | 87 程序 / **104 构建**（47 SM100 + 57 SM107）SASS 完全一致 |

---

## 4. 确定性复现实验与验证

本课题提供纯 Python 3.12 标准库复现实验：
```bash
python experiments/flashinfer-trtllm-gen-sliding-varwindow-fmha-cake-minimax-h3-small-dense-gemm/run.py --check
```
实验完整验证了 `TrtllmGenAttentionMaskType` 六态路由、`fixedWindowBinds` 因果快路径保持、`mChunkedAttentionSizeLog2` 修复、非因果滑动窗口拦截以及 Cake 三大 PR 共计 `-39,298` 行的精确指标。
