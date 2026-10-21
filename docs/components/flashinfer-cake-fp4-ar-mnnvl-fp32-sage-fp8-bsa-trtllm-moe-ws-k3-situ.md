# FlashInfer Cake NVFP4-Transport All-Reduce + RMSNorm 分架构程序、MNNVL One-Shot FP32 平方防溢出、Sage-FP8 块稀疏注意力、TRT-LLM-Gen 调用方持有 MoE 工作区与 Kimi-K3 SiTU 头文件折叠架构解析

> **更新日期**：2026-10-22 · **阶段**：Phase 53 (`priority: 62`) · **分类**：`operator-kernel`  
> **关联实验**：[`experiments/flashinfer-cake-fp4-ar-mnnvl-fp32-sage-fp8-bsa-trtllm-moe-ws-k3-situ/run.py`](../../experiments/flashinfer-cake-fp4-ar-mnnvl-fp32-sage-fp8-bsa-trtllm-moe-ws-k3-situ/run.py)

---

## 1. 一手上游来源（First-Hand Sources）

| 仓库与提交 | 日期 (UTC) | 核心技术贡献 |
|---|---|---|
| [`flashinfer-ai/flashinfer@067b36956d`](https://github.com/flashinfer-ai/flashinfer/commit/067b36956da21f9cb6cfbdad9fd6326c52fa3655) | 2026-10-21 | `perf(cake_comm)`：NVFP4 传输 All-Reduce + RMSNorm 的 `sm_100a`/`sm_103a`/`sm_107a` 分架构专属程序（`_rso`、`_rst`、`_qp`、`_sop`、`_pf`）与 Pass-1 置换平方和（#6954） |
| [`flashinfer-ai/flashinfer@1ff146212d`](https://github.com/flashinfer-ai/flashinfer/commit/1ff146212d7f0e5641c361f316a7f039633f98df) | 2026-10-21 | `fix(comm)`：MNNVL one-shot 融合 All-Reduce + RMSNorm 在求平方前先转 `fp32`，消除 `|x| > 255.94` 的 FP16 `inf -> 0` 全零崩溃（#6818） |
| [`flashinfer-ai/flashinfer@4e4c41da67`](https://github.com/flashinfer-ai/flashinfer/commit/4e4c41da67f6c463940e4c75d11207ce93d6c516) | 2026-10-20 | `perf(cake_sage)`：SM100/SM103 Sage-FP8 块稀疏注意力单次启动（`1.30x–1.40x`）与融合量化器（`1.23x`）加速（#6872） |
| [`flashinfer-ai/flashinfer@15c83f1c36`](https://github.com/flashinfer-ai/flashinfer/commit/15c83f1c36f5468c6fdc49e8f7f8ede247705cbb) | 2026-10-20 | `feat(moe)`：`trtllm_fused_moe_workspace_size` 与调用方持有 `workspace_buffer=`，B300 自调优峰值显存下降 `2,683.1 MiB`（#6825） |
| [`flashinfer-ai/flashinfer@75b30ffdbf`](https://github.com/flashinfer-ai/flashinfer/commit/75b30ffdbf8c4413b26ff12ba7cb4c6c7d5f4d46) | 2026-10-20 | `deslop(cake_megamoe)`：移除 SM120 1,090 行重复 `part002.cuh` 并重签 18 项 Manifest 摘要与 EP16 未调用辅助函数（#6898） |
| [`flashinfer-ai/flashinfer@e2456a4eea`](https://github.com/flashinfer-ai/flashinfer/commit/e2456a4eeafe3ab14159b8debebf0582e34d3ed4) | 2026-10-20 | `deslop(cake_kimi_k3_situ)`：Kimi-K3 NVFP4 SiTU 路由 MoE 抽取 103 个受控设备辅助函数头文件、折叠 `fc1:n8_s2a` 与共享 8 组 Host Binding，净减 `17,018` 行（#6879） |

---

## 2. 核心架构机制深挖

### 2.1 Cake NVFP4-Transport All-Reduce + RMSNorm 分架构程序（`067b36956d`）

在 `flashinfer.comm.cake_fp4_allreduce` 中，TP2 one-shot、TP4 two-shot 以及 6 种与 Residual Add + Gemma RMSNorm 融合的 TP2 变体（输入 `BF16` / `NVFP4`，输出 `BF16` / `NVFP4` / `MXFP8`）原本在 `sm_100a`（B200）、`sm_103a`（GB300）和 `sm_107a` 上共用同一套程序。PR #6954 将注册表扩展为 **37 个程序（每个架构 20 个，新增 22 个、淘汰 16 个）**，引入五种针对微架构特性的编译形式：

1. **`_rso`（`sm_103a` TP2 one-shot BF16）**：在第一次扫过对端数据包时“飞行中（in-flight）”读取每行 scale word，替代原先在扫描开始前单独轮询完成的开销（`1K x 6144` 延迟从 `12.99 us` 降至 `12.04 us`，**`1.0789x`**）。
2. **`_rst`（`sm_107a` TP2 one-shot BF16）**：对已消费 Lamport 槽位的内核内清零改用普通每线程 store 替代 bulk store（几何平均 **`1.0121x`**）。
3. **`_qp`（`sm_100a` / `sm_103a` BF16 输入、NVFP4/MXFP8 输出）**：Pass-2 量化改用打包双宽（packed two-wide）算术替代标量 FP32 量化树（`sm_107a` 上未带来提速故不启用）。
4. **`_sop`（Pass-1 置换平方和）**：在 Pass-1 中按 lane 加载分块的自然寄存器顺序累加平方，而非按行自然顺序重排；结合 `4K` 行的残差预取孪生核 `_pf`，在 `sm_100a` 上实现 **`1.0718x`** 几何平均加速（`8K x 6144` `NVFP4 -> NVFP4` 从 `77.36 us` 降至 `70.70 us`，峰值 **`1.0942x`**）。

### 2.2 MNNVL One-Shot 融合 All-Reduce + RMSNorm 的 FP32 平方防溢出（`1ff146212d`）

`include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh` 中的 `oneshot_allreduce_fusion_kernel` 原本按 `toFloat<T>(packedAccum.elements[i] * packedAccum.elements[i])` 计算平方和。当 `T = half`（FP16，最大可表示数 `65504`）时，任何满足 $|x| > \sqrt{65504} \approx 255.94$ 的激活值在 `half` 乘法下会溢出为 `inf`，导致 `rsqrtf(inf) = 0`，整行归一化输出全部坍缩为 `0`！

- **生产故障复现**：vLLM 0.25.1 在 TP > 1 时启用 `fuse_allreduce_rms` 并在解码批次（$\le 1\text{ MiB}$）选择 MNNVL one-shot。Qwen3.5-27B 在 FP16 TP=2 下约有 **9.5%** 的解码 token 在最终层残差流激活超过 `256`，导致所有生成坍缩为 `!!!!`（全零 logits），RL 验证集得分为 **`0.000`**。
- **修复**：改为先执行 `float const value = toFloat<T>(packedAccum.elements[i]); threadSum += value * value;`，与 two-shot 核对齐，RL 验证集得分恢复至 **`0.698`**（与 BF16 完全一致）。

### 2.3 SM100/SM103 Sage-FP8 块稀疏注意力与量化器（`4e4c41da67`）

`flashinfer/cake_sage_block_sparse_attention_backend` 将 SM100/SM103 的 Sage-FP8 块稀疏注意力（`head_dim=128`，E4M3 Q 每 token 缩放、K 每 16-token 缩放、V 每通道缩放，64 行查询块对 `sel` 个 64-token KV 块）从“描述符上传 + 内核”2 次启动合并为 **1 次启动**，并将 20 个 SM120 特化核迁入统一包中：
- 在 GB300 (`sm_103a`) 上，`S_q = S_k = 4096` 注意力前向加速 **`1.30x–1.40x`**（例如 `A4: B=1, H=8, sel=6` 从 `0.0219 ms` 降至 `0.0156 ms`；`A3: B=8, H=32, sel=58` 从 `1.8138 ms` 降至 `1.3691 ms`）；
- 融合量化器 `quantize_qk_vamax` + `quantize_v`（`Q1`）从 `0.2610 ms` 降至 `0.2127 ms`（**`1.23x`**）。

### 2.4 TRT-LLM-Gen 调用方持有 MoE 工作区与 Kimi-K3 SiTU 折叠（`15c83f1c36`、`75b30ffdbf`、`e2456a4eea`）

1. **`trtllm_fused_moe_workspace_size`（`15c83f1c36`）**：为 11 种量化模式的 `trtllm_*_moe` 引入 `query_workspace_size` 干跑（dry-run）模式，仅记录 runner 调用前分配的临时张量总大小，并支持通过 `workspace_buffer=` 复用调用方缓冲区。在 B300 上对 `16384 tokens, H=7168, I=2048, E=256, top-8` 进行自调优时，峰值显存分配从 **`3,579.9 MiB` 骤降至 `896.8 MiB`（节省 `2,683.1 MiB` / `-74.95%`）**。
2. **SM120 MegaMoE 与 Kimi-K3 SiTU 精简（`75b30ffdbf`、`e2456a4eea`）**：删除 SM120 字节完全重复的 `part002.cuh`（`1,090` 行）并重封 `18` 个 Manifest 校验和；将 `csrc/fused_moe/cake_kimi_k3_situ/` 的 103 个设备辅助函数抽取到按宏守卫（`#ifdef CAKE_SITU_NEED_*`）的 `cake_kimi_k3_nvfp4_situ_device.cuh`，合并 `fc1:n8_s2a` 的 SM100/SM103 分支，并通过 `CAKE_SITU_KERNEL` / `CAKE_SITU_SHIM` 共享 8 组 Host Binding，在保持 42 个程序与 73 个架构构建 SASS 100% 恒等的前提下，代码量从 **`75,311` 行降至 `58,293` 行（净减 `17,018` 行 / `-22.60%`）**。
