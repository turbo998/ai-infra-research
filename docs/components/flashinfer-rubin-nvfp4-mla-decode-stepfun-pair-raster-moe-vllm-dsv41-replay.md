# FlashInfer Rubin (`sm_107a`) Cake NVFP4 MLA 解码、StepFun Pair-Raster/2-CTA MoE 与 vLLM DeepSeek-V4.1 Replay 修复深挖

> **English Companion**: [flashinfer-rubin-nvfp4-mla-decode-stepfun-pair-raster-moe-vllm-dsv41-replay.en.md](flashinfer-rubin-nvfp4-mla-decode-stepfun-pair-raster-moe-vllm-dsv41-replay.en.md)  
> **Phase 41 (`priority: 50`) · 确定性复现实验**: [`experiments/flashinfer_rubin_nvfp4_mla_stepfun_moe_vllm_dsv41/run.py`](../../experiments/flashinfer_rubin_nvfp4_mla_stepfun_moe_vllm_dsv41/run.py)  
> **7-Card 4:5 视觉报告**: [`docs/reports/ai-infra-phase41-benchmark-cards.html`](../reports/ai-infra-phase41-benchmark-cards.html)

---

## 一、核心架构结论（Executive Summary）

1. **FlashInfer Rubin (`sm_107a` / R200) Cake NVFP4 MLA Paged Decode（`PR #6931` / `adca7b87d2`）**：
   - 面向 NVIDIA R200（`sm_107a` / Compute Capability `10.7`，`208 SMs`，`120 MiB L2`，CUDA 13.4 PTX ISA 9.4）首次引入针对 Kimi-K3 / DeepSeek-V3 吸收式 MLA 几何（`512` latent + `64` rope 通道）的 **Cake 稠密 NVFP4 Paged Decode** 后端。
   - **紧凑存储与动态行缩放契约**：KV Cache 每 Token 仅占 **`352 B`**（`256 B` E2M1 打包 `ckv_cache` + `32 B` E4M3 块缩放 `ckv_sf_cache` + `64 B` E4M3 `kpe_cache`，Page Size = `64`）；Query 量化器 `quantize_mla_nvfp4_query` 将 BF16 `[.., 576]` 切分为 **`356 B/row`**（`256 B q_nope + 32 B q_sf + 64 B q_rope + 4 B FP32 q_scale`），行级缩放因子 `q_scale = max(amax(nope)/(6*448), amax(rope)*kpe_scale/(448*ckv_scale))` 使 RoPE 内积直接落入 Latent Logit 量纲。
   - **混合 4-CTA / 2-CTA Cluster 协同与零分配 CUDA Graph**：Kernel 以 4-CTA Cluster（Rank 0-1 在 Tensor Memory 持有 `128` Query 行执行两次 `kind::mxf4nvf4` `K=128` + 一次 `kind::f8f6f4` RoPE MMA；Rank 2-3 片上将 V 转为 E4M3 执行 `kind::mxf8f6f4` `K=64 M256 N256` MMA）为主体，并在剩余 SM 上填充 2-CTA Pair Cluster（`208 SM` 对应 `50` 个 4-CTA + `4` 个 2-CTA；`212 SM` 对应 `46` 个 4-CTA + `14` 个 2-CTA）。在 R200 实测 16 组配置中，相对 TokenSpeed MLA 2 (2.0.3) 取得 **`1.0436x`** 几何平均加速（Q4 子集 **`1.1039x`**，在 `H96-Q4-B64-KV128K` 达 **`1.2483x`** 峰值加速：`1623.8 -> 1300.8 us`）。
2. **FlashInfer Cake StepFun SwiGLU Fused MoE 全精度族重生成（`PR #6934` / `fd4de5acce`）**：
   - 针对 StepFun SwiGLU 融合 MoE（`hidden=4096, intermediate=1536, 64 experts, top_k=8`），在 Blackwell `sm_100a`（B200）与 `sm_103a`（GB300）上重生成全部 5 个精度族（`bf16`, `fp8`, `mxfp8`, `nvfp4`, `nvfp4_bf16tok`，每架构 **`111` 个 Kernel**）。
   - **BF16 FC1 Pair-Raster Tile-128 与 NVFP4 2-CTA FC2 Tile-256**：BF16 FC1 将相邻两个 `M=128` Tile 配对光栅化共享 `N=128` 权重块，使 L2 权重读取流量降低 **`2.0x`**；`nvfp4_bf16tok` FC2 采用 2-CTA Tile-256 Kernel 替代单 CTA，在 `T=1024` 下取得 B200 **`1.0323x`**（`211.0 -> 204.4 us`）与 GB300 **`1.0245x`**（`196.8 -> 192.1 us`）端到端加速。
3. **vLLM DeepSeek-V4.1 `DecoderReplayLayers` 目标 Token 评分行保留修复（`PR #62652` / `85752a43c3`）**：
   - 修复 `DeepseekV41ModelState.add_request` 仅检查 `params.prompt_logprobs is not None` 而遗漏 `params.prompt_logprob_token_ids is not None` 的缺陷。修复后，目标 Token 评分请求（无论单块 `300` Token 还是分块 `150+150` Token）在 Replay 层完整保留全部 **`300` 行**，避免被局部注意力窗口（`WINDOW=64`）截断丢失 **`236` 行**（单块）或 **`172` 行**（双块）。

---

## 二、一手上游提交与证据映射（First-Hand Upstream Ledger）

| 上游仓库 | 提交 / PR | 日期 | 核心改动模块 | 证据 ID |
|---|---|---|---|---|
| `flashinfer-ai/flashinfer` | [`adca7b87d2` (`#6931`)](https://github.com/flashinfer-ai/flashinfer/commit/adca7b87d23d426e7ae0eae6df3992d6b44532e8) | `2026-10-21` | `flashinfer/experimental/cake_rubin_mla_nvfp4_decode/{cake_backend.py,cake_jit.py}` | `EV-2026-355` |
| `flashinfer-ai/flashinfer` | [`fd4de5acce` (`#6934`)](https://github.com/flashinfer-ai/flashinfer/commit/fd4de5acce18fd1c747c8425555f082dd4452cc3) | `2026-10-21` | `csrc/cake_stepfun_moe/{dispatch.inc,jit_Compile.py}` | `EV-2026-356` |
| `vllm-project/vllm` | [`85752a43c3` (`#62652`)](https://github.com/vllm-project/vllm/commit/85752a43c3cee76d9234704e189cc2ca69f24712) | `2026-10-21` | `vllm/models/deepseek_v41/nvidia/model_state.py` | `EV-2026-357` |

---

## 三、架构机制深挖（Architectural Deep Dive）

### 3.1 Rubin (`sm_107a`) Cake NVFP4 MLA Paged Decode 混合 Cluster 流水线

```
BF16 Query [B, q_len, H, 576]
  │  quantize_mla_nvfp4_query (1 warp/row, 356 B/row)
  ▼
q_nope (256B E2M1) | q_sf (32B E4M3) | q_rope (64B E4M3) | q_scale (4B FP32)
  │
  ├─► 4-CTA Cluster (50 clusters on 208 SMs):
  │     Ranks 0-1 (Tensor Memory 128 Q rows/CTA):
  │       2x kind::mxf4nvf4 (K=128) + 1x kind::f8f6f4 RoPE -> Softmax Anchor -> Push P (E4M3 + UE8M0)
  │     Ranks 2-3 (On-chip V E2M1->E4M3 conversion):
  │       kind::mxf8f6f4 (K=64, M256, N256) accumulate O under row anchor
  │
  └─► 2-CTA Pair Cluster (4 pairs on 208 SMs):
        Handles 128-row half-tiles / double items at 0.6x 4-CTA tile time -> Split-KV Merge (4/2/1 warps/row)
```

- **4-Slot 256B 对齐工作区契约**：`workspace_bytes(plan)` 在单一调用方 `uint8` 缓冲区上按 `256 B` 边界依次切分 `partial_o`（`BF16 [rows, max_splits, 512]`）、`partial_lse`（`FP32 [rows, max_splits]`）、`lse_scratch`（`FP32 [rows]`）与 `plan_block`（`int32 [items*6 + B + 1 + rows]`）。当 `H=96, Q=4, B=64`（`rows=24,576`）时，Query 工作区为 **`8,749,056 B`（`8.34 MiB`）**，最大 Decode 工作区上限为 **`1,617,348,608 B`（`1,542.42 MiB`）**。
- **Decode Context Parallelism (DCP) 交错映射**：本地第 `k` 个 Key 在全局序列的位置严格遵循 `((k // I) * W + r) * I + (k % I)`（`W=cp_world, r=cp_rank, I=cp_interleave`），无可见 Key 的 Rank 输出 `out = 0, lse = -inf` 并通过自然对数 LSE 跨 Rank 归约。

### 3.2 StepFun SwiGLU Fused MoE Pair-Raster FC1 与 2-CTA FC2

- **BF16 FC1 Pair-Rasterization**：对于 `H=4096, I=1536` 的 Gate+Up 投影（`2*I = 3072`，共 `24` 个 `N=128` 权重块），传统单 M-Tile 调度对每对 M-Tile 需从 L2 读取 `2 * 24 * 1 MiB = 48 MiB` 权重；Pair-Raster Tile-128 在同一 CTA 组内配对两个 M-Tile 复用同一 `N=128` 权重块，将 L2 权重带宽需求减半至 **`24 MiB`（`2.0x` 复用比）**。
- **2-CTA `nvfp4_bf16tok` FC2 Tile-256**：利用 Blackwell `cta_group::2` MMA 将 Down 投影（`I=1536 -> H=4096`）扩展为 `Tile-256`，消除单 CTA 在高路由 Token 数（`T=1024, 2048`）下的尾部分裂开销。

### 3.3 vLLM DeepSeek-V4.1 `prompt_logprob_token_ids` Replay 保护

- DeepSeek-V4.1 的 `DecoderReplayLayers` 在正常 Prefill 时会将每个 Chunk 的 Hidden States 裁剪至局部注意力窗口 `WINDOW=64` 以节省显存；但当客户端仅传入 `prompt_logprob_token_ids`（未传 `prompt_logprobs`）请求目标 Token 评分时，旧逻辑 `_keeps_rows_np[req_index] = params.prompt_logprobs is not None` 为 `False`，导致 `300` Token Chunk 被错误截断为 `64` 行。`PR #62652` 将条件补全为 `params.prompt_logprobs is not None or params.prompt_logprob_token_ids is not None`，确保目标评分 Chunk 100% 保留。

---

## 四、标准库确定性复现与验证指标（Deterministic Reproduction Metrics）

| 评估维度 | 指标名称 | 确定性复现值 / 实测值 | 验证状态 |
|---|---|---|---|
| Rubin NVFP4 MLA | KV Cache 每 Token 字节 / Query 每行字节 | `352 B` / `356 B` | `PASS` |
| Rubin NVFP4 MLA | R200 (`208 SM`) 16 配置总几何平均 / Q4 几何平均加速 | `1.0436x` / `1.1039x` | `PASS` |
| Rubin NVFP4 MLA | `H96-Q4-B64-KV128K` 峰值加速 (`MLA2 -> Cake`) | `1.2483x` (`1623.8 -> 1300.8 us`) | `PASS` |
| StepFun MoE | 每架构生成 Kernel 数 / FC1 L2 权重读取缩减比 | `111` / `2.0x` | `PASS` |
| StepFun MoE | `nvfp4_bf16tok` `T=1024` 加速比 (`B200` / `GB300`) | `1.0323x` / `1.0245x` | `PASS` |
| vLLM DSV4.1 Replay | 目标评分 Replay 保留行数（单块 `300` / 双块 `150+150`） | `300` (`+236`) / `[150, 150]` (`+172`) | `PASS` |

---

## 五、生产采纳建议（Production Guidance）

1. **Rubin R200 / SM107 部署门禁**：仅在 `batch >= 4`（特别是 MTP `q_len=4`）时启用 `cake_rubin_mla_nvfp4_decode`；在 `B=1` 极小批量下 TokenSpeed MLA 2 仍快约 `6%~10%`，应结合路由层按 `batch_size` 动态分派。
2. **CUDA Graph 捕获规范**：务必在 Graph Capture 外部调用 `prepare_cake_mla_nvfp4_paged_decode(..., seq_lens_host=...)` 显式传入 Host 端长度列表，避免隐式 `.cpu()` 同步破坏 Graph 捕获。
3. **vLLM DeepSeek-V4.1 评分服务升级**：凡使用 `prompt_logprob_token_ids` 进行奖励模型打分或 PPL 评测的集群，必须包含 `85752a43c3` 补丁，防止长 Prompt 分块重放时静默丢弃前部 Token 行。
