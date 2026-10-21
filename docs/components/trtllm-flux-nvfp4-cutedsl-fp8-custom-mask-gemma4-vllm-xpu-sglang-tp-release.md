# TensorRT-LLM FLUX.1-dev NVFP4 + CUTEDSL FP8 注意力、Triton 自定义掩码 FP8、Gemma4 异构 KV 缓存、vLLM XPU 投机解码与 SGLang 跨 TP 协同 KV 释放深挖

> **研究阶段**: Phase 47 (`priority: 56`) · **问题编号**: `q-phase47-trtllm-flux-nvfp4-cutedsl-fp8-custom-mask-gemma4-vllm-xpu-sglang-tp-release` · **确定性复现脚本**: [`experiments/trtllm_flux_nvfp4_cutedsl_fp8_custom_mask_gemma4_vllm_sglang/run.py`](../../experiments/trtllm_flux_nvfp4_cutedsl_fp8_custom_mask_gemma4_vllm_sglang/run.py)

---

## 1. 核心架构问题与一手提交总览

在多模态扩散生成（FLUX.1-dev）、异构注意力架构（Gemma 4）、Intel XPU 投机解码与分离式（Disaggregated）注意力 TP 服务中，TensorRT-LLM、vLLM 与 SGLang 于 2026-10-21 落地了七项关键工程突破：

| 仓库与提交 | 日期 | 核心架构变更 | 实测性能 / 正确性收益 |
|---|---|---|---|
| `TensorRT-LLM@5d6fbb2` (`#18396`) | 2026-10-21 | FLUX.1-dev 静态 NVFP4 线性层 + CUTEDSL 逐张量 FP8 MHA（含 GEMM+GELU 与 QK Norm+RoPE+FP8 融合） | GB300 单步延迟：`1024x1024` **`54.82 -> 31.26 ms` (`1.75x`, `-43.0%`)**；`2048x2048` **`278.63 -> 163.46 ms` (`1.70x`, `-41.3%`)**，LPIPS 零损失 |
| `TensorRT-LLM@5f90e89` (`#20387`) | 2026-10-21 | `triton_prefill_with_custom_mask` 原位 FP8 缓存前缀反量化与输出 E4M3 饱和截断 `[-448, 448]` | 免除 `Medusa` / `Eagle` 树注意力预填充时的整池 KV Cache 临时拷贝 |
| `TensorRT-LLM@e8127c7` (`#18547`) | 2026-10-21 | Gemma4 `per_layer_config` 异构头维（`256` vs `512`）解析 + 自动启用 `KVCacheManagerV2` 与 FlashInfer FA2 | 修复非 `SM100f` GPU 上全局层 `head_dim=512` (`GQA=16`) 无法回退 MMHA 的崩溃 |
| `vllm@35d0166` (`#62244`) | 2026-10-21 | Intel XPU `AttentionCGSupport.ALWAYS` + 零参数置信头自适应验证（Eagle/OSA） | 8×Intel Arc Pro B60 Sonnet BS32：`K=5/7` 吞吐提升 **`+11.10% / +11.70%`**（峰值 **`468.55 tok/s`**），延迟降低 **`-8.17% / -10.98%`** |
| `vllm@550c1cb` (`#59703`) | 2026-10-21 | `OffloadingEventsTracker` 支持 `block_size=0` 占位符写入并在 `reset_prefix_cache` 跨周期保留 `kv_cache_spec_kind` | 修复混合注意力或前缀缓存重置后 `take_events()` 丢失 KV 卸载事件元数据的 Bug |
| `sglang@b57b5e2` (`#45453`) | 2026-10-21 | 分离式解码 `DecodeTransferQueue.pop_transferred` 引入 `ReduceOp.MAX` 跨 Attention-TP 同步释放 | 消除不同 TP Rank 因轮询时机差几毫秒导致的 KV 空闲池分叉与死锁 |
| `sglang@5e79333` (`#45439`) | 2026-10-21 | Rust `sglang-radix-tree` 拆分 `WALK -> OVERLAP -> COMMIT -> TAIL` 四阶段插入并导出 `Evicted` 动作 | 支持外部异步存储分层（Hierarchical Storage）在精确前缀拆分边界接管逐出 |

---

## 2. TensorRT-LLM FLUX.1-dev NVFP4 + CUTEDSL 逐张量 FP8 MHA (`#18396`)

```
  FLUX.1-dev Fused DiT QK Preprocess + CUTEDSL Per-Tensor FP8 MHA Pipeline (#18396):
  Packed BF16 QKV ---> [fusedDiTQKNormRopeQuantFp8Kernel: 1 warp/head, RMSNorm + RoPE + E4M3] ---> Q_fp8, K_fp8
                       [torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor]              ---> V_fp8
                       ---> [CUTEDSL BlackWell FMHA (1.295 ms vs TRTLLM-gen 1.474 ms / FA4 1.515 ms)]
```

1. **两项微内核融合消除重尾访存**：
   - **CUTEDSL NVFP4 GEMM + GELU Epilogue 融合**：将独立 GEMM (`97.690 us`) + `gelu_tanh` (`17.406 us`，合计 `115.096 us`) 融合为单内核 **`80.662 us`**（**`1.43x`**）；
   - **`fusedDiTQKNormRopeQuantFp8Kernel` 融合**：每个 Warp 负责一个 Token 的单个 Q/K/V Head，在寄存器内完成 BF16 向量加载、Warp 级平方和归约、`rsqrtf` RMSNorm、交错/非交错 RoPE 旋转，并直接乘以静态 `1.0f / dequant_scale` 写入 `__nv_fp8_e4m3`，将 QK 预处理 + 3 次静态量化从 `457.424 us` 压缩至 **`305.162 us`**（**`1.50x`**）。
2. **CUTEDSL FMHA 硬件瓶颈剖析**：在 `2048x2048` (`B=1, S=16896, H=24, D=128`) 下，CUTEDSL FMHA 仅耗时 **`1.295 ms`**，显著优于 `TRTLLM-gen` (`1.474 ms`, `+13.8%`) 与 `FA4` (`1.515 ms`, `+17.0%`)。NSight Compute 剖析显示该内核并非受限于 Tensor Core 或 HBM 带宽，而是受限于 **超越函数单元（XU Active `63.70%`）与寄存器压力（`168 regs/thread`，每块仅驻留 384 线程，每调度器 Eligible Warps `0.8667`）**。

---

## 3. Triton 自定义掩码 FP8 注意力与 Gemma4 异构 KV 几何 (`#20387`, `#18547`)

- **Triton 自定义掩码原位 FP8 反量化与饱和截断（`#20387`）**：在 `triton_prefill_with_custom_mask` 中，当缓存前缀页为 `float8_e4m3fn` 时，内核在寄存器内执行 `(k.to(tl.float32) * kv_scale).to(q.dtype)`，并在写回 FP8 输出前执行确定性有限饱和截断 `tl.minimum(tl.maximum(acc / out_scale, -448.0), 448.0)`，彻底避免将整个分页 KV 池复制转换为 FP16/BF16。
- **Gemma4 异构层 KV 几何与后端路由（`#18547`）**：Gemma 4 的滑动窗口层（`head_dim=256, num_kv_heads=8`）与全局注意力层（`head_dim=512, num_kv_heads=1, GQA=16`）具有不同 KV 页步长。`_has_variable_kv_cache_geometry()` 检测到层间差异后自动启用支持异构页池的 `KVCacheManagerV2`，并在非 `SM100f` GPU 上自动将默认注意力后端切至 `FLASHINFER`（规避仅支持 `head_dim <= 256` 的原生 MMHA）。

---

## 4. vLLM XPU 投机解码与 SGLang 跨 Attention-TP 协同 KV 释放 (`#62244`, `#59703`, `#45453`, `#45439`)

- **vLLM Intel XPU Eagle/OSA 全图化与自适应验证（`#62244`）**：将 `XPUAttentionBackend` 标记为 `AttentionCGSupport.ALWAYS` 以捕获完整 Eagle Drafter 图，并在 `worker_xpu_osa.py` 中启用无需额外训练参数的置信度验证，在 8×Arc Pro B60 上实现 **`468.55 tok/s`**（`K=7` 吞吐 **`+11.70%`**，E2E 延迟 **`-10.98%`**，GSM8K 精度零退化）。
- **SGLang 分离式解码跨 TP 协同 KV 释放（`#45453`）**：当启用 Attention-TP (`tp_size > 1` 且未开启 DP-Attention) 时，各 Rank 独立轮询 Mooncake/NIXL 传输完成状态会导致某 Rank 提前释放 KV 槽位并产生分页分配死锁。`pop_transferred` 通过 `torch.distributed.all_reduce(..., op=ReduceOp.MAX)` 强制所有 TP Rank 仅在全组均完成传输后同步释放 KV 内存。
