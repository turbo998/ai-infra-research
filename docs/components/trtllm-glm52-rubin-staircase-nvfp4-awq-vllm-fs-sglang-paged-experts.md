# TensorRT-LLM GLM-5.2 Rubin (`SM107`) 阶梯式低延迟解码与 NVFP4_AWQ MegaMoE、vLLM 共享文件系统幂等 KV 卸载与 SGLang 全量化按需分页专家

> **关联研究问题 (`question_id`)**：`trtllm-glm52-rubin-awq-megamoe-vllm-fs-offload-sglang-paged-experts`（Phase 76，`priority: 85`）  
> **先修依赖 (`depends_on`)**：`trtllm-qwen3vl-graph-gemma4-vllm-cublas-tail-sglang-msa-nvfp4`  
> **关联证据 ID**：`ev-trtllm-glm52-rubin-awq-vllm-fs-sglang-paged-experts-2026`  
> **本地可复现实验**：`experiments/trtllm_glm52_rubin_staircase_nvfp4_awq_vllm_fs_sglang_paged_experts/run.py`  
> **英文版文档**：[`trtllm-glm52-rubin-staircase-nvfp4-awq-vllm-fs-sglang-paged-experts.en.md`](trtllm-glm52-rubin-staircase-nvfp4-awq-vllm-fs-sglang-paged-experts.en.md)

---

## 1. 核心架构问题与背景

面向下一代 GPU 架构（NVIDIA Rubin `SM107` / Blackwell `SM100`）、多实例共享存储 KV Cache 卸载以及超大规模 MoE 主机专家按需分页，三大开源推理引擎在 2026 年 10 月下旬落地了关键生产特性：

1. **GLM-5.2 (`GlmMoeDsa`) 在 Rubin (`SM107`, TP16) 上的超低延迟阶梯式投机解码**：在小批次解码 ($M \le 8$) 与多轮 MTP 投机采样下，GLM-5.2 前三层稠密 MLP (`H=6144`, TP16 单卡 `I=768`) 受限于权重带宽 (`27 MiB/rank`) 与三次独立 kernel 启动开销；同时 MTP 阶梯投机循环中局部 argmax、TP16 AllGather 与草稿槽位回写需要多次微小的胶水 kernel 与同步，且 DeepGEMM 在 `SM 10.7` (Rubin) 上缺少向 `100f` 架构族的映射。
2. **MegaMoE CuteDSL 对 `NVFP4_AWQ` / `NVFP4_ARC` 预量化缩放因子的缺失**：对于 Kimi-K2.5 等使用 AWQ/ARC 通道平滑的 NVFP4 模型，若在动态激活量化为 FP4 前未乘上 `fc31_act_scale`（`w1_pre_quant_scale` / `w3_pre_quant_scale`），离群通道会直接饱和 `E2M1` 动态范围（$\pm 6.0$），导致严重精度坍塌。
3. **vLLM 多引擎共享文件系统 KV 卸载的 `EEXIST` 竞态失败**：当多个 vLLM 实例并发将相同内容哈希的 KV 块写入共享 POSIX/NFS 文件系统（先写临时文件再 `rename(tmp, dest)`）时，部分共享存储对目标已存在的 `rename` 返回 `EEXIST` (`FileExistsError`)，导致落败的卸载任务报错并残留孤儿 `.tmp` 文件。
4. **SGLang 按需分页专家 (`--enable-paged-experts`) 的量化格式局限与字节对齐断言**：此前 `paged_experts` 仅支持非量化、block-FP8 和 GPTQ，无法覆盖生产常见的 AWQ、MXFP4、NVFP4、compressed-tensors 与 W8A8；且 `paged_experts_gather_kernel` 强制要求每行字节数被 4 整除 (`row_bytes % 4 == 0`)，遇到 2 字节对齐的子字缩放表（如 22 字节 scale 行）即触发断言崩溃。此外，混合 Mamba/GDN 模型在开启 HiCache 时缺少主机端 Mamba 状态池占用指标，导致主机 Mamba 池先于主机 KV 池耗尽而静默丢弃前缀匹配。

---

## 2. 第一手源码与提交审计

| 仓库与提交 | 合并日期 | 证据层级 | 核心源码路径与改动重点 |
|---|---|---|---|
| `NVIDIA/TensorRT-LLM@e707e62fc4d3` (PR `#20517`) | `2026-10-24` | `implementation` | `cpp/tensorrt_llm/kernels/glm52Kernels/glmDenseMlpSmallM.{h,cu}`、`glmDraftSampleGlue.{h,cu}`、`glmDsaHistTopK.{h,cu}`、`jit_utils.py`：实现 Rubin `SM107` TP16 下 `M<=8` 的流式权重融合 MLP（将权重放在 `m16n8k16` 的 `M` 侧、激活放在 `N` 侧）、16B Lamport 包融合投机采样胶水 kernel、直方图快速 TopK，并将 DeepGEMM `10.x (minor > 3)` 映射至 `100f`。 |
| `NVIDIA/TensorRT-LLM@ecae3c97e2b6` (PR `#20823`) & `9666d7427a8c` (PR `#20774`) | `2026-10-25` | `implementation` | `mega_moe_cute_dsl.py`、`fused_moe_cute_dsl.py`、`w4a16_nvfp4.py`、`modeling_kimi_linear.py`：在 NVFP4 激活量化及 W4A16 路径前注入 `x = x * fc31_act_scale` 并在流式专家加载中搬运 `w1/w3_pre_quant_scale`；`loraManager.cpp` 在重载适配器时保留原请求的 LoRA 格式。 |
| `vllm-project/vllm@1757f6720c3c` (PR `#61390`), `503e4e16ad8d` (`#63126`) & `4a81a5097631` (`#55576`) | `2026-10-25` | `implementation` | `csrc/kv_offload/fs_io.cpp` + `vllm/v1/kv_offload/tiering/fs/io.py`：在 `rename` 抛出 `EEXIST` / `FileExistsError` 时清理 `tmp_path` 并返回 `0`（幂等成功）；捕获异步 lookup 异常（`#63126`）并为 NIXL `OBJ` 层启用传输遥测（`#55576`）。 |
| `sgl-project/sglang@2d3de82633b0` (PR `#45787`), `26a1e5110d4a` (`#43727`) & `97220dd67437` (`#46173`) | `2026-10-25` | `implementation` | `python/sglang/srt/layers/moe/paged_experts/` + `sgl-kernel/csrc/expert_specialization/paged_experts.cu`：将按需分页专家扩展至 12 种量化方法，`paged_experts_gather_kernel` 改为按偶数 `row_bytes` 选择 `16B (float4)` / `4B` / `2B (uint16_t)` 拷贝；`metrics_reporter.py` 导出 `sglang:hicache_host_mamba_used_states` 与 `total_states`（`#43727`）。 |

---

## 3. 关键实现机制与架构推导

### 3.1 TensorRT-LLM GLM-5.2 Rubin (`SM107`) 小批次融合 MLP 与 Lamport 投机采样胶水
1. **MMA 操作数互换的 `glmDenseMlpSmallM` ($M \le 8, H = 6144, I_{\text{local}} = 768$)**：
   - 标准 Tensor Core `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` 的 `M` 维为 16、`N` 维为 8。当解码有效 Token 数 $M \le 8$ 时，若将 Token 放在 `M` 维会浪费至少一半 Tensor Core 行；`glmDenseMlpSmallM` **将权重行放在 MMA 的 `M` 侧（16 列输出通道），将 $\le 8$ 个激活行放在 MMA 的 `N` 侧**。
   - 网格划分为 `64` 个 N-tile ($\text{TILE\_N}=12 \implies 64 \times 12 = 768$) $\times$ `SPLIT_K = 4` $= 256$ 个 CTA，单次流式读取 `Gate+Up` (`18 MiB`) 与 `Down` (`9 MiB`) 共 `27 MiB` BF16 权重，在芯片内通过共享内存槽完成 `Gate/Up SPLIT_K` 归约、`SiLU(gate) * up` 门控与 `Down` 投影原子累加。
2. **`glmDraftSampleGlue` 的 16 字节 Lamport 包**：
   - 将局部最大 logit 归约、TP16 NVLink 16-byte Lamport 包（携带 `(bestVal, bestIdx, epoch, negZeroTag)`）跨卡广播、全局赢家选择以及 `position_ids` / `draft_tokens` 更新（支持 `kNone`、`kIncrement`、`kRewind` 三种位置模式）压缩为每轮投机步单次 `1 CTA` 启动。

### 3.2 MegaMoE CuteDSL 的 `NVFP4_AWQ` 预量化平滑
在 `MegaMoECuteDSL` 与 `FusedMoECuteDSL` 中，当 `quant_config.quant_mode == NVFP4_AWQ` 时，输入激活 $\mathbf{x} \in \mathbb{R}^{M \times K}$ 在进入 `torch.ops.trtllm.fp4_quantize` 之前执行逐通道平滑：
$$\tilde{\mathbf{x}} = \mathbf{x} \odot \mathbf{s}_{\text{fc31}}, \quad (\mathbf{q}_{\text{fp4}}, \mathbf{s}_{\text{sf}}) = \text{FP4Quantize}(\tilde{\mathbf{x}}, g_{\text{fc31}}, \text{sf\_vec\_size}=16)$$
在我们的离线算例中，原始通道峰值 $\max |\mathbf{x}| = 8.0$ 经 $\mathbf{s}_{\text{fc31}}$ 平滑后降至 $\max |\tilde{\mathbf{x}}| = 1.0$（动态范围压缩 `8.0x`），完全落入 E2M1 高分辨率区间。

### 3.3 vLLM 共享文件系统幂等 `rename` 与 SGLang 子字长 `2B` 专家拷贝
- **vLLM `fs_io.cpp` (`#61390`)**：当 $W=8$ 个实例并发卸载 $B=64$ 个相同前缀哈希的 KV 块时，旧实现会产生 $(8-1)\times 64 = 448$ 次 `EEXIST` 报错并遗留 `448` 个 `.tmp` 孤儿文件；修复后在 `errno == EEXIST` 时直接 `unlink(tmp_path)` 并返回 `0`（`0` 失败、`0` 孤儿文件）。
- **SGLang `paged_experts_gather_kernel` (`#45787`)**：将参数从 4 字节 `words` 改为偶数 `row_bytes`，按对齐粒度三级分发：
  $$\text{copy\_width}(B_{\text{row}}) = \begin{cases} 16\text{ B }(\texttt{float4}), & B_{\text{row}} \equiv 0 \pmod{16} \\ 4\text{ B }(\texttt{uint32\_t}), & B_{\text{row}} \equiv 0 \pmod{4} \\ 2\text{ B }(\texttt{uint16\_t}), & B_{\text{row}} \equiv 0 \pmod{2} \end{cases}$$
  从而安全支持 `22 B` 等非 4 倍数的 MXFP4/NVFP4/AWQ 子字缩放表行拷贝。
- **SGLang HiCache 主机 Mamba 池监控 (`#43727`)**：当设备 Mamba 池较小（如 64 槽位，`--hicache-ratio=2.0` 对应主机 128 槽位），而主机 KV 池可容纳 512 个前缀时，第 129..200 个前缀会触发主机 Mamba LRU 驱逐，造成 **72 个主机 KV 前缀成为无法命中的“孤儿前缀”**。导出 `sglang:hicache_host_mamba_used_states` 与 `total_states` 使运维可直接告警并调大 Mamba 池配比。
