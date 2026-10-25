# TensorRT-LLM 多模态编码器 CUDA Graph 与 Gemma4 预算、vLLM cuBLAS 清理与 Offloading 尾块及 SGLang MSA-v1 NVFP4 KV 核函数

> English version: [trtllm-qwen3vl-graph-gemma4-vllm-cublas-tail-sglang-msa-nvfp4.en.md](trtllm-qwen3vl-graph-gemma4-vllm-cublas-tail-sglang-msa-nvfp4.en.md)

## 1. 背景与核心问题

在生产级多模态与混合注意力推理引擎（TensorRT-LLM、vLLM、SGLang）中，系统架构师面临五类跨层耦合挑战：

1. **多模态视觉编码器 CUDA Graph 捕获的 Padding 污染与零长度拒绝 (`TensorRT-LLM #20213`)**：全注意力视觉编码器（Qwen3-VL、RADIO、GLM5-Next）若直接将 padding token 追加到最后一个真实图像上下文末尾，会导致真实视觉 token 对 padding 行产生注意力泄漏；而若将 dummy context 设为空（0 token），底层 attention backend 又会拒绝零长度上下文。
2. **Gemma4 多模态图像/视频与音频编码器的按项预算失真 (`TensorRT-LLM #20663`)**：Gemma4 视觉塔的内部激活张量大小取决于 **填充后 (padded) 的 patch 总数** 而非有效 patch 数，若调度器仅按池化后的有效 soft token 数预算显存，会在高分辨率多图批次下触发 OOM。
3. **vLLM 权重缓存守护进程残留显存与 EAGLE/MTP 混合 KV Offloading 尾块对齐 (`vLLM #63451, #61019, #60084`)**：`WeightCacheDaemon` 在导出 CUDA IPC 句柄并销毁通信组后，ATen 缓存的 cuBLAS workspace 仍常驻 GPU；同时在 Mamba + Full-Attention + EAGLE/MTP 混合模型中，核心调度器在精细命中后丢弃一个哈希对齐单元 (`cache_hit_alignment_tokens`)，若 OffloadingConnector 仍按 Mamba 截点存储或查询 Attention 尾块，会导致永远无法命中；而在 ROCm 上，指向 32 GiB 共享 KV 缓存切片的单层视图会因底层 `storage` 超出 2 GiB 而丧失 Triton 32 位 `buffer_load`/`buffer_store` 加速。
4. **SGLang `MSA-v1` NVFP4 KV 量化中 IEEE 除法与硬件近似倒数的比特漂移 (`SGLang #46139, #46140`)**：在 B300 (`sm_103a`) 上对 16 元素块执行 NVFP4 KV 量化时，若用精确 IEEE 除法替代 `flashinfer::nvfp4_kv_quantize` 基准所使用的 `rcp.approx.ftz.f32`，~1 ULP 的误差会跨越 E2M1 最近偶数舍入边界，导致约 `0.1%` 的输出半字节 (nibble) 发生漂移。
5. **PD 分离下的 Decode 端 RadixCache 命中与全提示词 KV 校验和冲突 (`SGLang #39729`)**：当开启 `--disaggregation-decode-enable-radix-cache` 时，Prefill 仅传输未命中的后缀 KV，而全提示词 KV digest 覆盖整个序列，导致每次前缀命中都被误判为传输损坏。

---

## 2. 第一手源码实现路径审计

### 2.1 TensorRT-LLM `EncoderGraphIOAdapter` 隔离 Dummy Context 与 Gemma4 预算 (`TensorRT-LLM@7d391a06d7fb`, #20213; `abb94ab8fc1c`, #20663)

- **隔离式 1-Token Dummy Padding Context (#20213)**：`tensorrt_llm/_torch/models/ modeling_multimodal_encoder_graph.py` 抽象出 `EncoderGraphIOAdapter` 与共享 `MultimodalEncoderMixin`，统一支持 **Qwen3-VL**、**RADIO (`vision`)** 与 **GLM5-Next (`image`)**。当 `enable_padding=True` 时，针对每个配置的真实预算桶 `(total_tokens, num_contexts)`，捕获图键强制扩展为：
  $$\text{captured\_total\_tokens} = \text{bucket.total\_tokens} + 1, \quad \text{captured\_num\_contexts} = \text{bucket.num\_contexts} + 1$$
  预留的 `1` 个额外 token 确保即便真实请求恰好填满 `bucket.total_tokens`，隔离的 dummy padding context 也至少包含 1 个 token，既消除了跨上下文注意力污染，又满足了 attention backend 非空上下文约束。
- **Gemma4 多模态按项预算公式 (#20663)**：在 `modeling_gemma4mm.py` 中，`get_mm_encoder_item_metadata` 为每张图像、每个视频帧或每段音频生成精确的 `MultimodalEncoderItemMetadata`：
  $$\text{padded\_patches\_per\_frame} = \text{max\_soft\_tokens} \times (\text{pooling\_kernel\_size})^2, \quad \text{AUDIO\_SUBSAMPLING\_FACTOR} = 4$$
  例如当 `max_soft_tokens = 256, pooling_kernel_size = 3` 时，单帧编码器输入固定按 $256 \times 3^2 = 2304$ 个 padded patches 计入编码器激活预算，而音频塔经过两层 stride-2 卷积下采样后每 4 个 mel 帧保留 1 个编码 token（如 `3000` mel 帧对应 `750` 个音频 token）。

### 2.2 vLLM cuBLAS 工作区释放、EAGLE Offloading 尾块证明裕度与 ROCm 32 位指针范围 (`vllm@b3466364b69d`, #63451; `fc0cfcdc05b1`, #61019; `a55937f25528`, #60084)

- **`WeightCacheDaemon.release_distributed()` 清理 cuBLAS 工作区 (#63451)**：在 `destroy_model_parallel()` 与 `destroy_distributed_environment()` 之后、调用 `torch.accelerator.empty_cache()` 之前，显式执行 `torch._C._cuda_clearCublasWorkspaces()`，释放预热阶段由 ATen 独立缓存的 cuBLAS scratch 显存。
- **`OffloadingConnector` EAGLE/MTP 尾块证明裕度 (#61019)**：在混合 Mamba + Attention + EAGLE 模型（如 Qwen3-Next MTP）中，对于 30-token 提示词且 Mamba 截点位于 `boundary = 24` 时，核心调度器在精细命中后回退一个对齐单元。`scheduler._partial_tail_key_tokens(group, boundary)` 对非 EAGLE 组（Mamba）保持 `24`，而对 EAGLE 共享的 Attention 组按核心对齐粒度偏移：
  $$\text{proof\_tokens} = \text{boundary} + (\text{cache\_hit\_alignment\_tokens} \;\text{if not None else}\; \text{block\_size})$$
  当 `cache_hit_alignment_tokens = 4` 时为 $24 + 4 = 28$；当为 `16` 或 `None`（回退至 `block_size = 16`）时为 $24 + 16 = 40$。
- **ROCm `set_triton_ptr_range` 恢复 32 位 Buffer 寻址 (#60084)**：Triton AMD 后端仅在指针可访问范围 `< 2 GiB` (`(1 << 31) - 1`) 时生成 32 位 `buffer_load`/`buffer_store` 指令，且默认检查底层 `storage` 总大小（例如 64 层共 32 GiB 会导致退化为 64 位寻址）。`set_triton_ptr_range(kv_cache, key_cache, value_cache)` 为单层视图绑定 `view.ptr_range = functools.partial(int, region_end - view.data_ptr())`（如单层 512 MiB $< 2$ GiB），无损恢复 32 位硬件 buffer 寻址。

### 2.3 SGLang `MSA-v1` NVFP4 KV 核函数、MiniMax-M3.1 NVFP4 与 PD 分离 KV 校验和 (`sglang@174bd2976d34`, #46139; `a99c7d050fae`, #46140; `5d11819f3a7a`, #39729)

- **`MSA-v1` NVFP4 KV 量化与 `rcp.approx.ftz.f32` 对齐 (#46139)**：`nvfp4_store_kv_index.cuh` 将每 `kNvfp4BlockSize = 16` 个 BF16 元素量化为 `kNvfp4BlockBytes = 8` 字节（每字节打包 2 个 E2M1 码字，`cvt.rn.satfinite.e2m1x2.f32`）外加 1 字节 UE4M3 块缩放，在 `4096 x 128` 头部张量上将显存从 `1,048,576 B` 压缩至 `294,912 B`（压缩比 **`3.5556x`**）。关键在于严格复用 PTX `rcp.approx.ftz.f32` 计算 `sf = rcp(gs) * (amax * rcp(6))` 与 `osc = rcp(sf * gs)`，消除了使用 IEEE 精确除法时在 B300 上实测出现的 **`~0.1%`** 输出 nibble 漂移。
- **`MiniMaxM3Nvfp4Config` 跨代投影降级 (#46140)**：静态 M3.1 权重采用 `weight_block_size = [1, 32]` MXFP8 投影与 `group_size = 16` NVFP4 路由专家（`index_k_quant_type = "nvfp4"`, `kv_cache_quant_algo = "FP8"`）。当部署在非 Blackwell 平台 (`not get_platform().is_blackwell`) 时，自动在加载期设置 `convert_mxfp8_to_block_fp8 = True` 将投影转换为 `[128, 128]` block-FP8。
- **PD 分离 KV Checksum 安全门控与按 Rank 错位注入 (#39729)**：当 `disaggregation_decode_enable_radix_cache` 开启时自动禁用全序列 KV checksum 以避免后缀传输误报；引入 `kv_layout_signature(kv_args)` 校验 Prefill/Decode 布局签名一致性；并将测试错位注入器改为按 `(random_seed, attn_tp_rank)` 独立哈希播种，确保多 TP rank 下的 mismatch all-reduce 路径得到真实覆盖。

---

## 3. 可复现 CPU/Offline 验证实验与量化汇总

```bash
python experiments/trtllm_qwen3vl_graph_gemma4_vllm_cublas_tail_sglang_msa_nvfp4/run.py --check
python -m unittest tests/test_phase75_experiments.py
```

| 引擎 / 特性 | 关键公式 / 机制 | 基准 / 修复前行为 | 修复后定量指标 |
|---|---|---|---|
| TensorRT-LLM `EncoderGraphIOAdapter` (#20213) | `(total_tokens + 1, num_contexts + 1)` | Padding 混入末个上下文或空上下文报错 | 预留 **1 token / 1 context** 隔离 dummy padding |
| TensorRT-LLM Gemma4 MM 预算 (#20663) | `max_soft_tokens * pooling_kernel_size^2` | 仅按池化输出 token 估算激活 | `256 * 3^2 = 2304` padded patches/frame；音频 4x 下采样 |
| vLLM Offloading EAGLE 尾块 (#61019) | `boundary + alignment_or_block` | Attention 与 Mamba 同用 `boundary=24` 失配 | 对齐粒度 `4 / 16 / None` 下精确映射至 **`28 / 40 / 40`** |
| vLLM ROCm Triton KV View (#60084) | `set_triton_ptr_range` 单层距离绑定 | 32 GiB 共享存储被迫退化为 64 位指针 | 单层 `512 MiB < 2 GiB`，100% 恢复 **32-bit `buffer_load`** |
| SGLang `MSA-v1` NVFP4 KV (#46139) | `16` 元素块 + `rcp.approx.ftz.f32` | IEEE 除法导致 `~0.1%` nibble 漂移 | 零比特漂移，KV 显存压缩比 **`3.5556x`** |

---

## 4. 面向解决方案架构师的设计练习与可证伪问题

1. **架构设计练习（ROCm 分层 KV 缓存容量上限）**：在 MI300X (192 GiB HBM3) 上使用 vLLM `TRITON_ATTN` 后端并通过 `set_triton_ptr_range` 为每层绑定独立 `ptr_range` 时，若模型共 80 层、KV 缓存采用 FP8 (`1 B/elem`)，单层 KV 缓存最大可分配多少 GiB 而不触发 Triton AMD 后端从 32 位 `buffer_load` 回退到 64 位寻址？
2. **可证伪后续问题**：在 SGLang PD 分离启用 Decode RadixCache 的场景下，若将 `KvChecksumComputer` 改为基于传输 block 列表的增量 Merkle/XOR 摘要而非全提示词摘要，主机端签名汇总与 RDMA 元数据开销会增加多少微秒？
