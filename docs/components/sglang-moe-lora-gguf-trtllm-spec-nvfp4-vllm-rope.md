# SGLang MoE LoRA 虚拟专家路由与 GGUF 次正规数保真、TensorRT-LLM DeepSeek-V4 NVFP4 尾块填充与 DeviceCopyBuffer 前缀守护及 vLLM Muse Spark 共享 FP32 RoPE 与 DeepSeek-VL2 裁剪哈希隔离

> **English Edition**: [sglang-moe-lora-gguf-trtllm-spec-nvfp4-vllm-rope.en.md](sglang-moe-lora-gguf-trtllm-spec-nvfp4-vllm-rope.en.md)  
> **Phase 63 · Priority 72 · Track B (`question-203`)** · 核验日：`2026-10-24`（UTC+08:00）

---

## 1. 核心架构问题与背景

在多租户 MoE LoRA 服务、Blackwell NVFP4 稀疏 MLA 预填充、分布式投机解码（Speculative Decoding）以及多模态前缀缓存中，SGLang、TensorRT-LLM 与 vLLM 近期修复了六项高隐蔽性的系统故障模式：

1. **SGLang MoE LoRA 虚拟专家路由越界与已禁用 Adapter 穿透**：在 `moe_lora_align_block_size` 与 `_fused_virtual_topk_ids_kernel` 中，当某请求的 `lora_id` 对应 slot 在当前投影层未启用（`adapter_enabled[lora_id] == 0`）或 `lora_id >= num_slots` 时，旧实现未将其映射为无效虚拟专家 `-1`，甚至在 `SHARED_OUTER` 共享专家路径上将无效 `lora_id` 钳制（clamp）到 slot `0`，导致未挂载 LoRA 的请求污染 slot `0` 的增量输出。
2. **SGLang GGUF MoE 融合加权核的负专家索引越界与 SM90+ FTZ 精度漂移**：当 top-$k$ 路由含被掩码的负专家 ID（`ids[token, slot] < 0`）时，未置零对应输出行；同时在 SM90+（Hopper/Blackwell）上普通乘法指令受编译期 `-ftz=true` 影响将次正规数（subnormal）刷新为零，偏离 `torch.mul` 的 IEEE-754 行为。
3. **TensorRT-LLM DeepSeek-V4 NVFP4 稀疏 MLA FMHA 页尾越界读取**：`nvfp4_compress_fp8_scratch` 仅按精确 `num_rows` 分配，而 `sparse_mla_fmha` 核始终以 `_SPARSE_MLA_FMHA_KV_TILE_SIZE = 128` 行的完整 KV tile 加载，当某批次最后一段长度不是 128 的整数倍时触发 HBM 越界读取（OOB read）。
4. **TensorRT-LLM CUDA Graph 投机解码 `DeviceCopyBuffer` 冗余 H2D 拷贝与尾部状态冲刷**：在两步（2-step）模型投机解码循环中，每步无条件重拷主机元数据不仅增加 PCIe/NVLink H2D 延迟，而且用较短前缀覆写固定设备缓冲时若冲刷尾部槽位，会导致 draft 模型丢失残留上下文。
5. **vLLM ROCm Muse Spark 目标层重复构建 32–40 份 FP32 RoPE 缓存**：在 speculative decoding 启用时，每个非 draft 目标层各自通过 `get_rope` 分配独立的 FP32 `cos_sin_cache`（在 $128\text{k}$ 上下文下每层 `64 MiB`，40 层重复占用 **`2.5 GiB`**）。
6. **vLLM DeepSeek-VL2 动态分辨率裁剪（Cropping）多模态缓存哈希碰撞**：DeepSeek-VL2 仅在单请求图像数 $\le 2$ 时启用局部裁剪（cropping），当图像数 $\ge 3$ 时关闭裁剪（整图缩放到单一全局视图）。由于 `_get_mm_hashes` 未将 `cropping = len(images) <= 2` 混入哈希键，同一张图像在 2 图请求中生成的 1000+ 裁剪视觉 token 会被 3 图请求错误命中，直接破坏视觉特征长度与语义。

---

## 2. 上游源码实现路径与关键机制

### 2.1 SGLang：MoE LoRA 虚拟专家路由契约（`d0022e1df3` / `#45681`）与 GGUF `__fmul_rn` 保真（`e6451c1a95` / `#45916`）

1. **虚拟专家 ID 严格门控（`compute_virtual_id`）**：在 `sgl-kernel/csrc/lora/moe_lora_align_kernel.cu` 中新增 `adapter_is_enabled<HasAdapterEnabled>(lora_id, adapter_enabled, num_slots, enabled_stride)`：
   $$\text{virtual\_id}(e, l) = \begin{cases} -1, & \text{if } l < 0 \lor l \ge S \lor \neg\text{enabled}[l] \lor e < 0 \\ l, & \text{else if } \text{SHARED\_OUTER} \\ -1, & \text{else if } e \ge E_{\text{weight}} \\ e + l \cdot E_{\text{weight}}, & \text{otherwise} \end{cases}$$
   同时新增 `return_inactive_prefix_count=True`，将 `[total_padded_tokens, leading_negative_expert_blocks]` 双元组返回给 SGMM/Cutlass 调度器，直接跳过排在最前面的无效 `-1` 专家块；在 `fused_moe_lora_gate_up_group` 中对 `route_bucket_ids` 在转 `int32` 前执行 `0 <= slot < MAX_LORAS` 与 `0 <= group < GROUPS_PER_SLOT` 显式边界检查。
2. **GGUF MoE 非 FTZ 加权（`apply_gguf_moe_weights_kernel`）**：在 `sgl-kernel/csrc/moe/gguf_moe_weights_kernel.cu` 中，当 `expert_id < 0` 时直接向 `output` 写 `0.0f`；当 `expert_id >= 0` 时，在 `__CUDA_ARCH__ >= 900` 下调用硬件内建函数 `__fmul_rn(value, weight)` 强制遵循 IEEE-754 最近偶数舍入且禁止 FTZ，确保次正规数（如 $10^{-39} \times 0.5$）与 PyTorch `torch.mul` 逐位一致。
3. **Radix Tree 多态缓存动作（`005333d029` / `#45444`）与 Quark FP8 路由（`9d7ad308bd` / `#45909`）**：Rust `sglang-radix-tree` 将紧耦合的枚举重构为 `Box<dyn CacheAction>` trait 对象（通过 `requires_insert_barrier()` 判断写屏障）；`QuarkW8A8Fp8MoEMethod` 在 HIP/ROCm 上当输入非 FP8 或非连续时自动降级至 `rocm_fused_experts_tkw1` / Triton。

### 2.2 TensorRT-LLM：DSv4 NVFP4 KV-Tile 零填充（`36d8aef55e` / `#20119`）与 `DeviceCopyBuffer` 前缀跳过（`dd63b36fcf` / `#19989`）

1. **NVFP4 稀疏 MLA FMHA 尾块零填充**：在 `tensorrt_llm/_torch/modules/fused_moe/quantization.py` 等 MLA 路径中，定义 `_SPARSE_MLA_FMHA_KV_TILE_SIZE = 128`，将 `nvfp4_compress_fp8_scratch` 分配尺寸从 `(num_rows, head_dim)` 扩展为 `(num_rows + 128, head_dim)` 并预置零。以 $N_{\text{rows}}=1024, D=576$ 为例，仅增加 `128 * 576 = 73,728 B`（`72 KiB`，`+12.5%` scratch 缓冲），彻底消除段尾 tile 的越界读取。
2. **`DeviceCopyBuffer` 前缀相等性跳过与尾部保留**：在 `tensorrt_llm/_torch/speculative/model_drafter.py` 中，封装固定设备地址的 `DeviceCopyBuffer`：
   - 当 `force=False` 且主机张量 `flat`（长度 $n$）与上次拷贝记录的前缀满足 `torch.equal(self._last_host[:n], flat)` 时，**完全跳过 `copy_(..., non_blocking=True)`**。
   - 当部分前缀更新（$n < \text{len}(\text{buffer})$）时，仅覆写 `self._device[:n]` 与 `self._last_host[:n]`，严格保留 `[n:]` 尾部残留状态；而当 C++ `MambaHybridCacheManager` 直接原地修改设备缓冲后，调用方传入 `force=True` 强制刷新。
   - 同时在同提交的 Kimi K3（1T MoE, `960G` 权重）部署指南中，明确量化了 `DEP16`（非专家层 BF16 复制 `114 GB/rank` + MXFP4 专家 EP16 `90 GB/rank` $\approx 210\text{ GB/rank}$）与 `TEP16`（非专家层 TP16 切分 $\approx 115\text{ GB/rank}$，预留 **~75 GB/rank** 给 KV Cache）的显存账本。

### 2.3 vLLM：Muse Spark 共享 FP32 RoPE（`b1c66e5545` / `#62757`）、CPU Pooling 免 KV 校验（`c3117c7441` / `#59405`）与 DeepSeek-VL2 裁剪哈希（`23f546eee0` / `#63120`）

1. **Muse Spark 跨层共享 FP32 `cos_sin_cache`**：在 `vllm/model_executor/models/muse_spark.py` 中，`MuseSparkTextModel` 在首个非 draft 目标层构建 `get_rope` 后，将其 `_ shared_rope` 引用注入后续全部 $L-1$ 个目标层。对于 $L=40, T_{\max}=131,072, d_{\text{head}}=128$ 的 FP32 缓存（单份 `64 MiB`），直接节省 **`2,496 MiB`（`2.4375 GiB`，$40\times$ 缩减）** GPU 显存。
2. **CPU Pooling Runner 免 KV 内存预检**：在 `vllm/v1/worker/cpu_worker.py` 中，当 `model_config.runner_type == "pooling"`（Embedding / Reranker 模型）时跳过 `available_cpu_memory < total_kv_cache_bytes` 断言，消除无 KV Cache 模型在紧凑容器内存下的误报崩溃。
3. **DeepSeek-VL2 `cropping` 哈希隔离**：在 `_get_mm_hashes` 中传入 `MultiModalHasher.hash_kwargs(model_id=..., image=img, cropping=len(images) <= 2)`，保证 $\le 2$ 图（启用局部裁剪）与 $\ge 3$ 图（全局单视图）生成互不冲突的多模态前缀缓存键。

---

## 3. 本项目确定性实验验证与架构师设计建议

| 验证子项（`experiments/sglang-moe-lora-gguf-trtllm-spec-nvfp4-vllm-rope/run.py`） | 关键输入参数 | 确定性验证结果 | 架构师落地准则 |
|---|---|---|---|
| SGLang MoE LoRA `compute_virtual_id` (`d0022e1df3`) | `num_slots=4, num_experts=8, enabled=[1,0,1,1]` | 禁用 slot `1` $\to -1$；越界 slot `4` $\to -1$；有效 `(e=5, l=2)` $\to 21$ | 启用 `return_inactive_prefix_count` 跳过 `-1` 前缀块，严禁将无效 slot clamp 到 `0` |
| TRT-LLM NVFP4 FMHA Scratch Padding (`36d8aef55e`) | `num_rows=1024, head_dim=576, tile=128` | 尾部零填充 `73,728 B`（`72 KiB`），总计 `663,552 B` | 稀疏 MLA KV 压缩缓冲必须额外预留 `128 * head_dim` 零行以对齐 TMA tile |
| TRT-LLM `DeviceCopyBuffer` (`dd63b36fcf`) | 5 步投机解码元数据序列（含短前缀与 `force=True`） | 实际入队 `3` 次拷贝、跳过 `2` 次冗余拷贝，尾部 `[30, 40]` 完整保留 | CUDA Graph 内嵌缓冲应按前缀相等性短路 H2D，但 C++ 侧原地写后须传 `force=True` |
| vLLM Muse Spark Shared RoPE (`b1c66e5545`) | $L=40, T_{\max}=131072, d=128$ (`float32`) | 单份 `64.0 MiB`，节省 **`2,496.0 MiB`**（$40\times$ 缩减） | 同构层的 RoPE `cos_sin_cache` 应在 Model 级单例化共享（注意避开异构 draft 层） |
| vLLM DeepSeek-VL2 Cropping Hash (`23f546eee0`) | 同一图像在 $N=2$ vs $N=3$ 请求 | `cropping=True` 与 `cropping=False` SHA-256 前缀完全隔离 | 任何依赖请求级上下文（如图片总数、动态分辨率档位）的视觉预处理器必须将该条件纳入哈希键 |
