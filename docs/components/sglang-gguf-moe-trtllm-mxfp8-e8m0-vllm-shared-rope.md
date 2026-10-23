# SGLang GGUF MoE 权重核与流排空、TensorRT-LLM MXFP8 E8M0 次正规数 NaN 修复及 vLLM 共享 RoPE 缓存

> **Phase 66 · Priority 75 · Track B（推理引擎、量化算子与生命周期治理）**  
> **覆盖上游源码提交**：`sgl-project/sglang@e6451c1a956d`、`sgl-project/sglang@194abb8e54a8`、`NVIDIA/TensorRT-LLM@348347ce3e93`、`vllm-project/vllm@b1c66e55450d`、`vllm-project/vllm@d143a9de3915`  
> **配套确定性实验**：`experiments/sglang-gguf-moe-trtllm-mxfp8-e8m0-vllm-shared-rope/run.py` / `summary.json`

---

## 1. 架构背景与核心工程问题

在量化 MoE 推理（GGUF / MXFP8）、多模态解码流管理以及 ROCm 大模型长序列部署中，IEEE-754 次正规数（Subnormal Float）、多 Warp 共享内存索引以及进程优雅关闭预算构成了三个极具隐蔽性的生产级故障源：

1. **TensorRT-LLM MXFP8 `E8M0` 最小缩放因子（`2^-127`）触发 `rcp.approx.ftz.f32` 全块 `NaN` 崩溃**：在 `cpp/tensorrt_llm/kernels/quantization.cuh` 的 `cvt_warp_fp16_to_mxfp8` 中，`E8M0` 编码的最小缩放因子为 `2^-127`（当原始字节 `u8 == 0` 时）。而在 IEEE-754 FP32 中，最小正规数为 `2^-126`，因此 `2^-127`（位模式 `0x00400000`，约 `5.877e-39`）在 FP32 中属于**次正规数（Subnormal）**！当 MoE 中间层（如近零 SwiGLU 激活块）的最大绝对值 `vecMax` 落在 `(0, 448 * 2^-127]`（即 `(0, 2.633e-36]`）区间时，`vecMax != 0.f` 守卫放行，但 `reciprocal_approximate_ftz(SFValue)` 底层的 `rcp.approx.ftz.f32` 会将 FP32 次正规数输入直接冲刷为 `+0.0`，返回 `+inf`；随后块内任何 `0.0` 元素乘以 `outputScale (+inf)` 立即产生 `0.0 * inf = NaN`，污染整个残差流。
2. **SGLang GGUF MoE 多 Warp `tile_y_ds` 共享内存未初始化与 SM90 Fast-Math 次正规数冲刷**：在 `python/sglang/kernels/aot/csrc/quantization/gguf/moe.cuh` 的 `moe_q` 中，当 `mmq_x > nwarps`（例如 `mmq_x = 64, nwarps = 8`）时，旧代码仅写入了 `tile_y_ds[threadIdx.y * ...]`（即仅填充前 `8` 行，其余 `56` 行共 **`87.5%`** 的激活缩放因子保持未初始化状态）；同时在 SM90 开启 `-use_fast_math` 时，普通乘法 `value * weight` 会冲刷次正规数专家权重乘积。此外，在 `cosmos3_simulation/dreams.py` 中，`CausalDecodeQueue.__exit__` 在 `self.outputs.clear()` 之前调用 `self.stream.synchronize()`，导致释放锁页内存（Pinned Outputs）时记录到流上的 Host Allocator 事件在上下文退出后仍未排空。
3. **vLLM ROCm Muse Spark 逐层重复分配 FP32 RoPE 缓存与 `EngineCore` 立即中止后的资源释放超时**：在 `vllm/models/muse_spark/amd/model.py` 中，目标模型每一层独立实例化 `RotaryEmbedding` 并按 `max(config.max_position_embeddings, max_model_len)` 分配 FP32 `cos_sin_cache`（例如 32 层、`max_position_embeddings=131072`、`head_dim=128` 消耗 **`2,048 MiB`** 显存）；此外当 `request_timeout == 0` 触发立即中止时，非 ROCm 平台缺少通过 `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` 配置设备资源清理宽限期的机制。

---

## 2. 第一手源码实现路径与关键修复机制

### 2.1 TensorRT-LLM MXFP8 `exp2f_rcp(fp8SFVal)` 指数倒数与 Ray 非 MPI `mapping.world_size`（`TensorRT-LLM@348347ce3e93`）

- **MXFP8 `E8M0` 次正规数安全倒数**：在 `cpp/tensorrt_llm/kernels/quantization.cuh` 中，移除将 `tmpSFVal` 解码为 `float SFValue` 再调用 `reciprocal_approximate_ftz(SFValue)` 的路径，直接对 `uint8_t fp8SFVal` 调用 `exp2f_rcp(fp8SFVal)`：
  $$\text{outputScale} = \begin{cases} 2^{127 - \text{fp8SFVal}}, & \text{vecMax} \neq 0 \\ 0.0, & \text{vecMax} = 0 \end{cases}$$
  当 `fp8SFVal == 0` 时，$2^{127 - 0} = 2^{127} \approx 1.7014 \times 10^{38}$，这是完全合法的 **FP32 正规数（Normal Float）**！块内 `0.0` 元素乘以 $2^{127}$ 严格等于 `0.0`，而非零最大元素 $448 \times 2^{-127} \times 2^{127} = 448.0$，彻底根除近零 MoE SwiGLU 块的 `NaN` 爆炸。
- **Ray 编排器下自定义编译 Pass 的 `world_size` 修复**：在 `tensorrt_llm/_torch/compilation/backend.py` 的 `build_custom_passes()` 中，由于 Ray 会设置 `TLLM_DISABLE_MPI` 导致 `tensorrt_llm.mpi_world_size()` 恒返回 `1`，现改为优先读取 `mapping.world_size if mapping is not None else tensorrt_llm.mpi_world_size()`，恢复多卡 AllReduce/UserBuffers 融合 Pass；同时在 `piecewise_optimizer.py` 中新增 `TLLM_PWCG_NVTX_PROBE=1` 环境变量，用 `torch.cuda.nvtx.range_push("pwcg_runner")` 标注分段 CUDA Graph 间的 Host 胶水开销。

### 2.2 SGLang GGUF MoE `apply_gguf_moe_weights_kernel`、多 Warp 缩放加载与流排空顺序（`sglang@e6451c1a956d`、`sglang@194abb8e54a8`）

- **多 Warp `tile_y_ds` 完整步进填充**：在 `python/sglang/kernels/aot/csrc/quantization/gguf/moe.cuh` 中，将单次写入改为 `#pragma unroll for (int i = 0; i < mmq_x; i += nwarps)` 循环，每次计算 `col_y_eff = token_offs[i / nwarps] / top_k` 并写入 `tile_y_ds[(threadIdx.y + i) * (WARP_SIZE_GGUF / QI8_1) + kby]`，确保全部 `64/64` 行缩放因子完整加载。
- **`apply_gguf_moe_weights_kernel` 路由负 ID 归零与 PTX `mul.rn.f32`**：在 `gguf_moe_weights.cuh` 与 `moe_sum.cu` 中，首先使用 `at::assert_no_overlap(outputs, weights/ids)` 校验内存无重叠；当 `ids[token * id_stride0 + slot * id_stride1] < 0` 时直接将该行置零；对有效路由槽位，在 CUDA 路径下显式执行内联汇编 `asm("mul.rn.f32 %0, %1, %2;" : "=f"(product) : "f"(value), "f"(weight));`，绕过 SM90 `-use_fast_math` 的 FTZ（Flush-to-Zero），保留次正规数乘积精度。
- **Cosmos Dreams 解码流排空顺序**：在 `dreams.py` 的 `CausalDecodeQueue.__exit__` 中，将 `self.stream.synchronize()` 移入 `finally:` 块的最末尾（位于 `self.vae.reset_causal_decode_state()` 和 `self.outputs.clear()` 之后），确保释放锁页输出张量时向该流注册的 Host Allocator 事件在退出前被完整同步排空。

### 2.3 vLLM ROCm Muse Spark 共享 FP32 RoPE 缓存与 `EngineCore` 关闭宽限期（`vllm@b1c66e55450d`、`vllm@d143a9de3915`）

- **跨层共享 `max_model_len` FP32 RoPE 缓存**：在 `vllm/models/muse_spark/amd/model.py` 中，非 Draft 目标层（`self.shares_rotary_emb = self.use_rope and not is_draft`）通过 `get_rope(..., max_position=vllm_config.model_config.max_model_len, dtype=torch.float)` 共享单一的全局 FP32 `RotaryEmbedding` 实例，并在 `_qk_norm_rope_kv_insert` 中断言 `cos_sin_cache.dtype == torch.float32`。在 32 层、`config.max_position_embeddings=131072`、部署 `max_model_len=8192`、`head_dim=128` 场景下，RoPE 缓存显存从 **`2,048.0 MiB`** 骤降至 **`4.0 MiB`**（节省 **`2,044.0 MiB`**，压缩比 **`512.0x`**）。
- **`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` 宽限期控制**：在 `vllm/v1/engine/utils.py` 的 `get_engine_process_shutdown_timeout()` 中，当且仅当 `request_timeout == 0 and process_timeout == 0`（即立即中止请求且无外层剩余预算）时，优先返回 `float(envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS)`（若已设置），否则在 ROCm 下返回默认 `15.0s`；而当 `request_timeout > 0` 且外层排空预算已耗尽（`process_timeout == 0`）时，绝不重新授予额外宽限期，严格守护排空截止时间（Deadline）。

---

## 3. 确定性实验结果与对比矩阵

运行 `python experiments/sglang-gguf-moe-trtllm-mxfp8-e8m0-vllm-shared-rope/run.py --check` 得到以下验证结果：

| 评估维度 | 修复前基线 (Legacy) | 修复后方案 (Phase 66) | 关键量化指标 / 正确性保障 |
| :--- | :--- | :--- | :--- |
| **TRT-LLM MXFP8 `E8M0=0` (`2^-127`) 倒数** | `rcp.approx.ftz.f32(2^-127) -> +inf` | `exp2f_rcp(0) -> 2^127` (`1.701e38`) | 块内含 `0.0` 时消除 `0 * inf = NaN` |
| **TRT-LLM Ray (`TLLM_DISABLE_MPI`) 编译 Pass** | `mpi_world_size() == 1`（跳过分布式融合） | 读取 `mapping.world_size` | 恢复多卡自定义编译 Pass 与 `pwcg_runner` NVTX |
| **SGLang GGUF MoE `tile_y_ds` (`mmq_x=64, nwarps=8`)** | 仅填充 `8/64` 行（`87.5%` 未初始化） | 循环步进填充 `64/64` 行 + `mul.rn.f32` | 消除多 Warp 量化比例尺垃圾值与 SM90 FTZ 误差 |
| **vLLM Muse Spark 32 层 RoPE 显存 (`L=8192`)** | `2,048.0 MiB`（32 层 × `131072` 独立缓存） | `4.0 MiB`（全局共享单份 `8192` FP32 缓存） | 节省 **`2,044.0 MiB`** 显存（压缩 **`512.0x`**） |
| **vLLM `EngineCore` 关闭超时 (`req=0, proc=0`)** | CUDA 固定 `0s`（强制 Kill 易残留显存） | 遵循 `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` | 支持自定义清理窗口且不破坏 `req>0` 排空截止时间 |

---

## 4. 面向解决方案架构师的设计练习与可证伪问题

1. **数值边界练习**：为什么在 MXFP8 量化中，仅用 `vecMax != 0.f` 无法防止 `reciprocal_approximate_ftz(SFValue)` 产生 `+inf`？请给出触发该 Bug 的精确 `vecMax` 浮点数区间。  
   - **解答**：因为 `E8M0` 的最小可表示比例尺为 $2^{-127}$，对应块最大可表示正数为 $448 \times 2^{-127} \approx 2.6331 \times 10^{-36}$。只要块内最大元素 `vecMax` 落在 $(0, 448 \times 2^{-127}]$ 区间内（这在正规数或次正规数 FP16/BF16->FP32 转换后完全非零），`__nv_cvt_float_to_e8m0` 向上取整得到的指数编码仍为 `0`（即 $2^{-127}$），而 $2^{-127}$ 在 FP32 中是次正规数，被 `ftz` 冲刷为 `0.0` 后求倒数必然得到 `+inf`。
2. **可证伪假设**：在 vLLM 中设置 `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=30` 后，如果外层管理器设置了 `request_timeout=7` 秒用于优雅排空请求，且 7 秒耗尽后传入 `process_timeout=0`，`get_engine_process_shutdown_timeout(7, 0)` 是否会额外再给 30 秒？  
   - **验证结论**：**不会**。函数仅在 `request_timeout == 0 and process_timeout == 0` 时读取 `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`；当 `request_timeout == 7` 时直接返回 `0`，防止破坏请求排空的硬截止时间。
