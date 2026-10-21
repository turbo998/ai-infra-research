# TensorRT-LLM MiniMax-M3 三项式 BF16 拆分路由器与 FLUX FP8 融合算子、SGLang ROCm Quark AITER ASM FP4 缩放重排与繁忙健康探针应答、及 vLLM MoRIIO 步屏障读取等待深度解析

> **演进阶段**：Phase 45 (`priority: 54`) · **上游一数据源**：`NVIDIA/TensorRT-LLM@522daa8cdf` (`#20699`), `5d6fbb2062` (`#18396`), `5f90e8939d` (`#20387`), `sgl-project/sglang@6e98c84c60` (`#39598`), `ed268dd320` (`#45628`), `ca72dad953` (`#45475`), `vllm-project/vllm@db3b0f0fae` (`#60808`), `35d01660ff` (`#62244`)  
> **确定性复现脚本**：`experiments/trtllm_minimax_m3_router_flux_sglang_quark_vllm_moriio/run.py --check`

---

## 1. 核心架构发现与量化结论

1. **TensorRT-LLM MiniMax-M3 3 项式 BF16 残差拆分路由器 (`#20699`, `522daa8cdf`)**：
   - MiniMax-M3 MoE 路由器权重存储在 FP32 中；若直接在 `allow_tf32=True` 下运行 `F.linear(y.float(), weight)`，cuBLAS TF32 Tensor Core 会将 FP32 尾数截断为 10 bit，改变 Top-K 专家路由边界；
   - `split_router_weight_bf16(weight, terms=3)` 在权重加载阶段将 $W \in \mathbb{R}^{E \times H}$ 迭代拆分为 3 个残差 BF16 分量 $W = W_0 + W_1 + W_2$（每个 BF16 含 8 位有效尾数，3 项共精确覆盖 IEEE-754 FP32 的 **24 位有效尾数**），并在推理时通过单次 BF16 Tensor Core GEMM（FP32 累加）计算 $\text{cat}([y, y, y], -1) \cdot [W_0, W_1, W_2]^T$，使最大绝对误差从 TF32 的 $1.16 \times 10^{-4}$ 降至 **`0.0`（`< 1e-9`）**。
2. **TensorRT-LLM FLUX.1-dev SM103 融合 QK-Norm+RoPE+FP8 量化 (`#18396`, `5d6fbb2062`) 与自定义掩码 FP8 上下文注意力 (`#20387`, `5f90e8939d`)**：
   - `fusedDiTQKNormRopeQuantFp8Kernel` 以每个 warp 处理一个 Q/K/V 头的粒度融合 per-head RMSNorm、复数旋转位置编码（RoPE）与静态 E4M3 量化输出 `(q_out, k_out, v_out)`，并将 `cutedsl` 加入 `nvfp4_gemm_allowed_backends` 以在 SM103 上默认启用 CuTeDSL NVFP4 GEMM；
   - Triton 自定义掩码上下文注意力新增 `kv_scale_quant_orig`（在注意力内循环中执行 `k = (k.to(tl.float32) * kv_scale).to(q.dtype)`）与 `out_scale`（写回前缩放并转为 `float8_e4m3fn`）。
3. **SGLang ROCm Quark AITER ASM FP4 7 维缩放重排与安全门控 (`#39598`, `6e98c84c60`)、DeepSeek-V4.1 视觉偏置路由模拟 (`#45628`, `ed268dd320`) 及 Rust 繁忙调度器健康探针直回 (`#45475`, `ca72dad953`)**：
   - `_swizzle_asm_fp4_scale` 将满足 `rows % 32 == 0, cols % 8 == 0` 的 E8M0 缩放张量按 `(rows // 32, 2, 16, cols // 8, 2, 4, 1)` 重塑并执行双射置换 `.permute(0, 3, 5, 2, 4, 1, 6)`，同时将激活缩放按 `256` 行对齐填充 `0x7F`（即 E8M0 的 $2^{127-127} = 1.0$），并对 MI355X 未调优区间 `N <= 128` 且 `2 <= M <= 64` 强制回退至非 ASM 路径以规避错行缺陷；
   - DeepSeek-V4.1 `_finish_vision_topk` 在视觉偏置专家选择后接入 `apply_routing_simulation`，仅对路由专家槽位 `num_routed_topk = top_k - num_shared_topk_slots` 施加工作负载模拟；
   - 当调度器处于忙碌状态跳过健康探针时，Rust 服务模式通过 `rust_State.push_health_ack()` 直接向响应环写入 `frame_health_ack()` 并清空积压探针，避免 `sgl-front` 超时断连。
4. **vLLM ROCm MoRIIO 步屏障异步读取等待 (`#60808`, `db3b0f0fae`) 与 Intel XPU 在线投机解码接受率估计 (`#62244`, `35d01660ff`)**：
   - 将 MoRIIO KV 块与 Mamba SSM 状态的异步 RDMA READ 完成等待从逐层 `wait_for_layer_load` 前移至模型前向开始前的 `start_load_kv`（`_await_reads_issued_this_step`），并将超时转换从 `int(timeout * 1000)` 改为 `math.ceil(timeout * 1000)`（防止 `< 1 ms` 超时向下取整为 `0` 退化为非阻塞轮询）。

---

## 2. 确定性实验验证汇总

| 验证维度 | 上游提交 | 核心不变量 | 实验验证结果 |
|---|---|---|---|
| MiniMax-M3 3-Term BF16 拆分 | `TensorRT-LLM@522daa8cdf` | 3 项 BF16 还原 24-bit FP32 尾数 | `abs_error_split_bf16 = 0.0` vs `abs_error_tf32 = 1.16e-4` |
| Quark AITER ASM FP4 缩放重排 | `sglang@6e98c84c60` | `32×8` 块 7D 置换双射 + `N <= 128` 门控 | `swizzle_bijection_verified = True`, `0x7F` 填充 |
| Rust 繁忙调度器健康探针应答 | `sglang@ca72dad953` | 跳过探针时写 `frame_health_ack` 并清空队列 | `rust_busy_probe_single_ack = True` |
| vLLM MoRIIO 步屏障读取等待 | `vllm@db3b0f0fae` | 前向开始前挂起读取归零 + `ceil` 毫秒超时 | `new_sub_ms_timeout_ms = 1`, `pending = 0` |
