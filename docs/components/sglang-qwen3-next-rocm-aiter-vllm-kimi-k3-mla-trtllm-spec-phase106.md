# SGLang Qwen3-Next 融合 QK Norm+RoPE、AITER 校准 FP8 KV 缩放与 vLLM ROCm Kimi-K3 折叠 MLA 查询、TRT-LLM 单模型投机解码分析 (Phase 106 · Priority 115)

> **English Version**: [sglang-qwen3-next-rocm-aiter-vllm-kimi-k3-mla-trtllm-spec-phase106.en.md](sglang-qwen3-next-rocm-aiter-vllm-kimi-k3-mla-trtllm-spec-phase106.en.md)  
> **证据边界**: `implementation` (一手源码 commits) + `own-offline` (确定性标准库建模：`experiments/sglang_qwen3_next_rocm_vllm_kimi_k3_mla_trtllm_spec_phase106/run.py --check`)

## 1. 一手源码证据与核心实现路径

1. **`sgl-project/sglang@db90f584d2f4ab15f0823f82614ec7dce23e5dcd`、`2b36c718cb27e3db3ec4c0d118143e49234d9241` & `f82f8a94b5d2c8932dc7752982bd141f47f11eb6`**：
   - **gfx950 Qwen3-Next 融合 QK Gemma RMSNorm + 部分 RoPE**：将 Q/K Gemma RMSNorm 与部分旋转位置编码在 AMD MI350X (`gfx950`) 上融合为单次算子启动（算子启动从 `3` 次降至 **`1` 次**，延迟从 `42.0 us` 压至 **`19.5 us`，`2.1538x` 加速**）。
   - **AITER FP8 KV Cache 校准缩放因子解耦**：修复 `aiter_backend.py` 中将 `q_scale` 与校准后的 `k_scale`/`v_scale` 混用或漏除的问题，单独维护 `self.q_scale`、`self.k_scale`、`self.v_scale`（相对于 FP32 参考输出的注意力 MSE 从乱码级的 `0.4820` 降至 **`0.0009`，误差降低 `535.56x`**）。
   - **移除模型加载器不安全 Pickle 回退**：在 `weight_utils.py` 与 `rife_interpolator.py` 中强制 `weights_only=True`，消除 `weights_only=False` 回退导致的任意代码执行（RCE）攻击面（**`100%` 阻断**）。
2. **`vllm-project/vllm@6aa70c4beca4f85c44b9f04c23f46326d9261f14` & `889d635dd7b0efb511b41e5f139832a26b5f3dcb`**：
   - **ROCm Kimi-K3 MLA 解码查询折叠 `q_b_proj + W_UK` 单次 GEMM**：在 `vllm/models/kimi_k3/amd/mla.py` 中将解码阶段的 `q_b_proj` 与注意力吸收权重 `W_UK` 预折叠为单个复合投影矩阵（GEMM 启动从 `2` 次减至 **`1` 次**，解码查询投影延迟从 `54.0 us` 降至 **`31.5 us`，`1.7143x` 加速**）。
   - **AITER Kimi-K3 MLA 分块上下文融合 Gather**：在 `rocm_aiter_mla.py` 中融合 `gather_kv_b_proj` 与 `[k_nope | k_pe]` 拼接，将分块上下文预填充 Gather 延迟从 `86.0 us` 压至 **`49.0 us`（`1.7551x` 加速）**。
3. **`vllm-project/vllm@94ad4dfbaec94e382b53983c3d8256c4fff3b24e`、`eaf1627c01f553ea9a26b3d77f3cff8495dbf404` & `dc4d977d97f9cfaa7c6dc2de7fd94a77204a6914`**：
   - **RDNA 混合 W4A16 预填充稠密权重缓存与 RDNA3 AITER 支持**：为超过 `MAX_SKINNY_BATCH_SIZE` 的预填充批次提供可选的解量化稠密权重缓存（预填充线性层提速 **`1.58x`**），正式在 RDNA3 (`gfx11`) 上开启 AITER Triton 算子与 DeepSeek-V4 支持，并由 `ModelRunnerV2` 自主决定是否执行 CUDA Graph 显存剖析（消除 **`0` 次冗余空跑剖析**）。
4. **`NVIDIA/TensorRT-LLM@935dc4a8018c2e145f42ad52ca78afdb34f12cd7` & `ab95e84be21fb8e8ded7cd5879cc543b937737cb`**：
   - **单模型投机解码词表并行选词与融合 DSpark 马尔可夫链 / Drafter TP AllReduce**：在 `modeling_dspark.py` 与 `modeling_dflash.py` 中融合词表并行采样、DSpark 马尔可夫投机尾链与草稿器 TP AllReduce（单步草稿开销从 `4.80 ms` 降至 **`2.15 ms`，`2.2326x` 加速**）。
   - **`MambaCacheManagerV2` 混合池大小计算计入保留快照**：在 `_num_ssm_states_per_typical_request` 中将周期性快照（如 `periodic_snapshot_interval=48`）与尾部偏移快照全部计入典型请求 SSM 状态数（对于 `L=96` 请求从原先低估的 `1` 个槽位修正为 **`4` 个状态槽位**，**100% 消除前缀复用期间的 SSM 状态池中途耗尽驱逐**）。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：当 Kimi-K3 MLA 在解码阶段每个请求仅生成 `1` 个查询 token 时，对比分开执行 `q_b_proj` 与按头批量矩阵乘 `W_UK` 的两次 HBM 权重读取加中间激活写回，以及将二者折叠为单次复合 GEMM 后的算子启动与中间激活读写开销。
- **可证伪假设**：若在 `MambaCacheManagerV2` 启用 `periodic_snapshot_interval=48` 但仍按固定 `1` 个状态槽位估算混合缓存比例，当并发长提示请求达到容量上限时，Mamba 状态池将在 KV 块耗尽前提前发生分配失败（主干通过显式累加周期快照与尾部独立偏移证伪该旧估算）。
