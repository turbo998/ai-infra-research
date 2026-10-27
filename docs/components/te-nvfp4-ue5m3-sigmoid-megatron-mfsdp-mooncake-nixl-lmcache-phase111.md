# Phase 111 Track C：TransformerEngine NVFP4-UE5M3 块缩放与偏置路由极小 Sigmoid 分数保留、Megatron-LM MFSDP v2 检查点与多轴 QuantizedDBuffer、verl 粘性 DP 会话绑定，以及 Mooncake/NIXL/LMCache 零拷贝与 2D 传输

> **公开来源与证据边界**：本文基于 TransformerEngine、Megatron-LM、verl、Mooncake、NIXL 与 LMCache 公开 GitHub 仓库一手提交及本项目合成离线验证（`experiments/te_nvfp4_ue5m3_megatron_mfsdp_mooncake_nixl_lmcache_phase111/run.py`）。

---

## 1. TransformerEngine：NVFP4 + FP8 `UE5M3` 块缩放、偏置路由极小 Sigmoid 保真与 `no_grad` UB AG 重叠

### 1.1 NVFP4 + FP8 `UE5M3` 块缩放原型（`NVIDIA/TransformerEngine@d0b4b32c8e90`, `#3325`）
- **实现路径**：在 CUDA $\ge 13.4$（`CUDA_VERSION >= 13040`）下引入 `DType::kFloat8UE5M3`（`__nv_fp8_ue5m3`，无符号 5 位指数 + 3 位尾数）作为 NVFP4（`float4e2m1`）的块缩放因子类型，接入 cuDNN 的 `e5m3` 融合 GEMM 与 `wgrad` API（要求张量尺寸按 `256` 对齐，非 `256` 对齐或 Grouped GEMM 时回退至通用 `wgrad` / 稠密 GEMM 路径）。

### 1.2 偏置融合路由保留极小 Sigmoid 分数（`NVIDIA/TransformerEngine@5e9946406256`, `#3637`）
- **数值丢失根因**：在 `fused_topk_with_score_function.cu` 中，当使用 `score_function = "sigmoid"`（`ScoreFunc == 0`）并启用负载均衡 `expert_bias` 时，选专家阶段计算 $s'_i = \sigma(z_i) + b_i$，此前在 top-k 选出后通过 $s'_i - b_i$ 还原原始分数。当 $\sigma(z_i)$ 极小（例如 $z = -18 \Rightarrow \sigma(z) \approx 1.523\times 10^{-8}$）而 $b_i = 0.75$ 时，FP32 加法 $\sigma(z_i) + b_i$ 直接将尾数抹零，再减去 $b_i$ 得到精确的 `0.0`，导致梯度与路由权重失真。
- **修复机制**：当 `ScoreFunc == 0` 时，直接从已保存的无偏中间缓冲区 `intermediate_output[pos_offset + topk_indices[i]]` 读取原始 $\sigma(z_i)$；而 `sqrtsoftplus`（`ScoreFunc == 2`，中间值保存的是 logit）继续执行减偏置还原。

### 1.3 `LayerNormLinear` 在 `torch.no_grad()` 下启用 Userbuffers AG 重叠（`6eb41bd20f16`, `#3546`）
- 将 `layernorm_linear.py` 中的 `ub_overlap_ag_fprop and is_grad_enabled and not return_layernorm_output` 简化为 `ub_overlap_ag_fprop and not return_layernorm_output`，使推理/评测及无需梯度的前向传播同样享受 Userbuffers All-Gather + GEMM 重叠。

---

## 2. Megatron-LM MFSDP v2 检查点与多轴 `QuantizedDBuffer`，以及 verl 多轮粘性 DP 绑定

### 2.1 MFSDP v2 专家并行 MoE 检查点恢复（`NVIDIA/Megatron-LM@db3a6cd8a03e`, `#8230`）
- **实现路径**：
  1. `FullyShardedDataParallel.load_state_dict` 在调用 `super().load_state_dict` 将 DCP 检查点写入 FP32 主权重（main weights）DTensor 后，立即调用 `sync_model_weights_from_main_weights(self.module.parameters())` 刷新独立存储的 BF16/量化前向计算缓冲区；
  2. 在 `FullyShardedOptimizer.__init__` 中，针对旧版 TE（`< 2.18.0`）过滤掉空本地分片（empty local shards，在 EP MoE 某些 rank 上常见）的问题，回退扫描 `model_chunk.parameters()` 以保持全 rank 一致的 `_param_to_fqn` 映射。

### 2.2 `QuantizedDBuffer.allgather` 多轴支持（`NVIDIA/Megatron-LM@e6913b8f3225`, `#7727`）
- 允许 `QuantizedDBuffer.allgather(mesh_axis)` 接收 `int | Iterable[int]`，并在遍历各量化平面（数据平面与缩放平面）前将可迭代对象物化为 `tuple(mesh_axis)`，避免传入一次性生成器（generator）时在第二个平面被耗尽为空。

### 2.3 verl 多轮 Agent Rollout 粘性 vLLM DP Rank 绑定（`volcengine/verl@9058412fb308`, `#8179`）
- **实现路径**：`LLMServerClient.generate` 透传 `session_id = request_id`；`vLLMHttpServer` 使用跨进程稳定的 CRC32 计算：
  $$\text{data\_parallel\_rank} = \operatorname{crc32}(\text{session\_id}) \bmod \text{data\_parallel\_size}$$
  并在 PD 分离（`_pd_dispatch`）下同时向 prefill 与 decode 传递 `session_id`。作者在 GB200 上实测 Nemotron-H 30B-A3B（SWE-bench Pro 多轮回放，单节点 DP4/EP4）：前缀缓存命中率从 **0.81 提升至 0.86**，长会话完成时间加快 **10–15%**，完整 RL 训练累计前缀命中率达 **0.93**。

---

## 3. Mooncake、NIXL 与 LMCache：零暂存 GPU 直读、条带化 `ObjectRoute`、子区间 RDMA Token 与 2D 批量拷贝

### 3.1 Mooncake Store 远端 GPU 读取免暂存与条带化 `ObjectRoute`
- **NVLink vs RDMA 直读判据（`kvcache-ai/Mooncake@9c0528d9b486`, `#3431`）**：`RealClient::can_use_direct_memory_read` 根据所选副本描述符的 `protocol_` 分流：当 `replica_protocol == "nvlink"` 时直接返回 `true`（NVLink 支持直接读入普通未注册 CUDA 设备分配，免去主机/暂存拷贝）；当 `replica_protocol == "rdma"` 时校验目标缓冲区位于已注册内存区间且 `size <= region->size - region->offset`。
- **条带化 `ObjectRoute`（`kvcache-ai/Mooncake@f8c900715e23`, `#4585`）**：在 `mooncake-store/include/metadata/object_route.h` 中将锁绑定到键槽（`KeySlot`）而非对象实例（移除 `ObjectEntry`），并为每次发布分配单调递增的 `Generation`，严格执行“条带锁（Stripe Lock）仅用于查找/创建/删除槽位、持有条带锁时绝不等待槽位锁、单线程最多持有一个槽位锁”的无死锁不变式。

### 3.2 NIXL `cuObject` 子区间 RDMA Token 签发（`ai-dynamo/nixl@a88b4ddc604a`, `#2369`）
- **问题与修复**：`cuMemObjGetRDMAToken(base, size, offset, op)` 仅接受注册基地址 `base` 加字节偏移 `offset`。此前 `SharedCuObjClient` 直接传入子区间指针 `ptr` 且 `offset=0`，导致在单一大注册缓冲区（如 LMCache MP 竞技场或分页 KV 池）内传输子块时以 `rc=1`（`NIXL_ERR_BACKEND`）失败。修复后在 `registrations_` 映射中查找包含 `[ptr, ptr + size)` 的注册项 `[base, base + reg_size)`，并以 `(base, size, ptr - base, op)` 签发 Token。

### 3.3 LMCache 多进程 Block-Outermost 布局 2D 批量直拷（`LMCache/LMCache@0d441be069cc`, `#5506`）
- **实现路径**：在 `csrc/cuda/mp_mem_kernels.cu` 中，当引擎 KV 缓存采用 vLLM 的块最外层（block-outermost）布局、使得同一内核组各层在同一块号与 KV 平面之间具有恒定字节跨距（constant pitch）时，将原先每层一条的 `cudaMemcpyBatchAsync` 1D 描述符折叠为跨全层的单条 2D 拷贝（`cudaMemcpy3DBatchAsync`，`width = block_bytes, height = num_layers`），在 32 层模型下将拷贝描述符数量压缩 **32x**。

---

## 4. 本项目可复现实验验证（`experiments/te_nvfp4_ue5m3_megatron_mfsdp_mooncake_nixl_lmcache_phase111`）

运行 `python experiments/te_nvfp4_ue5m3_megatron_mfsdp_mooncake_nixl_lmcache_phase111/run.py --check` 验证：
- FP32 下 $\sigma(-18.0) \approx 1.523\times 10^{-8}$ 加上 `0.75` 偏置再减去 `0.75` 后归零（`0.0`），从无偏中间缓冲区读取则 100% 保真；
- `QuantizedDBuffer.allgather` 物化 `tuple(mesh_axis)` 后数据平面与缩放平面均获得 `[0, 1]` 轴；verl CRC32 绑定使同一会话全部 6 轮严格落入同一 DP rank；
- Mooncake NVLink 未注册设备缓冲直读判定为 `True`、RDMA 未注册判定为 `False`；NIXL 在 `4 MiB` 竞技场 `+2 MiB` 处精确导出 `offset = 2,097,152`；LMCache 2D 批量拷贝将 `32 层 x 2 KV x 16 块` 的描述符从 `1024` 降至 `32`（**32x** 缩减）。
