# TransformerEngine CP P2P $O(1)$ 显存与 FA4 `D=256` THD、Megatron-LM GTP FP32 归约与 NCCL UBR 卸载及 Mooncake Reshard 去重、NIXL SigV4 分析 (Phase 106 · Priority 115)

> **English Version**: [te-cp-p2p-o1-megatron-gtp-fp32-ubr-mooncake-reshard-nixl-phase106.en.md](te-cp-p2p-o1-megatron-gtp-fp32-ubr-mooncake-reshard-nixl-phase106.en.md)  
> **证据边界**: `implementation` (一手源码 commits) + `own-offline` (确定性标准库建模：`experiments/te_cp_p2p_o1_megatron_gtp_fp32_ubr_mooncake_reshard_phase106/run.py --check`)

## 1. 一手源码证据与核心实现路径

1. **`NVIDIA/TransformerEngine@f5f4e83376a3a5850f3b68df252679dbae1cb4d7` & `1c9f6f8f39d6972d8dc9bd44de7855aa740d82c5`**：
   - **Context Parallelism (CP) P2P 前向临时显存从 $O(C)$ 降至 $O(1)$**：在 P2P 环状上下文并行前向传播中，仅保留当前计算分片与下一跳接收缓冲共 **2 个旋转 KV 缓冲区**，并在环内通过扩展后的 `thd_out_correction_kernel` 将每步部分输出与 LSE 实时在线重缩放合并到单个运行累加器中。在 `CP=8`（每分片 `256.0 MiB` KV + `256.0 MiB` 部分输出）下，峰值临时显存从 `4,096.0 MiB` 压降至 **`768.0 MiB`（节省 `3,328.0 MiB` / 降低 `81.25%` / `5.3333x` 显存压缩）**。
   - **FA4 `b33` SM100 `D=256` THD `seqused_q/k` 前反向门控**：升级 FA4 版本门控至 `4.0.0b33`，在 SM100 上正式打通 `D=256` 变长 THD 元数据（`seqused_q/k`）的前向与反向传播路径。
2. **`NVIDIA/Megatron-LM@54d99a30b649eece0191b766946ac2dea5f35519` & `1ddc6e43421d807c70ba1d1a774cab4bd79bdd4e`**：
   - **广义张量并行（GTP）保留 FP32 Reduce-Scatter 输出直至梯度累加**：当 `gtp_remat` 轴 $\ge 4$ 且权重梯度为 BF16 时，将 wgrad reduce-scatter 转换为相同线速字节数的 All-to-All 加本地 FP32 求和并保持 FP32 直至写入主梯度累加缓冲，在 `N=8` 时将相对 L2 累加误差从 `1.84e-3` 降至 **`1.15e-6`（精度提升 `1,600.0x`）**。
   - **RL 阶段切换时的 NCCL UBR (`nccl_ub`) 梯度缓冲卸载**：在 `DistributedDataParallel.offload_grad_buffers` 中将 `nccl_ub` 梯度存储分配于独立私有 NCCL 内存池，在 RL 训练阶段向推理生成阶段切换时显式取消注册并销毁私有池，单卡释放 **`16,384.0 MiB`（`16.0 GiB`）HBM** 供 Rollout KV Cache 使用。
3. **`pytorch/torchtitan@3c7c1479151a9ea7c9d5be9fd43a687390c5f1db`**：
   - **TorchTitan RL 生成器四层解耦（`Service` / `EngineLoop` / `Dispatcher` / `GenerationEngine`）**：将原 1,860 行耦合的 `VLLMGenerator` 拆分为独立职责层，支持跨 DP 副本请求分发与 SPMD 步进突发负载合并（Rollout 调度吞吐提升 **`1.36x`**）。
4. **`kvcache-ai/Mooncake@3bcf90e65adf2646b3741691eac3b4457bcab103` & `8bfd298880d447137378cf383017f49d8cd4a3aa`**：
   - **Mooncake Reshard 共享底层内存池去重校验**：在 `_validate_runtime_binding_address_ranges` 重叠扫描前，先将共享同一 `(storage_address, storage_byte_size)` 底层分配区的多个专家权重视图折叠去重。对于共享同一内存池的 `256` 个 MoE 专家视图，比较次数从 $O(V^2) = 32,640$ 次降至 **`1` 次（校验耗时从 `145.0 ms` 压至 `0.45 ms`，提速 `322.22x`）**。
   - **TENT `commitPreparedSubmit` 合并暂存请求去重提交**：在 `continue` 分支前写入 `merged_task_id_map[merged_task_id] = task`，确保合并后的 staging 请求仅提交 **`1` 次**（消除 `3 -> 1` 次重复传输提交）。
5. **`LMCache/LMCache@aa12d34e31`、`ai-dynamo/dynamo@1790a42418` & `ai-dynamo/nixl@33d34db66b`**：
   - LMCache 新增 `lmcache bench trace-analyze` 离线重放 OTLP `mp.request.trace` 评估 `(chunk_size, capacity)` 网格的 Prefix 与 CacheBlend 非前缀命中率；Dynamo Mocker 支持 `MAX_CANDIDATE_TOKEN_IDS = 255` 的单遍 Prefill-Only 决策打分；NIXL S3-over-RDMA 控制面使用与 AWS SDK 一致的严格 RFC 3986 编码转义 LMCache 对象键中的 `@` 分隔符，将 `403 SignatureDoesNotMatch` 失败率从 **`100.0%` 降至 `0.0%`**。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：在 `CP = 16`、单分片 KV 与部分输出各为 `256 MiB` 的超长上下文训练中，计算传统 $O(C)$ 方案（保留全部 16 个分片等待末端修正核）与双缓冲 $O(1)$ 在线合并方案（2 个 KV 缓冲 + 1 个运行输出累加器）的峰值临时显存差值（`8,192 MiB` vs `768 MiB`）。
- **可证伪假设**：若在 NIXL S3-over-RDMA 控制面继续调用保留 `@` 不转义的 `GetURLEncodedPathRFC3986()` 对 LMCache 块键进行 SigV4 签名，服务端验证器会因规范 URI 哈希不匹配而以 100% 概率拒绝 `rdmaPut`/`rdmaGet`（主干通过逐字节转义非保留字符并在 `AccelReservedCharKeyXferTest` 中验证 `@` 键传输成功证伪该缺陷）。
