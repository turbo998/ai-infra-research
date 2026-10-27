# Megatron-LM 推理专家张量并行 (ETP)、TransformerEngine CuTeDSL 评估 GEMM、TorchTitan EMA 重分片与 Mooncake/NIXL 状态机分析 (Phase 108 · Priority 117)

> **English Version**: [megatron-etp-te-cutedsl-torchtitan-ema-mooncake-tent-nixl-phase108.en.md](megatron-etp-te-cutedsl-torchtitan-ema-mooncake-tent-nixl-phase108.en.md)  
> **证据边界**: `implementation` (一手源码 commits) + `own-offline` (确定性标准库建模：`experiments/megatron_etp_te_cutedsl_torchtitan_ema_mooncake_tent_nixl_phase108/run.py --check`)

## 1. 一手源码证据与核心实现路径

1. **`NVIDIA/Megatron-LM@2abdd0d1c6` (`#4743`: `megatron/core/inference/contexts/dynamic_context.py`, `dynamic_engine.py`, `token_dispatcher_inference.py`, `vllm_fused_moe.py`)与 `d61d8d44e2` (`#7710`)**：
   - **Megatron-Core 动态推理引擎支持专家张量并行（Expert Tensor Parallelism, `tp_ep`）**：在 `DynamicInferenceContext`、`DynamicInferenceEngine` 与 `token_dispatcher_inference.py` 中引入复合通信组 `tp_ep`（大小等于 `ETP × EP`，当 `ETP == 1` 时退化为 `ep` 组）。对于 `128` 专家模型，在 `EP = 8, ETP = 2`（`tp_ep = 16`）配置下，单卡专家权重显存占用从纯 `EP = 8` 的 `18.0 GiB` 降至 **`9.0 GiB`（`2.0x` 显存削减）**，同时保持 ZMQ/NCCL 协调器与双向 token 聚合/分发语义一致；`#7710` 将 MFSDP v2 适配器测试拆分为独立模块并在 `mcore_utils.py` 中清理 `_clear_sharding_prop_cache`。
2. **`NVIDIA/TransformerEngine@0c4d9610ac` (`#3666`: `transformer_engine/pytorch/ops/fused/grouped_mlp.py`) 与 `e394b124ff` (`#3655`)**：
   - **CuTeDSL 单组 `DenseGEMM` 推理/评估快路径启用**：在 `no_grad` 或 `frozen_weights` 评估模式下（无需计算权重梯度 $dW$），激活张量 `grouped_fc2_x` 仅生成行方向量化数据（`rowwise_data` / `rowwise_scale_inv`）而省略列方向转置量化缓冲区（`columnwise_data is None`）。`#3666` 移除对 `columnwise_data` 的冗余非空要求，在 `mxfp8` 与 `nvfp4_rht` 评估路径上直接激活 CuTeDSL `DenseGEMM`，将评估阶段激活量化显存开销降低 **`50.0%`**（`4096 × 4096` 下从 `34,603,008 B` 降至 **`17,301,504 B`**）；`#3655` 在分布式测试 `init_process_group` 中显式绑定 `device_id=device` 并执行确定性 `destroy_process_group()`。
3. **`pytorch/torchtitan@dc9ea005e4` (`#5357`: `torchtitan/components/optim/ema.py`)**：
   - **非均匀分片下卸载式 EMA 检查点全局元数据保留**：当形状为 `5 × 4` 的参数张量在 `2` 个 rank 上按行分片时，各 rank 本地分片行数分别为 `3` 和 `2`。旧版 `_materialize_dtensor` 在调用 `DTensor.from_local` 时未传入全局形状，导致两个 rank 分别推断出冲突的全局形状 `[6, 4]` 与 `[4, 4]` 并被 DCP 检查点规划器拒绝；`#5357` 显式传入在线参数的 `shape=p.shape` 与 `stride=p.stride()`（`[5, 4]` 与 `[4, 1]`），确保非整除分片下 EMA 保存与重分片加载 100% 通过。
4. **`kvcache-ai/Mooncake@dd4371ade4` (`#4736`: `mooncake-store/include/master_service.h`) 与 `8a4d4f68ca` (`#4904`: `gds_transport.cpp`)**：
   - **Mooncake Store HA 恢复生命周期状态机与 Intel XPU `cuFile` GDS 传输**：`#4736` 实现 RFC #4700 显式恢复状态机（`Unrestored -> Restoring -> ReadyForServing -> Serving`），强制单飞（Single-Flight）恢复准入、在进入 `Serving` 后拒绝迟到恢复请求，并在恢复期间门控后台工作线程；`#4904` 新增 `USE_XPU_GDS` 构建选项，通过 Intel `xFile`（兼容 `cufile.h` + `libcufile.so`）在 Intel XPU 上启用 GPUDirect Storage，并在 CMake 层与 `USE_CUDA` 互斥以避免单进程双 `libcufile.so` 符号冲突。
5. **`LMCache/LMCache@8510ceb4be` (`#5927`)、`ai-dynamo/dynamo@88fcef9260` (`#17353`) 与 `ai-dynamo/nixl@e053fc4fd4` (`#2522`)**：
   - LMCache `#5927` 重构 `lmcache/inprocess/v1/manager.py` 的 `_safe_close`，在组件关闭超时后调用 `executor.shutdown(wait=False, cancel_futures=True)` 继续后续步骤而非阻塞挂起；Dynamo `#17353` 为 `mooncake_bench` 新增主机内存层（KVCR lower-tier）回放模式并分离调度器滞后、队列等待与查找延迟；NIXL `#2522` 新增 `traceXferError`（`nixl::xfer.error`）及每次 `postXferReq` / `releaseXferReq` 的状态记录，实现 **100% 终态传输可观测性**。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：当一个全局形状为 `5 × 4` 的 FP32 参数在 `world_size = 2` 上按第 0 维进行 `Shard(0)` 切分时，Rank 0 持有 `3 × 4`、Rank 1 持有 `2 × 4`。解释为何 `DTensor.from_local(local_tensor, mesh, [Shard(0)])` 在不显式指定 `shape=(5, 4), stride=(4, 1)` 时会在 Rank 0 和 Rank 1 上分别计算出 `6 × 4` 与 `4 × 4` 的矛盾全局元数据。
- **可证伪假设**：在 TransformerEngine 的 `no_grad` / `frozen_weights` 评估模式下，若仍要求 `grouped_fc2_x.columnwise_data` 非空才允许进入 CuTeDSL 单组 `DenseGEMM`，所有仅生成行方向量化激活的推理/评估调用都将无谓回退至慢速路径或被迫分配双倍量化缓冲（被 `TransformerEngine #3666` 证伪并修复）。
