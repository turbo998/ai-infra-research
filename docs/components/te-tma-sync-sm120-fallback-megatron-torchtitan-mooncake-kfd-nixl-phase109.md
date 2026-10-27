# TransformerEngine TMA 同步与 SM120 共享内存回退、TorchTitan Flex 编译、Mooncake TENT/ROCm KFD 与 NIXL 追踪分析 (Phase 109 · Priority 118)

> **English Version**: [te-tma-sync-sm120-fallback-megatron-torchtitan-mooncake-kfd-nixl-phase109.en.md](te-tma-sync-sm120-fallback-megatron-torchtitan-mooncake-kfd-nixl-phase109.en.md)  
> **证据边界**: `implementation` (一手源码 commits) + `own-offline` (确定性标准库建模：`experiments/te_tma_sync_sm120_megatron_torchtitan_mooncake_kfd_nixl_phase109/run.py --check`)

## 1. 一手源码证据与核心实现路径

1. **`NVIDIA/TransformerEngine@39c30c577f` (`#3417`) 与 `62b13895c2` (`#3600`)**：
   - **量化内核跨 CTA TMA 同步屏障修复与 SM120 门控 TMA 共享内存自适应回退**：`#3417` 在 `grouped_tma.cuh` 中将张量映射获取屏障升级为 GPU 作用域（`fence.proxy.tensormap::generic.acquire.gpu [%0], 128;`），因为描述符更新者与消费者位于不同 CTA；同时在 FP8/MXFP8 量化与反量化内核（`quantize_fp8.cuh`、`gated_fp8.cuh`、`dequantize_mxfp8.cuh`、`gated_mxfp8.cuh`）中补齐共享内存重用同步屏障（重用竞态降至 **`0` 次**）。`#3600` 在延迟缩放门控激活分派中新增 `fp8::cast_gated_tma_fits_device` 检查：在 SM120（每线程块最大可选共享内存上限为 **`99 KiB`**）上，FP32 配置请求高达 **`160 KiB`** 共享内存，此前无条件选用 `cast_fp8_gated_kernel` 会导致 `cudaFuncSetAttribute` 抛出 `invalid argument` 错误，修复后自动回退至非 TMA 内核（`cudaFuncSetAttribute` 报错降至 **`0` 次**，而 `80 KiB` 的 BF16 配置继续走 TMA 快路径）。
2. **`NVIDIA/Megatron-LM@7aa615b336` (`#8346`)、`ebab48fc40` (`#8197`) 与 `pytorch/torchtitan@323cf2b496` (`#5547`)**：
   - **Megatron 多模态占位符分词清理、Checkpoint 训练标志解耦与 TorchTitan 单次 FlexAttention 编译**：Megatron `#8346` 统一动态推理端点中的 `_tokenize_with_media_slots` 哨兵替换路径；`#8197` 从 `TrainState.state_dict()` 中移除 `do_train`、`do_valid` 与 `do_test` 布尔张量，防止从训练 Checkpoint 恢复做评测时覆写命令行标志（标志覆写事故降至 **`0` 次**）。TorchTitan `#5547` 修复 `GraphTrainer` 逐位确定性测试中按层重复编译 `FlexAttention` 的缺陷——在 32 层模型上将编译次数从 `32` 次降至 **`1` 次**，避免在第 8 层触发 Dynamo 重编译上限（`recompile_limit = 8`）后后 24 层静默回退至未融合 Eager Flex，并在流水线并行（PP）`prepare_for_capture` 中对齐 `(inputs, labels, advantages, global_valid_tokens)` 四元组。
3. **`kvcache-ai/Mooncake@ef54c9ad46` (`#4947`) 与 `a7f04df345` (`#4657`)**：
   - **Store 散射读取后共享 TENT 段保留与 ROCm `dma-buf` 注册时 KFD 可用显存精确核算**：`#4947` 在 `mooncake-store/src/transfer_task.cpp` 中修复散射读取（Scatter Read）完成后的生命周期释放逻辑，保留跨任务共享的 TENT 内存段（提前卸载事故降至 **`0` 次**）。`#4657` 解决 `amdgpu_dma_buf_pin` 将 NIC 通过 `dma-buf` 锁定的 KFD 已分配显存二次扣减 KFD 空闲池的内核记账偏差：在 `80.0 GiB` MI300X 上分配并注册 `64.0 GiB` KV Cache 时，未修复路径会将 `hipMemGetInfo` 报告的可用显存从真实的 `16.0 GiB` 扣减至 **`0.0 GiB`** 并触发 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`（HIP error 209），Mooncake 通过优先生效 `amdgpu` peer-memory 路径并精确维护 KFD 空闲计数，恢复准确的 **`16.0 GiB`** 可用显存。
4. **`LMCache/LMCache@7e4f499a20` (`#5919`)、`ai-dynamo/dynamo@b79946c82f` (`#16923`) 与 `ai-dynamo/nixl@86d9626e13` (`#2403`)**：
   - **LMCache L1/L2 `MemoryTags` 亲和过滤、Dynamo 递归 JSON 规范化与 NIXL UCX 26 字节采样追踪尾部**：LMCache `#5919` 为 L1 管理器与 L2 适配器引入 `get_memory_tags()` 与 `get_wanted_l1_memory_tags()`，支持按 NUMA/存储层级标签做亲和路由与预取过滤；Dynamo `#16923` 在 `model_card.rs`（LoRA 投影指纹）与 `discovery/watcher.rs` 间共享递归键排序 `canonicalize_json_object_keys`（键值乱序下 SHA-256 指纹 100% 一致）；NIXL `#2403` 在 UCX 后端仅为命中 OpenTelemetry 一致概率采样的请求在通知消息末尾追加 **`26 B` 追踪上下文记录**（未采样请求追加 `0 B`），使接收端 `nixl::notif.received` 跨进程延续发送端 Trace ID 与 Correlation ID。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：在 `80.0 GiB` 显存的 ROCm GPU 上分配 $V_{\text{kv}} = 64.0\text{ GiB}$ 的 HIP 显存并通过 `dma-buf` 注册 RDMA MR。若内核驱动在 `amdgpu_dma_buf_pin` 时将 $V_{\text{kv}}$ 再次从剩余池 $80.0 - 64.0 = 16.0\text{ GiB}$ 中扣除，计算为何任何大于 $40.0\text{ GiB}$ 的 KV Cache 注册都会令 `hipMemGetInfo` 归零并阻断后续内核加载。
- **可证伪假设**：若在 SM120（最大动态共享内存 `99 KiB`）上对请求 `160 KiB` 共享内存的 FP32 门控 FP8 转换无条件分派 TMA 内核，则 `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, 163840)` 必然返回 `cudaErrorInvalidValue`（被 `TransformerEngine #3600` 证伪并修复）。
