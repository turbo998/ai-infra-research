# Megatron-LM FP32 Token 梯度缩放、TransformerEngine MXFP8/BOLT 链接、Mooncake EP CUDA JIT 与 NIXL AMD XGMI 零拷贝（Phase 118 / Priority 127）

> **证据分类**：Upstream `implementation`（10 个上游公开提交，2026-10-05 至 2026-10-27）+ 本仓库 `own-offline` 确定性数值/带宽模型（`experiments/megatron_fp32_token_scale_te_mxfp8_mooncake_ep_jit_dynamo_nixl_phase118/run.py`）。本文不包含未标注的真实 GPU 硬件重测声明。

## 1. 核心架构问题与背景

在大规模分布式训练（Megatron-LM、TransformerEngine、verl FSDPTurbo）与跨节点 KV/专家传输引擎（Mooncake、Dynamo、NIXL）中，近期解决了五项深层工程缺陷：

1. **Megatron-LM 低精度梯度缓冲区的 Token 缩放因子截断误差**：在按全局有效 Token 数对梯度进行归一化（$\text{scale} = 1 / N_{\text{tokens}}$）时，若将 `scaling_factor` 作为 Python `float` 或提前转为低精度标量再乘以 BF16/FP8/NVFP4 梯度缓冲区，当全局 Token 数较大（如百万级 Token）且非 2 的幂时，$\text{scale}$ 的尾数截断会向全部梯度注入系统性偏差。
2. **TransformerEngine MXFP8 ONNX 块量化属性缺失与 AArch64 BOLT 链接失败**：`onnx_quantize_mxfp8_symbolic` 导出 `TRT_MXFP8DynamicQuantize` 时遗漏了块量化轴属性；而在 Grace/ARM64 (`AArch64`) 上开启 BOLT 重定位（`--emit-relocs`）构建 NCCL EP 子模块时，`LINKER:--no-fix-cortex-a53-843419` 传递方式导致 `nvcc`/CMake 链接报错。
3. **Mooncake EP 预编译包膨胀与 Store DSL 探针租约耦合**：Mooncake EP 的分发（Dispatch）、合并（Combine）与弹性缓冲区内核跨 Hidden Size、Top-K、World Size 存在数百种模板特化，全量 AOT 打包会导致 Wheel 体积急剧膨胀；同时 `MasterService::BatchProbeKey` 在 `#4314` 引入租约策略后，测试场景未显式传入免租约默认策略 `{}`（`ProbeLeaseMode::None`）。
4. **Dynamo Operator 对 Volcano 的硬绑定与 NIXL 在 AMD GPU 上回退至主机内存拷贝**：
   - Dynamo Kubernetes Operator 此前将 LeaderWorkerSet (LWS) 的可用性探测与 Volcano `podgroups` RBAC/Scheme 强耦合，导致仅部署 Grove 或原生 LWS 的集群因缺少 Volcano CRD 而无法启用 LWS；
   - UCX 的 `rocm_ipc` 传输层默认将最小零拷贝阈值设为 `ROCM_IPC_MIN_ZCOPY = 128` 字节（针对小消息 MPI 集合通信），而 UCP 的 RMA `put/get` 零拷贝协议直接拒绝任何 `min_zcopy > 0` 的通道！这导致 NIXL 在 AMD GPU 间的 VRAM-to-VRAM 传输完全无法走 `rocm_ipc`（XGMI），被迫退化为经由主机内存（Host Bounce Buffer）分段拷贝。

---

## 2. 上游一手实现剖析（10 个零重叠 Commit）

| # | 仓库与提交 | 日期 | 核心实现路径与系统机制 |
|---|---|---|---|
| 1 | `NVIDIA/Megatron-LM@53015a0d7b` (`#8532`) | 2026-10-27 | `megatron/core/distributed/param_and_grad_buffer.py` & `distributed_data_parallel.py`：将 `scale_gradients(self, scaling_factor: torch.Tensor)` 的缩放因子保持为 FP32 `torch.Tensor` 传入底层缩放内核，并新增 `test_gradient_scaling.py` 纳入确定性内核清单。 |
| 2 | `NVIDIA/TransformerEngine@9d4bd38678` (`#3592`) | 2026-10-06 | `transformer_engine/pytorch/onnx_extensions.py`：修复 `onnx_quantize_mxfp8_symbolic` 中 `TRT_MXFP8DynamicQuantize` 的块量化属性导出，确保 TensorRT 解析 MXFP8 E8M0 块缩放维度一致。 |
| 3 | `NVIDIA/TransformerEngine@f68a64a373` (`#3627`) | 2026-10-05 | `setup.py` & `transformer_engine/common/CMakeLists.txt`：在 `NVTE_TARGET_IS_ARM64` 且启用 BOLT 时，统一向 `nvcc` 与 CMake 传递 `-mno-fix-cortex-a53-843419`，修复 AArch64 NCCL EP 链接失败。 |
| 4 | `volcengine/verl@4d3ed12a76` (`#8400`) | 2026-10-27 | `examples/grpo_trainer/run_qwen3_5_35b_fsdp_turbo.sh`：适配昇腾 NPU 上 FSDPTurbo PR #197 的分发器接口变更，将 Qwen3.5-35B GRPO 训练的 `ep_dispatcher` 从 `eager`/`fused` 切换为 `all2all`，并独立配置 Actor 与 Reference 模型的 `use_fused_ops`。 |
| 5 | `kvcache-ai/Mooncake@dfd1b716d5` (`#4581`) | 2026-10-27 | `docs/source/deployment/mooncake-ep-jit.md` & `mooncake-ep/include/elastic/mooncake_ep_elastic_combine_official.cuh`：为 Mooncake EP CUDA 后端引入运行时 `nvcc` JIT 编译与磁盘 `cubin` 缓存（覆盖 Dispatch、Combine 与 Elastic Buffer 内核），避免在发布包中预置全量模板组合。 |
| 6 | `kvcache-ai/Mooncake@74944a7772` (`#4703`) | 2026-10-19 | `mooncake-transfer-engine/benchmark/tent_backend.cpp` & `tests/mlu_data_test.cpp`：在 `tebench` 中新增寒武纪 MLU 设备显存 (`USE_MLU=ON`) 与 `parseSegTypeMix("dram,vram")` 混合段基准测试支持，显式开启主机-设备拷贝校验。 |
| 7 | `kvcache-ai/Mooncake@19e0ab28f1` (`#4729`) | 2026-10-17 | `mooncake-store/tests/master_service/dsl/scenario.cpp`：修复 `BatchProbeKey` 调用点，显式传入默认 `{}` 租约策略（`ProbeLeaseMode::None`），保持批量 Key 存在性探测的无租约轻量语义。 |
| 8 | `ai-dynamo/dynamo@acef8afe0a` (`#17070`) | 2026-10-27 | `deploy/operator/cmd/main.go` & `deploy/helm/charts/platform/components/operator/files/role.yaml`：从 Dynamo Operator 中移除 `volcanoscheme` 导入及 `scheduling.volcano.sh/podgroups` 的 7 项 RBAC 动词权限，使 LWS 可用性与 Volcano 完全解耦。 |
| 9 | `ai-dynamo/nixl@bc24e80c91` (`#2371`) | 2026-10-12 | `src/plugins/ucx/ucx_utils.cpp`：在 NIXL 创建的 UCX Context 中显式设置 `ROCM_IPC_MIN_ZCOPY=0`（与 `cuda_ipc` 对齐，且尊重用户显式设置的 `UCX_ROCM_IPC_MIN_ZCOPY`），使 UCP `put/get` 零拷贝协议接纳 `rocm_ipc` 通道，打通 AMD GPU 间直接 XGMI RMA 传输。 |
| 10 | `ai-dynamo/nixl@9fbfcbb6f2` (`#2402`) | 2026-10-12 | `src/core/telemetry/telemetry.h` & `docs/telemetry.md`：当启用 `prometheus_mp` 且未设置 `NIXL_TELEMETRY_MULTIPROC_DIR` 时，自动默认使用 `/dev/shm/nixl-telemetry-<euid>-<port>`；若目录冲突或端口已被占用则降级为进程内无输出 Sink，不再中断 NIXL Agent 构造。 |

---

## 3. 关键公式、精度/带宽分析与离线验证结果

### 3.1 Megatron-LM FP32 Token 缩放误差与 NIXL AMD XGMI 零拷贝带宽增益
1. **Token 缩放因子的有效尾数精度**：设全局 Token 数为 $N_{\text{tokens}} = 1,054,720$，真实缩放因子为 $s = 1 / N_{\text{tokens}} \approx 9.481189 \times 10^{-7}$。BF16 仅有 7 位显式尾数（相对步长 $2^{-7} \approx 0.78\%$），将 $s$ 量化为 BF16 会产生约 $0.193\%$ 的相对截断误差（`megatron_token_scale_bf16_rel_err = 0.00193`）；保持 `scaling_factor` 为 FP32 `torch.Tensor`（23 位显式尾数）可将缩放乘积相对误差降至 $2 \times 10^{-8}$，消除 **99.999%** 的缩放因子截断误差。
2. **NIXL UCX `ROCM_IPC_MIN_ZCOPY=0` 链路切换**：UCP RMA 零拷贝协议要求候选传输通道满足：
   $$\text{lane.min\_zcopy} = 0$$
   由于 UCX `rocm_ipc` 默认 `min_zcopy = 128 B`，NIXL 的 VRAM-to-VRAM 传输被拒之门外并退化为 Host DRAM 分段拷贝（受限于 PCIe Gen5 x16 往返有效带宽 $\approx 42\text{ GB/s}$）。在 `nixlUcxContext` 中注入 `ROCM_IPC_MIN_ZCOPY=0` 后，传输直接走 AMD Instinct MI300X/MI355X 的 XGMI 互连（单向有效 RMA 达 $\approx 336\text{ GB/s}$），有效带宽提升 **$8.0\times$**（`nixl_xgmi_vs_host_stage_bw_gain_ratio = 8.0`）。

### 3.2 离线模型验证汇总（`run.py --check`）

- `megatron_token_scale_bf16_rel_err`: `0.00193` vs `megatron_token_scale_fp32_rel_err`: `2e-08`（`megatron_token_scale_error_reduction_ratio = 1.0`）
- `mooncake_ep_aot_variants_avoided`: `256`（按需 CUDA JIT 编译与磁盘 `cubin` 缓存节省约 `320.0 MiB` 预编译包体积）
- `mooncake_batch_probe_leases_acquired`: `0`（`ProbeLeaseMode::None` 实现零租约开销的批量存在性探测）
- `dynamo_lws_volcano_rbac_verbs_required`: `0`（移除 7 个 Volcano `podgroups` RBAC 动词依赖）
- `nixl_rocm_ipc_min_zcopy_bytes`: `0`（激活 AMD GPU 间 XGMI 零拷贝 RMA，相比 Host Staging 提升 `8.0x` 带宽）
- `nixl_telemetry_unset_dir_init_failures`: `0`（默认 `/dev/shm/nixl-telemetry-<euid>-<port>` 并在冲突时无损降级）

---

## 4. 面向云解决方案架构师的练习与可证伪问题

1. **网络与 KV 传输计算练习**：在 AMD MI300X/MI355X 节点内执行分离式推理（Disaggregated Prefill-Decode）时，单次请求需跨 GPU 传输 `128 MiB` 的 KV Cache。对比修复前（因 `ROCM_IPC_MIN_ZCOPY=128` 退化至 Host Staging，`42 GB/s`）与修复后（`ROCM_IPC_MIN_ZCOPY=0` 走 XGMI 零拷贝，`336 GB/s`）的单次 KV 传输延迟。（答案：$128\text{ MiB} = 0.1342\text{ GB}$；修复前耗时 $0.1342 / 42 \approx 3.20\text{ ms}$，修复后耗时 $0.1342 / 336 \approx 0.40\text{ ms}$，单请求 TTFT 传输开销直降 $2.80\text{ ms}$）。
2. **可证伪假设**：在 NIXL 中设置 `ROCM_IPC_MIN_ZCOPY=0` 是否会劣化同一进程内 MPI 小消息集合通信的性能？不会：`#2371` 仅在 NIXL 自身创建的私有 `nixlUcxContext` 上调用 `config.modify("ROCM_IPC_MIN_ZCOPY", "0")`，并不修改全局环境变量，因此同一进程内的 MPI 或其他 UCX Context 仍保持默认的 128 字节阈值。
