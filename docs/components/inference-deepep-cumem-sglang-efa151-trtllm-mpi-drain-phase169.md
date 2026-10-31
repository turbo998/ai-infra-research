# 推理与专家并行进程间通信 Phase 169：DeepEP `cuMem` 跨进程共享内存管理、TRT-LLM 子进程派生前排空缓存 MPI 池、SGLang EFA 1.51.0 升级及 vLLM ROCm CI 镜像收敛与 AITER MXFP4 测试激活

## 1. 背景与核心问题

在多进程专家并行（EP）、MPI 分布式推理编排与异构 ROCm/EFA 容器交付中，推理系统面临三项工程瓶颈：
1. **DeepEP 传统 `cudaIpcGetMemHandle` 的静态映射限制**：在节点内跨进程共享通信缓冲区时，传统 CUDA Runtime IPC 句柄不支持虚拟地址按需扩缩与细粒度物理页重映射，限制了动态 EP 缓冲池的弹性管理。
2. **TensorRT-LLM 缓存 MPI 进程池在派生子进程时的死锁（`nvbugs/6700875`）**：当测试或服务主进程持有已缓存的 MPI Pool 时，若不预先排空（Drain）直接启动新子进程，子进程会继承未清理的 MPI/UCX 文件描述符与上下文状态并引发死锁。
3. **SGLang EFA 驱动栈版本滞后与 vLLM ROCm 多层 CI 镜像膨胀及 AITER MXFP4 Padding 测试跳过**：AWS EFA PD 分离容器需升级至 `EFA installer 1.51.0`；vLLM ROCm 原先维护 4 层冗余 Docker 镜像层级拖慢构建流水线，且 AITER MXFP4 矩阵/Token Padding 关键测试长期处于跳过状态。

---

## 2. 上游源码机制深挖（11 条零重叠提交）

### 2.1 DeepEP：基于 CUDA Driver `cuMem` API 的跨进程共享内存与旧版兼容
- **`9249c25796` (`2025-11-04`, `#217`)**：在 `csrc/deep_ep.{cpp,hpp}` 与 `deep_ep/buffer.py` 中引入 CUDA Driver Virtual Memory Management（`cuMemCreate`、`cuMemExportToShareableHandle`、`cuMemMap`）管理跨进程共享内存，以 `2 MiB` 粒度支持零停顿重映射。
- **`bfded34800` (`2025-10-29`, `#469`)**：补齐 DeepEP 对旧版本 Python 与 PyTorch 的向后兼容支持。

### 2.2 TensorRT-LLM：派生子进程前排空缓存 MPI 池与 CBTS 覆盖率阶段保留
- **`9d3b492125` (`2026-10-29`, `#20331`)**：在 `tests/test_common/session_reuse{_hooks}.py` 中强制在启动新子进程前排空所有已缓存的 MPI 进程池（修复 `nvbugs/6700875`）。
- **`c69d879b4d` (`2026-10-30`, `#21093`)**：在 CBTS 覆盖率层级中保留自动触发的流水线阶段。

### 2.3 SGLang & vLLM：EFA 1.51.0、DSpark MLA 权重固定种子与 ROCm CI/AITER MXFP4
- **`e337bb10b5` (`2026-10-30`, `#46892`) & `e4bb0478c5` (`2026-10-30`, `#47540`) & `02cd9ff0a2` (`#47395`) & `a8559ca775` (`#47424`)**：SGLang 将 `docker/efa.Dockerfile` 升级至 `EFA installer 1.51.0`；固定 DSpark MLA 草稿权重随机种子；修复 AMD Dockerfile-only PR 的 AITER 重建过滤并为 `chunked-abort` 测试调度器注入 `output_streamer`。
- **`bba4d9afec` (`2026-10-29`, `#61939`) & `e3f03f4f89` (`2026-10-29`, `#62996`) & `9957f74617` (`#64362`)**：vLLM 将 ROCm CI/Release 镜像层级从 `4` 层折叠收敛为单层统一流水线（`4.0x` 镜像层级缩减），重新激活 2 组 AITER MXFP4 Padding 测试，并为 OpenVLA 补齐 Chat Template 回退。

---

## 3. 确定性离线建模与验证结果

通过 `experiments/deepep_cumem_shared_sglang_efa151_trtllm_mpi_drain_phase169/run.py` 验证：

| 子系统与机制 | 修复/优化前基线 | Phase 169 表现 | 核心收益 |
|---|---|---|---|
| DeepEP 跨进程共享内存 (`8` ranks/node, `2 MiB` 粒度) | 传统 `cudaIpc` 重映射停顿 `8` 次 | `cuMem` 共享句柄重映射停顿 **`0` 次** | 支持虚拟地址弹性拼接与零拷贝跨进程共享 |
| TRT-LLM 子进程派生 (`4` 个缓存 MPI 池) | `4` 次 MPI 状态继承死锁 | 派生前排空后 **`0` 次死锁** | 根除多进程与 MPI 混用死锁 (`nvbugs/6700875`) |
| SGLang EFA & vLLM ROCm CI / AITER MXFP4 | EFA 旧版；ROCm `4` 层镜像 | **EFA `1.51.0`**；镜像层级压缩 **`4.0x`** | 恢复 `2` 组 AITER MXFP4 Padding 测试门禁 |
