# NVIDIA NIXL UCX `vram_memtype_hint`, Libfabric `cuMemCreate` Fallback & Empty Notification Simulator / 模拟器

## 中文说明

本确定性标准库模拟器（`run.py`）基于 `ai-dynamo/nixl` 提交 `466ebb1ccc5a`、`1f4afe6b60b2` 与 `b55e79152fd6` 验证：
1. **UCX `vram_memtype_hint` 位掩码解析与保守自动提示 (`resolveMemoryTypeConfig`)**：仅当 `ucp_context_query(UCP_ATTR_FIELD_MEMORY_TYPES)` 呈现纯 `ZE_DEVICE`（无 `CUDA` / `ROCM`）时自动提示 `UCS_MEMORY_TYPE_ZE_DEVICE`，其余混合或 CUDA 环境保持 `UCS_MEMORY_TYPE_UNKNOWN`；显式 `cuda-managed` 提示可修复 Host 误判，未提示且被误判为 `UCS_MEMORY_TYPE_HOST` 时立即 `ucp_mem_unmap` 并报错终止。
2. **Libfabric `cuMemCreate` 无上下文虚拟内存的主上下文回退 (`cudaUpdateCtxPtr`)**：当 `cuPointerGetAttribute(..., CU_POINTER_ATTRIBUTE_CONTEXT)` 返回 `ctx == nullptr` 时返回 `-1`，触发 `cudaSetDevice(mem.devId)` 绑定设备 Primary Context，修复 `fi_mr_regattr(FI_HMEM_CUDA)` 注册失败。
3. **Python 空通知 (`b""`) 一次性消费语义 (`check_remote_xfer_done`)**：对比旧版 `if message:`（`bool(b"") == False` 导致空屏障通知永久残留并引发幽灵重消费）与新版 `if message is not None:` 的精确单次移除。

## English Summary

This deterministic Python standard-library simulator (`run.py`) models `ai-dynamo/nixl` commits `466ebb1ccc5a`, `1f4afe6b60b2`, and `b55e79152fd6`:
1. **UCX `vram_memtype_hint` Bitmask Resolution (`resolveMemoryTypeConfig`)**: Verifies conservative `AUTO` hinting (`ze-device` only on ZE-only stacks, `unknown` on CUDA/mixed stacks), explicit `cuda-managed` recovery, and mandatory `ucp_mem_unmap` abort on `UCS_MEMORY_TYPE_HOST` misdetection.
2. **Libfabric Context-Independent (`cuMemCreate`) Primary-Context Fallback (`cudaUpdateCtxPtr`)**: Models `ctx == nullptr` returning `-1` so `nixlLibfabricEngine::regMem` invokes `cudaSetDevice(mem.devId)` before `fi_mr_regattr`.
3. **Python Empty Notification (`b""`) Single-Consumption Guarantee (`check_remote_xfer_done`)**: Proves how `if message is not None:` eliminates the infinite re-consumption bug triggered by `b""` under `if message:`.

```bash
python experiments/nixl_ucx_vram_vmm_notif_sim/run.py --check
```
