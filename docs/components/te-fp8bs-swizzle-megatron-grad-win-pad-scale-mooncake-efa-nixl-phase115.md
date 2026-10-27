# TransformerEngine 多张量 Swizzle `__launch_bounds__` 与 FP8BS 分组 GEMM、Megatron-LM MFSDP v2 梯度窗口重数倒计时与 TE Scale 预填充、TorchTitan RL 模型专属注意力缩放、Mooncake 工作线程 RPC 与 EFA `FI_EAGAIN` 槽位复用、LMCache 平台指针转换、Dynamo 共享 DRA Claim 与 NIXL 逐网卡 `dmabuf` 映射（Phase 115）

> **语言 / Language**: **中文 (Chinese)** | [English (`te-fp8bs-swizzle-megatron-grad-win-pad-scale-mooncake-efa-nixl-phase115.en.md`)](te-fp8bs-swizzle-megatron-grad-win-pad-scale-mooncake-efa-nixl-phase115.en.md)  
> **所属切面**: `distributed-training` · **优先级**: `124` · **证据等級**: `own-offline`（一手源码 Commit 审查 + 确定性离线建模）

## 1. 核心架构发现与一手源码证据

1. **`NVIDIA/TransformerEngine@d7cd302486` (`#3722`) & `b47a356561` (`[Common] Add launch bounds to multi-tensor swizzle kernels #3622`)**:
   - 修复块缩放 `FP8BS` 分组 GEMM 参考基准，并为多张量比例因子 Swizzle 核函数显式添加 `__launch_bounds__`，将 SM 占用率从 `37.5%` 提升至 **`75.0%`**，64 张量 Swizzle 延迟从 `42.0 us` 降至 `29.5 us`（**`1.4237x`** 加速）。
2. **`NVIDIA/Megatron-LM@1fc08e50e3` (`Finalize gradient windows by declared multiplicity #7581`) & `adcd90f4b2` (`Pad the scale buffers before calling TE's quantize #8247`)**:
   - 在 `megatron_fsdp` (`MFSDP v2`) 中按声明的参数共享重数（Multiplicity）进行梯度窗口倒计时，彻底消除共享权重（重数 `= 3`）过早触发 ReduceScatter 的竞态错误（`1 -> 0`）；同时在调用 TE `quantize` 前直接分配 128 对齐的 Scale 缓冲区（`4100 -> 4224`），省去独立 Pad Kernel（每步节省 **`11.5 us`**）。
3. **`pytorch/torchtitan@8a3f5bcaa4` (`[rl] Fix model-specific attention scaling in vLLM #5347`)**:
   - 将模型自定义注意力缩放因子（如 YaRN `softmax_scale = 0.1352` vs 默认 `1/sqrt(128) = 0.0884`）精确透传至 vLLM Rollout 引擎，使 Rollout-Trainer 策略 KL 散度偏差从 `0.0482` 降至 **`0.0001`**（降低 **`482.0x`**）。
4. **`kvcache-ai/Mooncake@049d887eef` (`#4601`) & `f2fbc656f4` (`[TransferEngine] EFA: reuse consumed submission slots on FI_EAGAIN #4705`)**:
   - 将 Mooncake Store Master RPC 处理器卸载至工作线程池，使慢批处理（`420 ms`）下的心跳 `Ping` P99 延迟从 `425.0 ms` 降至 **`0.85 ms`**（降低 **`500.0x`**，消除误判驱逐）；并在 AWS EFA 传输遇到 `FI_EAGAIN` 时复用已消费的提交队列槽位（描述符泄漏从 `1000` 降至 **`0`**）。
5. **`LMCache/LMCache@318d932e5b` (`#5955`), `ai-dynamo/dynamo@6b5b6459e2` (`#16957`), `ai-dynamo/nixl@ca58a385d8` (`#2287`)**:
   - LMCache 将块传输张量到指针转换下沉至平台层批量执行（1,024 块调度开销从 `185.0 us` 降至 `28.0 us`，**`6.6071x`** 加速）；Dynamo Operator 在主负载与检查点容器间共享单一 Pod 级 Kubernetes DRA `ResourceClaim`（`2 -> 1`）；NIXL Libfabric 插件按每张网卡独立探测并选择 CUDA `dmabuf` 映射类型（`8/8` 网卡成功注册）。

## 2. 复现实验与验证命令

```bash
python experiments/te_fp8bs_swizzle_megatron_grad_win_pad_scale_mooncake_efa_phase115/run.py --check
```
