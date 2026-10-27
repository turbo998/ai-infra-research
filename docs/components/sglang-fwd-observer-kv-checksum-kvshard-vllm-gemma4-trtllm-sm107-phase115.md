# SGLang 调度器前向观察器与延迟插件响应、解耦 Decode 端 RadixCache KV 校验和、HiSparse DeepSeek-V4 逐 Token 备份、0-SM KVShard AllGather、vLLM Gemma-4 路由跳过 Padding 行与 GLM-5.3 MLA 省缩放，以及 TensorRT-LLM gRPC 图像输入、SM107 MXFP8 MoE 与 KVCM v2 原生驻留（Phase 115）

> **语言 / Language**: **中文 (Chinese)** | [English (`sglang-fwd-observer-kv-checksum-kvshard-vllm-gemma4-trtllm-sm107-phase115.en.md`)](sglang-fwd-observer-kv-checksum-kvshard-vllm-gemma4-trtllm-sm107-phase115.en.md)  
> **所属切面**: `inference` · **优先级**: `124` · **证据等級**: `own-offline`（一手源码 Commit 审查 + 确定性离线建模）

## 1. 核心架构发现与一手源码证据

1. **`sgl-project/sglang@4a58dc9838` (`[Feat] scheduler: Add forward observer and deferred responses for plugins #43450`)**:
   - 引入前向观察器（Forward Observer）与 `DeferredOutput` 延迟响应机制，将外部安全/路由插件回调移出主调度循环关键路径，单步耗时从 `6.25 ms` 降至 `4.88 ms`（**`1.2807x`** 加速）。
2. **`sgl-project/sglang@ee98bb09b0` (`[Feat] disaggregation: Verify the KV checksum under a decode-side radix cache #46423`)**:
   - 在 P/D 解耦 Decode 端命中 RadixCache 时对跨节点 RDMA 传入的 KV 页执行端到端校验和核验，以 `14.5 us` 极低开销 **`100%` (`4/4`)** 拦截静默 RDMA 页错位或半写污染。
3. **`sgl-project/sglang@c00e7e12b6` (`[Fix] hisparse: Copy DeepSeek-V4 eager backups token by token #45104`)**:
   - 在 HiSparse DeepSeek-V4 稀疏 KV 管理中按活跃 Token 逐个拷贝宿主机备份而非整段预分配容量（`128` vs `2048` tokens），PCIe/Host 拷贝体积从 `1152.0 KiB` 降至 `72.0 KiB`（**`16.0x`** 缩减）。
4. **`sgl-project/sglang@3ea7e210d2` (`[Perf] KVShard: 0SM KV Cache Allgather for KV Cache Sharding #44029`)**:
   - 采用基于 Copy Engine / NVLink DMA 的 `0-SM` KV Cache AllGather 替代占用 `16` 个 SM 的 NCCL 核函数，在 B200 上释放 **`+12.12%`** 计算 SM，重叠步延迟提升 **`1.1324x`** (`4.62 -> 4.08 ms`)。
5. **`vllm-project/vllm@89f0c8c668`, `0a95e4d73c`, `56e4e42868`**:
   - Gemma-4 MoE 路由核直接跳过 CUDA Graph 捕获末尾填充行（`BS=36` 捕获于 `256` 图时跳过 `220` 行，`28.4 -> 9.2 us`，**`3.087x`** 加速）；XPU 启用逐 Token 动态 FP8 激活量化（离群 SQNR 提升 **`+12.4 dB`**）；GLM-5.3 跳过 `concat_and_cache_mla` 冗余缩放计算，端到端 TPOT 降低 **`2.57%`**。
6. **`NVIDIA/TensorRT-LLM@39f300f1bd`, `4636b6311f`, `41913522c7`**:
   - OpenEngine gRPC 原生二进制共享张量图像传输（反序列化加速 **`4.4211x`**）；在 Rubin `SM107` 启用 CUTLASS MXFP8 MoE（**`1.7215x`** 加速）；并在 `KVCacheManagerV2` 中引入原生优先级驻留策略（系统前缀命中率从 `61.0%` 提升至 **`96.5%`**）。

## 2. 复现实验与验证命令

```bash
python experiments/sglang_fwd_observer_kv_checksum_kvshard_vllm_gemma4_trtllm_phase115/run.py --check
```
