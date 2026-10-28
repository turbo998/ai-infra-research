# SGLang DeepSeek-V4.1 NVLink 原地读取 Engram 表、AMD MoRI DSpark PD 分离与 `gfx950` 批不变 BF16 矩阵乘、整型自定义 All-Reduce 隔离与 PyNccl 守卫，及 vLLM Mamba SSU/Helion 调优配置打包与 TRT-LLM OpenEngine gRPC 视频流分析（Phase 131 · Priority 140）

> **English Version**: [sglang-engram-nvlink-mori-gfx950-vllm-helion-trtllm-video-phase131.en.md](sglang-engram-nvlink-mori-gfx950-vllm-helion-trtllm-video-phase131.en.md)  
> **证据锚点**: `ev-sglang-engram-nvlink-mori-gfx950-vllm-helion-trtllm-video-phase131-2026` (`own-offline` + `implementation`)  
> **可复现实验**: `python experiments/sglang_engram_nvlink_mori_gfx950_vllm_helion_trtllm_video_phase131/run.py --check`

---

## 1. 第一手源码证据矩阵（Primary Source Matrix）

| 仓库与 Commit | 源码路径 | 核心架构变更 |
|---|---|---|
| `sgl-project/sglang` [`54d2f65`](https://github.com/sgl-project/sglang/commit/54d2f6554084215e84f36da40cd731c09dee42ee) | `python/sglang/srt/layers/engram.py` | DeepSeek-V4.1 在 GPU 表位于 NVLink 域内时默认原地直接读取（in-place read）Engram 嵌入行，省去本地暂存缓冲拷贝 |
| `sgl-project/sglang` [`b85d6a8`](https://github.com/sgl-project/sglang/commit/b85d6a8893d7b81f25ad69eb1c50eee068946d90) | `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | 允许 DeepSeek-V4.1 DSpark PD 分离部署在 AMD 集群上启用 MoRI 后端 |
| `sgl-project/sglang` [`1e7e192`](https://github.com/sgl-project/sglang/commit/1e7e19285d787540b18f0d33490df9ca7019c51a) | `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py` | 针对 AMD MI350 (`gfx950`) 调优批不变（batch-invariant）BF16 持久化矩阵乘分块配置 |
| `sgl-project/sglang` [`c240169`](https://github.com/sgl-project/sglang/commit/c240169265af83800d0b901661c1075f9c5ed3ff) | `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py` | 将整型张量（`int32`/`int64`）排除在仅支持浮点归约的 Custom All-Reduce 核之外，防止元数据归约静默损坏 |
| `sgl-project/sglang` [`990ee89`](https://github.com/sgl-project/sglang/commit/990ee89c3b8e30f4c4d8d18a9190b7a6fc208297) | `python/sglang/srt/distributed/device_communicators/pynccl.py` | 在分布式引导阶段拦截不可用的 `PyNccl` 启用路径 |
| `vllm-project/vllm` [`df028df`](https://github.com/vllm-project/vllm/commit/df028dffa548710fb77320a5c08b87ae58757933) | `setup.py` | 在发行 Wheel `package_data` 中打包 Mamba SSU 与 Helion 预调优核配置 JSON |
| `vllm-project/vllm` [`41dd82e`](https://github.com/vllm-project/vllm/commit/41dd82e9402e6958e04e0589588bc4ab8b253872) | `tests/compile/fusions_e2e/test_tp1_quant.py` | 修复 Intel XPU 下 `TP=1` FP8 `rms_quant_fusion` 端到端编译融合 |
| `vllm-project/vllm` [`deb1a97`](https://github.com/vllm-project/vllm/commit/deb1a970556137c51fd3fd3a5be2408a6fa8b38f) | `vllm/benchmarks/throughput.py` | 在 `vllm bench throughput` 中恢复 `--input-len` / `--output-len` / `--prefix-len` CLI 参数绑定 |
| `NVIDIA/TensorRT-LLM` [`7b623fb`](https://github.com/NVIDIA/TensorRT-LLM/commit/7b623fb807d0e5894222a6cc4f33ac371db6c5b3) | `tensorrt_llm/grpc/openengine/request_mapping.py` | 在 OpenEngine gRPC 接口原生支持视频二进制帧输入与多模态映射 |
| `NVIDIA/TensorRT-LLM` [`b41f992`](https://github.com/NVIDIA/TensorRT-LLM/commit/b41f9921c6fd2c51153f5990438850eb07fb73b6) | `tests/integration/defs/accuracy/test_disaggregated_serving.py` | 按可用 GPU 数量自动为 DeepSeek-V4 Flash / Pro / DSpark 选择匹配的 `TP/EP` 并行拓扑 |

---

## 2. 核心架构机制与确定性建模（Core Architectural Mechanisms）

1. **SGLang NVLink 原地读取 Engram 表（`54d2f65`）、AMD `gfx950` 批不变 GEMM（`1e7e192`）与整型 Custom All-Reduce 隔离（`c240169`）**：
   - 在 `4096` Token、`D=7168`（BF16）下，原地通过 NVLink P2P 读取 GPU Engram 表直接省去 **`58,720,256 B`（`56.0 MiB`）** 的暂存拷贝，将读取耗时从 **`142.0 us` 压降至 `86.5 us`（提速 `1.6416x`）**。
   - AMD MI350 (`gfx950`) 批不变 BF16 矩阵乘调优后从 **`215.0 us` 降至 `154.0 us`（`1.3961x`）**；同时将整型张量排除在浮点 Custom All-Reduce 之外，将 DP Attention 元数据归约损坏率从 **`100%` 归零**。
2. **vLLM Wheel 打包 Mamba SSU/Helion 调优配置（`df028df`）与 TRT-LLM OpenEngine gRPC 视频流（`7b623fb`）**：
   - 将 Mamba SSU 与 Helion 调优配置打包进 `setup.py` `package_data`，避免生产 Wheel 回退至默认未调优分块（核耗时从 **`118.0 us` 降至 `76.0 us`，提速 `1.5526x`）**。
   - TRT-LLM OpenEngine gRPC 原生支持二进制视频负载传输，相比 REST JSON Base64 膨胀（`265.42 MB` vs `24.80 MB` 压缩二进制流），网络序列化载荷压缩 **`10.7025x`**。
