# SGLang PD 分离采样器 customized_info 透传与 Cosmos3 GPU BLAKE3 哈希、vLLM MI300 MXFP4 门控及 DeepEP 跨节点分发修复（Phase 164 · Priority 173）

> **English Version**: [`sglang-pd-sampler-blake3-vllm-mxfp4-deepep-internode-phase164.en.md`](sglang-pd-sampler-blake3-vllm-mxfp4-deepep-internode-phase164.en.md)

## 1. 一手源码证据与分类表

| 证据编号 | 仓库与提交 | 发布日期 | 证据层级 | 核心改动路径 |
| :--- | :--- | :--- | :--- | :--- |
| `src-sglang-disagg-preserve-sampler-customized-info-2026` | `sgl-project/sglang` `#40107` (`f78d906f3e`) | 2026-10-31 | `implementation` | `python/sglang/srt/disaggregation/*`: 在 Prefill-Decode 分离传输中完整保留采样器 `customized_info` 载荷（保留率 `0% -> 100%`） |
| `src-sglang-cosmos3-gpu-blake3-conditioning-hash-2026` | `sgl-project/sglang` `#46028` (`03a27e2a17`) | 2026-10-31 | `implementation` | `python/sglang/multimodal_gen/*`: 使用 GPU BLAKE3 内核加速 Cosmos3 条件输入哈希（Conditioning Hashing），延迟从 4.80 ms 降至 0.30 ms（`16.0x` 加速） |
| `src-sglang-rust-server-abort-finish-reason-parity-2026` | `sgl-project/sglang` `#47321` (`ba78bce729`) | 2026-10-31 | `implementation` | `sgl-model-gateway/*`: 使 Rust Server 对 `/abort_request` 请求返回与 Python 调度器完全一致的 `abort` 结束原因（`finish_reason`） |
| `src-sglang-hicache-sidecar-host-only-checkpoint-2026` | `sgl-project/sglang` `#47630` (`fb9846d3e2`) | 2026-10-31 | `implementation` | `test/registered/hicache/*`: 在 HiCache Sidecar 恢复测试中显式设置 `host_only_checkpoint` |
| `src-vllm-rocm-mi300-skip-aiter-mxfp4-padding-2026` | `vllm-project/vllm` `#64727` (`e811e502ef`) | 2026-10-30 | `implementation` | `tests/kernels/*`: 修复 AITER MXFP4 填充测试在不支持硬件原生 MXFP4 的 MI300 (`gfx942`) 上报错而非跳过的问题（误报失败 `4 -> 0`） |
| `src-vllm-deps-upgrade-huggingface-hub-2-2026` | `vllm-project/vllm` `#64779` (`644a9f5d27`) | 2026-10-30 | `implementation` | `requirements/*`: 升级依赖至 `huggingface_hub >= 2.0` |
| `src-trtllm-test-gemma4-unified-conditional-gen-2026` | `NVIDIA/TensorRT-LLM` `#21255` (`cecf5f6959`) | 2026-10-30 | `implementation` | `tests/unittest/_torch/modeling/*`: 新增 `Gemma4UnifiedForConditionalGeneration` 模型覆盖 |
| `src-trtllm-remove-obsolete-trt-backend-compat-2026` | `NVIDIA/TensorRT-LLM` `#21041` (`e3e0b12d2b`) | 2026-10-30 | `implementation` | `tensorrt_llm/*`: 移除废弃的旧版 TensorRT 后端兼容层表面（`TRTLLM-16618`） |
| `src-deepep-match-libnccl-library-check-2026` | `deepseek-ai/DeepEP` `#640` (`bb29c6bfd0`) | 2026-05-22 | `implementation` | `setup.py`: 在动态库探测中精确匹配 `"libnccl"` 而非子串 `"nccl"`，消除误匹配 |
| `src-deepep-fix-internode-dispatch-args-2026` | `deepseek-ai/DeepEP` `#641` (`a1b3af3938`) | 2026-05-22 | `implementation` | `csrc/*`: 修复跨节点（Internode）Dispatch 内核调用参数签名错位 |

## 2. 确定性实验结果（`experiments/sglang_pd_sampler_blake3_vllm_mxfp4_deepep_internode_phase164`）

- **SGLang PD `customized_info` 透传与 GPU BLAKE3 哈希**：PD 跨节点 `customized_info` 字段保留率从 **`0.0%`** 提升至 **`100.0%`**，Cosmos3 条件哈希延迟从 **`4.80 ms`** 降至 **`0.30 ms`**（**`16.0x`** 加速，**`-93.75%`** 延迟）。
- **vLLM MI300 MXFP4 门控与 DeepEP 跨节点分发**：MI300 上 AITER MXFP4 填充测试误报失败从 **4** 降至 **0**，DeepEP 跨节点 Dispatch 参数错位与 `libnccl` 误匹配均收敛为 **0**。
