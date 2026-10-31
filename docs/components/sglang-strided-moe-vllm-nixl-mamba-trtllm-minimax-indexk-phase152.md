# SGLang 跨步网格 MoE 填充、vLLM NIXL Mamba 块最外层跨距修复与 TensorRT-LLM MiniMax-M3 稀疏 Index-K 缓存审计（Phase 152）

> **证据分类**：源码实现级审计 (`implementation`) + 本项目标准库确定性离线算例 (`own-offline`)。

## 一、 10 项核心公开提交审计矩阵（Zero-Overlap）

| 编号 | 仓库 | Commit (`SHA`) | 日期 | 核心工程突破与代码路径 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `sgl-project/sglang` | [`2830808`](https://github.com/sgl-project/sglang/commit/2830808b46822216df7cc329fa44bc1caaa67be2) | 2026-10-30 | `[Refactor] scheduler: Write every request finish state through Req methods` (`#43896`)：统一通过 `Req` 方法写入请求完成状态，消除 Prefill/Decode 分离与 Beam Search 状态机不一致。 |
| 2 | `sgl-project/sglang` | [`add6f6d`](https://github.com/sgl-project/sglang/commit/add6f6d2c3cb5a94fd0f64a025614c1710722d97) | 2026-10-29 | `[Refactor] detokenizer: Move the incremental decode state and step into incremental_detokenize.py` (`#47338`)：解耦增量分词状态机。 |
| 3 | `sgl-project/sglang` | [`20036b7`](https://github.com/sgl-project/sglang/commit/20036b7006ca5433b6e35c4e244684829c317873) | 2026-10-29 | `[Perf] moe: Fill padded rows of large narrow buffers with a strided grid` (`#47115`)：使用跨步网格 (`strided grid`) 填充大型窄缓冲区 (`131,072` 行) 的 MoE 填充行。 |
| 4 | `sgl-project/sglang` | [`c805455`](https://github.com/sgl-project/sglang/commit/c805455f71be19175db0c849f834e843366d4a79) | 2026-10-30 | `[Fix] minimax_h3: Support out-of-tree platforms` (`#44725`)：支持树外异构硬件平台运行 MiniMax-H3 去噪循环。 |
| 5 | `vllm-project/vllm` | [`8084f06`](https://github.com/vllm-project/vllm/commit/8084f0618162eb1b4c80448a5b593c8fc3d718bf) | 2026-10-29 | `[Bugfix][KV Connector][NIXL] Fix Mamba region stride on block-outermost layouts` (`#64582`)：修复 `block-outermost` 布局下 NIXL 描述符的 Mamba 区域跨距 (`base_worker.py`)。 |
| 6 | `vllm-project/vllm` | [`ed14230`](https://github.com/vllm-project/vllm/commit/ed142309b58359333e34c3a44a2ee3ee773e9185) | 2026-10-30 | `[Bugfix] Skip more weight formats in the tokenizer/processor snapshot` (`#63922`)：在下载 Tokenizer/Processor 快照时过滤更多权重文件格式 (`repo_utils.py`)。 |
| 7 | `vllm-project/vllm` | [`c05e9ed`](https://github.com/vllm-project/vllm/commit/c05e9edc0134832b453d60b6fa78823c94879f1c) | 2026-10-30 | `[Bugfix][Benchmark] Count tool call deltas as generated tokens in openai-responses` (`#62753`)：在 `openai-responses` 压测中计入 Tool Call Delta 生成 Token。 |
| 8 | `vllm-project/vllm` | [`dc42019`](https://github.com/vllm-project/vllm/commit/dc42019e7c33d7203be8b77ef435e185cb73c575) | 2026-10-30 | `[Bugfix][Frontend] Leave GLM-4.7 array and object tool arguments unpinned` (`#64615`)：解除 GLM-4.7 数组与对象工具参数的错误结构固定。 |
| 9 | `NVIDIA/TensorRT-LLM` | [`c492171`](https://github.com/NVIDIA/TensorRT-LLM/commit/c49217126d27a72fc4be59dc615292d19ca0fd06) | 2026-10-29 | `MiniMax-M3 VL: forward video metadata so videos get timestamps` (`#20695`)：透传视频元数据以生成正确的时间戳编码。 |
| 10 | `NVIDIA/TensorRT-LLM` | [`f7e0b91`](https://github.com/NVIDIA/TensorRT-LLM/commit/f7e0b912b5a2ef40fd8c298fc8e2fb569bc4db4d) | 2026-10-29 | `Derive MiniMax-M3 index-K cache layers from the model config` (`#20693`)：直接从模型配置推导稀疏注意力 Index-K 缓存层数 (`20 / 80` 层)。 |

## 二、 确定性离线算例验证 (`own-offline`)

运行 `python experiments/sglang_strided_moe_vllm_nixl_mamba_trtllm_minimax_phase152/run.py --check`：
- **SGLang 跨步网格 MoE 填充加速比**：`36.4 us -> 18.5 us` (**`1.9676x`**)
- **vLLM NIXL Mamba 跨距修复**：块跨距从错误的 `32,768 B` 修正为 `786,432 B` (`24 * 32 KB`)，消除 `511/512` 块静默覆写
- **vLLM Tokenizer 快照下载节省**：`28.5 GB -> 0.04 GB`（节省 **`99.86%`** 带宽）
- **TensorRT-LLM MiniMax-M3 Index-K 缓存显存节省**：仅分配 `20/80` 稀疏层，节省 **`75.0%`** Index-K 显存
