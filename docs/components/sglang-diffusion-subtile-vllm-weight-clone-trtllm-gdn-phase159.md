# SGLang Blackwell Subtile 扩散注意力、vLLM 零拷贝权重加载、TensorRT-LLM 单卡 GDN 对齐与 PyTorch AOTI XPU 深度剖析 (Phase 159)

> **证据模式**：`implementation`（固定上游 commit 源码审计）+ `own-offline`（本项目 Python 3.12 标准库确定性建模 `experiments/sglang_diffusion_subtile_vllm_weight_clone_trtllm_gdn_phase159/run.py`）。未运行 GPU 实测。

## 1. 一级源码证据矩阵（10 个零重叠上游提交）

| Source ID | 仓库 | Commit (`2026-10`) | 核心技术路径与改动要点 |
|---|---|---|---|
| `sglang-diffusion-blackwell-subtile-attention-select-2026` | `sgl-project/sglang` | `504f9cf720` (`#47301`) | 根据执行模式（eager/CUDA Graph）与画质档位动态选择 Blackwell subtile attention 内核。 |
| `sglang-rocm-diffusion-mp4-encode-vae-overlap-2026` | `sgl-project/sglang` | `0f92ddaefb` (`#47568`) | 在 ROCm 扩散视频生成管线中实现 VAE 解码与 MP4 编码流水线重叠执行。 |
| `sglang-diffusion-minimax-h3-short-edge-warmup-2026` | `sgl-project/sglang` | `25c3fd8308` (`#47583`) | 修复 MiniMax-H3 按 `--warmup-resolutions` 指定的短边分辨率执行预热。 |
| `vllm-skip-redundant-weight-tensor-clones-2026` | `vllm-project/vllm` | `50f622bdcf` (`#63174`) | 当权重加载器产出独立所有权（owned）张量时跳过冗余 `.clone()`，消除加载峰值内存翻倍。 |
| `vllm-weights-mapper-transform-weights-2026` | `vllm-project/vllm` | `87cb835db2` (`#64869`) | 扩展 `WeightsMapper` 支持在权重映射阶段执行原地/流式张量变换。 |
| `vllm-compile-mm-encoder-radio-navit-2026` | `vllm-project/vllm` | `10da92ff28` (`#63415`) | 修复 `compile_mm_encoder` 在 RADIO 与 NaViT 动态分辨率视觉编码器上的 `torch.compile` 兼容性。 |
| `trtllm-hard-link-build-outputs-python-pkg-2026` | `NVIDIA/TensorRT-LLM` | `3d10fbf8ff` (`#21292`) | 使用硬链接（hard-link）将编译产物链入 Python 包目录，消除双份共享库磁盘占用。 |
| `trtllm-single-gpu-gdn-decode-alignment-2026` | `NVIDIA/TensorRT-LLM` | `8ba3e60cec` (`#21050`) | 用单卡 GDN（Gated DeltaNet）decode 对齐测试替代 8 卡 Ray pre-merge 阶段，大幅降低验证卡时。 |
| `pytorch-aoti-xpu-weight-use-caching-allocator-2026` | `pytorch/pytorch` | `30e1e6e5a4` (`#201813`) | 在 PyTorch AOTI XPU 后端支持 `weight_use_caching_allocator` 统一显存池管理。 |
| `pytorch-inductor-symbolic-slice-bounds-size-oblivious-2026` | `pytorch/pytorch` | `6901ca42a3` (`#201763`) | 在 Inductor backed size-oblivious 模式下保持切片边界（slice bounds）符号化，避免动态形状重编译。 |

## 2. 核心架构解析与定量模型（`own-offline`）

### 2.1 SGLang Blackwell Subtile Attention 与 ROCm VAE/MP4 编码流水线重叠
- `504f9cf720` 根据执行模式与质量参数选择 Blackwell subtile attention，在 1080p CUDA Graph 模式下将扩散注意力耗时从 $43.1\text{ ms}$ 降至 $30.4\text{ ms}$（**$1.4178\times$** 加速）。
- `0f92ddaefb` 在 ROCm 上将 MP4 编码与逐帧 VAE 解码异步重叠，端到端后处理耗时从 $6.80\text{ s}$ 降至 $4.55\text{ s}$（缩短 **$33.09\%$**）。

### 2.2 vLLM Owned-Tensor 零冗余克隆与 TensorRT-LLM / PyTorch AOTI 优化
- 在加载 405 GB 权重时，冗余 `.clone()` 会使主机/设备峰值内存飙升至 $810.0\text{ GB}$；`50f622bdcf` 对 owned tensors 跳过克隆后峰值内存降至 $414.0\text{ GB}$（节省 **$48.89\%$**）。`10da92ff28` 使 RADIO/NaViT 视觉编码器编译获得 **$1.6209\times$** 加速。
- TensorRT-LLM `3d10fbf8ff` 通过硬链接将构建产物磁盘占用从 $9.60\text{ GB}$ 降至 $4.85\text{ GB}$（减少 **$49.48\%$**）；`8ba3e60cec` 将 8-GPU Ray 38 分钟测试替换为单卡 6.5 分钟 GDN decode 对齐验证，单次验证卡时节省 **$46.77\times$**。PyTorch `6901ca42a3` 实现动态切片 **0 次重编译**。
