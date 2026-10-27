# FlashInfer Cake FMHA NVFP4 分页上下文、QSA 工作区复用、Rubin SSD 与 CUTLASS/TileLang 边界分析 (Phase 108 · Priority 117)

> **English Version**: [flashinfer-cake-fmha-qsa-ssd-sm90-megamoe-cutlass-tilelang-phase108.en.md](flashinfer-cake-fmha-qsa-ssd-sm90-megamoe-cutlass-tilelang-phase108.en.md)  
> **证据边界**: `implementation` (一手源码 commits) + `own-offline` (确定性标准库建模：`experiments/flashinfer_cake_fmha_qsa_ssd_sm90_megamoe_phase108/run.py --check`)

## 1. 一手源码证据与核心实现路径

1. **`flashinfer-ai/flashinfer@07cc0d60a9` (`#7512`: `csrc/cake_fmha/cuda/context_*/sm_100a|sm_103a/*.cu`, `csrc/cake_fmha/registry.json`)**：
   - **Blackwell (`sm_100a` / `sm_103a`) 20 个 Cake FMHA 分页上下文（Prefill）注意力程序**：在 `trtllm_batch_context_with_kv_cache(backend="cake")` 路径下新增覆盖 `BF16`、`FP8` 与 `NVFP4` KV Cache 的 20 个 `head_dim = 128` 右下角因果掩码内核。针对 `page_size = 32` 采用 `l2_swizzle = 1, num_m_blocks = 11, tok_per_stage = 25`；针对 `page_size = 1024` 采用 `l2_swizzle = 8, num_m_blocks = 6, tok_per_stage = 25`。在 `seq_len = 8192, num_kv_heads = 8, head_dim = 128` 下，`NVFP4` KV（每元素 `0.5 B` + 每 16 元素 `1 B` FP8 E4M3 缩放因子 = `0.5625 B/elem`）将单层 KV 读取字节从 BF16 的 `33,554,432 B` 降至 **`9,437,184 B`（`3.5556x` 带宽压缩，较 FP8 提升 `1.7778x`）**。
2. **`flashinfer-ai/flashinfer@fa8fb88e0f` (`#7514`: `flashinfer/experimental/cake_qsa_nvfp4/cake_backend.py`)**：
   - **Cake QSA NVFP4 稀疏解码跨调用工作区切片（Workspace Carve）复用**：针对 3 个 `tcgen05`（CC 10.0/10.3/10.7）与 3 个 `mma.sync`（CC 8.0/9.0/12.0）稀疏解码程序，首次传入 256 字节对齐的 `workspace_buffer` 时校验大小并清零 `uint32 [rows * num_kv_heads]` 计数器，后续解码步直接复用已校验切片（`1,000` 步内校验与初始化从 `1,000` 次降至 **`1` 次，降低 `99.9%` 宿主端分派开销**）；且当 `splits == 1` 时跳过 split-KV 局部缓冲区切片（`single_split_allocates_partials = False`）。
3. **`flashinfer-ai/flashinfer@a8b8245be2` (`#7499`) 与 `5b732d7271` (`#7488`)**：
   - **Rubin (`sm_107a`) 56 个 Mamba2 SSD 联合扫描程序与 Blackwell 4-head/CTA 选择性状态更新**：`#7499` 为 `cake_ssd_combined` 交付 `sm_107a` 精确扫描（融合单次启动、带因式分解持久段预处理的双核及尾核）、三阶段块并行与回溯块并行共 56 个内核；`#7488` 在 `sm_100a`/`sm_103a` 上通过精简 acquire TMA 同步、流水化归约与每 CTA 4 头（`__launch_bounds__(288)`）优化 Nemotron-H / Granite-4.0-H 的 Mamba2 解码步。
4. **`flashinfer-ai/flashinfer@a3452cd405` (`#7438`: `flashinfer/cudnn/prefill.py`) 与 `479c38bbfc` (`#6999`: `flashinfer/moe_ep/`)**：
   - **cuDNN Prefill 显式有效长度与物理偏移解耦 + Hopper SM90 NVFP4×FP8 CuTe DSL MegaMoE**：`#7438` 将逻辑有效序列长度 `seq_lens_q` 与物理紧凑偏移 `q_offsets` 解耦——当 `q_offsets = [0, 320, 512]` 且显式传入 `seq_lens_q = [300, 177]` 时，旧实现按相邻偏移推断出 `320 + 192 = 512` 行（泄漏 **`35` 行尾部填充垃圾数据**），修复后严格处理 `477` 个有效行；`#6999` 在 `moe_ep.MoEEpLayer` 中集成 SM90 NVFP4 权重 × FP8 E4M3 激活单次提升（Single-Promotion）CuTe DSL MegaMoE 后端。
5. **`NVIDIA/cutlass@a2b29f847b` (`#3464`)、`1d7cde107f` (`#3278`) 与 `tile-ai/tilelang@876a59e6eb` (`#2818`)、`b69e0c2c0c` (`#3576`)**：
   - CUTLASS `#3464` 在 `tvm_ffi_builder.py` 的 `find_env_stream` 中递归遍历 `spec.TupleParam`；`#3278` 在 `PredicatedTileIterator` 中以 `if (!ScatterD)` 保护 `byte_pointer += params_.increment_group`，消除 `ScatterD` 模式下跨线程组 `6,144 B` 的基址指针漂移。TileLang `#2818` 拦截非整除 MMA K 原子（如 `chunk = 8` 不满足 `K % 16 == 0`），强制显式零填充至 `K = 16` 或回退至带边界保护的 FMA；`#3576` 为 CuTeDSL `CumSum1D` 增加 `thread_offset` 模板参数以支持非零起始线程子集前缀和。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：对于 `seq_len = 8,192`、`num_kv_heads = 8`、`head_dim = 128` 的单层上下文注意力，计算 BF16 KV（`2 B/elem`）与 NVFP4 KV（`0.5 B/elem` 数据 + 每 16 个元素 `1 B` FP8 缩放因子 = `0.5625 B/elem`）的总读取字节数，并验证为何带宽压缩比恒为 $2.0 / 0.5625 = 32 / 9 \approx 3.5556\times$。
- **可证伪假设**：若在 CUDA Graph 重放中为紧凑变长请求预分配固定槽位 `q_offsets = [0, 320, 512]`，而将 `actual_seq_lens_q = [300, 177]` 忽略并由相邻偏移差推导掩码，则每个请求末尾的填充槽位（共 `20 + 15 = 35` 行）将参与注意力计算并污染输出激活（被 `flashinfer #7438` 复现实验证伪并修复）。
