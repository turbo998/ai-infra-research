# FlashInfer Cake KDA TF32 24->9 内核折叠、四族 FP8 GEMM 56 行路由表、StepFun MoE `cake-build/1` 222 项清单、MiniMax H3 QKV 8 变体打包与 cuDNN Paged Prefill HND 布局契约

> **Phase 55 (`priority: 64`) · 一手源码与 Commit 级深挖**
> - `flashinfer-ai/flashinfer@9f7a290987b69277a9031b1c6091810d83cfed3b` (`#6876`)
> - `flashinfer-ai/flashinfer@272ed2a6cf2fe66356b31b91635c26220b819fc8` (`#6862`)
> - `flashinfer-ai/flashinfer@e1b052de722e193324559ba2639b26992c7b9f50` (`#6871`)
> - `flashinfer-ai/flashinfer@5ea501832bf7e37a7acde903a32329d580c0e9b4` (`#6869`)
> - `flashinfer-ai/flashinfer@013f02e75ddf8430ad010b965b2d5047c370f576` (`#6846`)

---

## 1. Cake KDA TF32 预填充内核簇折叠 (`#6876`)

在 Kimi Delta Attention (KDA) TF32 预填充内核族中，`#6876` 延续 `#6758`（BF16 半区）的三项收敛机制：
1. **共享设备头文件 (`csrc/kda/tf32/cake_kda_device.cuh`, `1,039` 行)**：将 34 个 TF32 内核源文件中重复内联的 `elect_sync()`、`mbarrier_init()` 等底层 PTX 辅助函数抽取至单一头文件；
2. **24 个克隆内核折叠为 9 个共享内核体**：通过编译宏 `CAKE_KDA_KERNEL_SYMBOL` 与 `CAKE_KDA_WRITE_CHECKPOINTS`（`0` 或 `1`）参数化同一源码，使 `csrc/kda/tf32/` 从 **35 个文件 / 97,009 行** 降至 **21 个文件 / 41,231 行**（净减 **57.50%**），同时从 1 个宿主绑定模板动态渲染全部 34 个 TF32 绑定（删除 `9,497` 行重复绑定代码）；
3. **移除死代码与不可达探针**：清理 `prepare_fwd`、`KDAInputABI`、`_rebind_tensor_signature` 以及 15 个强制探针启动类与 19 个孤立常量，共享运行时从 `6,478` 行缩减至 `5,832` 行。

---

## 2. 四族 Cake FP8 GEMM 56 行路由表与 StepFun SwiGLU Fused-MoE (`#6862`, `#6871`)

- **四族 Cake FP8 算子收敛 (`#6862`)**：覆盖 `batch_deepgemm_fp8`（`38` 个程序，`sm_100a`/`sm_103a` 共 `67` 个模块）、`cake_grouped_fp8_gemm`（`15` 个程序，其中 3 个 `deepk_c2` 变体合并为 1 份源码）、`cake_grouped_fp8_fused_silu_quant`（`5` 个程序）与 `cake_grouped_mxfp8_quantize`（`2` 个程序，FP16 直接由 BF16 源码编译），共计 **60 个程序**；将 `batch_deepgemm_fp8` 的两条长链分支替换为 **56 行表驱动路由**，加载器代码从 `6,381` 行骤降至 `1,548` 行（缩减 **75.74%**），全 PR 净删 `13,777` 行。
- **StepFun SwiGLU Fused-MoE `cake-build/1` 二阶段收敛 (`#6871`)**：针对 `H=4096, I=1536, E=64, top_k=8`（支持 `bf16 / fp8 / mxfp8 / nvfp4`），包含 `111` 个程序（`routing 52 + FC1 26 + FC2 27 + requant 2 + finalize 4`），跨越 `sm_100a` 与 `sm_103a` 共 **222 条 `build.json` 内容寻址清单项**；共享 `47` 份仅编译宏不同的 TVM-FFI 宿主绑定，并将 `cake_jit.py`（`8,064 -> 1,212` 行）与 `cake_launch.py`（`5,340 -> 21` 行）改为从注册表自动派生。

---

## 3. MiniMax H3 融合 QKV 量化打包与 cuDNN Paged Prefill HND 契约 (`#6869`, `#6846`)

- **MiniMax H3 `minimax_h3_qkv_quantize_pack` (`#6869`)**：提供 `{MXFP8, NVFP4} × P ∈ {1, 2, 4, 8}` 共 **8 个变体**（覆盖 `sm_100a` B200 与 `sm_103a` GB300），统一接入树内 `cake-build` 程序加载器与独立 `cake_ffi_checks.h`（89 行），在 B200 与 GB300 的 6 组生産形状（`M=33,472..109,952`）上实现 **1.000x 几何平均无损加速**。
- **cuDNN Paged Prefill HND 布局修复 (`#6846`)**：基准测试传入形状为 `[pages, heads=8, page_size=16, dim=128]`（真实 `HND` 布局）的 K/V 缓存，但错误声明为 `"NHD"`；在 `#5350` 启用 `NHD -> HND` 轴交换后，第 1 轴与第 2 轴被错误转置为 `kv_heads=16, page_size=8`，导致 `qo_heads=40` 时因 `40 % 16 != 0` 触发 GQA 头数整除报错。修正声明为 `"HND"` 后恢复 `40 / 8 = 5` 的合法 GQA 分组。

---

## 4. 确定性复现实验验证

运行 `python experiments/flashinfer-cake-kda-tf32-fp8-gemm-stepfun-moe-minimax-h3-cudnn-hnd/run.py --check` 全部不变量通过（详见 `summary.json`）。
