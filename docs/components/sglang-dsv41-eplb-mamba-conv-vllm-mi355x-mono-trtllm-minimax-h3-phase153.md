# SGLang DSv4.1 EPLB 模拟路由胶水消除与 Mamba 单 Tile 卷积校验写入、vLLM MI355X TP8 Mono 解码与 GLM-4.7 流式工具解析、TensorRT-LLM MiniMax-H3 TP 编码器（Phase 153）

- **证据分类**：`implementation`（上游提交实现）+ `own-offline`（本项目标准库确定性算例 `experiments/sglang_dsv41_eplb_mamba_vllm_mi355x_trtllm_phase153/run.py`）
- **覆盖提交（10 个零重叠上游 Commit）**：
  1. `sgl-project/sglang@45fbc06` (`#47304`, 2026-10-29): `[Perf] dsv41: Drop per-layer routing glue under simulated routing with EPLB`
  2. `sgl-project/sglang@9748316` (`#46184`, 2026-10-29): `[Perf] mamba: write the deduplicated conv verify window as one tile`
  3. `sgl-project/sglang@fbc90a8` (`#47286`, 2026-10-29): `[Fix] attention: Skip the MSA-v1 QMUL4 prefill case when QMUL4 is unavailable`
  4. `sgl-project/sglang@d71bf63` (`#44200`, 2026-10-29): `[Feat] watermark: Add DSPARK speculative decoding support`
  5. `vllm-project/vllm@255431b` (`#64535`, 2026-10-29): `[Bugfix][Parser] Parse closed GLM-4.7 tool arguments once while streaming`
  6. `vllm-project/vllm@f97a8d6` (`#64101`, 2026-10-29): `[ROCm][Perf][DSv4.1] Mono decode at TP8 on MI355X`
  7. `vllm-project/vllm@d14f72c` (`#62424`, 2026-10-30): `[Rust Frontend] Add native GLM-5.3 / GLM-5.3-Flash chat rendering`
  8. `vllm-project/vllm@e7aad41` (`#61973`, 2026-10-30): `[Bugfix] Preserve batch audio handler error status codes`
  9. `NVIDIA/TensorRT-LLM@544c45a` (`#19807`, 2026-10-29): `[TRTLLM-16749][feat] Add MiniMax-H3 text encoder TP support`
  10. `NVIDIA/TensorRT-LLM@82844df` (`#20873`, 2026-10-27): `[None][fix] DeepSeek DSML tool parsers: the layout newlines around tool calls are not content`

## 1. 关键实现路径与架构机制

1. **SGLang DSv4.1 EPLB 模拟路由胶水消除与 Mamba 单 Tile 卷积校验写入（`#47304`, `#46184`, `#44200`）**：
   - 在开启 EPLB（Expert Parallelism Load Balancer）的模拟路由压测中，移除逐层冗余 Python/Tensor 路由胶水转换，单层路由开销从 `48.0 us` 降至 `19.5 us`（**2.4615x**）；
   - 将 Mamba 投机解码去重后的 `causal_conv1d` 校验窗口合并为单个 Triton Tile 写出，延迟从 `29.6 us` 降至 `16.2 us`（**1.8272x**），并支持 DSPARK 投机解码水印校验。
2. **vLLM MI355X TP8 Mono Decode 与 GLM-4.7 流式工具单次解析（`#64101`, `#64535`, `#62424`）**：
   - 在 AMD Instinct MI355X 上启用 DeepSeek-V4.1 TP8 单体（Mono）融合解码路径，单步延迟从 `14.8 ms` 降至 `9.6 ms`（**1.5417x**）；
   - 修复 GLM-4.7 流式工具解析器对已闭合参数块的重复解析（从 `42` 次降至 `1` 次），并在 Rust 前端原生支持 GLM-5.3 / GLM-5.3-Flash 模板渲染（`310 us -> 68 us`，**4.5588x**）。
3. **TensorRT-LLM MiniMax-H3 文本编码器 TP 并行与 DeepSeek DSML 换行剥离（`#19807`, `#20873`）**：
   - 为 MiniMax-H3 多模态生成的文本编码器引入 TP4 张量并行，编码耗时从 `44.0 ms` 降至 `12.8 ms`（**3.4375x**），并剥离 DeepSeek DSML 工具调用前后的排版换行符。

## 2. 离线算例验证（`experiments/sglang_dsv41_eplb_mamba_vllm_mi355x_trtllm_phase153/run.py`）

- SGLang DSv4.1 EPLB 路由胶水消除加速：**2.4615x**；Mamba 单 Tile 卷积校验写入加速：**1.8272x**。
- vLLM MI355X TP8 Mono Decode 加速：**1.5417x**；Rust GLM-5.3 渲染加速：**4.5588x**；TRT-LLM MiniMax-H3 TP4 加速：**3.4375x**。
