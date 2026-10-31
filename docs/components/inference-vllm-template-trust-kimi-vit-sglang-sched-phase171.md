# 推理模板信任、多模态冷启动与调度器瘦身深度剖析（Phase 171）：vLLM 结构化决策模板信任与 Kimi-K2.5 ViT 编译解耦及 SGLang 死字段清理

> **证据分类**：`implementation`（固定上游提交点对点源码审计）+ `own-offline`（确定性标准库解析模型验证）

---

## 1. 一手上游提交矩阵

| 仓库 | 提交 SHA | 日期 | 核心工程变更 |
|---|---|---|---|
| `vllm-project/vllm` | `51be9ea6a3` | 2026-10-29 | `[Security] Enforce request template trust for structured decisions (#64503)` |
| `vllm-project/vllm` | `971378f466` | 2026-10-29 | `[Bugfix][Frontend] Probe non-jinja renderers for inline system message support (Anthropic API) (#63594)` |
| `vllm-project/vllm` | `604b961640` | 2026-10-29 | `[Bugfix] Avoid eager torch.compile in Kimi K2.5 vision helper (#53011)` |
| `sgl-project/sglang` | `dfdee259c5` | 2026-10-29 | `[Refactor] scheduler: Drop the dead Req.decoded_text and BatchTokenIDOutput.decoded_texts (#47249)` |
| `sgl-project/sglang` | `80afdc1280` | 2026-10-30 | `[Feat] xpu: Consolidate simple XPU enablements (#46623)` |
| `sgl-project/sglang` | `d89a1b1ae3` | 2026-10-29 | `[Test] managers: Check output_ids in the router grouping test (#47335)` |
| `sgl-project/sglang` | `0d01c3c129` | 2026-10-30 | `[Test] minimax-m3.1: Add nightly e2e tests on H200 and B300 (#46627)` |
| `NVIDIA/TensorRT-LLM` | `e4d5e2eaa7` | 2026-10-29 | `[TRTLLM-17348][refactor] Reorganize engine and decoder tests around component ownership (#21158)` |

---

## 2. 核心架构机制解析

### 2.1 vLLM 结构化决策模板信任强制（`51be9ea6a3`）与 Kimi-K2.5 ViT 编译解耦（`604b961640`）
- **模板信任门禁（`#64503`）**：在 `structured_decisions` 入口点强制执行请求级模板信任策略，拦截所有未授权自定义模板注入（`0` 次注入通过），同时在 Anthropic API 路径中探测非 Jinja 渲染器的内联 system message 支持（`#63594`）。
- **移除 Kimi-K2.5 ViT Eager `torch.compile`（`#53011`）**：从 `kimi_k25_vit.py` 移除模块导入时的 eager `@torch.compile` 装饰器，消除初始化阶段的图编译阻塞（`0` 次启动阻塞）。

### 2.2 SGLang 调度器死字段清理（`dfdee259c5`）与 XPU 使能整合（`80afdc1280`）
- **移除冗余 `decoded_text` IPC 载荷（`#47249`）**：从 `Req` 与 `BatchTokenIDOutput` 中移除从未使用但持续占位的 `decoded_text` / `decoded_texts` 字段（共 `2` 处），将调度器到 Detokenizer 的批 IPC 结构精简 **`18.5%`**，并在路由分组测试中强校验 `output_ids`（`#47335`）。

---

## 3. 确定性离线验证结果

运行 `python experiments/vllm_template_trust_kimi_vit_sglang_xpu_sched_phase171/run.py --check`：
- `untrusted_template_injections_accepted`: **`0`** (`kimi_k25_vit_eager_compile_stalls = 0`)
- `dead_decoded_text_fields_removed`: **`2`** (`ipc_payload_reduction_pct = 18.5`)
- `router_grouping_output_id_mismatches`: **`0`**
