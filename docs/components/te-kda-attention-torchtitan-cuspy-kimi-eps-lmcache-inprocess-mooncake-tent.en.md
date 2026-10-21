# TransformerEngine KDA Linear Attention, TorchTitan Cuspy Profiler & Kimi `rms_norm_eps` Fix, LMCache/Dynamo Decoupling & Mooncake TENT Diagnostics

> **Primary Upstream Commits**:
> - [`NVIDIA/TransformerEngine@b9715e5`](https://github.com/NVIDIA/TransformerEngine/commit/b9715e55a9161c703da3b8cd1d8117c2257f356e) (`#3548`, 2026-10-14) — *KDA attention (`KimiDeltaAttention` via cuDNN FROST on SM100/SM103/SM107)*
> - [`NVIDIA/TransformerEngine@c3081a5`](https://github.com/NVIDIA/TransformerEngine/commit/c3081a5f2db9de31eab51689a25c70ea48068aa6) (`#3470`, 2026-10-14) — *[All] Guard THD learnable dSink on older cuDNN*
> - [`pytorch/torchtitan@789740c`](https://github.com/pytorch/torchtitan/commit/789740c02578d288ee6e5a5651c6f96162bf9b84) (`#5525`), [`35f4a2e`](https://github.com/pytorch/torchtitan/commit/35f4a2e166586302ae6da957656a539759b20121) (`#5450`), [`d06d37f`](https://github.com/pytorch/torchtitan/commit/d06d37f53cb206ad008fb5869ae36a5c46adba50) (`#5548`, 2026-10-20) — *Cuspy profiler backend, `torch.accelerator` migration, and Kimi K2.x `rms_norm_eps=1e-5` fix*
> - [`LMCache/LMCache@eb2f3df`](https://github.com/LMCache/LMCache/commit/eb2f3df13f6e49e8a92c464f161dc3c456da3b42) (`#5925`), [`e6534c6`](https://github.com/LMCache/LMCache/commit/e6534c635e747b2c66ac2cb5342236d0ce62f312) (`#5939`, 2026-10-21) & [`ai-dynamo/dynamo@45a4e56`](https://github.com/ai-dynamo/dynamo/commit/45a4e56a5f4dd4111d937ac36f38ac669677468d) (`#16324`, 2026-10-21) — *Isolate `lmcache.inprocess` & retire KVBM in favor of `dynamo.vllm.pd_connector`*
> - [`kvcache-ai/Mooncake@7540f58`](https://github.com/kvcache-ai/Mooncake/commit/7540f5826ed32cb755b1db027a2c609adb53f024) (`#4154`, 2026-10-21) & [`8952021`](https://github.com/kvcache-ai/Mooncake/commit/89520212377af7114bbba4177021ba092d78ed11) (`#4611`, 2026-10-20) — *[TENT] Add diagnostic core/CLI & preserve `int64`/`bool` in `typed_ragged` with empty rows*

---

## 1. TransformerEngine `KimiDeltaAttention` & Narrowed cuDNN THD Guard

### 1.1 Non-Commuting Per-Key-Channel Decay in KDA (`#3548`)
`KimiDeltaAttention` generalizes Gated DeltaNet's scalar decay to a per-key-channel vector $\alpha_t = \exp(g_t) \in (0, 1]^{d_{qk}}$ with scalar write strength $\beta_t$:
$$S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top, \quad o_t = \text{scale} \cdot q_t S_t$$
- **Non-Commutativity**: Because $\mathrm{Diag}(\alpha_t)$ does not commute with $(I - \beta_t k_t k_t^\top)$ when $g_t$ varies across channels, decay must be applied before the delta-rule erase term reads the state. Collapsing $g_t$ to a scalar recovers exact commutativity (`0.0` difference).
- **cuDNN FROST Runtime**: Targets SM100/SM103/SM107 for `head_dim in (64, 128)` with transposed state `[batch, heads, v_head_dim, qk_head_dim]`.

### 1.2 Narrowed Pre-cuDNN 9.26.0 THD Learnable-Softmax Guard (`#3470`)
`TransformerEngine@c3081a5` restricts the `cudnn_runtime_version < 92600` backward rejection for THD `NVTE_LEARNABLE_SOFTMAX` to `head_dim_v not in {64, 128, 256}`, preserving specialized kernels for 64, 128, and 256.

---

## 2. TorchTitan Kimi K2.x `rms_norm_eps` Fix, Cuspy Profiler & `torch.accelerator`

- **Kimi K2.x Decoder-Layer `rms_norm_eps=1e-5` (`#5548`)**: Passing `norm_eps=1e-5` to `build_mla_moe_layers` for `Moonlight-16B-A3B`, `Kimi-VL-A3B`, and `Kimi-K2.5` (while keeping `1e-6` for DeepSeek-V3 and MLA q/kv norms) eliminates layer-0 normalized input drift of **`1.0%` / `1.2%` / `4.8%` median** and **`34%–40%` max** on published token embeddings.
- **Cuspy CUPTI Profiler (`#5525`)**: Adds `CuspyProfilerConfig` for CUPTI PM sampling, power/clock/thermal counters, CUDA graph dependency edges, and `pftrace` Perfetto output.
- **`torch.accelerator` Stream & Event Unification (`#5450`)**: Migrates EMA CPU offload (`params_read` event holding compute stream only until H2D+lerp finishes so trailing D2H overlaps compute), RL weight staging, and `PagedStashManager` to `torch.accelerator`, `torch.Stream`, and `torch.Event`.

---

## 3. LMCache / Dynamo Isolation & Mooncake TENT Diagnostics / Ragged Dtypes

- **LMCache & Dynamo (`#5925` / `#5939` / `#16324`)**: Isolates deprecated in-process LMCache under `lmcache.inprocess` and removes 7 `kvbm-*` crates/wheels from Dynamo CI/release while migrating `PdConnector` to `dynamo.vllm.pd_connector`.
- **Mooncake TENT Diagnostics & Redactor (`#4154`)**: Introduces `tent diagnostics version [--json]` with exit-code precedence **`4` (`tool.internal`) > `3` (`engine.process` / `tool.probe_missing`) > `1` (check fail) > `0`** (`2` for CLI parse error) and consistent pseudonym redaction across fields and free text via `Redactor::sanitizeText()`.
- **Mooncake `typed_ragged` Empty-Row Fix (`#4611`)**: `_typed_ragged_row_dtypes` ignores empty Python `list/tuple` rows so `[[1, 2], [], [2**53 + 1]]` stays `int64` (`typed_ragged`) without `float64` precision loss, and `[[True, False], []]` stays `msgpack_ragged`.

---

## 4. Deterministic Reproduction

```bash
python experiments/te-kda-attention-torchtitan-cuspy-kimi-eps-lmcache-inprocess-mooncake-tent/run.py --check
```
