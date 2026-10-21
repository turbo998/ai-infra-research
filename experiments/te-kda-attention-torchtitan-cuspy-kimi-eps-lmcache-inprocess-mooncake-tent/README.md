# TE `KimiDeltaAttention` (KDA), TorchTitan Kimi K2.x `rms_norm_eps=1e-5` & `CuspyProfilerConfig`, LMCache `inprocess`, Dynamo KVBM Retirement & Mooncake TENT Experiment

Deterministic Python 3.12 standard-library verification for Phase 50 Topic 3 (`te-kda-attention-torchtitan-cuspy-kimi-eps-lmcache-inprocess-mooncake-tent`):

- **`TransformerEngine#3548` (`b9715e55a9`) & `#3470` (`c3081a5f2d`)**: Verifies non-commuting per-key-channel `KimiDeltaAttention` recurrence $S_t = (I - \beta_t k_t k_t^\top)\mathrm{Diag}(\exp(g_t))S_{t-1} + \beta_t k_t v_t^\top$ (`0.388656` non-commutativity Frobenius gap vs GDN) and narrowed pre-cuDNN 9.26 THD learnable-softmax guard (`head_dim_v not in {64, 128, 256}`).
- **`torchtitan#5548` (`d06d37f53c`), `#5525` (`789740c025`), `#5450` (`35f4a2e166`)**: Verifies Kimi K2.x (`Moonlight-16B-A3B`, `Kimi-VL-A3B`, `Kimi-K2.5`) `rms_norm_eps=1e-5` (`0.0` drift vs `34–40%` max drift under `1e-6`), `CuspyProfilerConfig`, and `torch.accelerator`.
- **`LMCache#5925`/`#5939`, `dynamo#16324` (`45a4e56a5f`), `Mooncake#4154` (`7540f5826e`) & `#4611` (`8952021237`)**: Verifies `lmcache.inprocess`, Dynamo KVBM retirement (`7` crates removed), Mooncake TENT CLI exit-code lattice (`4 > 3 > 1 > 0`) & pseudonym `Redactor`, and `_typed_ragged_row_dtypes` preserving `int64` (`2**53 + 1`) and `bool` across empty rows `[]`.

```powershell
python experiments\te-kda-attention-torchtitan-cuspy-kimi-eps-lmcache-inprocess-mooncake-tent\run.py --check
```
