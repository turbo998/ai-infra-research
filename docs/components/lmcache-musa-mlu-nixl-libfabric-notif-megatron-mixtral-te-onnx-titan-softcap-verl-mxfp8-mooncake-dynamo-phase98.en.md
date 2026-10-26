# LMCache MUSA/MLU Backends, NIXL Libfabric $O(1)$ Keyed Notification, Mooncake Single-Clock RDMA Doorbell, Megatron Mixtral Version Check, TE ONNX 0-D Scale, TorchTitan Compiled Softcap, veRL Blackwell MXFP8 RL & Dynamo GPT-OSS v2 (Phase 98)

> **中文版本**: [lmcache-musa-mlu-nixl-libfabric-notif-megatron-mixtral-te-onnx-titan-softcap-verl-mxfp8-mooncake-dynamo-phase98.md](lmcache-musa-mlu-nixl-libfabric-notif-megatron-mixtral-te-onnx-titan-softcap-verl-mxfp8-mooncake-dynamo-phase98.md)
> **Priority**: `107` · **Question ID**: `lmcache-musa-mlu-nixl-libfabric-notif-megatron-mixtral-te-onnx-titan-softcap-verl-mxfp8-mooncake-dynamo-phase98` · **Evidence Class**: `implementation` + `author-benchmark` + `own-offline`

---

## 1. Executive Summary & Architectural Findings

Across 10 zero-overlap upstream commits in LMCache, NIXL, Megatron-LM, TransformerEngine, TorchTitan, veRL, Mooncake, and Dynamo, Phase 98 establishes the following systems invariants:

1. **Mooncake `RdmaEndPoint::submitPostSend` Single Clock Read per QP WR Chain (`Mooncake#4793`, `dd1217b6c4`)**: Stamping every slice with its own `clock_gettime(CLOCK_REALTIME)` call (`~19 ns` per WR) while holding the endpoint write lock `lock_` dominated WR chain construction and delayed the RDMA doorbell. Reading `getCurrentTimeInNano()` once per chain drops a 256-WR call (two 128-WR chains) from `6.3 us` to `1.9 us` (**`3.316x` speedup**, `25.0 -> 7.5 ns/WR`).
2. **NIXL Libfabric $O(1)$ Keyed Pending-Notification Lookup (`nixl#2436`, `46ac6014e9`) & `kvbench` CA Bundle Forwarding (`nixl#2395`, `1a068bea62`)**: Replaces the full $O(N)$ `checkPendingNotifications()` map scan under `receiver_tracking_mutex_` with `checkPendingNotification(makePendingKey(sender_peer_idx, notif_xfer_id))`, cutting map inspections across 256 pending notifications from `65,536` to `256` (**`256x` reduction**), while `kvbench` preserves `--obj_ca_bundle` in `NIXLBench.defaults()`.
3. **LMCache MUSA `KVFormatSpec` Convergence & Cambricon MLU Backend (`LMCache#5573`, `76bc242df7`; `#6023`, `a2cd25df48`)**: Converges MUSA `kv_format.py` onto `get_spec_class` and adds `MluDeviceSpec` with lazy `torch_mlu` probing, expanding LMCache's unified multiprocess device abstraction to 7 hardware backends (`cpu, cuda, hpu, mlu, musa, neuron, npu`).
4. **Megatron-LM Mixtral HF Loader Semantic Version Check for Transformers 5.x (`Megatron-LM#7937`, `888887d35e`)**: Replaces `major >= 4 and minor >= 36` (which rejected `transformers 5.0.0` because `minor=0 < 36`) with `packaging.version.Version(transformers.__version__) >= Version('4.36.0')`.
5. **TransformerEngine ONNX 0-D Scalar Scale Metadata & TorchTitan Compiled MuseGlimmer Softcap (`TransformerEngine#3659`, `33f3b35619`; `torchtitan#5630`, `18e92bae92`)**:
   - TE changes `onnx_cs_quantize_fp8_op`'s fake tensor scale output from `(1,)` to a 0-D scalar `()` to match `tex.quantize`;
   - TorchTitan applies `@local_compile("muse_glimmer_logit_postprocess", batch_invariant=True)` to MuseGlimmer softcap layers, recomputing `tanh` in backward from saved BF16 logits (`2 B/elem`) instead of saving FP32 `tanh` outputs (`4 B/elem`, **`-50%` activation memory traffic**) and documenting `temp * (1/temp)` cancellation (`4 -> 2` FP32 roundings).
6. **veRL Blackwell End-to-End MXFP8 RL Recipe & Dynamo GPT-OSS v2 Structural Tag Parser (`verl#8127`, `d1390eea0c`; `dynamo#17146`, `100cce2e48`)**:
   - veRL adds the Blackwell `fp8_recipe: mxfp8` Megatron training + vLLM/SGLang MXFP8 rollout recipe for `Qwen3-30B-A3B` (`32 E4M3 + 1 E8M0 = 33 B` vs `64 B` BF16, **`1.939x` compression**);
   - Dynamo upgrades `dynamo-parsers-v2` to `0.7.35` with `dynamo-structural-tag`, rejects unsupported `reasoning_effort="none"` on `gpt_oss`, and bypasses v1 Harmony `<|call|>` structural tags to avoid XGrammar stop-token conflicts.

---

## 2. Primary Source Evidence & Quantitative Metrics

| # | Evidence ID / Commit | Mechanism & Hardware Contract | Quantitative Gain / Correctness Invariant |
| :--- | :--- | :--- | :--- |
| 1 | `lmcache-musa-converge-kv-format-onto-kvformatspec-5573-76bc242-2026` (`76bc242df7`) | `musa/kv_format.py`: delegate to `get_spec_class` class methods | 100% API convergence with `KVFormatSpec` |
| 2 | `lmcache-mlu-add-cambricon-mlu-device-backend-for-mp-mode-6023-a2cd25d-2026` (`a2cd25df48`) | `device_backend.py`: `MluDeviceSpec` with lazy `torch_mlu` probe | 7 unified device backends in LMCache MP mode |
| 3 | `nixl-perf-libfabric-check-only-the-affected-pending-n-2436-46ac601-2026` (`46ac6014e9`) | `nixl_libfabric_engine.cpp`: `checkPendingNotification(key)` $O(1)$ lookup | Pending map inspections cut `65,536 -> 256` (`256x` reduction) |
| 4 | `nixl-fix-kvbench-preserve-obj-ca-bundle-in-nixlbench-2395-1a068be-2026` (`1a068bea62`) | `nixlbench/main.py`: include `obj_ca_bundle` in `NIXLBench.defaults()` | Preserves custom S3/OBJ TLS CA bundle path |
| 5 | `megatron-lm-fix-mixtral-loader-version-check-for-transformer-7937-888887d-2026` (`888887d35e`) | `loader_mixtral_hf.py`: `Version(transformers.__version__) >= Version('4.36.0')` | Accepts `transformers 5.0.0+` (`minor=0`) without false failure |
| 6 | `transformerengine-onnx-fix-layernorm-execution-and-fp8-scale-fake-3659-33f3b35-2026` (`33f3b35619`) | `onnx_extensions.py`: `torch.empty((), dtype=float32)` 0-D scalar scale | Eliminates `(1,)` vs `()` fake-tensor mismatch under `torch.compile` |
| 7 | `torchtitan-compile-muse-glimmer-softcap-5630-18e92ba-2026` (`18e92bae92`) | `muse_glimmer/model.py`: `@local_compile(..., batch_invariant=True)` | Saves BF16 logits (`2 B`) instead of FP32 `tanh` (`4 B`), `-50%` traffic |
| 8 | `verl-megatron-feat-mxfp8-training-recipe-and-end-to-e-d1390ee-2026` (`d1390eea0c`) | `docs/low_precision/mxfp8.md` + `run_qwen3_30b_a3b_megatron.sh` | Blackwell MXFP8 (`33 B / 32 elems`) gives `1.939x` compression vs BF16 |
| 9 | `mooncake-transferengine-read-the-clock-once-per-rdma-post-dd1217b-2026` (`dd1217b6c4`) | `rdma_endpoint.cpp`: single `getCurrentTimeInNano()` per QP WR chain | 256-WR call drops `6.3 -> 1.9 us` (`3.316x` speedup, `25 -> 7.5 ns/WR`) |
| 10 | `dynamo-feat-frontend-integrate-gpt-oss-v2-parser-17146-100cce2-2026` (`100cce2e48`) | `structural_tag.rs` + `preprocessor.rs`: `dynamo-parsers-v2 0.7.35` | Rejects `gpt_oss` `reasoning_effort=none`; avoids v1 `<|call|>` conflict |

---

## 3. Architect Design Exercises & Falsifiable Questions

1. **Calculation Exercise (RDMA `ibv_post_send` Clock-Read Amortization)**: When `RdmaEndPoint::submitPostSend` builds two 128-WR chains (`256` WRs total) under `lock_`, each `clock_gettime(CLOCK_REALTIME)` call costs `~19 ns` while WR/SGE pointer setup costs `~6 ns`. Compute the total lock hold time before (`256 * 25 ns = 6.4 us`, measured `6.3 us`) and after (`256 * 6 ns + 2 * 19 ns = 1.574 us`, measured `1.9 us`), and explain why sharing one timestamp across all WRs in a single `ibv_post_send` chain preserves timeout accuracy.
2. **Falsifiable Hypothesis**: Is `major >= 4 and minor >= 36` a valid semantic version check for `>= 4.36.0`? Megatron-LM `#7937` proves **No**: any `5.x` release with `x < 36` (such as `5.0.0` or `5.8.1`) evaluates `minor >= 36` as `False`.

---

## 4. Reproducible Offline Experiment

- **Experiment Script**: `experiments/lmcache_musa_mlu_nixl_libfabric_notif_megatron_mixtral_te_onnx_titan_softcap_verl_mxfp8_mooncake_dynamo_phase98/run.py`
- **Verification Command**: `python experiments\lmcache_musa_mlu_nixl_libfabric_notif_megatron_mixtral_te_onnx_titan_softcap_verl_mxfp8_mooncake_dynamo_phase98\run.py --check`
- **Unit Test**: `tests/test_phase98_experiments.py`
