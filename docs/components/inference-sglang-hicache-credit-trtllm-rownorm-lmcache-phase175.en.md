# Inference Systems Deep Dive (Phase 175): SGLang HiCache Protected Page-Cache Credit, TensorRT-LLM RowLinearNorm Fusion & LMCache Deterministic KV Fill

## 1. Architectural Motivation

Production hierarchical KV caching (HiCache), tensor-parallel norm fusion, and disaggregated KV benchmarking depend on three exact accounting and verification mechanisms:
1. **Over-Withholding Page-Cache Credit in SGLang HiCache**: Deducting unprotected, evictable bytes from HiCache page-cache credit prematurely exhausts admission quota and throttles prefix cache offload.
2. **Corner-Case `RowLinear + Norm` Fusion in TensorRT-LLM**: Small-dimension tensor-parallel shapes (`TP=2, hidden=16, seqlen=2`) stress vector alignment and reduction boundaries in fused row-linear norm kernels.
3. **Deterministic Identity KV Payloads & XPU Kimi Linear in LMCache**: Filling benchmark KV buffers from deterministic request identities allows zero-overhead silent data corruption (SDC) checks during high-throughput KV transfers, while Intel XPU deployments require engine-driven Kimi Linear coverage.

## 2. Primary Source Evidence & Implementation Mechanics

| Repository | Commit SHA | Mechanism | Architectural Impact |
| :--- | :--- | :--- | :--- |
| `sgl-project/sglang` | `181380682b50` | `[Fix] hicache`: Withhold only protected bytes from page-cache credit | Prevents false credit exhaustion in HiCache (`1.0` credit accounting accuracy) |
| `sgl-project/sglang` | `1a50817ed056` / `a81065456339` / `98a7f7e175c3` | Rename NPU hardware docs to A2/A3 Series products and sync SGLang blog cards | Standardizes Ascend A2/A3 hardware terminology across serving docs |
| `NVIDIA/TensorRT-LLM` | `a42093fc319d` / `87ce6f1aa903` | Unwaive `test_row_linear_norm_fusion[2-hidden:16-seqlen:2]` and clean up GPU architecture handling | Verifies `1.0` pass rate on narrow-hidden TP RowLinear + Norm fusion |
| `LMCache/LMCache` | `58c2ace2c2ab` / `b6432b3f3915` | Fill benchmark request KV from deterministic identities & add XPU engine-driven tiny Kimi Linear coverage | Enables `1.0` reproducible KV payload verification and XPU Kimi Linear validation |
| `vllm-project/vllm` | `15a9677306b1` / `4a8dc13adc0e` / `c5e37f1922f4` / `fc84d05acb20` | Enroll 92 remaining Sampler, Compile/ABI, Distributed/EP, and Spec Decode jobs in automatic sharding | Completes `1.0` automatic shard balancing across vLLM's CI test suites |

## 3. Quantitative Verification (`sglang_hicache_protected_credit_trtllm_row_norm_lmcache_det_kv_phase175`)

Validated via `experiments/sglang_hicache_protected_credit_trtllm_row_norm_lmcache_det_kv_phase175/run.py --check`:
- `sglang_hicache_protected_bytes_only_credit_withhold_accuracy_ratio`: `1.0`
- `trtllm_row_linear_norm_fusion_tp2_h16_s2_pass_ratio`: `1.0`
- `lmcache_xpu_kimi_linear_engine_coverage_ratio`: `1.0`
- `lmcache_bench_deterministic_identity_kv_fill_reproducibility_ratio`: `1.0`
- `vllm_auto_sharded_ci_matrix_coverage_ratio`: `1.0`

## 4. Solutions Architect Takeaways

1. **Protected-Only HiCache Credit Accounting**: Only withhold page-cache credits for actively pinned/protected KV pages so unpinned cached pages remain available for opportunistic hits without blocking new offloads.
2. **Deterministic Identity KV Benchmarking**: Use deterministic identity KV payloads in LMCache stress tests to detect page-ordering bugs across RDMA/NIXL fabrics.
