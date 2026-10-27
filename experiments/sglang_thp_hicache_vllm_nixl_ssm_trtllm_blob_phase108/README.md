# SGLang Parallel THP HiCache & DP-Attention Padding, vLLM Hybrid SSM NIXL Regions & TensorRT-LLM Blob Backend (Phase 108)

Reproducible stdlib-only model validating:
- `sgl-project/sglang@75468e4b4f` (`#43978`): Parallel THP-backed (`2MB` / `1GB`) host KV pool prefaulting.
- `sgl-project/sglang@58eb1740d2` (`#45796`): Out-of-place host list padding for DP-attention to preserve pre-planned DSA metadata rows.
- `vllm-project/vllm@4ddc7aa40f` (`#62317`): NIXL KV connector per-group region filtering (`_group_region_indices`) in hybrid Attention + SSM models.
- `NVIDIA/TensorRT-LLM@b351c407bb` (`#20884`) & `c554d4486d` (`#20227`): Disaggregated `BlobBackend` batched registration + bounded queueing and deterministic MoE finalize selection.

```bash
python experiments/sglang_thp_hicache_vllm_nixl_ssm_trtllm_blob_phase108/run.py --check
```
