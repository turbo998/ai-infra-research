# vLLM V1 Triton Rejection Sampler (`BLOCK_SIZE=8192` Gumbel-Max) & FLy Entropy-Gated Deferred Verification

Deterministic stdlib reproduction of `vllm-project/vllm` (`vllm/v1/sample/rejection_sampler.py`: `sample_recovered_tokens_kernel`, `generate_uniform_probs` `float64` guard) and `vllm/v1/spec_decode/fly.py` (`compute_fly_entropy`, `apply_fly_greedy_acceptance_kernel`, `apply_fly_random_acceptance_kernel`).

## Run & Verify

```bash
python experiments/vllm_v1_rejection_sampler_spec_decode/run.py --check
```
