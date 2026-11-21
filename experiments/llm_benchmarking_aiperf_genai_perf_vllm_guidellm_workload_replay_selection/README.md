# LLM Benchmarking & PoC Acceptance Protocol: AIPerf/GenAI-Perf, vLLM Bench, GuideLLM & Anti-Gaming Trace Replay

Deterministic quantitative evaluation model for [`docs/components/llm-benchmarking-aiperf-genai-perf-vllm-guidellm-workload-replay-selection-faq.md`](../../docs/components/llm-benchmarking-aiperf-genai-perf-vllm-guidellm-workload-replay-selection-faq.md) and [`.en.md`](../../docs/components/llm-benchmarking-aiperf-genai-perf-vllm-guidellm-workload-replay-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/llm_benchmarking_aiperf_genai_perf_vllm_guidellm_workload_replay_selection/run.py
python experiments/llm_benchmarking_aiperf_genai_perf_vllm_guidellm_workload_replay_selection/run.py --check
python -m unittest tests/test_llm_benchmarking_aiperf_genai_perf_vllm_guidellm_workload_replay_selection.py
```
