# Enterprise Multi-Model Router (`RouteLLM` Matrix Factorization) & Logprob-Margin Abstention Cascade (`R462`)

Deterministic evaluation suite comparing four multi-model LLM routing and cascade architectures across `60M` monthly enterprise requests (`4` candidate model tiers: `8B SLM`, `32B Dense`, `70B/Sonnet`, `671B/R1 Reasoning`):

1. **Tier 1**: 100% Frontier Model Default Routing (`100.0%` quality retention, `100.0%` frontier call share, `$240,000/mo` inference spend)
2. **Tier 2**: Prompt-Length / Regex Heuristic Router + Uncalibrated SLM Verbal Self-Ask (`76.4%` quality retention, `21.8%` silent SLM hallucination, `180.0 ms` router overhead, `$84,000/mo` spend)
3. **Tier 3**: Standalone Embedding Cosine Pre-Router Only (`91.2%` quality retention, `7.4%` silent SLM hallucination on OOD queries, `12.0 ms` overhead, `$96,000/mo` spend)
4. **Tier 4**: Matrix-Factorization Win-Rate Pre-Router + First-16-Token Logprob Margin Abstention Cascade (`99.2%` quality retention, `0.3%` silent SLM hallucination, `8.5 ms` overhead, `22.0%` frontier call share, `$64,000/mo` spend)

## Usage

```bash
python experiments/enterprise_llm_router_cascade_matrix_factorization_abstention_selection/run.py
python experiments/enterprise_llm_router_cascade_matrix_factorization_abstention_selection/run.py --check
python -m unittest tests/test_enterprise_llm_router_cascade_matrix_factorization_abstention_selection.py
```
