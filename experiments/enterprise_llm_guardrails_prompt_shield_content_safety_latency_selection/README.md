# Enterprise AI Guardrails, Prompt Shield & Content Moderation Architecture Selection (`R472`)

Deterministic evaluation suite comparing four enterprise AI guardrail, prompt injection defense, and RAG groundedness verification architectures across `45,000,000` monthly LLM requests (`280ms` baseline main-model prefill TTFT, `30ms` max guardrail latency budget):

1. **Tier 1**: Naive Regex & Keyword Blacklists Only (`31.5%` jailbreak block rate, `0.0%` hallucination catch, `14.2%` false positives, `$142,000/mo` incident + ops TCO)
2. **Tier 2**: Synchronous Frontier LLM-as-a-Judge Before & After Generation (`97.4%` block rate, `2,450.0ms` added P99 TTFT overhead breaking streaming, `$186,000/mo` TCO)
3. **Tier 3**: Sequential Pre-Call + Post-Call Small Guard Models (`Llama-Guard-3-8B / Cloud Safety API` in Serial) (`96.8%` block rate, `380.0ms` added P99 TTFT, `$68,000/mo` TCO)
4. **Tier 4**: Parallel Speculative Input Shield (`Azure Prompt Shields / DeBERTa + Llama-Guard`) + Async Sliding-Window SSE Stream Gate + NLI Groundedness Verifier (`99.4%` jailbreak block rate, `97.8%` hallucination catch rate, `18.0ms` net P99 TTFT overhead, `0.6%` false positives, `$29,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_llm_guardrails_prompt_shield_content_safety_latency_selection/run.py
python experiments/enterprise_llm_guardrails_prompt_shield_content_safety_latency_selection/run.py --check
python -m unittest tests/test_enterprise_llm_guardrails_prompt_shield_content_safety_latency_selection.py
```
