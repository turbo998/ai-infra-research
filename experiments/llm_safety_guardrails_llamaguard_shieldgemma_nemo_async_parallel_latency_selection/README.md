# LLM Safety Guardrails Architecture Selection Experiment (`Llama-Guard-3 / ShieldGemma` vs. `NeMo / Azure Content Safety` & Async Parallel Streaming)

Reproducible parametric model comparing 4 enterprise safety guardrail execution patterns across user-perceived first-token latency (`TTFT`), added guardrail latency penalty, prompt-injection/toxicity recall, and monthly guardrail GPU overhead.

## Run
```bash
python experiments/llm_safety_guardrails_llamaguard_shieldgemma_nemo_async_parallel_latency_selection/run.py
python experiments/llm_safety_guardrails_llamaguard_shieldgemma_nemo_async_parallel_latency_selection/run.py --check
```
