# Experiment #584: Enterprise LLM Guardrails & Parallel Speculative Safety Gateway Sizer

Reproduces the `+6ms` net TTFT penalty of parallel speculative input guardrails (`22ms` overlapped with `240ms` LLM prefill) vs. `+340ms` serial guardrails, and `2× L40S` sizing for `100 QPS` (`docs/components/enterprise-llm-guardrails-prompt-shield-safety-gateway-sizing-faq.md`).
