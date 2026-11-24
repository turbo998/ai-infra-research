# Experiment #585: Enterprise AI Observability OTel + ClickHouse ZSTD + 5% Stratified Judge Sizer

Reproduces the `560 GB/month` ClickHouse `ZSTD(3)` storage footprint (`7.5:1` compression on `4,200 GB` raw traces for `100M` traces/month), `$1,800/mo` self-hosted Langfuse v3 cost (`-96.4%` vs. `$50,000/mo` SaaS), and `2× Spot L40S` `5%` stratified online judge sizing (`docs/components/enterprise-ai-observability-otel-langfuse-judge-slo-sizing-faq.md`).
