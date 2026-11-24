# Experiment #597: Enterprise Structured Output (XGrammar/llguidance + Jump-Forward) Sizer

Reproduces the `26.5%` decode step reduction (`235` vs. `320` steps via Jump-Forward deterministic key injection), `4.57x` P99 latency improvement (`980ms` vs. `4,480ms` retry loops), and `33.3%` GPU fleet reduction (`8× H200` vs. `12× H200` at `100 QPS`) (`docs/components/enterprise-structured-output-xgrammar-llguidance-json-sizing-faq.md`).
