# Enterprise LLM & Agent Security Selection Experiment (`Spotlighting` + `CaMeL / Dual-LLM Control-Data Separation` + `Taint Tracking` vs. `Flat Prompt Concatenation`)

Reproducible security evaluation model comparing 4 indirect prompt injection (`XPIA`) defense architectures across attack success rate (`ASR`), data exfiltration rate, benign task utility retention, and policy latency overhead.

## Run
```bash
python experiments/llm_security_indirect_prompt_injection_spotlighting_camel_privilege_isolation_selection/run.py
python experiments/llm_security_indirect_prompt_injection_spotlighting_camel_privilege_isolation_selection/run.py --check
```
