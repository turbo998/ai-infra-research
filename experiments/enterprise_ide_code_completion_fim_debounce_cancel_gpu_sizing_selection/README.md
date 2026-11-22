# Enterprise 10,000-Developer IDE Code Completion (`FIM <= 200ms`, `Syntax-Aware Debounce`, `In-Flight Cancel` & `PLD Speculative` Sizer) (`#491`)

Reproducible analytical model comparing Un-Debounced Shared 70B Pools (`96 GPUs`, `1,840ms` latency, `11.2%` acceptance), Fixed 300ms Debounce, Dedicated `7B/14B` FIM Pools, and `Tree-Sitter Syntax-Aware Debounce + HTTP/2 Cancel + 7B/14B FP8 + PLD Speculative + File-Prefix Cache` (`8 GPUs` for 10,000 developers, `138ms P99` latency, `38.6%` acceptance).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_ide_code_completion_fim_debounce_cancel_gpu_sizing_selection/run.py --check`
