# LLMOps Continuous Evaluation & LLM-as-a-Judge Selection Experiment (`Frontier API Judge` vs. `Prometheus-2 / Qwen-14B-Judge` vs. `DeBERTa NLI + RAGAS Triad`)

Reproducible accuracy, bias, and TCO model comparing 4 continuous evaluation architectures across `5,000,000` monthly production RAG/Agent traces.

## Run
```bash
python experiments/llm_continuous_eval_llm_as_judge_prometheus_ragas_canary_rollout_selection/run.py
python experiments/llm_continuous_eval_llm_as_judge_prometheus_ragas_canary_rollout_selection/run.py --check
```
