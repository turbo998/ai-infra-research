# Enterprise LLM Knowledge Distillation & Synthetic Data Pipeline Selection Experiment (`SeqKD` vs. `Offline Top-64 Logit KD` vs. `On-Policy GKD / MiniLLM Reverse KL`)

Reproducible student accuracy retention, logprob storage footprint, distillation compute cost, and annual serving TCO model for distilling a `72B / 671B` teacher into an `8B` production student model.

## Run
```bash
python experiments/llm_knowledge_distillation_logit_kd_minillm_synthetic_data_pipeline_selection/run.py
python experiments/llm_knowledge_distillation_logit_kd_minillm_synthetic_data_pipeline_selection/run.py --check
```
