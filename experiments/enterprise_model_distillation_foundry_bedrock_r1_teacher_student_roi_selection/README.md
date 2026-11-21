# Enterprise Model Distillation & Synthetic Data Factory Selection (`Azure AI Foundry Distillation` vs. `AWS Bedrock Model Distillation` vs. `Rejection-Sampled R1 + Top-64 Logit KD`)

Reproducible analytical model comparing four enterprise model distillation and teacher-student cascade architectures at 10B tokens/month across `domain accuracy`, `P95 latency`, `one-time distillation cost`, and `annual serving TCO`.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_model_distillation_foundry_bedrock_r1_teacher_student_roi_selection/run.py`
