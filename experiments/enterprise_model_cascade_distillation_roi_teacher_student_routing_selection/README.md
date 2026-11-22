# Enterprise Teacher-Student Model Distillation ROI & Pre-Routed Cascade (`671B Teacher -> 32B/8B Distilled Student + RouteLLM`) (`#488`)

Reproducible analytical model comparing 100% `671B` Frontier Serving (`64 GPUs`, `$115.2k/mo`), Un-Distilled Small Models (`74.2%` accuracy), Sequential Retry Cascades, and a `Domain-Distilled 32B/8B Student (82% traffic) + 671B Teacher (18% traffic)` pre-routed flywheel (`95.1%` accuracy, `16 GPUs`, `$28.8k/mo`, `0.64-day` distillation break-even).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_model_cascade_distillation_roi_teacher_student_routing_selection/run.py --check`
