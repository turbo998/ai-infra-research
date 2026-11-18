# Enterprise Workforce Adaptive Learning, BKT/DKT Knowledge Tracing & Socratic Tutor Agent (`R434`)

Deterministic evaluation suite comparing four enterprise engineering upskilling, hands-on cloud lab auto-grading, and knowledge tracing architectures across `25,000` active engineers, `480` skill-DAG competency nodes, and `110,000` monthly hands-on lab sessions:

1. **Tier 1**: Legacy Static Video LMS + Multiple-Choice Quiz (`24.0%` 90-day retention, `31.5%` false pass rate, `42.0h` time-to-competency, `$380,000/mo` wasted engineering time & rework cost)
2. **Tier 2**: Unconstrained Direct-Answer General LLM Chatbot (`19.5%` 90-day retention, `78.4%` direct-answer spoiler leak, `26.0%` false pass, `28.0h` time-to-competency, `$425,000/mo` cost)
3. **Tier 3**: Courseware Vector RAG + LLM-as-a-Judge Code Grader (`48.0%` 90-day retention, `22.5%` spoiler leak, `14.8%` prompt-injection false pass, `24.5h` time-to-competency, `$195,000/mo` cost)
4. **Tier 4**: Prerequisite Skill-DAG `BKT/DKT` + 3-Step Socratic Anti-Spoiler `FSM` + Ephemeral `gVisor/K8s` Sandbox `pytest/OpenTelemetry` Grader + `FSRS` Spaced Repetition (`86.5%` 90-day retention, `0.0%` spoiler leak, `0.0%` false pass, `11.5h` time-to-competency, `$44,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_adaptive_learning_knowledge_tracing_socratic_tutor_selection/run.py
python experiments/enterprise_adaptive_learning_knowledge_tracing_socratic_tutor_selection/run.py --check
python -m unittest tests/test_enterprise_adaptive_learning_knowledge_tracing_socratic_tutor_selection.py
```
