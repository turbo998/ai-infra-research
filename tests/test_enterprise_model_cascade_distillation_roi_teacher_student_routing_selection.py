from pathlib import Path
from experiments.enterprise_model_cascade_distillation_roi_teacher_student_routing_selection.run import build_results


def test_enterprise_model_cascade_distillation_roi_teacher_student_routing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_model_cascade_distillation_roi_teacher_student_routing_selection"
    tiers = res["cascade_tiers"]
    assert len(tiers) == 4
    assert tiers[3]["monthly_serving_cost_usd"] <= tiers[0]["monthly_serving_cost_usd"] * 0.25
    assert tiers[3]["domain_task_accuracy_pct"] > 95.0

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-model-cascade-distillation-roi-teacher-student-routing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-model-cascade-distillation-roi-teacher-student-routing-selection-faq.en.md").exists()
    assert (root / "docs/h5/model-cascade-distillation-roi-routing-sizer-lab.html").exists()
