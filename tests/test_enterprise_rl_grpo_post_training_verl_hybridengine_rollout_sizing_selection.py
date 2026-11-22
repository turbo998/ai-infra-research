from pathlib import Path
from experiments.enterprise_rl_grpo_post_training_verl_hybridengine_rollout_sizing_selection.run import build_results


def test_enterprise_rl_grpo_post_training_verl_hybridengine_rollout_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_rl_grpo_post_training_verl_hybridengine_rollout_sizing_selection"
    archs = res["rl_architectures"]
    assert len(archs) == 4
    assert archs[3]["grpo_steps_per_hour_64gpu"] > archs[0]["grpo_steps_per_hour_64gpu"] * 7
    assert archs[3]["weight_sync_reshard_sec_per_step"] < 2.0

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-rl-grpo-post-training-verl-hybridengine-rollout-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-rl-grpo-post-training-verl-hybridengine-rollout-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/rl-grpo-post-training-verl-rollout-sizer-lab.html").exists()
