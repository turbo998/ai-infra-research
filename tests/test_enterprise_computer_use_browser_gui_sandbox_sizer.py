from __future__ import annotations

from experiments.enterprise_computer_use_browser_gui_sandbox_sizer.run import (
    simulate_computer_use_agent_infra,
)


def test_computer_use_browser_gui_sandbox_sizer() -> None:
    res = simulate_computer_use_agent_infra()
    gm = res["grounding_modes"]
    assert gm["naive_full_1080p_screenshot_history"]["cumulative_visual_tokens_per_15step_task"] == 30300
    assert gm["hybrid_a11y_dom_plus_omniparser_roi_crop"]["cumulative_visual_tokens_per_15step_task"] == 6540
    assert res["summary_metrics"]["visual_token_reduction_pct"] == 78.4
    assert res["summary_metrics"]["vlm_gpu_reduction_pct"] == 78.6
    assert res["summary_metrics"]["sandbox_ram_density_gain_x"] == 4.44
