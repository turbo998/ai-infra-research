from pathlib import Path
from experiments.enterprise_video_generation_dit_usp_teacache_vae_gpu_sizing_selection.run import build_results


def test_enterprise_video_generation_dit_usp_teacache_vae_gpu_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_video_generation_dit_usp_teacache_vae_gpu_sizing_selection"
    stacks = res["dit_stacks"]
    assert len(stacks) == 4
    assert stacks[3]["video_clip_generation_latency_sec"] < stacks[0]["video_clip_generation_latency_sec"] * 0.05
    assert stacks[3]["videos_per_hour_per_8gpu_node"] > stacks[0]["videos_per_hour_per_8gpu_node"] * 3

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-video-generation-dit-usp-teacache-vae-gpu-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-video-generation-dit-usp-teacache-vae-gpu-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/video-generation-dit-usp-teacache-gpu-sizer-lab.html").exists()
