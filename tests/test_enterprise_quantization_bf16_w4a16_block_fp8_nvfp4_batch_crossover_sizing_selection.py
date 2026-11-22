from pathlib import Path
from experiments.enterprise_quantization_bf16_w4a16_block_fp8_nvfp4_batch_crossover_sizing_selection.run import build_results


def test_enterprise_quantization_bf16_w4a16_block_fp8_nvfp4_batch_crossover_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_quantization_bf16_w4a16_block_fp8_nvfp4_batch_crossover_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[2]["batch_64_goodput_tok_s"] > tiers[1]["batch_64_goodput_tok_s"] > tiers[0]["batch_64_goodput_tok_s"]
    assert tiers[3]["batch_64_goodput_tok_s"] > tiers[2]["batch_64_goodput_tok_s"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-quantization-bf16-w4a16-block-fp8-nvfp4-batch-crossover-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-quantization-bf16-w4a16-block-fp8-nvfp4-batch-crossover-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/quantization-bf16-w4a16-block-fp8-nvfp4-crossover-lab.html").exists()
