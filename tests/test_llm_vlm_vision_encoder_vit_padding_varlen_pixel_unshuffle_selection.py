import unittest
from experiments.llm_vlm_vision_encoder_vit_padding_varlen_pixel_unshuffle_selection.run import (
    evaluate_vlm_vit_varlen_pixel_unshuffle_tiers,
)


class TestVlmVitVarlenPixelUnshuffleSelection(unittest.TestCase):
    def test_vlm_vit_tiers(self) -> None:
        data = evaluate_vlm_vit_varlen_pixel_unshuffle_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_padded_3d_vit_plus_uncompressed_1to1_llm_visual_tokens"]
        t4 = tiers["tier4_varlen_windowed_vit_plus_2x2_pixel_unshuffle_and_nvjpeg"]
        self.assertEqual(t1["llm_visual_tokens_per_highres_img"], 14000)
        self.assertEqual(t4["llm_visual_tokens_per_highres_img"], 3500)
        self.assertLess(t4["end_to_end_vlm_ttft_ms"], 120.0)


if __name__ == "__main__":
    unittest.main()
