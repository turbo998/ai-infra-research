import unittest

from experiments.vllm_v1_rejection_sampler_spec_decode.run import _check, run_simulation


class TestVLLMV1RejectionSamplerSpecDecode(unittest.TestCase):
    def test_rejection_sampler_and_fly_verification(self) -> None:
        res = run_simulation()
        _check(res)
        self.assertTrue(res["gumbel_max_and_fp64_guard"]["unnormalized_argmax_equals_normalized_argmax"])
        self.assertGreater(res["vocab_roofline"]["vocab_128256"]["hbm_traffic_reduction_x"], 4.4)


if __name__ == "__main__":
    unittest.main()
