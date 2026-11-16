import unittest
from experiments.llm_pretraining_mmap_indexed_dataset_feistel_o1_shuffle_memory_selection.run import evaluate_mmap_feistel_o1_shuffle


class TestMmapFeistelO1Shuffle(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_mmap_feistel_o1_shuffle()
        self.assertEqual(res["round"], 308)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_feistel_bijective_o1_permutation_plus_mmap_madv_dontneed"]["node_shuffle_index_ram_gb"],
            0.01,
        )
        self.assertLess(
            tiers["tier4_feistel_bijective_o1_permutation_plus_mmap_madv_dontneed"]["epoch_init_shuffle_sec"],
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
