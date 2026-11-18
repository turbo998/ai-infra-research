"""Unit tests for GB200 / B200 Multi-Node NVLink (MNNVL), IMEX Channels & Multi-Tenant Fabric Isolation Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.gb200_mnnvl_imex_channels_nvlink_isolation_selection.run import (
    evaluate_gb200_mnnvl_imex_channels_nvlink_isolation_matrix,
)


class TestGb200MnnvlImexChannelsNvlinkIsolationSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_gb200_mnnvl_imex_channels_nvlink_isolation_matrix()
        self.assertEqual(data["experiment_id"], "gb200_mnnvl_imex_channels_nvlink_isolation_selection")
        self.assertEqual(data["round"], 182)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertEqual(
            tiers["tier2_shared_global_imex_channel0_single_partition"]["cross_tenant_hbm_exposure_pct"],
            100.0,
        )
        self.assertEqual(
            tiers["tier4_k8s_dra_imex_plus_nmxc_hardware_nvlink_partition"]["cross_tenant_hbm_exposure_pct"],
            0.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_busbw_speedup_ratio"],
            15.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "gb200_mnnvl_imex_channels_nvlink_isolation_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_gb200_mnnvl_imex_channels_nvlink_isolation_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
