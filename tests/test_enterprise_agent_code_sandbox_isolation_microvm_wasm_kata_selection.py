from __future__ import annotations

import unittest
from experiments.enterprise_agent_code_sandbox_isolation_microvm_wasm_kata_selection.run import build_payload


class TestEnterpriseAgentCodeSandboxIsolationMicrovmWasmKataSelection(unittest.TestCase):
    def test_tier4_zero_escape_full_compatibility_fast_boot_and_zero_secret_exposure(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertEqual(tier4["host_kernel_and_lateral_network_escape_risk_pct"], 0.0)
        self.assertEqual(tier4["native_python_c_extension_compatibility_pct"], 100.0)
        self.assertGreater(
            tier4["native_python_c_extension_compatibility_pct"],
            tier2["native_python_c_extension_compatibility_pct"] + 40.0,
        )
        self.assertLess(tier4["sandbox_cold_start_p95_ms"], 150.0)
        self.assertLess(tier4["sandbox_cold_start_p95_ms"], tier3["sandbox_cold_start_p95_ms"] * 0.01)
        self.assertEqual(tier4["sandbox_credential_theft_exposure_pct"], 0.0)
        self.assertLess(
            tier4["monthly_sandbox_infra_and_security_tco_usd"],
            tier1["monthly_sandbox_infra_and_security_tco_usd"] * 0.25,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
