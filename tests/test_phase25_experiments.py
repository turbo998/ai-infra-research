import subprocess
import sys
import unittest
from pathlib import Path

from experiments.fa4_sm100_hd256_bwd_m64_sim.run import (
    accum_tile_crd2idx,
    build_summary as build_fa4_summary,
    verify_summary as verify_fa4_summary,
)
from experiments.nixl_ucx_vram_vmm_notif_sim.run import (
    build_summary as build_nixl_summary,
    verify_summary as verify_nixl_summary,
)
from experiments.megatron_ckpt_without_output_mfsdp_sim.run import (
    build_summary as build_megatron_summary,
    verify_summary as verify_megatron_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase25Experiments(unittest.TestCase):
    def test_fa4_sm100_hd256_bwd_m64_summary_and_cli(self) -> None:
        summary = build_fa4_summary()
        verify_fa4_summary(summary)
        cfg = summary["configs"]["hd256_2cta_m64_serial_valid"]
        self.assertTrue(cfg["fits_in_512_cols"])
        self.assertEqual(cfg["tmem_end"], 512)
        self.assertFalse(cfg["has_sdS_xchg"])
        self.assertEqual(accum_tile_crd2idx("ROW_MAJOR", 64, 256, 3, 7), 3 * 256 + 7)

        res = subprocess.run(
            [sys.executable, str(REPO_ROOT / "experiments" / "fa4_sm100_hd256_bwd_m64_sim" / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("CHECK PASSED", res.stdout)

    def test_nixl_ucx_vram_vmm_notif_summary_and_cli(self) -> None:
        summary = build_nixl_summary()
        verify_nixl_summary(summary)
        self.assertEqual(
            summary["ucx_scenarios"]["ze_only_auto_hints_ze_device"]["resolved_hint"],
            "ze-device",
        )
        self.assertEqual(
            summary["notif_cases"]["empty_barrier_notif"]["patched_remaining_notifs"],
            0,
        )

        res = subprocess.run(
            [sys.executable, str(REPO_ROOT / "experiments" / "nixl_ucx_vram_vmm_notif_sim" / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("CHECK PASSED", res.stdout)

    def test_megatron_ckpt_without_output_mfsdp_summary_and_cli(self) -> None:
        summary = build_megatron_summary()
        verify_megatron_summary(summary)
        self.assertEqual(summary["checkpoint_without_output"]["final_state"], "DONE")
        self.assertFalse(summary["mfsdp_frozen_and_microbatch"]["frozen_patched_use_after_free"])

        res = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "experiments" / "megatron_ckpt_without_output_mfsdp_sim" / "run.py"),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("CHECK PASSED", res.stdout)


if __name__ == "__main__":
    unittest.main()
