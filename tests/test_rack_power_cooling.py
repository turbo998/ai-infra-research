"""Unit tests for experiments/rack_power_cooling/run.py."""

import json
from pathlib import Path
import unittest

from experiments.rack_power_cooling.run import (
    HERE,
    evaluate_dcgm_rack_fault_domain,
    evaluate_innovator_rack_power_profile,
    evaluate_rack_cooling_thermodynamics,
    evaluate_slurm_power_save_and_energy,
    evaluate_suite,
    main,
)


class RackPowerCoolingTest(unittest.TestCase):
    def test_committed_artifacts_match_check(self) -> None:
        self.assertEqual(main(["--check"]), 0)

    def test_dlc_vs_degraded_coolant_thermodynamics(self) -> None:
        nom = evaluate_rack_cooling_thermodynamics(
            num_gpus_per_rack=72,
            gpu_tdp_watts=1200.0,
            num_cpus_per_rack=36,
            cpu_tdp_watts=300.0,
            num_nvswitches_per_rack=18,
            nvswitch_power_watts=400.0,
            nic_and_aux_power_per_rack_watts=14400.0,
            psu_efficiency=0.94,
            liquid_capture_fraction=0.88,
            coolant_type="water",
            coolant_inlet_temp_c=32.0,
            coolant_delta_t_c=12.0,
            air_inlet_temp_c=24.0,
            air_delta_t_c=15.0,
            gpu_thermal_resistance_jc_plus_coldplate_k_per_w=0.028,
            gpu_max_operating_temp_c=83.0,
            gpu_slowdown_temp_c=87.0,
            gpu_shutdown_temp_c=92.0,
        )
        self.assertAlmostEqual(nom["it_payload_kw"], 118.8)
        self.assertAlmostEqual(nom["rack_wall_power_kw"], 126.383, places=3)
        self.assertEqual(nom["clock_throttle_state"], "NOMINAL_NO_THERMAL_THROTTLE")
        self.assertGreater(nom["temp_margin_to_max_op_c"], 5.0)

        deg = evaluate_rack_cooling_thermodynamics(
            num_gpus_per_rack=72,
            gpu_tdp_watts=1200.0,
            num_cpus_per_rack=36,
            cpu_tdp_watts=300.0,
            num_nvswitches_per_rack=18,
            nvswitch_power_watts=400.0,
            nic_and_aux_power_per_rack_watts=14400.0,
            psu_efficiency=0.94,
            liquid_capture_fraction=0.88,
            coolant_type="pg25",
            coolant_inlet_temp_c=45.0,
            coolant_delta_t_c=10.0,
            air_inlet_temp_c=27.0,
            air_delta_t_c=15.0,
            gpu_thermal_resistance_jc_plus_coldplate_k_per_w=0.028,
            gpu_max_operating_temp_c=83.0,
            gpu_slowdown_temp_c=87.0,
            gpu_shutdown_temp_c=92.0,
        )
        self.assertEqual(deg["clock_throttle_state"], "DCGM_CLOCKS_EVENT_REASON_HW_THERMAL")
        self.assertGreater(deg["required_coolant_lpm"], nom["required_coolant_lpm"])

    def test_slurm_token_bucket_and_rapl_wraparound(self) -> None:
        res = evaluate_slurm_power_save_and_energy(
            suspend_rate_per_min=30,
            resume_rate_per_min=12,
            power_save_interval_sec=15,
            power_save_min_interval_sec=5,
            nodes_to_wake=64,
            node_peak_power_kw=10.2,
            exc_node_cnt_target=16,
            active_nodes_in_pool=10,
            idle_nodes_in_pool=20,
            rapl_power_unit_raw_bits_12_8=14,
            rapl_is_dram_server_specific_unit=True,
            rapl_counter_prev_raw=4290000000,
            rapl_counter_curr_raw=12000000,
            xcc_sd650v2_poll_elapsed_sec=30,
            xcc_sd650v2_prev_watts=2200,
            xcc_sd650v2_curr_watts=9800,
            pm_counters_prev_joules=1000000,
            pm_counters_curr_joules=1242000,
        )
        self.assertEqual(res["resume_max_tokens"], 3)
        self.assertEqual(res["resume_refill_period_ms"], 5000)
        self.assertAlmostEqual(res["wake_duration_sec"], 305.0)
        self.assertEqual(res["idle_nodes_kept_awake"], 6)
        self.assertEqual(res["idle_nodes_eligible_for_suspend"], 14)
        self.assertTrue(res["rapl_wraparound_occurred"])
        self.assertEqual(res["rapl_delta_ticks"], (1 << 32) - 4290000000 + 12000000)
        self.assertAlmostEqual(res["rapl_unit_microjoules"], (0.5 ** 16) * 1e6, places=5)
        self.assertEqual(res["xcc_trapezoidal_joules"], 180000)
        self.assertEqual(res["pm_exact_joules"], 242000)
        self.assertEqual(res["xcc_vs_pm_error_joules"], -62000)

    def test_dcgm_blackwell_vs_pre_blackwell_and_imex_faults(self) -> None:
        bw_warn = evaluate_dcgm_rack_fault_domain(
            is_blackwell_or_newer=True,
            thermal_violation_start_us=100,
            thermal_violation_end_us=100,
            power_violation_start_us=100,
            power_violation_end_us=100,
            cpu_temp_start_c=70.0,
            cpu_temp_end_c=72.0,
            cpu_temp_warning_c=85.0,
            cpu_temp_critical_c=95.0,
            cpu_power_watts=200.0,
            cpu_power_limit_watts=300.0,
            imex_domain_status="UP",
            imex_daemon_status=5,
            nvswitch_non_fatal_errors=0,
            nvswitch_fatal_errors=0,
            nvswitch_link_down_count=0,
            fabric_manager_status="Success",
            fabric_health_degraded_bw=False,
            fabric_health_route_unhealthy=False,
            nvlink_recovery_successful_delta=2,
            nvlink_recovery_failed_delta=0,
            gpu_recovery_action="NONE",
        )
        self.assertEqual(bw_warn["nvlink_error_health"], "WARN")
        self.assertEqual(bw_warn["overall_health"], "WARN")
        self.assertFalse(bw_warn["mnnvl_rack_clique_compromised"])

        pre_bw_fail = evaluate_dcgm_rack_fault_domain(
            is_blackwell_or_newer=False,
            thermal_violation_start_us=100,
            thermal_violation_end_us=100,
            power_violation_start_us=100,
            power_violation_end_us=100,
            cpu_temp_start_c=70.0,
            cpu_temp_end_c=72.0,
            cpu_temp_warning_c=85.0,
            cpu_temp_critical_c=95.0,
            cpu_power_watts=200.0,
            cpu_power_limit_watts=300.0,
            imex_domain_status="UP",
            imex_daemon_status=5,
            nvswitch_non_fatal_errors=0,
            nvswitch_fatal_errors=0,
            nvswitch_link_down_count=0,
            fabric_manager_status="Success",
            fabric_health_degraded_bw=False,
            fabric_health_route_unhealthy=False,
            nvlink_recovery_successful_delta=2,
            nvlink_recovery_failed_delta=0,
            gpu_recovery_action="NONE",
        )
        self.assertEqual(pre_bw_fail["nvlink_error_health"], "FAIL")
        self.assertEqual(pre_bw_fail["overall_health"], "FAIL")

    def test_innovator_rack_power_profile(self) -> None:
        groq = evaluate_innovator_rack_power_profile(
            architecture_kind="groq_lpu_v1_air_rack",
            rack_wall_power_kw=32.0,
            cooling_medium="air",
            coolant_delta_t_c=12.0,
            air_delta_t_c=15.0,
            effective_batch1_tokens_per_sec_per_rack=2400.0,
            effective_high_batch_tokens_per_sec_per_rack=9600.0,
        )
        self.assertEqual(groq["required_coolant_lpm"], 0.0)
        self.assertGreater(groq["required_airflow_cfm"], 3700.0)
        self.assertAlmostEqual(groq["joules_per_token_batch1"], 13.3333, places=4)

    def test_validation_rejects_bad_inputs(self) -> None:
        spec = json.loads((HERE / "input.json").read_text(encoding="utf-8"))
        spec["scenarios"].append(dict(spec["scenarios"][0]))
        with self.assertRaises(ValueError):
            evaluate_suite(spec)


if __name__ == "__main__":
    unittest.main()
