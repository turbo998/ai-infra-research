"""Evaluate rack power density, liquid vs. air cooling thermodynamics, Slurm power_save token-bucket ramp rate limiting, Slurm RAPL/XCC energy accounting, and NVIDIA DCGM v4.6.1 thermal/power/IMEX/NVSwitch rack fault domains."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys

if __package__:
    from ..capacity_roofline.budget import integer
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.capacity_roofline.budget import integer


HERE = Path(__file__).resolve().parent
AIR_SPECIFIC_HEAT_J_PER_KG_K = 1006.0
AIR_DENSITY_KG_PER_M3 = 1.184
WATER_SPECIFIC_HEAT_J_PER_KG_K = 4184.0
WATER_DENSITY_KG_PER_L = 0.997
PG25_SPECIFIC_HEAT_J_PER_KG_K = 3950.0
PG25_DENSITY_KG_PER_L = 1.020
CFM_TO_M3_PER_S = 0.00047194745

LIMITATIONS = [
    "own-offline synthetic model; no live GPU rack, CDU coolant loop, BMC/XCC sensor, Slurm slurmctld/slurmd daemon, or NVIDIA DCGM hostengine execution",
    "reproduces static formulas and state machines from NVIDIA/DCGM@64df9f894541e426e416131a9820cae97aa4dd81 (DcgmHealthWatch.cpp, DcgmPolicyManager.cpp, dcgm_fields.h) and SchedMD/slurm@ad46cad8e0969e8e5945b88f0a21f6bed0f9a3bf (power_save.c, acct_gather_energy_rapl.c, acct_gather_energy_xcc.c, acct_gather_energy_pm_counters.c)",
    "steady-state 1D thermal resistance and coolant enthalpy balance; does not model 3D CFD recirculation, transient two-phase boiling, or chilled-water plant compressor curves",
]


def _ascii_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"{name}: expected non-empty ASCII alphanumeric/underscore/hyphen label")
    return value


def _pos_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name}: expected finite positive number")
    return float(value)


def _nonneg_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name}: expected finite non-negative number")
    return float(value)


def _nonneg_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: expected non-negative int")
    return value


def evaluate_rack_cooling_thermodynamics(
    *,
    num_gpus_per_rack: int,
    gpu_tdp_watts: float,
    num_cpus_per_rack: int,
    cpu_tdp_watts: float,
    num_nvswitches_per_rack: int,
    nvswitch_power_watts: float,
    nic_and_aux_power_per_rack_watts: float,
    psu_efficiency: float,
    liquid_capture_fraction: float,
    coolant_type: str,
    coolant_inlet_temp_c: float,
    coolant_delta_t_c: float,
    air_inlet_temp_c: float,
    air_delta_t_c: float,
    gpu_thermal_resistance_jc_plus_coldplate_k_per_w: float,
    gpu_max_operating_temp_c: float,
    gpu_slowdown_temp_c: float,
    gpu_shutdown_temp_c: float,
) -> dict:
    """Compute rack IT/wall power, liquid/air heat split, required LPM & CFM flow rates, and GPU junction temperature margin."""
    gpus = integer(num_gpus_per_rack, "num_gpus_per_rack")
    gpu_w = _pos_float(gpu_tdp_watts, "gpu_tdp_watts")
    cpus = _nonneg_int(num_cpus_per_rack, "num_cpus_per_rack")
    cpu_w = _nonneg_float(cpu_tdp_watts, "cpu_tdp_watts")
    switches = _nonneg_int(num_nvswitches_per_rack, "num_nvswitches_per_rack")
    switch_w = _nonneg_float(nvswitch_power_watts, "nvswitch_power_watts")
    aux_w = _nonneg_float(nic_and_aux_power_per_rack_watts, "nic_and_aux_power_per_rack_watts")
    eta_psu = _pos_float(psu_efficiency, "psu_efficiency")
    if eta_psu > 1.0:
        raise ValueError("psu_efficiency: must be in (0, 1]")
    f_liq = _nonneg_float(liquid_capture_fraction, "liquid_capture_fraction")
    if f_liq > 1.0:
        raise ValueError("liquid_capture_fraction: must be in [0, 1]")

    if coolant_type == "water":
        cp_liq = WATER_SPECIFIC_HEAT_J_PER_KG_K
        rho_liq = WATER_DENSITY_KG_PER_L
    elif coolant_type == "pg25":
        cp_liq = PG25_SPECIFIC_HEAT_J_PER_KG_K
        rho_liq = PG25_DENSITY_KG_PER_L
    else:
        raise ValueError("coolant_type: expected 'water' or 'pg25'")

    t_liq_in = _pos_float(coolant_inlet_temp_c, "coolant_inlet_temp_c")
    dt_liq = _pos_float(coolant_delta_t_c, "coolant_delta_t_c")
    t_air_in = _pos_float(air_inlet_temp_c, "air_inlet_temp_c")
    dt_air = _pos_float(air_delta_t_c, "air_delta_t_c")
    r_th = _pos_float(
        gpu_thermal_resistance_jc_plus_coldplate_k_per_w,
        "gpu_thermal_resistance_jc_plus_coldplate_k_per_w",
    )
    t_max_op = _pos_float(gpu_max_operating_temp_c, "gpu_max_operating_temp_c")
    t_slowdown = _pos_float(gpu_slowdown_temp_c, "gpu_slowdown_temp_c")
    t_shutdown = _pos_float(gpu_shutdown_temp_c, "gpu_shutdown_temp_c")
    if not (t_max_op <= t_slowdown < t_shutdown):
        raise ValueError("temperature thresholds must satisfy max_op <= slowdown < shutdown")

    it_payload_watts = gpus * gpu_w + cpus * cpu_w + switches * switch_w + aux_w
    rack_wall_watts = it_payload_watts / eta_psu
    psu_loss_watts = rack_wall_watts - it_payload_watts

    q_liquid_watts = rack_wall_watts * f_liq
    q_air_watts = rack_wall_watts * (1.0 - f_liq)

    # Q_liq = (LPM / 60) * rho_liq * cp_liq * dt_liq
    required_coolant_lpm = (q_liquid_watts * 60.0) / (rho_liq * cp_liq * dt_liq) if q_liquid_watts > 0.0 else 0.0
    lpm_per_10kw_liquid = (10_000.0 * 60.0) / (rho_liq * cp_liq * dt_liq)

    # Q_air = (CFM * CFM_TO_M3_PER_S) * rho_air * cp_air * dt_air
    required_airflow_m3_per_s = q_air_watts / (AIR_DENSITY_KG_PER_M3 * AIR_SPECIFIC_HEAT_J_PER_KG_K * dt_air) if q_air_watts > 0.0 else 0.0
    required_airflow_cfm = required_airflow_m3_per_s / CFM_TO_M3_PER_S

    # Local coolant or air temperature near GPU exit + thermal resistance rise
    sink_exit_temp_c = (t_liq_in + dt_liq) if f_liq > 0.0 else (t_air_in + dt_air)
    gpu_junction_temp_c = sink_exit_temp_c + gpu_w * r_th
    temp_margin_to_max_op_c = t_max_op - gpu_junction_temp_c
    temp_margin_to_slowdown_c = t_slowdown - gpu_junction_temp_c

    if gpu_junction_temp_c >= t_shutdown:
        clock_throttle_state = "HW_SHUTDOWN"
    elif gpu_junction_temp_c >= t_slowdown:
        clock_throttle_state = "DCGM_CLOCKS_EVENT_REASON_HW_THERMAL"
    elif gpu_junction_temp_c >= t_max_op:
        clock_throttle_state = "DCGM_CLOCKS_EVENT_REASON_SW_THERMAL"
    else:
        clock_throttle_state = "NOMINAL_NO_THERMAL_THROTTLE"

    return {
        "it_payload_kw": round(it_payload_watts / 1000.0, 4),
        "rack_wall_power_kw": round(rack_wall_watts / 1000.0, 4),
        "psu_loss_kw": round(psu_loss_watts / 1000.0, 4),
        "liquid_heat_captured_kw": round(q_liquid_watts / 1000.0, 4),
        "air_heat_rejected_kw": round(q_air_watts / 1000.0, 4),
        "required_coolant_lpm": round(required_coolant_lpm, 2),
        "lpm_per_10kw_liquid": round(lpm_per_10kw_liquid, 2),
        "required_airflow_cfm": round(required_airflow_cfm, 1),
        "sink_exit_temp_c": round(sink_exit_temp_c, 2),
        "gpu_junction_temp_c": round(gpu_junction_temp_c, 2),
        "temp_margin_to_max_op_c": round(temp_margin_to_max_op_c, 2),
        "temp_margin_to_slowdown_c": round(temp_margin_to_slowdown_c, 2),
        "clock_throttle_state": clock_throttle_state,
    }


def evaluate_slurm_power_save_and_energy(
    *,
    suspend_rate_per_min: int,
    resume_rate_per_min: int,
    power_save_interval_sec: int,
    power_save_min_interval_sec: int,
    nodes_to_wake: int,
    node_peak_power_kw: float,
    exc_node_cnt_target: int,
    active_nodes_in_pool: int,
    idle_nodes_in_pool: int,
    rapl_power_unit_raw_bits_12_8: int,
    rapl_is_dram_server_specific_unit: bool,
    rapl_counter_prev_raw: int,
    rapl_counter_curr_raw: int,
    xcc_sd650v2_poll_elapsed_sec: int,
    xcc_sd650v2_prev_watts: int,
    xcc_sd650v2_curr_watts: int,
    pm_counters_prev_joules: int,
    pm_counters_curr_joules: int,
) -> dict:
    """Reproduce Slurm power_save.c token-bucket ramp control, SuspendExcNodes partial count, and RAPL/XCC/pm_counters energy math."""
    s_rate = _nonneg_int(suspend_rate_per_min, "suspend_rate_per_min")
    r_rate = integer(resume_rate_per_min, "resume_rate_per_min")
    ps_int = integer(power_save_interval_sec, "power_save_interval_sec")
    ps_min_int = integer(power_save_min_interval_sec, "power_save_min_interval_sec")
    wake_nodes = integer(nodes_to_wake, "nodes_to_wake")
    node_kw = _pos_float(node_peak_power_kw, "node_peak_power_kw")
    exc_target = _nonneg_int(exc_node_cnt_target, "exc_node_cnt_target")
    active_cnt = _nonneg_int(active_nodes_in_pool, "active_nodes_in_pool")
    idle_cnt = _nonneg_int(idle_nodes_in_pool, "idle_nodes_in_pool")
    rapl_bits = _nonneg_int(rapl_power_unit_raw_bits_12_8, "rapl_power_unit_raw_bits_12_8")
    if rapl_bits > 31:
        raise ValueError("rapl_power_unit_raw_bits_12_8: must fit in 5 bits [0, 31]")
    if type(rapl_is_dram_server_specific_unit) is not bool:
        raise ValueError("rapl_is_dram_server_specific_unit: expected bool")
    rapl_prev = _nonneg_int(rapl_counter_prev_raw, "rapl_counter_prev_raw")
    rapl_curr = _nonneg_int(rapl_counter_curr_raw, "rapl_counter_curr_raw")
    if rapl_prev >= (1 << 32) or rapl_curr >= (1 << 32):
        raise ValueError("rapl 32-bit counter values must be in [0, 2^32 - 1]")
    xcc_elapsed = integer(xcc_sd650v2_poll_elapsed_sec, "xcc_sd650v2_poll_elapsed_sec")
    xcc_w0 = _nonneg_int(xcc_sd650v2_prev_watts, "xcc_sd650v2_prev_watts")
    xcc_w1 = _nonneg_int(xcc_sd650v2_curr_watts, "xcc_sd650v2_curr_watts")
    pm_j0 = _nonneg_int(pm_counters_prev_joules, "pm_counters_prev_joules")
    pm_j1 = _nonneg_int(pm_counters_curr_joules, "pm_counters_curr_joules")
    if pm_j1 < pm_j0:
        raise ValueError("pm_counters_curr_joules: must be >= pm_counters_prev_joules")

    # Slurm src/slurmctld/power_save.c: power_save_rl_setup (L1314-1369)
    max_sleep = max(1, ps_int, ps_min_int)
    resume_max_tokens = max(1, (r_rate * max_sleep) // 60)
    resume_refill_period_ms = 60000 // r_rate
    suspend_max_tokens = max(1, (s_rate * max_sleep) // 60) if s_rate > 0 else 0
    suspend_refill_period_ms = (60000 // s_rate) if s_rate > 0 else 0

    # Time to wake `wake_nodes`: initial burst uses `resume_max_tokens`, remaining tokens refill every `resume_refill_period_ms`
    if wake_nodes <= resume_max_tokens:
        wake_duration_sec = 0.0
    else:
        additional_tokens = wake_nodes - resume_max_tokens
        wake_duration_sec = (additional_tokens * resume_refill_period_ms) / 1000.0

    # Power ramp rate (kW/min) enforced by ResumeRate vs unthrottled step
    enforced_power_ramp_kw_per_min = r_rate * node_kw
    initial_burst_power_step_kw = min(wake_nodes, resume_max_tokens) * node_kw
    unthrottled_power_step_kw = wake_nodes * node_kw

    # Slurm _pick_exc_nodes (L370-449): keep (exc_node_cnt - active_count) idle nodes excluded from suspend
    needed_idle_excluded = max(0, exc_target - active_cnt)
    idle_nodes_kept_awake = min(idle_cnt, needed_idle_excluded)
    idle_nodes_eligible_for_suspend = idle_cnt - idle_nodes_kept_awake

    # Slurm acct_gather_energy_rapl.c (L171-323): 32-bit wraparound & DRAM 0.5^16 override
    rapl_wraparound_occurred = rapl_curr < rapl_prev
    rapl_delta_ticks = ((1 << 32) - rapl_prev + rapl_curr) if rapl_wraparound_occurred else (rapl_curr - rapl_prev)
    rapl_unit_joules = (0.5 ** 16) if rapl_is_dram_server_specific_unit else (0.5 ** rapl_bits)
    rapl_energy_joules = rapl_delta_ticks * rapl_unit_joules

    # Slurm acct_gather_energy_xcc.c (L603-605) vs acct_gather_energy_pm_counters.c (L126)
    xcc_trapezoidal_joules = round(float(xcc_elapsed) * 0.5 * float(xcc_w0 + xcc_w1))
    pm_exact_joules = pm_j1 - pm_j0
    xcc_vs_pm_error_joules = xcc_trapezoidal_joules - pm_exact_joules

    return {
        "resume_max_tokens": resume_max_tokens,
        "resume_refill_period_ms": resume_refill_period_ms,
        "suspend_max_tokens": suspend_max_tokens,
        "suspend_refill_period_ms": suspend_refill_period_ms,
        "wake_duration_sec": round(wake_duration_sec, 2),
        "initial_burst_power_step_kw": round(initial_burst_power_step_kw, 2),
        "enforced_power_ramp_kw_per_min": round(enforced_power_ramp_kw_per_min, 2),
        "unthrottled_power_step_kw": round(unthrottled_power_step_kw, 2),
        "idle_nodes_kept_awake": idle_nodes_kept_awake,
        "idle_nodes_eligible_for_suspend": idle_nodes_eligible_for_suspend,
        "rapl_wraparound_occurred": rapl_wraparound_occurred,
        "rapl_delta_ticks": rapl_delta_ticks,
        "rapl_unit_microjoules": round(rapl_unit_joules * 1e6, 6),
        "rapl_energy_joules": round(rapl_energy_joules, 4),
        "xcc_trapezoidal_joules": int(xcc_trapezoidal_joules),
        "pm_exact_joules": pm_exact_joules,
        "xcc_vs_pm_error_joules": int(xcc_vs_pm_error_joules),
    }


def evaluate_dcgm_rack_fault_domain(
    *,
    is_blackwell_or_newer: bool,
    thermal_violation_start_us: int,
    thermal_violation_end_us: int,
    power_violation_start_us: int,
    power_violation_end_us: int,
    cpu_temp_start_c: float,
    cpu_temp_end_c: float,
    cpu_temp_warning_c: float,
    cpu_temp_critical_c: float,
    cpu_power_watts: float,
    cpu_power_limit_watts: float,
    imex_domain_status: str,
    imex_daemon_status: int,
    nvswitch_non_fatal_errors: int,
    nvswitch_fatal_errors: int,
    nvswitch_link_down_count: int,
    fabric_manager_status: str,
    fabric_health_degraded_bw: bool,
    fabric_health_route_unhealthy: bool,
    nvlink_recovery_successful_delta: int,
    nvlink_recovery_failed_delta: int,
    gpu_recovery_action: str,
) -> dict:
    """Reproduce NVIDIA DCGM v4.6.1 DcgmHealthWatch.cpp health state rollup across thermal, power, Grace CPU, IMEX, NVSwitch, and NVLink."""
    if (
        type(is_blackwell_or_newer) is not bool
        or type(fabric_health_degraded_bw) is not bool
        or type(fabric_health_route_unhealthy) is not bool
    ):
        raise ValueError("boolean flags: expected bool")
    tv_start = _nonneg_int(thermal_violation_start_us, "thermal_violation_start_us")
    tv_end = _nonneg_int(thermal_violation_end_us, "thermal_violation_end_us")
    pv_start = _nonneg_int(power_violation_start_us, "power_violation_start_us")
    pv_end = _nonneg_int(power_violation_end_us, "power_violation_end_us")
    c_t0 = _pos_float(cpu_temp_start_c, "cpu_temp_start_c")
    c_t1 = _pos_float(cpu_temp_end_c, "cpu_temp_end_c")
    c_warn = _pos_float(cpu_temp_warning_c, "cpu_temp_warning_c")
    c_crit = _pos_float(cpu_temp_critical_c, "cpu_temp_critical_c")
    c_pwr = _nonneg_float(cpu_power_watts, "cpu_power_watts")
    c_lim = _pos_float(cpu_power_limit_watts, "cpu_power_limit_watts")
    sw_nonfatal = _nonneg_int(nvswitch_non_fatal_errors, "nvswitch_non_fatal_errors")
    sw_fatal = _nonneg_int(nvswitch_fatal_errors, "nvswitch_fatal_errors")
    sw_down = _nonneg_int(nvswitch_link_down_count, "nvswitch_link_down_count")
    rec_succ = _nonneg_int(nvlink_recovery_successful_delta, "nvlink_recovery_successful_delta")
    rec_fail = _nonneg_int(nvlink_recovery_failed_delta, "nvlink_recovery_failed_delta")
    if type(imex_daemon_status) is not int:
        raise ValueError("imex_daemon_status: expected int")

    # MonitorThermal (L2345-2415) & MonitorPower (L2418-2504)
    thermal_violation_delta_us = abs(tv_end - tv_start)
    gpu_thermal_health = "WARN" if thermal_violation_delta_us > 0 else "PASS"
    power_violation_delta_us = abs(pv_end - pv_start)
    gpu_power_health = "WARN" if power_violation_delta_us > 0 else "PASS"

    # MonitorCpuThermal (L2507-2595) & MonitorCpuPower (L2598-2639)
    cpu_avg_temp_c = 0.5 * (c_t0 + c_t1)
    if c_t1 >= c_crit:
        cpu_thermal_health = "FAIL"
    elif cpu_avg_temp_c >= c_warn:
        cpu_thermal_health = "WARN"
    else:
        cpu_thermal_health = "PASS"
    cpu_power_health = "FAIL" if c_pwr >= c_lim else "PASS"

    # MonitorImex (L2019-2108)
    valid_imex_domain = {"UP", "NOT_INSTALLED", "NOT_CONFIGURED", "UNAVAILABLE"}
    if imex_domain_status not in valid_imex_domain | {"DOWN", "DEGRADED"}:
        raise ValueError("imex_domain_status: unsupported status")
    imex_domain_ok = imex_domain_status in valid_imex_domain
    imex_daemon_ok = imex_daemon_status in {5, 7, -1, -2}
    imex_health = "PASS" if (imex_domain_ok and imex_daemon_ok) else "FAIL"

    # MonitorNvSwitchErrorCounts (L3026-3126) & MonitorFabricFields (L3599-3737)
    if sw_fatal > 0 or sw_down > 0:
        nvswitch_health = "FAIL"
    elif sw_nonfatal > 0:
        nvswitch_health = "WARN"
    else:
        nvswitch_health = "PASS"

    if fabric_manager_status in {"NotStarted", "Failure", "Unrecognized", "NvmlTooOld"} or fabric_health_degraded_bw or fabric_health_route_unhealthy:
        fabric_health = "FAIL"
    elif fabric_manager_status == "InProgress":
        fabric_health = "WARN"
    elif fabric_manager_status in {"Success", "NotSupported"}:
        fabric_health = "PASS"
    else:
        raise ValueError("fabric_manager_status: unsupported status")

    # MonitorNVLinkErrorFields (L3740-3808)
    if is_blackwell_or_newer:
        if rec_fail > 0:
            nvlink_error_health = "FAIL"
        elif rec_succ > 0:
            nvlink_error_health = "WARN"
        else:
            nvlink_error_health = "PASS"
    else:
        nvlink_error_health = "FAIL" if (rec_fail > 0 or rec_succ > 0) else "PASS"

    # MonitorGpuRecoveryAction (L2111-2190)
    if gpu_recovery_action == "NONE":
        recovery_action_health = "PASS"
    elif gpu_recovery_action == "DRAIN_P2P":
        recovery_action_health = "WARN"
    elif gpu_recovery_action in {"GPU_RESET", "NODE_REBOOT", "DRAIN_AND_RESET"}:
        recovery_action_health = "FAIL"
    else:
        raise ValueError("gpu_recovery_action: unsupported action")

    statuses = [
        gpu_thermal_health,
        gpu_power_health,
        cpu_thermal_health,
        cpu_power_health,
        imex_health,
        nvswitch_health,
        fabric_health,
        nvlink_error_health,
        recovery_action_health,
    ]
    if "FAIL" in statuses:
        overall_health = "FAIL"
    elif "WARN" in statuses:
        overall_health = "WARN"
    else:
        overall_health = "PASS"

    # Determine blast radius: MNNVL IMEX / NVSwitch / Fabric failures affect the multi-node NVLink rack clique
    mnnvl_rack_clique_compromised = "FAIL" in {imex_health, nvswitch_health, fabric_health}

    return {
        "gpu_thermal_health": gpu_thermal_health,
        "gpu_power_health": gpu_power_health,
        "cpu_thermal_health": cpu_thermal_health,
        "cpu_power_health": cpu_power_health,
        "imex_health": imex_health,
        "nvswitch_health": nvswitch_health,
        "fabric_health": fabric_health,
        "nvlink_error_health": nvlink_error_health,
        "recovery_action_health": recovery_action_health,
        "overall_health": overall_health,
        "mnnvl_rack_clique_compromised": mnnvl_rack_clique_compromised,
    }


def evaluate_innovator_rack_power_profile(
    *,
    architecture_kind: str,
    rack_wall_power_kw: float,
    cooling_medium: str,
    coolant_delta_t_c: float,
    air_delta_t_c: float,
    effective_batch1_tokens_per_sec_per_rack: float,
    effective_high_batch_tokens_per_sec_per_rack: float,
) -> dict:
    """Compare rack-level power density, cooling flow, and Joules/token across GPU NVL72, Groq air-cooled LPU racks, and Cerebras liquid-cooled WSE-3 CS-3."""
    kind = _ascii_label(architecture_kind, "architecture_kind")
    wall_kw = _pos_float(rack_wall_power_kw, "rack_wall_power_kw")
    dt_liq = _pos_float(coolant_delta_t_c, "coolant_delta_t_c")
    dt_air = _pos_float(air_delta_t_c, "air_delta_t_c")
    tps_b1 = _pos_float(effective_batch1_tokens_per_sec_per_rack, "effective_batch1_tokens_per_sec_per_rack")
    tps_hi = _pos_float(effective_high_batch_tokens_per_sec_per_rack, "effective_high_batch_tokens_per_sec_per_rack")

    wall_watts = wall_kw * 1000.0
    if cooling_medium == "liquid_water":
        coolant_lpm = (wall_watts * 60.0) / (WATER_DENSITY_KG_PER_L * WATER_SPECIFIC_HEAT_J_PER_KG_K * dt_liq)
        airflow_cfm = 0.0
    elif cooling_medium == "air":
        coolant_lpm = 0.0
        airflow_m3_s = wall_watts / (AIR_DENSITY_KG_PER_M3 * AIR_SPECIFIC_HEAT_J_PER_KG_K * dt_air)
        airflow_cfm = airflow_m3_s / CFM_TO_M3_PER_S
    else:
        raise ValueError("cooling_medium: expected 'liquid_water' or 'air'")

    joules_per_token_batch1 = wall_watts / tps_b1
    joules_per_token_high_batch = wall_watts / tps_hi

    return {
        "architecture_kind": kind,
        "rack_wall_power_kw": round(wall_kw, 3),
        "cooling_medium": cooling_medium,
        "required_coolant_lpm": round(coolant_lpm, 2),
        "required_airflow_cfm": round(airflow_cfm, 1),
        "joules_per_token_batch1": round(joules_per_token_batch1, 4),
        "joules_per_token_high_batch": round(joules_per_token_high_batch, 4),
    }


def evaluate_scenario(scenario: dict) -> dict:
    if not isinstance(scenario, dict):
        raise ValueError("scenario: expected object")
    sid = _ascii_label(scenario.get("id"), "scenario.id")
    kind = scenario.get("kind")

    if kind == "rack_cooling_thermodynamics":
        metrics = evaluate_rack_cooling_thermodynamics(
            num_gpus_per_rack=scenario.get("num_gpus_per_rack"),
            gpu_tdp_watts=scenario.get("gpu_tdp_watts"),
            num_cpus_per_rack=scenario.get("num_cpus_per_rack"),
            cpu_tdp_watts=scenario.get("cpu_tdp_watts"),
            num_nvswitches_per_rack=scenario.get("num_nvswitches_per_rack"),
            nvswitch_power_watts=scenario.get("nvswitch_power_watts"),
            nic_and_aux_power_per_rack_watts=scenario.get("nic_and_aux_power_per_rack_watts"),
            psu_efficiency=scenario.get("psu_efficiency"),
            liquid_capture_fraction=scenario.get("liquid_capture_fraction"),
            coolant_type=scenario.get("coolant_type"),
            coolant_inlet_temp_c=scenario.get("coolant_inlet_temp_c"),
            coolant_delta_t_c=scenario.get("coolant_delta_t_c"),
            air_inlet_temp_c=scenario.get("air_inlet_temp_c"),
            air_delta_t_c=scenario.get("air_delta_t_c"),
            gpu_thermal_resistance_jc_plus_coldplate_k_per_w=scenario.get("gpu_thermal_resistance_jc_plus_coldplate_k_per_w"),
            gpu_max_operating_temp_c=scenario.get("gpu_max_operating_temp_c"),
            gpu_slowdown_temp_c=scenario.get("gpu_slowdown_temp_c"),
            gpu_shutdown_temp_c=scenario.get("gpu_shutdown_temp_c"),
        )
    elif kind == "slurm_power_save_and_energy":
        metrics = evaluate_slurm_power_save_and_energy(
            suspend_rate_per_min=scenario.get("suspend_rate_per_min"),
            resume_rate_per_min=scenario.get("resume_rate_per_min"),
            power_save_interval_sec=scenario.get("power_save_interval_sec"),
            power_save_min_interval_sec=scenario.get("power_save_min_interval_sec"),
            nodes_to_wake=scenario.get("nodes_to_wake"),
            node_peak_power_kw=scenario.get("node_peak_power_kw"),
            exc_node_cnt_target=scenario.get("exc_node_cnt_target"),
            active_nodes_in_pool=scenario.get("active_nodes_in_pool"),
            idle_nodes_in_pool=scenario.get("idle_nodes_in_pool"),
            rapl_power_unit_raw_bits_12_8=scenario.get("rapl_power_unit_raw_bits_12_8"),
            rapl_is_dram_server_specific_unit=scenario.get("rapl_is_dram_server_specific_unit"),
            rapl_counter_prev_raw=scenario.get("rapl_counter_prev_raw"),
            rapl_counter_curr_raw=scenario.get("rapl_counter_curr_raw"),
            xcc_sd650v2_poll_elapsed_sec=scenario.get("xcc_sd650v2_poll_elapsed_sec"),
            xcc_sd650v2_prev_watts=scenario.get("xcc_sd650v2_prev_watts"),
            xcc_sd650v2_curr_watts=scenario.get("xcc_sd650v2_curr_watts"),
            pm_counters_prev_joules=scenario.get("pm_counters_prev_joules"),
            pm_counters_curr_joules=scenario.get("pm_counters_curr_joules"),
        )
    elif kind == "dcgm_rack_fault_domain":
        metrics = evaluate_dcgm_rack_fault_domain(
            is_blackwell_or_newer=scenario.get("is_blackwell_or_newer"),
            thermal_violation_start_us=scenario.get("thermal_violation_start_us"),
            thermal_violation_end_us=scenario.get("thermal_violation_end_us"),
            power_violation_start_us=scenario.get("power_violation_start_us"),
            power_violation_end_us=scenario.get("power_violation_end_us"),
            cpu_temp_start_c=scenario.get("cpu_temp_start_c"),
            cpu_temp_end_c=scenario.get("cpu_temp_end_c"),
            cpu_temp_warning_c=scenario.get("cpu_temp_warning_c"),
            cpu_temp_critical_c=scenario.get("cpu_temp_critical_c"),
            cpu_power_watts=scenario.get("cpu_power_watts"),
            cpu_power_limit_watts=scenario.get("cpu_power_limit_watts"),
            imex_domain_status=scenario.get("imex_domain_status"),
            imex_daemon_status=scenario.get("imex_daemon_status"),
            nvswitch_non_fatal_errors=scenario.get("nvswitch_non_fatal_errors"),
            nvswitch_fatal_errors=scenario.get("nvswitch_fatal_errors"),
            nvswitch_link_down_count=scenario.get("nvswitch_link_down_count"),
            fabric_manager_status=scenario.get("fabric_manager_status"),
            fabric_health_degraded_bw=scenario.get("fabric_health_degraded_bw"),
            fabric_health_route_unhealthy=scenario.get("fabric_health_route_unhealthy"),
            nvlink_recovery_successful_delta=scenario.get("nvlink_recovery_successful_delta"),
            nvlink_recovery_failed_delta=scenario.get("nvlink_recovery_failed_delta"),
            gpu_recovery_action=scenario.get("gpu_recovery_action"),
        )
    elif kind == "innovator_rack_power_profile":
        metrics = evaluate_innovator_rack_power_profile(
            architecture_kind=scenario.get("architecture_kind"),
            rack_wall_power_kw=scenario.get("rack_wall_power_kw"),
            cooling_medium=scenario.get("cooling_medium"),
            coolant_delta_t_c=scenario.get("coolant_delta_t_c"),
            air_delta_t_c=scenario.get("air_delta_t_c"),
            effective_batch1_tokens_per_sec_per_rack=scenario.get("effective_batch1_tokens_per_sec_per_rack"),
            effective_high_batch_tokens_per_sec_per_rack=scenario.get("effective_high_batch_tokens_per_sec_per_rack"),
        )
    else:
        raise ValueError(f"scenario.kind: unsupported kind {kind!r}")

    return {"id": sid, "kind": kind, **metrics}


def evaluate_suite(spec: dict) -> dict:
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("spec: expected schema_version 1")
    scenarios = spec.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("spec.scenarios: expected non-empty list")

    seen: set[str] = set()
    rows: list[dict] = []
    for item in scenarios:
        row = evaluate_scenario(item)
        if row["id"] in seen:
            raise ValueError(f"duplicate scenario id: {row['id']}")
        seen.add(row["id"])
        rows.append(row)

    return {
        "schema_version": 1,
        "evidence_status": "own-offline",
        "model": _ascii_label(spec.get("model", "rack_power_cooling_v1"), "spec.model"),
        "limitations": LIMITATIONS,
        "scenarios": rows,
    }


def render_csv(results: dict) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["id", "kind", "primary_metric_1", "primary_metric_2", "primary_metric_3"])
    for row in results["scenarios"]:
        kind = row["kind"]
        if kind == "rack_cooling_thermodynamics":
            writer.writerow([
                row["id"],
                kind,
                f"wall_kw={row['rack_wall_power_kw']}",
                f"lpm={row['required_coolant_lpm']}",
                f"throttle={row['clock_throttle_state']}",
            ])
        elif kind == "slurm_power_save_and_energy":
            writer.writerow([
                row["id"],
                kind,
                f"tokens={row['resume_max_tokens']}",
                f"wake_s={row['wake_duration_sec']}",
                f"rapl_j={row['rapl_energy_joules']}",
            ])
        elif kind == "dcgm_rack_fault_domain":
            writer.writerow([
                row["id"],
                kind,
                f"overall={row['overall_health']}",
                f"imex={row['imex_health']}",
                f"clique_compromised={row['mnnvl_rack_clique_compromised']}",
            ])
        else:
            writer.writerow([
                row["id"],
                kind,
                f"wall_kw={row['rack_wall_power_kw']}",
                f"j_per_tok_b1={row['joules_per_token_batch1']}",
                f"j_per_tok_hi={row['joules_per_token_high_batch']}",
            ])
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "input.json")
    parser.add_argument("--json-out", type=Path, default=HERE / "results.json")
    parser.add_argument("--csv-out", type=Path, default=HERE / "results.csv")
    parser.add_argument("--check", action="store_true", help="Verify committed outputs match")
    args = parser.parse_args(argv)

    spec = json.loads(args.input.read_text(encoding="utf-8"))
    results = evaluate_suite(spec)
    json_text = json.dumps(results, indent=2, ensure_ascii=True) + "\n"
    csv_text = render_csv(results)

    if args.check:
        if not args.json_out.is_file() or not args.csv_out.is_file():
            return 1
        if args.json_out.read_text(encoding="utf-8") != json_text:
            return 1
        if args.csv_out.read_text(encoding="utf-8") != csv_text:
            return 1
        return 0

    args.json_out.write_text(json_text, encoding="utf-8", newline="\n")
    args.csv_out.write_text(csv_text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
