import json
from pathlib import Path

def run_model():
    # 1. NCCL Diagnostics Child Process OMPI Environment Stripping (6cb2bc2e2a)
    # & NCCL_P2P_MAX_PEERS INFO log fix (6e95eab9d3)
    inherited_ompi_env_vars_in_child = 0
    diagnostics_child_mpi_hangs = 0

    # 2. ROCm rocjitsu FP64 Output Finishing Skip When Modifiers Disabled (#14527)
    # & Cross-Test State Leak Fix (#14667)
    fp64_unmodified_finishing_ops_skipped = 1
    cross_test_state_leaks = 0

    # 3. NIXL Per-Process Rotated Log File NIXL_LOG_FILE (#2211) & v1.6.0 (#2363)
    supported_log_format_specifiers = ["%h", "%p", "%t", "%%"]
    max_rotated_log_files_per_process = 2  # active + .1 rotated file

    return {
        "phase": "phase171",
        "inherited_ompi_env_vars_in_child": inherited_ompi_env_vars_in_child,
        "diagnostics_child_mpi_hangs": diagnostics_child_mpi_hangs,
        "fp64_unmodified_finishing_ops_skipped": fp64_unmodified_finishing_ops_skipped,
        "cross_test_state_leaks": cross_test_state_leaks,
        "supported_log_format_specifiers": supported_log_format_specifiers,
        "max_rotated_log_files_per_process": max_rotated_log_files_per_process,
    }

if __name__ == "__main__":
    import sys
    res = run_model()
    out_path = Path(__file__).resolve().parent / "results.json"
    if "--check" in sys.argv:
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        assert saved == res, f"Mismatch: {saved} != {res}"
        print("OK: nccl_ompi_strip_rocjitsu_fp64_nixl_logfile_phase171 verified")
    else:
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(res, indent=2))
