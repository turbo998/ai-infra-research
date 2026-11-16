from __future__ import annotations

import json
from pathlib import Path


def evaluate_fim_token_healing_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Naive PSM Format Without Token Healing (Split BPE Boundary)",
            "mid_identifier_exact_match_pct": 57.8,
            "gpu_prefix_cache_hit_pct": 42.0,
            "p99_completion_ttft_ms": 48.5,
            "healing_cpu_overhead_ms": 0.00,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: SPM Format Without Token Healing (Still Fails on 'calculate_tot')",
            "mid_identifier_exact_match_pct": 60.4,
            "gpu_prefix_cache_hit_pct": 91.4,
            "p99_completion_ttft_ms": 11.3,
            "healing_cpu_overhead_ms": 0.00,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: PSM Format + Heuristic Whitespace Backtracking Only",
            "mid_identifier_exact_match_pct": 66.5,
            "gpu_prefix_cache_hit_pct": 42.0,
            "p99_completion_ttft_ms": 48.8,
            "healing_cpu_overhead_ms": 0.15,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: SPM Format + 1-Token BPE Backtrack + Marisa-Trie Prefix Bitmask (Token Healing)",
            "mid_identifier_exact_match_pct": 76.2,
            "gpu_prefix_cache_hit_pct": 91.4,
            "p99_completion_ttft_ms": 11.4,
            "healing_cpu_overhead_ms": 0.08,
            "decision": "Act automatically",
        },
    ]

    boundary_cases = [
        {
            "prompt_tail": "def calculate_tot",
            "expected": "al_cost(items):",
            "unhealed_output": "(items):  # Truncated identifier!",
            "healed_output": "al_cost(items):  # Sampled '_total' & stripped '_tot'",
        },
        {
            "prompt_tail": "url = 'https://",
            "expected": "github.com/vllm-project'",
            "unhealed_output": "/github.com'  # Double slash bug (':' + '//' split)!",
            "healed_output": "github.com/vllm-project'  # Backtracked '://' -> '://github'",
        },
        {
            "prompt_tail": "import torch.nn.func",
            "expected": "tional as F",
            "unhealed_output": " as F  # Refuses to complete 'functional'!",
            "healed_output": "tional as F  # Sampled 'functional' & stripped 'func'",
        },
        {
            "prompt_tail": "return np.lin",
            "expected": "alg.norm(x)",
            "unhealed_output": "space(0, 1)  # Biased away from 'linalg'",
            "healed_output": "alg.norm(x)  # Restores true training distribution",
        },
    ]

    summary = {
        "benchmark": "llm_code_fim_token_healing_bpe_prefix_boundary_backtracking_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "unhealed_accuracy_pct": 57.8,
        "healed_accuracy_pct": 76.2,
        "accuracy_gain_pt": round(76.2 - 57.8, 1),
        "psm_cache_hit_pct": 42.0,
        "spm_cache_hit_pct": 91.4,
        "psm_ttft_ms": 48.5,
        "spm_ttft_ms": 11.4,
        "ttft_speedup_x": round(48.5 / 11.4, 2),
        "tiers": tiers,
        "boundary_cases": boundary_cases,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_fim_token_healing_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
