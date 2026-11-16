from __future__ import annotations

import json
from pathlib import Path


def evaluate_checkpoint_reentrant_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Legacy checkpoint(use_reentrant=True) on Frozen-Embedding LoRA",
            "lora_gradient_survival_pct": 0.0,
            "recompute_step_overhead_pct": 33.0,
            "activation_hbm_gb": 12.40,
            "torch_compile_compatible": False,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: use_reentrant=True + enable_input_require_grads() Workaround (Full Recompute)",
            "lora_gradient_survival_pct": 100.0,
            "recompute_step_overhead_pct": 35.8,
            "activation_hbm_gb": 13.10,
            "torch_compile_compatible": False,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: use_reentrant=False (Full-Layer Activation Checkpointing)",
            "lora_gradient_survival_pct": 100.0,
            "recompute_step_overhead_pct": 31.2,
            "activation_hbm_gb": 12.40,
            "torch_compile_compatible": True,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: use_reentrant=False + Op-Level Selective Checkpoint (SAC: Save GEMMs, Recompute Attn/MLP-Act)",
            "lora_gradient_survival_pct": 100.0,
            "recompute_step_overhead_pct": 4.8,
            "activation_hbm_gb": 15.80,
            "torch_compile_compatible": True,
            "decision": "Act automatically",
        },
    ]

    failure_modes = [
        {
            "scenario": "LoRA / QLoRA with Frozen `embed_tokens` (`hidden_states.requires_grad == False`)",
            "reentrant_true": "Drops 100% of LoRA grads (`grad is None` / `does not have a grad_fn`)",
            "reentrant_false": "Tracks internal `lora_A/B` edges cleanly via `saved_tensors_hooks`",
        },
        {
            "scenario": "VLM with Frozen Vision Encoder Feeding trainable Projector + LLM",
            "reentrant_true": "Silently skips Projector & early LLM layer gradients",
            "reentrant_false": "100% gradient propagation without dummy `requires_grad_(True)`",
        },
        {
            "scenario": "DDP / FSDP with Shared or Tied Parameters Across Checkpointed Blocks",
            "reentrant_true": "Crashes: `Expected to mark a variable ready only once`",
            "reentrant_false": "Single autograd graph pass; zero duplicate hook firing",
        },
        {
            "scenario": "Wrapping Checkpointed Transformer Block in `torch.compile`",
            "reentrant_true": "Forces Dynamo graph breaks or nested autograd fallback",
            "reentrant_false": "Compiles seamlessly into fused AOTAutograd joint graph",
        },
    ]

    summary = {
        "benchmark": "pytorch_checkpoint_use_reentrant_true_lora_frozen_base_grad_drop_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "reentrant_true_lora_grad_survival_pct": 0.0,
        "reentrant_false_lora_grad_survival_pct": 100.0,
        "full_recompute_overhead_pct": 35.8,
        "selective_sac_overhead_pct": 4.8,
        "overhead_reduction_x": round(35.8 / 4.8, 1),
        "no_checkpoint_activation_gb": 56.40,
        "tier4_sac_activation_gb": 15.80,
        "activation_saved_pct": round((1.0 - 15.80 / 56.40) * 100.0, 1),
        "tiers": tiers,
        "failure_modes": failure_modes,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_checkpoint_reentrant_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
