# Experiment #595: Milestone Enterprise Post-Training (Full SFT vs. LoRA vs. GRPO) & Shared Cluster Sizer

Reproduces the `88.0%` VRAM reduction of `70B` `LoRA SFT` (`156 GB` on `2× H200` single node) vs. `Full-Parameter SFT` (`1,300 GB` on `16× H100` across 2 RDMA nodes) and demonstrates how `veRL GRPO` (`520 GB`, `82%` vLLM rollout time) runs overnight on borrowed inference nodes (`docs/components/enterprise-post-training-sft-lora-grpo-shared-cluster-sizing-faq.md`).
