# Experiment #586: Enterprise Multi-LoRA Serving (S-LoRA/vLLM) vs. 16K Few-Shot ICL & Dedicated Pods

Reproduces the `268.4 MB` `r=32` LoRA adapter footprint on `70B`, `4× H200` (`$13,600/mo`) sizing for `100` tenants at `60 QPS`, and `97.0%` / `60.0%` TCO savings vs. dedicated pods and `16K` few-shot ICL (`docs/components/enterprise-multi-lora-serving-slora-vs-icl-dedicated-sizing-faq.md`).
