# Experiment #593: Enterprise K8s GPU Scheduling (Hardware MIG vs. MPS vs. Time-Slicing & Kueue) Sizer

Reproduces the `57.9%` GPU fleet reduction (`16× H100` vs. `38× H100`, saving `$50,600/mo`) when packing `28` small production microservices onto `4× H100` (`7× 1g.10gb` Hardware MIG slices per GPU) and `14` dev notebooks onto `2× H100` (Time-Slicing) alongside `10× H100` full-GPU LLM pools (`docs/components/enterprise-k8s-gpu-scheduling-mig-mps-kueue-dra-sizing-faq.md`).
