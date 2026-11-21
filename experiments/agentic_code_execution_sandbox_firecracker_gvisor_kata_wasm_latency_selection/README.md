# Agentic Code Execution Sandbox Selection Experiment (`Firecracker MicroVM` vs. `gVisor` vs. `Kata` vs. `Wasm`)

Reproducible capacity and latency model comparing 4 sandbox isolation architectures for untrusted LLM-generated code execution (`500 executions/sec`, pre-loaded `Python + Pandas/NumPy/Playwright` runtime) across cold/snapshot startup latency, unique RSS memory per sandbox (`CoW` page sharing), host consolidation density, security compliance, and monthly TCO.

## Run
```bash
python experiments/agentic_code_execution_sandbox_firecracker_gvisor_kata_wasm_latency_selection/run.py
python experiments/agentic_code_execution_sandbox_firecracker_gvisor_kata_wasm_latency_selection/run.py --check
```
