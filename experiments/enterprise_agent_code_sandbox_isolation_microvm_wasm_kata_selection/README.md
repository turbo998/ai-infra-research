# Enterprise AI Agent Code Execution (`Code Interpreter`) & Browser Sandbox Isolation Selection (`R475`)

Deterministic evaluation suite comparing four enterprise AI Agent code execution and headless browser sandbox isolation architectures across `1,200,000` monthly sandbox sessions (`800` concurrent active sandboxes, `75%` requiring native Python C-extensions like `pandas/scipy/openpyxl/playwright`):

1. **Tier 1**: Standard Shared-Kernel Docker Containers (`runc`) or In-Process `exec()` (`18.5%` host kernel / lateral network escape risk, `100.0%` credential exposure via `os.environ`, `$162,000/mo` risk + infra TCO)
2. **Tier 2**: Pure WebAssembly (`Wasm / Pyodide`) User-Space Sandbox (`0.2%` escape risk, `58.0%` native C-extension compatibility, `45.0ms` cold start, `$74,000/mo` TCO)
3. **Tier 3**: Cold-Booted Dedicated Cloud VMs per Session (`0.0%` escape risk, `28,500ms` cold start, `45.0%` credential exposure, `$138,000/mo` TCO)
4. **Tier 4**: Hardware-Isolated MicroVMs (`Azure Container Apps Dynamic Sessions / E2B Firecracker / Kata Containers`) + Pre-Warmed Memory Snapshots + Egress Proxy Credential Injection (`0.0%` escape risk, `100.0%` native compatibility, `135.0ms` P95 startup, `0.0%` credential exposure, `$34,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_agent_code_sandbox_isolation_microvm_wasm_kata_selection/run.py
python experiments/enterprise_agent_code_sandbox_isolation_microvm_wasm_kata_selection/run.py --check
python -m unittest tests/test_enterprise_agent_code_sandbox_isolation_microvm_wasm_kata_selection.py
```
