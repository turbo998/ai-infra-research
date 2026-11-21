# Agentic Sandbox Egress Firewall & Secret Broker Selection Experiment (`Out-of-Sandbox L7 Secret Broker` vs. `Raw Env Vars & L3 Allowlists`)

Reproducible analytical model comparing 4 agentic sandbox network egress and credential isolation architectures against indirect prompt injection exfiltration (`Markdown image leak`, `DNS tunneling`, `print(os.environ)`, and `allowed-domain POST exfiltration`):
- **Tier 1 (`Unrestricted Outbound Egress + Raw Secrets in os.environ`)**: `1.5%` exfiltration block rate, `100.0%` raw credential theft risk.
- **Tier 2 (`L3/L4 IP & DNS Allowlist + Raw Secrets in os.environ`)**: `58.0%` exfiltration block rate (fails on allowed-domain exfiltration via `api.github.com` / `s3`), `100.0%` raw credential exposure inside the sandbox.
- **Tier 3 (`L7 Forward Proxy Path/Method ACL + Short-Lived Scoped Tokens`)**: `91.5%` block rate, `25.0%` token theft risk, `+1.4 ms` latency.
- **Tier 4 (`Out-of-Sandbox L7 Secret Broker + Path/Method ACL + CSP img-src Sinkhole`, Recommended)**: **`99.8%` exfiltration block rate**, **`0.0%` raw credential theft risk** (sandbox never holds real API keys!), and **`+1.8 ms` proxy overhead**.

## Run & Verify

```bash
python experiments/agentic_sandbox_egress_firewall_secret_broker_exfiltration_prevention_selection/run.py
python experiments/agentic_sandbox_egress_firewall_secret_broker_exfiltration_prevention_selection/run.py --check
```
