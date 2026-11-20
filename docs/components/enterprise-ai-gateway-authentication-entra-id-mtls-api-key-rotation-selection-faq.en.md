# Enterprise AI Gateway Zero-Trust Authentication, Secretless Workload Identity (`Entra Workload ID / AWS IRSA`), `0.3ms` Local `JWKS` Verification, `SPIFFE mTLS` & Dual-Slot Key Rotation Selection Guide: Eliminating Leaked `sk-...` Keys and `HTTP 401` Rotation Outages

> **Language / 语言**: **English (Current)** | [中文](./enterprise-ai-gateway-authentication-entra-id-mtls-api-key-rotation-selection-faq.md)  
> **Scope**: End-to-end zero-trust security reference architecture for Cloud & AI Solution Architects, CISOs, and Platform Security Engineers governing authentication, sender-constrained tokens (`RFC 8705 mTLS`), and zero-downtime credential rotation across `Azure APIM`, `Envoy AI Gateway`, `Kong`, and `LiteLLM`.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_ai_gateway_authentication_entra_id_mtls_api_key_rotation_selection/results.json`](../../experiments/enterprise_ai_gateway_authentication_entra_id_mtls_api_key_rotation_selection/results.json).

![Enterprise AI Gateway Zero-Trust Auth, Secretless OIDC & Key Rotation Summary](../../assets/svg/enterprise-ai-gateway-authentication-entra-id-mtls-api-key-rotation-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When **85 internal microservices, data science notebooks, and partner applications** connect to an enterprise AI Gateway, **credential sprawl and key-rotation outages** become a top security liability. Three flawed authentication patterns dominate early enterprise deployments:
1. **Trap #1: Static Non-Expiring `sk-...` API Keys Scattered in `.env` Files & Notebooks (Tier 1, `168-Hour` Leak Exposure Window, `18.5%` Rotation Outage Rate, `$175,000/month` Security/Leak TCO)**: A single static key accidentally committed to GitHub or left in a shared notebook gives attackers unrestricted weekend access to your `PTU / PAYGO` quota;
2. **Trap #2: Single-Slot Static Key Overwrites Every 90 Days (Tier 2, `11.2%` `HTTP 401` Service Outage Rate on Rotation Day)**: Overwriting the single active secret in Key Vault immediately invalidates the old key while running pods still hold the cached value in memory—crashing 11.2% of live traffic with `HTTP 401 Unauthorized`!
3. **Trap #3: Synchronous Remote OAuth2 Token Introspection on Every LLM Turn (Tier 3, `+88.0ms` P99 Latency Overhead)**: Calling the identity provider's `/oauth2/introspect` endpoint synchronously on every chat request adds 88ms of network latency and turns IdP rate limits into AI outages!

Evaluated across **85 enterprise services calling a shared AI Gateway**, the four authentication architectures compare as follows:

| Authentication Tier | Identity, Verification & Rotation Stack | Leak Blast Radius | Rotation `401` Outage Rate | Verify `P99` Overhead | Caller Non-Repudiation | Monthly Security & Leak TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Static `sk-...` Keys in `.env` & Notebooks** | Non-expiring static keys shared across repos & scripts | `168.0 hrs (7d)` ✕ *(weekend drain)* | `18.5%` ✕ *(manual config drift)* | `0.2 ms` | `12.0%` ✕ *(shared keys)* | `$175,000 / mo` ✕ *(leaks + toil)* | ⛔ Unacceptable credential leak & quota-theft risk |
| **🟨 Tier 2: Vault Static Keys + Single-Slot Overwrite** | Stores static key in Vault; overwrites slot every 90 days | `24.0 hrs` | `11.2%` ✕ *(instant 401 cliff)* | `1.5 ms` | `65.0%` | `$96,000 / mo` | ⛔ Single-slot overwrites break in-flight pods on rotation day |
| **🟦 Tier 3: OAuth2 JWT + Sync Remote `/introspect`** | Calls remote IdP introspection endpoint on every LLM call | `1.0 hr` | `0.8%` | **`88.0 ms` ✕** *(network hop)* | `98.0%` | `$62,000 / mo` | ⚠️ High latency & runtime coupling to IdP availability |
| **🟩 Tier 4: Secretless Workload `OIDC` (`15m JWT`) + `0.3ms` Local `JWKS` + `SPIFFE mTLS` + Dual-Slot Rotation** | **`disableLocalAuth: true` upstream + K8s OIDC `15m` JWT bound to `SPIFFE mTLS` (`0.3ms` local verify) + Dual-Slot overlap** | **`0.25 hrs (15m)` 🏆** *(mTLS sender-bound)* | **`0.0%` 🏆** *(24h dual-slot drain)* | **`0.3 ms` 🏆** *(in-memory JWKS)* | **`100.0%` 🏆** *(per-pod/user)* | **`$18,000 / mo` 🏆** *(-89.7% vs Tier 1)* | 🌟 **Recommended Enterprise Zero-Trust Auth Standard** |

> 💡 **Golden Rules for Enterprise AI Gateway Zero-Trust Security**:
> 1. **Enforce `disableLocalAuth: true` Upstream & 100% Secretless Workload Identity Downstream**: Disable static API keys on cloud LLM endpoints (`Azure OpenAI / Foundry`) so only the AI Gateway's **Managed Identity** can call them. Require internal microservices to authenticate via **Kubernetes Federated Workload Identity (`Entra Workload ID / AWS IRSA`)** issuing `15-minute` OIDC JWTs—zero static `sk-...` strings exist anywhere in code!
> 2. **Verify Asymmetric JWTs Locally in `0.3ms` (`Cached JWKS + RFC 8705 SPIFFE mTLS`)**: Cache the IdP's public `JWKS` in gateway memory for `0.3ms` signature verification, and bind tokens to the caller's **`SPIFFE mTLS` client certificate thumbprint (`cnf.x5t#S256`)** so leaked tokens cannot be replayed from outside the pod.
> 3. **Use a Dual-Slot (`Primary + Secondary`) Overlapping State Machine for External Clients**: Never overwrite an active key slot; maintain a 24-hour dual-key overlap and revoke the old slot only after telemetry confirms `0` traffic (`0.0%` rotation outage).

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Secretless Codebase, Zero-401 Rotation & Token Replay Defense Pain Points)

### Q1: How do 85 internal microservices authenticate to the AI Gateway and upstream Cloud LLMs with zero API keys in `.env` files or Key Vault?
- **The 3-Hop Secretless Federation Chain (`K8s Projected Token -> 15m OIDC JWT -> Managed Identity`)**:
  1. **Hop 1 (Pod Boot)**: Kubernetes mounts an ephemeral projected ServiceAccount token into the pod tmpfs (auto-rotated by kubelet);
  2. **Hop 2 (Client-to-Gateway)**: The application SDK (`DefaultAzureCredential / AWS STS`) exchanges the projected token for a **`15-minute` OIDC JWT** (`aud = api://ai-gateway`) and refreshes it in the background at `80%` TTL (`12 minutes`);
  3. **Hop 3 (Gateway-to-Cloud LLM)**: The AI Gateway validates the caller's JWT in `0.3ms` and calls `Azure OpenAI / Bedrock` using its **System-Assigned Managed Identity** (`disableLocalAuth: true`)—eliminating static keys end-to-end!

### Q2: How does the Dual-Slot (`Primary + Secondary`) state machine guarantee `0.0%` `HTTP 401` errors during credential rotation for legacy external clients?
- **4-Step Overlapping Drain Protocol**:
  - Instead of a single key field, every external client registration holds two active slots: `Slot_1 (Primary)` and `Slot_2 (Secondary)`.
  - On rotation day, the controller generates a new key in the idle `Slot_2` while leaving `Slot_1` **100% valid**. Downstream pods pick up `Slot_2` asynchronously over a 24-hour window. The controller only revokes `Slot_1` after Prometheus confirms `sum(rate(requests{slot="Slot_1"}[2h])) == 0`!

### Q3: What stops an attacker from replaying a `15-minute` Bearer JWT if it gets accidentally printed in a debug log?
- **Sender-Constrained Tokens via `RFC 8705` Mutual TLS (`SPIFFE/SPIRE mTLS`)**:
  - When a pod obtains its JWT inside the service mesh (`Istio / Linkerd / Cilium`), the token embeds the SHA-256 thumbprint of the pod's mTLS client certificate (`"cnf": {"x5t#S256": "..."}`).
  - The AI Gateway checks that the client certificate presented on the TLS socket matches `cnf.x5t#S256`. Without the pod's non-exportable private key, a stolen JWT string is completely unusable (`100.0%` non-repudiation)!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Secretless `OIDC + mTLS` Topology + Dual-Slot Rotation Sequence)

### 3.1 Enterprise AI Gateway Authentication & Key Rotation Decision Tree

```mermaid
flowchart TD
    A["Enterprise AI Gateway Authentication Selection<br/>(85+ Connected Services | Target: 0 Leaked Keys & 0% Rotation Outage)"] --> B{"Is the caller an internal cloud-native workload or an external legacy client?"}
    B -- "Static sk-... API keys in .env / notebooks (Tier 1)" --> C["⛔ 168-Hour Leak Blast Radius & $175K/Mo Risk!<br/>Committed notebook keys drain enterprise PTU/PAYGO quotas"]
    B -- "Internal K8s Microservices & Data Science Workbenches (90%)" --> D{"How does the AI Gateway verify OAuth2 / OIDC JWTs?"}
    D -- "Synchronous remote IdP /introspect call per turn (Tier 3)" --> E["⚠️ +88.0ms P99 Latency Overhead per Turn!<br/>Couples AI Gateway uptime to remote IdP rate limits"]
    D -- "In-Memory JWKS Verify (0.3ms) + SPIFFE mTLS Binding (Tier 4)" --> F["🟩 Tier 4A: 15m Secretless OIDC + RFC 8705 mTLS + disableLocalAuth 🏆<br/>(Zero Static Keys | 0.3ms Verify | Replay-Proof Certificate Binding)"]
    B -- "External Partner / Legacy Apps (10%)" --> G["🟩 Tier 4B: Dual-Slot (Primary + Secondary) 24h Overlapping Rotation 🏆<br/>(Drains old slot to 0 QPS before revocation -> 0.0% HTTP 401 Outage)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style G fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 End-to-End Secretless (`OIDC + SPIFFE mTLS -> 0.3ms Local JWKS -> Managed Identity`) Reference Architecture

```mermaid
flowchart LR
    subgraph Callers["1. Secretless & Dual-Slot Client Layer"]
        K8s["📦 85 Internal K8s Microservices<br/>Entra Workload ID / AWS IRSA<br/>(15m OIDC JWT + SPIFFE mTLS Cert)"]
        Legacy["🏢 External Partner Clients<br/>Dual-Slot Keys (Primary + Secondary)"]
    end

    subgraph AIGW["2. AI Gateway Zero-Trust Verifier (0.3ms P99)"]
        K8s -->|mTLS + Sender-Bound JWT| JWKS["⚡ In-Memory JWKS Verifier (0.3ms)<br/>1. Verify RS256/ES256 Signature<br/>2. Match cnf.x5t#S256 == TLS Peer Cert<br/>3. Check Redis JTI Revocation Bloom"]
        Legacy -->|24h Overlapping Window| Slots["🔄 Dual-Slot Rotation Engine<br/>Revokes old slot only at 0 QPS"]
    end

    subgraph Cloud["3. Upstream Cloud LLMs (disableLocalAuth: true)"]
        JWKS & Slots -->|Gateway System Managed Identity| Foundry["☁️ Azure OpenAI / Bedrock / Vertex<br/>(Static API Keys Disabled!)"]
    end
```

### 3.3 End-to-End Sequence: Zero-Downtime Dual-Slot (`Primary + Secondary`) Credential Rotation Without `HTTP 401` Errors

```mermaid
sequenceDiagram
    autonumber
    participant Rot as Rotation Controller (30-Day Cycle)
    participant GW as AI Gateway (Dual-Slot Auth Table)
    participant Vault as Secret Store / CSI Driver
    participant Pods as Client Fleet (20 Rolling Pods)

    Note over GW,Pods: Initial State: Slot_1 (Key_A_v1) serves 100% of traffic; Slot_2 is idle
    Rot->>GW: 1. Start rotation: Keep Slot_1 (`Key_A_v1`) active! Mint `Key_B_v2` in Slot_2
    Rot->>Vault: 2. Publish `Key_B_v2` to Vault; 20 client pods hot-reload `Key_B_v2` asynchronously over 4 hours
    Note over GW,Pods: During 24h overlap: BOTH `Key_A_v1` and `Key_B_v2` are accepted -> 0.0% HTTP 401 errors!
    Rot->>GW: 3. Query Prometheus: confirm `sum(rate(requests{slot="Slot_1"}[2h])) == 0`
    Rot->>GW: 4. Safely revoke `Key_A_v1` and promote `Slot_2 (Key_B_v2)` to Primary (100% zero-downtime rotation)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Static `sk-...` Keys | 🟨 Tier 2: Single-Slot Overwrite | 🟦 Tier 3: Remote `/introspect` | 🟩 Tier 4: Secretless `OIDC + mTLS` + Dual-Slot |
|---|---|---|---|---|
| **Compromise Blast Radius** | `168.0 hrs (7d)` ✕ | `24.0 hrs` | `1.0 hr` | 🏆 **`0.25 hrs (15 mins)` (`mTLS` certificate-bound)** |
| **Rotation `HTTP 401` Outage Rate** | `18.5%` ✕ (manual drift) | `11.2%` ✕ (instant overwrite) | `0.8%` | 🏆 **`0.0%` (80%-TTL OIDC refresh + dual-slot drain)** |
| **Auth Verify `P99` Overhead** | `0.2 ms` | `1.5 ms` | `88.0 ms` ✕ (network RTT) | 🏆 **`0.3 ms` (in-memory `JWKS` public-key verify)** |
| **Caller Non-Repudiation** | `12.0%` ✕ (shared keys) | `65.0%` | `98.0%` | 🏆 **`100.0%` (cryptographic per-pod/user identity)** |
| **Monthly Security & Leak TCO** | `$175,000 / mo` ✕ | `$96,000 / mo` | `$62,000 / mo` | **`$18,000 / mo` 🏆 (-89.7% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise AI Gateway Zero-Trust Auth, Secretless OIDC & Key Rotation Sizer](../reports/enterprise-ai-gateway-authentication-entra-id-mtls-api-key-rotation-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-ai-gateway-authentication-entra-id-mtls-api-key-rotation-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_ai_gateway_authentication_entra_id_mtls_api_key_rotation_selection/run.py --check
  python -m unittest tests/test_enterprise_ai_gateway_authentication_entra_id_mtls_api_key_rotation_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Set `disableLocalAuth: true` on All Upstream Cloud AI Resources (`Azure OpenAI / Foundry`)**: Physically disable raw provider API keys and authenticate the AI Gateway exclusively via Managed Identity / IAM Roles.
- [ ] **Migrate Internal Microservices to Secretless Federated Workload Identity (`Entra Workload ID / AWS IRSA`)**: Issue `15-minute` OIDC JWTs with automatic background renewal at `80%` TTL.
- [ ] **Validate JWT Signatures Locally Against Cached `JWKS` Public Keys (`0.3ms`)**: Never invoke a synchronous remote `/introspect` endpoint on the per-request data plane.
- [ ] **Bind Bearer Tokens to Service-Mesh `SPIFFE mTLS` Client Certificates (`RFC 8705 cnf.x5t#S256`)**: Render leaked token strings useless outside the originating pod.
- [ ] **Enforce a Dual-Slot (`Primary + Secondary`) Overlapping State Machine for External Client Keys**: Verify zero traffic on the old key slot for at least 2 hours before revocation to guarantee `0.0%` rotation outages.
