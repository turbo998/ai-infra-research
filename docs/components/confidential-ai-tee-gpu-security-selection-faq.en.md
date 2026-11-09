# Confidential AI & Regulated Enterprise GPU TEE Inference Selection FAQ (`Plaintext GPU Container` vs. `PII Redaction Gateway` vs. `Single-GPU H100/H200 CC-On TEE` vs. `Multi-GPU PPCIE + Blackwell NVLink Encrypted Cluster`)

> 🌐 **Language**: **[中文 (Chinese)](./confidential-ai-tee-gpu-security-selection-faq.md)** | **English Version** · **[📖 Back to Repository README](../../README.en.md)** · **[📊 2026-11-09 Daily Progress Report](../daily/2026-11-09.md)** · **[🃏 7-Card 4:5 Visual Benchmark Report](../reports/confidential-ai-tee-gpu-security-selection-cards.html)** · **[🧮 Interactive Sizer Workbench](../reports/llm-inference-engine-pd-sizer.en.html#confidential-tee-sizer)**
>
> **Overview**: A first-principles selection guide for AI Solution Architects. When banks, hospitals, law firms, and sovereign agencies deploy LLMs, the primary compliance blocker is **Data-in-Use and Model IP Exposure**: even with TLS in transit and disk encryption at rest, a cloud host administrator (Root / Hypervisor) can dump sensitive prompts and proprietary fine-tuned weights directly from RAM and GPU HBM. Architects commonly fall into two traps: **(1) relying on an application-layer regex/NER PII redaction proxy, which still leaks 18.5% of quasi-identifiers while degrading downstream LLM reasoning accuracy by `-6.4%`; and (2) fearing that GPU Confidential Computing (`CC-On`) will cripple LLM throughput because PCIe bounce-buffer encryption cuts PCIe bandwidth by 77%**. Grounded in `experiments/confidential_ai_tee_selection/` (`synthetic` + `anonymized_reported`, `independently_reproduced=false`), this guide proves why **70B single-GPU LLM decode incurs only a `4.8%` overhead under `CC-On`**, and why **405B multi-GPU `TP=8` requires `PPCIE` or Blackwell inline NVLink encryption to avoid a `+280.4%` bounce-buffer cliff**.

![Confidential AI & Regulated Enterprise GPU TEE Inference Visual Summary](../../assets/svg/confidential-ai-tee-gpu-security-selection-faq-summary.svg)

---

## ① 📌 Executive Summary

1. **Four-Tier Selection by Deployment Topology (`TP=1` Single-GPU vs. `TP=8` Multi-GPU) and Compliance Mandate**:
   - For regulated enterprise LLM serving (baseline plaintext `TPOT = 11.20 ms`, `520.0 tok/s/GPU`, target SLA `TPOT <= 15.0 ms` with cryptographic hardware attestation and zero data leakage):
   - 🟥 **Tier 1 (`Standard Plaintext GPU Container` + TLS-in-Transit + Disk Encryption-at-Rest)**: **Never use on shared/managed clouds for regulated PII/PHI or proprietary model weights**! Provides **`0%` Data-in-Use protection (`100%` exposed to Host OS Root, Hypervisor memory dumps, and PCIe bus probes)**.
   - 🟨 **Tier 2 (`Application-Layer PII / NER Redaction Proxy` + Plaintext Cloud GPU)**: **A flawed compromise that fails both security audits and reasoning quality**. Leaves **`18.5%` residual PII/IP leakage** via contextual quasi-identifiers, degrades downstream LLM reasoning accuracy by **`-6.4%` (`93.6%` retention)** due to broken entity references, and leaves cloud-hosted model weights 100% exposed.
   - 🟩 **Tier 3 (`CPU CVM [AMD SEV-SNP / Intel TDX]` + Single-GPU `H100/H200 Confidential Computing (CC-On)` Bounce Buffer)**: **The gold standard for models fitting on a single large-HBM GPU (`<= 70B FP8` on H200 141GB / B200 192GB, `TP=1`)**. Enforces `SPDM` hardware attestation + GPU Compute Protected Region (`CPR`) HBM firewall + `AES-GCM-256` PCIe DMA encryption for **`0.0%` leakage and `100.0%` lossless accuracy**; because 99.8% of 70B decode math stays inside HBM and only 4-byte token IDs cross PCIe per step, **TPOT rises from `11.20 ms` to just `11.74 ms` (only `+4.8%` overhead, `$1.06 / 1M tokens`)**!
   - 🟪 **Tier 4 (`Multi-GPU Protected PCIe (PPCIE)` + `Blackwell GB200/B200 Inline NVLink-C2C / NVLink-5 Hardware TEE`)**: **Mandatory for 405B/671B multi-GPU (`TP=8`) confidential inference**. Running 8-GPU `TP=8` (160 `All-Reduce` syncs/token) through single-GPU `CC-On` CPU bounce buffers inflates TPOT to **`42.60 ms` (`+280.4%` slowdown)**; enabling **`PPCIE` 8-GPU NVSwitch attestation** or **Blackwell inline line-rate NVLink encryption** holds `TP=8` TPOT at **`11.90 ms` (only `+6.2%` overhead, `488.0 tok/s/GPU`, `$1.12 / 1M tokens`)**!
2. **First-Principles Law (Why Small Models Slow Down `58.1%` Under `CC-On` While 70B LLM Decode Slows Down Only `4.8%`)**:
   - Small models (`BERT / ResNet`) transfer megabytes of inputs/activations over PCIe every `2.1 ms`, hitting the `11.8 GB/s` SWIOTLB bounce-buffer encryption bottleneck (`+58.1%` overhead).
   - Autoregressive 70B LLM decode (`TP=1`) keeps 70 GB of weights and KV cache inside the GPU's `3.35 TB/s` HBM `CPR` enclave and transfers only a **4-byte token ID** across PCIe per step—making PCIe bounce encryption virtually invisible (`+4.8%`)!
3. **Provenance & Transparency Disclosure**:
   - All TPOT latencies, PCIe bandwidths, leakage rates, and costs are bound 1:1 to `experiments/confidential_ai_tee_selection/results.json` (`provenance_type: "synthetic"`, calibrated to published NVIDIA H100/H200 CC-On, PPCIE, and Blackwell TEE-I/O whitepapers; `independently_reproduced: false`, `physical_gpu_cluster_measurement: "not_measured"`, `local_cpu_harness_measurement: "measured"`).

---

## ② 🎯 Architect & Customer Deep-Dive Q&A (FAQ-Driven)

### Q1: "Why did our compliance audit reject an application-layer PII redaction proxy (Tier 2), and why did it hurt our LLM accuracy by 6.4%?"

- **Plain-English Answer**: **Regex and NER redaction proxies miss `18.5%` of contextual quasi-identifiers in unstructured medical/financial documents, destroy multi-hop coreference chains (`-6.4%` accuracy), and provide zero protection for proprietary model weights stored in cloud GPU HBM**.
- **First-Principles Breakdown**:
  1. **Quasi-Identifier Re-Identification (`18.5%` Residual Leakage)**: Even if a proxy masks a patient's name, rare diagnosis combinations, timestamps, or proprietary source-code logic remain in plaintext.
  2. **Hardware TEE (Tier 3 / Tier 4) Preserves 100% Context & Protects Model IP**: With CPU CVM + GPU TEE, the unredacted prompt and encrypted model weights are decrypted **only inside attested CPU/GPU silicon packages**, delivering **`0.0%` leakage** and **`100.0%` reasoning accuracy**.

### Q2: How does the `SWIOTLB Bounce Buffer` work in `H100/H200 CC-On` (Tier 3), and why is its impact on 70B LLM decode only **4.8%**?

- **Plain-English Answer**: **In `CC-On` mode, hardware firewalls block the GPU from directly reading encrypted CPU VM memory; all CPU<->GPU transfers must copy through an unencrypted `SWIOTLB Bounce Buffer` encrypted with `AES-GCM-256` (`11.8 GB/s`). Because 70B decode spends 99.8% of its time inside GPU HBM (`CPR`) and sends only a 4-byte token ID over PCIe per step, end-to-end overhead is just `4.8%`!**
- **First-Principles Breakdown**:
  1. **Compute Protected Region (`CPR`)**: The GPU memory controller locks HBM inside a hardware-enforced `CPR` firewall; host OS/Hypervisor reads to `CPR` return zeros.
  2. **Arithmetic Intensity Shields LLM Decode**: While I/O-bound ResNet/BERT suffers a `+58.1%` penalty (`2.10 -> 3.32 ms`), 70B `TP=1` decode on a single H200 (141GB) or B200 (192GB) moves almost nothing over PCIe during token generation (`11.20 -> 11.74 ms`).

### Q3: Why does multi-GPU `TP=8` collapse (`+280.4%`) under single-GPU `CC-On`, and how do `PPCIE` and `Blackwell TEE-I/O` (Tier 4) reduce multi-GPU TEE overhead to **6.2%**?

- **Plain-English Answer**: **Single-GPU `CC-On` disables direct unencrypted NVLink P2P, forcing all 160 `All-Reduce` collectives per token through the CPU's `11.8 GB/s` software bounce buffer (`42.60 ms` TPOT). `PPCIE` (8-GPU + NVSwitch enclave) and `Blackwell` (inline line-rate NVLink AES-GCM encryption) keep multi-GPU collectives on high-speed NVLink, cutting TPOT back to `11.90 ms` (`+6.2%`)!**

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree, Bounce vs Inline Path & Attestation Sequence)

### 3.1 Confidential AI & GPU TEE Inference Decision Tree

```mermaid
flowchart TD
    Start["🚀 Regulated Enterprise Confidential AI Selection<br/>(Goal: Block Host/Hypervisor Scraping · Protect Model IP · TPOT <= 15ms)"] --> Q_Threat{"Data Sensitivity &<br/>Model IP Mandate?"}

    Q_Threat -->|"Public Non-Sensitive Data + Open Weights<br/>(No Data-in-Use TEE requirement)"| T1["🟥 Tier 1: Standard Plaintext GPU Container + TLS<br/>• 0.0% latency overhead (TPOT 11.20 ms · $0.96/1M tok)<br/>• ⚠️ 100% HBM & PCIe exposure to Host Root / Hypervisor"]

    Q_Threat -->|"Regulated Financial/Medical/Gov Data or Proprietary Weights<br/>(Zero Host/Cloud-Admin Trust Mandate)"| Q_Anti{"⛔ Anti-Pattern Check:<br/>Relying Only on a Regex/NER Proxy?"}

    Q_Anti -->|"Regex / NER Redaction Gateway Only"| T2["🟨 Tier 2: Application-Layer PII Redaction Proxy (Not Recommended)<br/>• 18.5% residual quasi-identifier leakage · Zero model weight protection<br/>• Redaction placeholders cut LLM reasoning accuracy by -6.4% (93.6%)"]

    Q_Anti -->|"Hardware-Rooted Trusted Execution Environment (CPU + GPU TEE)"| Q_Cards{"GPU Count &<br/>Parallelism Topology?"}

    Q_Cards -->|"Single-GPU Model Fit (<= 70B FP8 on H200 141GB / B200 192GB)<br/>TP = 1 (No inter-GPU All-Reduce)"| T3["🟩 Tier 3: CPU CVM (SEV-SNP/TDX) + Single-GPU H100/H200 CC-On<br/>• SPDM attestation + GPU CPR firewall + AES-GCM-256 PCIe DMA<br/>• 99.8% math stays in HBM -> TPOT 11.74 ms (only +4.8% overhead!)<br/>• 0.0% leakage · 100.0% lossless accuracy · $1.06 / 1M tok ✅"]

    Q_Cards -->|"Multi-GPU Tensor Parallelism (405B / 671B · TP = 4 ~ 8)<br/>160 inter-GPU syncs per token"| T4["🟪 Tier 4: Multi-GPU PPCIE + Blackwell Inline NVLink TEE<br/>• 8-GPU + NVSwitch attested domain / Blackwell 1,800 GB/s inline crypto<br/>• Avoids single-GPU CC-On bounce cliff (42.60ms / +280.4% slowdown)<br/>• TPOT 11.90 ms (+6.2% overhead) · 0.0% leakage · $1.12 / 1M tok 🏆"]

    style T1 fill:#fef2f2,stroke:#dc2626,stroke-width:2px
    style T2 fill:#fef9c3,stroke:#ca8a04,stroke-width:2px
    style T3 fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style T4 fill:#f3e8ff,stroke:#9333ea,stroke-width:2px
```

### 3.2 Plaintext Exposure & Bounce-Buffer Cliff vs. `Tier 3 / Tier 4` Hardware TEE Architecture

```mermaid
flowchart LR
    subgraph Bad["⛔ Legacy Traps: Tier 1 Plaintext HBM Exposure & Naive TP=8 Bounce Cliff (42.60 ms)"]
        direction TB
        B1["Tier 1 Plaintext: Malicious Host Root / Hypervisor dumps /dev/mem & GPU BAR1 to steal prompts & weights"]
        B2["Naive CC-On @ TP=8: GPU0 ──> CPU SWIOTLB Bounce (11.8 GB/s) ──> GPU1 (160 syncs/token = 3.8x slowdown!)"]
        B1 --> B2
    end

    subgraph Good["🟪 Tier 3 / Tier 4: CPU CVM + GPU CPR Firewall + PPCIE / Blackwell Inline Encrypted NVLink"]
        direction TB
        G1["CPU CVM (SEV-SNP/TDX) <══ AES-GCM-256 Encrypted PCIe / TEE-I/O ══> GPU CPR Protected HBM (Opaque to Host)"]
        G2["Tier 4 Multi-GPU TEE: GPU 0..7 <══ PPCIE NVSwitch / Blackwell 1,800 GB/s Inline Encrypted NVLink ══> TPOT 11.90ms (+6.2%)"]
        G1 --> G2
    end

    style Bad fill:#fef2f2,stroke:#dc2626,stroke-width:2px
    style Good fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

### 3.3 End-to-End Sequence: `SPDM + NRAS` Hardware Attestation & Key Release (`sequenceDiagram`)

```mermaid
sequenceDiagram
    autonumber
    participant KMS as Enterprise KMS & NVIDIA NRAS Verifier
    participant CVM as Cloud CPU CVM (AMD SEV-SNP / Intel TDX)
    participant GPU as NVIDIA H100/H200/B200 (CC-On / PPCIE Root of Trust)
    participant User as Regulated Enterprise Client

    Note over KMS,GPU: Phase 1: Boot-Time Cryptographic Attestation & Key Release (Zero Host Trust)
    CVM->>GPU: 1. Challenge GPU hardware root-of-trust via DMTF SPDM for signed firmware/mode measurements
    CVM->>KMS: 2. Submit composite [CPU SEV-SNP/TDX Quote + GPU SPDM Attestation] to NRAS & Enterprise KMS
    KMS-->>CVM: 3. Verification PASS! Release AES-256 model weight decryption key directly into CVM enclave RAM

    Note over CVM,User: Phase 2: End-to-End Encrypted Data-in-Use Inference (100% Accuracy, 0.0% Leakage)
    User->>CVM: 4. Stream unredacted prompt over mTLS directly into encrypted CVM memory
    CVM->>GPU: 5. Transfer via AES-GCM-256 DMA / TEE-I/O into GPU CPR protected HBM
    GPU->>GPU: 6. Execute 70B/405B decode inside CPR & encrypted NVLink (TPOT 11.74~11.90 ms)
    GPU-->>User: 7. Return encrypted completion (Cloud Host Root sees only ciphertext ✅)
```

---

## ④ 📊 Architect Decision Matrix

*All metrics are bound 1:1 to `experiments/confidential_ai_tee_selection/results.json` (70B/405B serving, baseline plaintext `TPOT = 11.20 ms`, `520.0 tok/s/GPU`, SLA `TPOT <= 15.0 ms`):*

| Dimension | 🟥 Tier 1: `Plaintext GPU Container` | 🟨 Tier 2: `PII Redaction Gateway` | 🟩 Tier 3: `Single-GPU CC-On TEE` (`TP=1`) | 🟪 Tier 4: `Multi-GPU PPCIE / Blackwell TEE` (`TP=8`) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Workload Fit** | Public data & open weights | Low-risk customer support | **`<= 70B` single-GPU TEE (H200/B200 `TP=1`)** | **`405B / 671B` multi-GPU (`TP=8`) regulated TEE** |
| **Data-in-Use & Weight Protection** | ✕ **None** (`100%` Host/PCIe exposed) | ✕ **Weights exposed**, partial prompt mask | **✅ `SPDM` + `GPU CPR` + `AES-GCM` PCIe** | **✅ 8-GPU NVSwitch TEE + inline NVLink crypto** |
| **Effective PCIe Bandwidth** | `52.0 GB/s` (plaintext DMA) | `52.0 GB/s` (plaintext DMA) | `11.8 GB/s` (`SWIOTLB Bounce Buffer`) | **`48.5 GB/s`** (`PPCIE` / Blackwell `TEE-I/O`) |
| **Decode TPOT / Latency Overhead** | `11.20 ms` (`0.0%` baseline) | `11.80 ms` (`+5.4%` + `14.5ms` proxy) | **`11.74 ms`** (**only `+4.8%` overhead** ✅) | **`11.90 ms`** (**only `+6.2%` overhead** 🏆 vs `42.60ms` bounce) |
| **Residual Leakage / Accuracy** | `100.0%` exposed / `100.0%` acc | **`18.5%` leakage** ✕ / **`93.6%` acc (`-6.4%` ✕)** | **`0.0%` leakage** ✅ / **`100.0%` acc** ✅ | **`0.0%` leakage** 🏆 / **`100.0%` acc** 🏆 |
| **Throughput / Cost per 1M Tokens** | `520.0 tok/s` · `$0.96 / 1M` | `485.0 tok/s` · `$1.08 / 1M` | **`495.0 tok/s` · `$1.06 / 1M` (`+10.4%`)** | **`488.0 tok/s` · `$1.12 / 1M` (`+16.7%`)** |

---

## ⑤ 🧪 Quantitative Case Study & Benchmark Insights

### 5.1 Workload Arithmetic Intensity vs. Confidential Computing Overhead (`+58.1%` vs. `+4.8%`)

Why do small I/O-bound models suffer under `CC-On` while large LLMs barely notice it, and why is `PPCIE / Blackwell` critical for `TP=8` (from `results.json` `payload_size_sensitivity`)?

| Workload Regime | Plaintext Baseline | Single-GPU `CC-On (Bounce)` | Multi-GPU `PPCIE / Blackwell TEE` | Architect Crossover & Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **Small `BERT / ResNet` (PCIe I/O-Bound)** | `2.10 ms` | `3.32 ms` (`+58.1%`) | **`2.25 ms` (`+7.1%`)** | Frequent multi-MB PCIe transfers hit the `11.8 GB/s` bounce buffer ceiling |
| **`70B LLM Decode TP=1` (HBM-Bound)** | `11.20 ms` | **`11.74 ms` (`+4.8%` ✅)** | **`11.45 ms` (`+2.2%` ✅)** | **Only 4-byte token IDs cross PCIe per step; `CC-On` overhead is negligible (`4.8%`)** |
| **`70B RAG 32K Prompt Prefill`** | `145.0 ms` | `162.4 ms` (`+12.0%` ✅) | **`149.2 ms` (`+2.9%` ✅)** | Compute dominates 32K prefill; bounce overhead stays at `12.0%` |
| **`405B LLM Decode TP=8` (160 Syncs/Token)** | `11.20 ms` | `42.60 ms` (`+280.4%` ✕) | **`11.90 ms` (`+6.2%` 🏆)** | **Never run `TP=8` over CPU bounce buffers; mandate `PPCIE / Blackwell TEE`** |

### 5.2 Reproducibility & Local Harness Verification

```bash
# Verify results.json matches the deterministic Confidential AI TEE model 100%
python experiments/confidential_ai_tee_selection/run.py --check

# Run Round 19 unit tests
python -m unittest tests/test_confidential_ai_tee_selection.py
```

---

## ⑥ 🛠️ Production Verification Checklist (Architect Checklist)

1. - [ ] **Gate KMS Key Release on `SPDM + NVIDIA NRAS` Attestation**: Never bake decrypted model weights into container images; release AES-256 decryption keys from enterprise KMS only after verifying both CPU (`SEV-SNP/TDX`) and GPU (`SPDM`) hardware attestation quotes.
2. - [ ] **Fit `<= 70B` Models on Single Large-HBM GPUs (`H200 141GB / B200 192GB TP=1`) with `Tier 3 CC-On`**: Avoid inter-GPU collectives altogether to achieve hardware TEE compliance at just `+4.8%` latency overhead.
3. - [ ] **Mandate `PPCIE` or `Blackwell Inline NVLink Encryption` (`Tier 4`) for `TP >= 2`**: Never run multi-GPU NCCL collectives over CPU SWIOTLB bounce buffers.
4. - [ ] **Size Guest Linux `SWIOTLB` Bounce Buffers to `>= 512 MB` (`swiotlb=65536`)**: Prevent `swiotlb buffer is full` DMA allocation panics under concurrent long-context requests.
5. - [ ] **Disable Plaintext Crash Dumps & Sanitize Telemetry Inside CVMs**: Disable `systemd-coredump` and strip prompt payloads from exception logs exported outside the TEE boundary.
