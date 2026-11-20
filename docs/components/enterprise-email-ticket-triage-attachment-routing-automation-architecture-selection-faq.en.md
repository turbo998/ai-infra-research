# Enterprise Shared-Mailbox & Ticket Triage Automation (`500K Emails/Mo`), Recursive Thread-Quote Stripping, `RFC 3834` Auto-Responder Loop Guard & Attachment VLM/OCR Routing Architecture Selection Guide

> **Language / 语言**: **English (Current)** | [中文](./enterprise-email-ticket-triage-attachment-routing-automation-architecture-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and sizing playbook for Cloud & AI Solution Architects, IT Service Management (`ServiceNow / Zendesk`) Leads, and Shared-Services Automation Engineers automating high-volume enterprise mailboxes (`support@`, `ap-invoices@`, `claims@`, `logistics@`).  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_email_ticket_triage_attachment_routing_automation_architecture_selection/results.json`](../../experiments/enterprise_email_ticket_triage_attachment_routing_automation_architecture_selection/results.json).

![Enterprise Shared-Mailbox & Ticket Triage Automation Summary](../../assets/svg/enterprise-email-ticket-triage-attachment-routing-automation-architecture-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Across enterprise customer support, accounts payable (`ap-invoices@`), insurance claims, and logistics desks, **`500,000+` monthly inbound emails (`42%` with PDF/Excel/image attachments, averaging `6,800 tokens` of nested `RE: FW:` history)** drive core workflows in `ServiceNow`, `Zendesk`, and `SAP`. Three architectural anti-patterns routinely derail production email triage agents:
1. **Trap #1: Feeding Raw Unstripped 15-Turn Email Threads to a Frontier LLM with Unchecked Auto-Reply (Tier 1, `74.0%` Wasted Quote/Banner Tokens, `14.5%` Duplicate Tickets on `Reply-All`, `3.8%` Infinite Bot-to-Bot Auto-Reply Loops, `$168,000/month` TCO)**:
   - A 15-turn corporate email thread carries `6,800 tokens` of duplicated `From:/Sent:` headers, legal disclaimers, and HTML tracking signatures when the user's new reply is just `180 tokens`. Worse, old resolved topics buried at the bottom of the quote chain confuse the LLM router!
   - When the agent auto-replies to an external sender whose mailbox has an Out-of-Office (OOO) or vendor ticketing auto-responder without **`RFC 3834` loop guards**, the two bots exchange `10,000+` emails in 15 minutes—getting the corporate SMTP domain blacklisted!
2. **Trap #2: Legacy Subject-Line Regex Rules + Naive Thread Splitting (Tier 2, `69.0%` Routing Accuracy, `18.2%` Duplicate Tickets, `$124,000/month` Re-Triage TCO)**: When 4 CC'd colleagues click `Reply-All` on an incident thread, naive ingestion spawns 4 separate ServiceNow tickets!
3. **Trap #3: Synchronous Frontier Vision LLM (`GPT-4o VLM`) on Every Email Page & Digital PDF/Excel Attachment (Tier 3, `18.4s` Webhook Lag, `$142,000/month` TCO)**.

Evaluated across **500,000 monthly inbound emails (`6,800` average raw tokens, `42%` attachment rate)**, the four email triage architectures compare as follows:

| Triage Tier | Loop Guard, Dedup & Attachment Stack | Routing & Extraction Accuracy | `Reply-All` Duplicate Tickets | Auto-Reply Loop Incidents | Prompt Token Savings | 500K-Email Monthly TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Raw Unstripped Thread to Frontier LLM** | Sends full `RE:/FW:` HTML chain; no `RFC 3834` loop breaker | `88.5%` *(distracted by old quotes)* | `14.5%` ✕ *(CC fork duplicates)* | `3.8%` ✕ *(OOO reply storms)* | `0.0%` ✕ *(`6,800` tok/email)* | `$168,000 / mo` ✕ *(token & merge waste)* | ⛔ Never send unstripped email threads or unguarded auto-replies |
| **🟨 Tier 2: Legacy Subject-Regex Rules** | Static keyword matching on subject lines | `69.0%` ✕ *(31% misrouted)* | `18.2%` ✕ | `1.2%` | `0.0%` | `$124,000 / mo` ✕ *(manual re-triage)* | ⛔ Blind to attachment payloads and multi-intent body text |
| **🟦 Tier 3: Synchronous Frontier `VLM` on All Pages** | Renders every email & digital PDF/XLSX as images for VLM | `93.2%` | `6.4%` | `0.8%` | `25.0%` | `$142,000 / mo` ✕ *(VLM token waste)* | ⛔ Paying vision tokens for native digital PDFs & spreadsheets |
| **🟩 Tier 4: `RFC 3834` Loop Guard + `References/SimHash` Dedup + `AST` Quote Stripper (`-74%`) + Tiered `85% SLM / 15% VLM`** | **Blocks `Auto-Submitted` + `2/24h` sender cap; strips quotes; merges CC forks (or splits on topic shift `<0.65`)** | **`96.8%` 🏆** *(clean delta focus)* | **`0.2%` 🏆** *(graph thread merge)* | **`0.0%` 🏆** *(zero bot loops)* | **`74.0%` 🏆** *(`6,800->1,768` tok)* | **`$19,500 / mo` 🏆** *(-88.4% vs Tier 1)* | 🌟 **Recommended Enterprise Email & Ticket Triage Standard** |

> 💡 **Golden Rules for Enterprise Email & Ticket Triage Architecture**:
> 1. **Dual-Layer `RFC 3834` Auto-Responder Loop Circuit Breaker**: **"Never let an AI agent argue with an Out-of-Office bot!"** Drop auto-replies for any inbound email carrying `Auto-Submitted: auto-replied / auto-generated`, `X-Auto-Response-Suppress`, `Precedence: bulk`, or `mailer-daemon / no-reply` senders; stamp all outbound agent emails with `Auto-Submitted: auto-replied`; and enforce a Redis circuit breaker of **`max 2 auto-replies per sender per 24 hours`** (`0.0%` loop storms).
> 2. **`In-Reply-To / References` DAG + Topic-Divergence Thread Forking**: Merge `Reply-All` CC forks into the existing ticket via SMTP `References` headers and quoted-block `SimHash` (`0.2%` duplicate tickets)—**unless** the parent ticket has been `CLOSED > 7 days` or the new message's embedding diverges (`cosine < 0.65`, i.e., customer hijacked an old thread for a brand-new issue), in which case automatically fork a new ticket!
> 3. **AST Quote-Chain Stripping (`-74%` Tokens) + `85% SLM / 15% VLM` Attachment Routing**: Strip nested `blockquote` / `Original Message` chains and legal footers so the model reads only the **new delta reply (`180 tokens`) + rolling ticket summary (`350 tokens`)**, cutting prompt tokens by **`74.0%`** while improving routing accuracy to **`96.8%`**!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Quote-Chain Bloat, Reply-All vs. Thread-Hijacking & Attachment Injection Pain Points)

### Q1: Why does stripping the 15-turn `RE: FW:` quote chain (`Tier 4`) simultaneously cut token costs by `74%` AND boost routing accuracy from `88.5%` to `96.8%`?
- **Eliminating Historical Topic Pollution**:
  - In a 15-turn thread, the bottom 5,000 tokens often discuss an issue that was already resolved last week (e.g., *"login password reset"*), while the latest top reply says *"Thanks, now please route my attached Q3 VAT invoice to Accounts Payable."*
  - An unstripped LLM frequently gets distracted by the repeated `"password reset"` tokens in the quote chain and misroutes the invoice to IT Support!
  - Tier 4 parses the HTML DOM (`gmail_quote`, Outlook `divRplyFwdMsg`, multi-language `On ... wrote:` markers) to isolate the **top delta message**, pairing it with a compact **`350-token` structured state summary** of the thread (`6,800 -> 1,768 tokens`, `-74.0%`).

### Q2: How do we distinguish between 4 colleagues clicking `Reply-All` on the same incident (which should merge into 1 ticket) vs. a customer replying to a 3-month-old closed email with a brand-new problem (which must open a new ticket)?
- **`References` Graph Matching + Semantic Topic-Divergence Gate (`cosine < 0.65`)**:
  - When an email arrives matching an existing ticket via `In-Reply-To / References` or subject/quote `SimHash`:
    1. If the existing ticket is **`OPEN / PENDING`** and `cosine(embed(delta_reply), ticket.summary_vec) >= 0.65` -> **Merge as a comment** on the existing ticket (`0.2%` duplicate tickets).
    2. If the existing ticket is **`CLOSED > 7 days`** OR `cosine < 0.65` (Thread Hijacking) -> **Fork a new ticket** linked via `related_to_ticket_id`!

### Q3: How do we process `210,000` monthly attachments (`42%` of `500K` emails) without paying `$142,000/month` for Frontier Vision LLMs or falling victim to hidden prompt injection inside PDFs?
- **Content Disarm & Reconstruction (`CDR`) Sandbox + `85%` Native/SLM vs. `15%` Frontier VLM + Read-Only Schema Output**:
  - `85%` of enterprise attachments are native digital PDFs, DOCX, or XLSX spreadsheets that are parsed deterministically + `Llama-3.1-8B` at a fraction of VLM cost; only the `15%` scanned/handwritten/damaged photos escalate to a Frontier VLM.
  - Furthermore, the extraction model outputs a **strictly typed data-only JSON schema** (`{ "category": "...", "po_number": "..." }`) with **zero tool-execution privileges**, neutralizing hidden white-text prompt injections inside untrusted attachments!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + `RFC 3834 + Quote Stripper + Tiered Attachment` Topology + Sequence)

### 3.1 Enterprise Email & Ticket Triage Automation Decision Tree

```mermaid
flowchart TD
    A["Enterprise Shared-Mailbox & Ticket Triage Selection<br/>(500K Inbound Emails/Mo | 6,800 Raw Thread Tokens | 42% with Attachments)"] --> B{"How are quote chains, Reply-All forks, auto-replies & attachments handled?"}
    B -- "Raw unstripped thread to Frontier LLM + naive auto-reply (Tier 1)" --> C["⛔ 74% Wasted Quote Tokens + 3.8% Auto-Reply Loops!<br/>14.5% duplicate tickets on Reply-All ($168K/mo TCO)"]
    B -- "Legacy subject-line regex rules + naive splitting (Tier 2)" --> D["⛔ 31.0% Misrouted Tickets + 18.2% Duplicates!<br/>Heavy manual re-triage across ServiceNow queues ($124K/mo)"]
    B -- "Synchronous Frontier VLM on every email & attachment page (Tier 3)" --> E["⛔ 18.4s Webhook Lag & $142K/mo VLM Bill!<br/>Wastes vision tokens on native digital PDFs & spreadsheets"]
    B -- "RFC 3834 Loop Guard + References/SimHash Dedup + Quote Stripper + 85/15 SLM/VLM (Tier 4)" --> F["🟩 Tier 4: RFC 3834 Guard + AST Quote Stripper (-74%) + Smart Merge/Fork 🏆<br/>(96.8% Accuracy | 0.0% Loops | 0.2% Dupes | $19.5K/mo, -88.4% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `RFC 3834` Loop Guard, `References/SimHash` Thread Deduplicator & `85/15` Attachment Router Architecture

```mermaid
flowchart LR
    subgraph Ingest["1. Loop Guard & Thread Graph Layer"]
        Mail["📧 Inbound Email<br/>(Graph / IMAP / SMTP)"] --> Loop["🛡️ RFC 3834 Loop Breaker<br/>Blocks Auto-Submitted / OOO<br/>+ Redis Cap (max 2/sender/24h)"]
        Loop --> Graph["🔗 Thread Merge / Fork Gate<br/>References DAG + SimHash<br/>+ Topic Divergence (cos >= 0.65)"]
    end

    subgraph Parse["2. Quote Stripper & Attachment Sandbox"]
        Graph --> Strip["✂️ AST Quote & Banner Stripper<br/>Extracts new delta reply only<br/>(6,800 -> 1,768 tokens, -74%!)"]
        Graph --> Attach["🧪 CDR Sandbox & Tiered Parser<br/>• 85% Digital PDF/XLSX -> Layout + 8B SLM<br/>• 15% Scanned/Photo -> Frontier VLM"]
    end

    subgraph Sync["3. Typed Extraction & ITSM Sync"]
        Strip & Attach --> Schema["🎯 Read-Only JSON Schema Extractor<br/>(96.8% Accuracy | Injection-Safe)"]
        Schema --> ITSM["🎫 ServiceNow / Zendesk / SAP<br/>Create/Comment + RFC 3834 Ack"]
    end
```

### 3.3 End-to-End Sequence: Quote Stripping, `OOO` Loop Blocking & `Reply-All` Thread Merging

```mermaid
sequenceDiagram
    autonumber
    participant Ext as Customer & CC Colleagues
    participant GW as RFC 3834 & Thread Dedup Gateway
    participant Strip as AST Quote Stripper & Attachment Router
    participant SLM as 8B SLM / Targeted VLM
    participant Snow as ServiceNow / Zendesk

    Ext->>GW: 1. Send 15-turn email thread (`6,800 tokens`) + PDF Invoice attachment
    GW->>GW: 2. Verify `Auto-Submitted == no`; check `References` (no existing open ticket)
    GW->>Strip: 3. Strip 14 nested quote blocks & legal footers -> keep `180-token` delta + parse PDF
    Strip->>SLM: 4. Extract typed JSON: `{intent: "AP_INVOICE_DISPUTE", po: "PO-8821", priority: "P2"}`
    SLM->>Snow: 5. Create ticket `INC-9001`; send Ack email stamped with `Auto-Submitted: auto-replied`
    Ext-->>GW: 6. Customer's mailbox fires back automatic `Out-of-Office (OOO)` reply
    GW->>GW: 7. ⚡ Detect `Auto-Submitted: auto-replied` header -> suppress auto-reply (`0.0%` loop storm!)
    Ext->>GW: 8. CC'd colleague clicks `Reply-All`: `"Adding the updated packing slip"`
    GW->>Snow: 9. Match `References` DAG + `cosine = 0.82 >= 0.65` -> append as comment on `INC-9001` (`0` duplicate tickets!)
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Raw Thread to LLM | 🟨 Tier 2: Subject Regex Rules | 🟦 Tier 3: Sync `VLM` Everything | 🟩 Tier 4: `RFC 3834` + Quote Strip + Smart Merge + `85/15` SLM/VLM |
|---|---|---|---|---|
| **Triage Routing Accuracy** | `88.5%` (polluted by old quotes) | `69.0%` ✕ (31% misrouted) | `93.2%` | 🏆 **`96.8%` (focused on delta reply + rolling summary)** |
| **`Reply-All` Duplicate Tickets** | `14.5%` ✕ | `18.2%` ✕ | `6.4%` | 🏆 **`0.2%` (`References/SimHash` merge + topic-shift fork)** |
| **Auto-Reply Loop Incidents** | `3.8%` ✕ (domain blacklist risk) | `1.2%` | `0.8%` | 🏆 **`0.0%` (`RFC 3834` header gate + `2/24h` Redis cap)** |
| **Prompt Token Reduction** | `0.0%` (`6,800` tok/email) | `0.0%` | `25.0%` | 🏆 **`74.0%` (`6,800 -> 1,768` tok via AST quote stripping)** |
| **500K-Email Monthly TCO** | `$168,000 / mo` ✕ | `$124,000 / mo` ✕ | `$142,000 / mo` ✕ | **`$19,500 / mo` 🏆 (-88.4% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Shared-Mailbox & Ticket Triage Automation Sizer](../reports/enterprise-email-ticket-triage-attachment-routing-automation-architecture-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-email-ticket-triage-attachment-routing-automation-architecture-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_email_ticket_triage_attachment_routing_automation_architecture_selection/run.py --check
  python -m unittest tests/test_enterprise_email_ticket_triage_attachment_routing_automation_architecture_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Enforce Dual-Layer `RFC 3834` Loop Prevention**: Suppress auto-replies on inbound `Auto-Submitted`, `X-Auto-Response-Suppress`, `Precedence: bulk`, and `no-reply@` messages; stamp outbound auto-replies with `Auto-Submitted: auto-replied`; and cap auto-replies at `2 per sender per 24h` in Redis.
- [ ] **Deduplicate `Reply-All` Forks with `In-Reply-To / References` DAG + Topic-Divergence Forking**: Append replies with `cosine >= 0.65` to the open parent ticket, while automatically forking a new ticket when a customer hijacks a closed thread for an unrelated issue.
- [ ] **Strip Nested HTML Quote Chains & Legal Disclaimers (`-74%` Prompt Tokens)**: Feed only the new delta reply plus the rolling structured ticket summary into the classifier.
- [ ] **Route Attachments via an `85%` Native/SLM vs. `15%` Frontier VLM Split Inside a `CDR` Sandbox**: Parse native digital PDFs and Excel sheets deterministically before escalating scanned or handwritten images to a VLM.
- [ ] **Restrict Email/Attachment Parsers to Read-Only Typed JSON Outputs**: Never grant untrusted inbound email parsers direct tool-calling permissions to approve payments or mutate ERP records.
