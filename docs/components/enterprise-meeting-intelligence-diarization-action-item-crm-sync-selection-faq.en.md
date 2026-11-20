# Enterprise Meeting Intelligence (`Teams / Zoom` Sales & Engineering Calls), Multi-Modal Speaker Diarization, ASR Hotword Phrase Boosting & Evidence-Grounded `CRM / Jira` Action-Item Sync Architecture Selection Guide

> **Language / 语言**: **English (Current)** | [中文](./enterprise-meeting-intelligence-diarization-action-item-crm-sync-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and sizing playbook for Cloud & AI Solution Architects, Revenue Operations (`Salesforce / Dynamics 365 MSX`) Leads, and Collaboration Platform Engineers building enterprise meeting intelligence across `100,000+` monthly recorded hours.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_meeting_intelligence_diarization_action_item_crm_sync_selection/results.json`](../../experiments/enterprise_meeting_intelligence_diarization_action_item_crm_sync_selection/results.json).

![Enterprise Meeting Intelligence, Diarization & CRM Sync Summary](../../assets/svg/enterprise-meeting-intelligence-diarization-action-item-crm-sync-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When enterprises process **`100,000+` monthly hours of Microsoft Teams and Zoom meetings (`6` participants/call, `35%` involving shared physical conference room microphones, `60%` syncing action items into `Salesforce / Dynamics 365 / Jira`)**, naive "download mixed MP3 -> run generic Whisper -> auto-write tasks to CRM" pipelines fail across three dimensions:
1. **Trap #1: Single-Channel Mixed Audio Confuses Buyer vs. Seller (Tier 1, `22.4%` Speaker Attribution Error `DER`, `28.5%` Domain SKU/Name Error, `19.2%` False CRM/Jira Action Items, `$195,000/month` TCO)**:
   - Without per-participant audio tracks or conference-room voiceprints, the pipeline misattributes a customer's price request (*"Can you give us a 20% discount?"*) to the Account Executive (*"Sales rep promised a 20% discount"*).
   - Generic unboosted ASR mangles `28.5%` of proprietary product SKUs, project codenames, and executive names into homophones that downstream LLM post-correction cannot recover.
   - Worse, unreviewed auto-sync writes casual brainstorming remarks directly into `Salesforce` and `Jira` (`19.2%` false action items)!
2. **Trap #2: Acoustic-Only Post-Hoc Clustering (`pyannote` on Mixed MP3 Only) (Tier 2, `11.8%` Speaker Error on Cross-Talk & Conference Rooms, `$128,000/month` TCO)**.
3. **Trap #3: Streaming Full 60-Minute Raw Audio Directly into a Frontier Native Audio LLM (`Tier 3`, `$420,000/month` Audio Token Bill — `17.1x` More Expensive Than Tier 4!)**.

Evaluated across **100,000 monthly enterprise meeting hours (`35%` hybrid conference room calls, `60%` CRM/Jira sync)**, the four architectures compare as follows:

| Architecture Tier | Diarization, Vocabulary & CRM Sync Stack | Speaker Attribution Error (`DER`) | Domain SKU / Jargon Error | False CRM / Jira Sync Rate | 100K-Hr Monthly TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Mixed Audio + Generic ASR + Blind CRM Write** | Single-channel MP3 to generic Whisper; auto-writes all LLM tasks | `22.4%` ✕ *(swaps buyer/seller)* | `28.5%` ✕ *(mangles SKUs)* | `19.2%` ✕ *(pollutes CRM)* | `$195,000 / mo` ✕ *(cleanup + deal risk)* | ⛔ Never use unboosted mixed-audio ASR or blind CRM auto-writes |
| **🟨 Tier 2: Acoustic-Only MP3 Clustering + LLM Fix** | `pyannote` clustering on mixed file + prompt typo correction | `11.8%` ✕ *(fails on overlap)* | `14.2%` ✕ *(LLM guesses wrong)* | `9.5%` ✕ | `$128,000 / mo` ✕ | ⛔ Cannot map anonymous `Speaker_0` to CRM contacts in hybrid rooms |
| **🟦 Tier 3: Full 60-Min Raw Audio to Frontier Audio LLM** | Streams raw audio tokens directly into Frontier Multimodal LLM | `8.4%` | `7.6%` | `6.2%` | **`$420,000 / mo` ✕** *(17.1x cost!)* | ⛔ Prohibitive audio token cost & lacks exact word-level timestamp anchors |
| **🟩 Tier 4: Per-Track `RTP` + Room Voiceprints (`ECAPA-TDNN`) + Pre-Call CRM Hotword Boost + `1-Click` Staged Card** | **Platform per-user RTP streams + calendar-scoped room voiceprints + ASR shallow-fusion phrase list + `[mm:ss]` approval card** | **`1.4%` 🏆** *(-21.0 pts vs Tier 1)* | **`2.2%` 🏆** *(`97.8%` SKU recall)* | **`0.4%` 🏆** *(staged 1-click card)* | **`$24,500 / mo` 🏆** *(-87.4% vs Tier 1)* | 🌟 **Recommended Enterprise Meeting Intelligence Standard** |

> 💡 **Golden Rules for Enterprise Meeting Intelligence Architecture**:
> 1. **Never Downmix Isolated Participant Audio Tracks**: Ingest **per-participant `RTP` audio streams** (`Teams Graph Communications API / Zoom Raw Audio`) for `100%` attribution of remote attendees, and apply **Calendar-Scoped `ECAPA-TDNN` Voiceprint Separation** only to shared physical conference room tracks (`1.4%` overall `DER`).
> 2. **Pre-Call Calendar + CRM Dynamic Hotword Boosting (`Shallow Fusion`)**: One minute before the call, fetch attendee names, customer account aliases, and active opportunity SKUs from `Salesforce / Dynamics 365` and inject them into the ASR decoder's **`Phrase List (boost=15.0)`**—slashing domain jargon error rates from `28.5%` to **`2.2%`**!
> 3. **Timestamp-Grounded Commitment Extraction + `1-Click Adaptive Card` CRM Gate**: Require every extracted action item to include an **explicit owner, action verb, target date, and verbatim `[mm:ss]` audio snippet**, staged via a **5-second `1-Click Teams/Slack Adaptive Card`** before mutating `Salesforce` or `Jira` (`0.4%` false sync rate)!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Hybrid Conference Rooms, Homophone SKU Corruption & CRM Pollution Pain Points)

### Q1: When 3 executives sit together in "Conference Room B" sharing one table microphone while 3 customers join remotely, how does Tier 4 achieve `1.4%` speaker attribution error (`DER`)?
- **Combining Platform Per-Participant `RTP` Tracks with Calendar-Constrained `ECAPA-TDNN` Voiceprints**:
  - Remote attendees already arrive on separate virtual audio tracks via the meeting bot API—giving zero cross-talk or attribution ambiguity.
  - For the shared `"Conference Room B"` audio track, instead of searching a 10,000-employee voiceprint database, the pipeline restricts the candidate pool to the **3 internal employees checked into that room on the Outlook/Google Calendar invite**, matching `192-d ECAPA-TDNN` speaker embeddings with `98.6%` accuracy!

### Q2: Why does LLM post-correction (`Tier 2`) fail to fix mangled product SKUs and executive names—and how does Pre-Call ASR Phrase Boosting achieve `97.8%` recall?
- **Acoustic Beam Pruning Is Irreversible Without Decoder Biasing**:
  - Once an unboosted ASR decoder prunes the rare token `"Kronos-4090"` during acoustic beam search and outputs `"cross 40 90"`, a text-only LLM has no acoustic signal left to recover the true product code.
  - Tier 4 queries `Salesforce / Dynamics 365 + Calendar` before the call starts, compiling a **150-term dynamic hotword list** (attendee names, account SKUs, competitor names, project codes) injected directly into the ASR decoder's **Trie/WFST shallow-fusion graph (`boost=15.0`)**, reducing domain term errors from `28.5%` to **`2.2%`**!

### Q3: Why do sales reps and engineers disable automatic CRM/Jira writeback—and how does Timestamp-Grounded Staged Approval reduce false tasks from `19.2%` to `0.4%`?
- **Filtering Exploratory Brainstorming + 5-Second `1-Click Adaptive Card` Review**:
  - A dedicated **Commitment Classifier** rejects tentative phrasing (*"we could potentially explore..."*) and retains only explicit commitments backed by a clickable **`[00:24:12 - 00:24:19]` audio replay link**.
  - Instead of silently mutating `Salesforce` or `Jira`, the bot sends the host an interactive **Teams/Slack Adaptive Card** within 10 seconds of call end—allowing 1-click confirmation in 5 seconds (`0.4%` noise)!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Multi-Modal Diarization & CRM Sync Topology + End-to-End Sequence)

### 3.1 Enterprise Meeting Intelligence & CRM/Jira Sync Decision Tree

```mermaid
flowchart TD
    A["Enterprise Meeting Intelligence Selection<br/>(100K Meeting Hours/Mo | 35% Hybrid Conference Rooms | 60% CRM/Jira Sync)"] --> B{"How are speakers separated, SKUs recognized & CRM tasks synced?"}
    B -- "Mixed single-channel MP3 + generic ASR + blind CRM auto-write (Tier 1)" --> C["⛔ 22.4% Speaker Error + 28.5% SKU Error + 19.2% CRM Noise!<br/>Swaps buyer & seller statements ($195K/mo TCO)"]
    B -- "Acoustic-only clustering on mixed MP3 + LLM text correction (Tier 2)" --> D["⛔ 11.8% Speaker Error in Conference Rooms!<br/>LLM cannot guess pruned SKU homophones (14.2% SKU error)"]
    B -- "Stream full 60-min raw audio directly to Frontier Audio LLM (Tier 3)" --> E["⛔ $420,000/mo Audio Token Bill (17.1x Tier 4)!<br/>Lacks exact word-level timestamp anchors for audio replay"]
    B -- "Per-Track RTP + Calendar Voiceprints + Pre-Call CRM Hotwords + 1-Click Card (Tier 4)" --> F["🟩 Tier 4: Per-Track RTP + Room Voiceprint + CRM Hotwords + Staged Card 🏆<br/>(1.4% DER | 2.2% SKU Error | 0.4% CRM Noise | $24.5K/mo, -87.4% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Per-Track `RTP` + Room Voiceprint, Pre-Call CRM Hotword Boost & Staged `1-Click` Sync Architecture

```mermaid
flowchart LR
    subgraph Pre["1. Pre-Call Context Loader"]
        CRM["📅 Calendar & CRM Opportunity<br/>Fetches attendee names, Account<br/>aliases & active product SKUs"] --> Boost["🎯 ASR Phrase List (Weight=15.0)<br/>Shallow-fusion decoder biasing<br/>(SKU error 28.5% -> 2.2%)"]
    end

    subgraph Diarize["2. Multi-Modal Diarization & ASR"]
        Call["🎙️ Teams / Zoom Call"] --> Split["🔀 Per-User RTP Tracks (Remote)<br/>+ Calendar-Scoped ECAPA-TDNN<br/>Room Voiceprints (1.4% DER!)"]
        Split & Boost --> ASR["📝 Word-Timestamped Streaming ASR"]
    end

    subgraph Action["3. Grounded Extraction & Staged Sync"]
        ASR --> Commit["🧠 Commitment Extractor<br/>Requires Owner + Verb + Date<br/>+ Verbatim [mm:ss] Audio Span"]
        Commit --> Card["🃏 1-Click Teams/Slack Approval Card<br/>7s audio preview + 5s sign-off"]
        Card --> Dest["💼 Salesforce / Dynamics 365 / Jira<br/>(0.4% false task rate)"]
    end
```

### 3.3 End-to-End Sequence: Pre-Call CRM Phrase Boosting, Hybrid Room Diarization & 5-Second `1-Click` Salesforce Sync

```mermaid
sequenceDiagram
    autonumber
    participant CRM as Calendar & Salesforce CRM
    participant Bot as Meeting Bot (Per-Track RTP + Voiceprint)
    participant ASR as Hotword-Boosted Streaming ASR
    participant LLM as Evidence-Grounded Action Extractor
    participant AE as Account Executive (Teams Card)

    CRM->>Bot: 1. T-1 min: Fetch invitees `[Sarah, VP Buyer]` + Opportunity SKUs `["Kronos-4090", "SOC2-Addendum"]`
    Bot->>ASR: 2. Register dynamic `Phrase List (boost=15.0)` in streaming ASR decoder
    Bot->>ASR: 3. Route remote buyer via isolated RTP track; split Conference Room B via `ECAPA-TDNN` voiceprints (`1.4% DER`)
    ASR->>LLM: 4. Emit speaker-labeled, word-timestamped transcript (`97.8%` exact SKU/name accuracy)
    LLM->>LLM: 5. Filter casual brainstorming; extract 2 firm commitments anchored to `[00:24:12 - 00:24:19]`
    LLM->>AE: 6. Push Teams Adaptive Card 10s after call with verbatim quote & `[▶ Play 7s Audio]` button
    AE->>CRM: 7. Click `[Approve & Sync to Salesforce (2 items)]` -> update Opportunity & tasks with `0.4%` noise!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Mixed Audio + Blind Write | 🟨 Tier 2: Acoustic MP3 Clustering | 🟦 Tier 3: Raw 60m Audio to LLM | 🟩 Tier 4: Per-Track + Voiceprint + Hotword + Card |
|---|---|---|---|---|
| **Speaker Attribution Error (`DER`)** | `22.4%` ✕ (swaps buyer/seller) | `11.8%` ✕ (fails in rooms) | `8.4%` | 🏆 **`1.4%` (per-user RTP + calendar room voiceprints)** |
| **Domain Jargon / SKU Error** | `28.5%` ✕ (homophone gibberish) | `14.2%` ✕ | `7.6%` | 🏆 **`2.2%` (pre-call CRM/Calendar ASR phrase boosting)** |
| **False CRM / Jira Sync Rate** | `19.2%` ✕ (pollutes pipeline) | `9.5%` ✕ | `6.2%` | 🏆 **`0.4%` (commitment filter + `[mm:ss]` 1-click card)** |
| **Word-Level Audio Replay Link** | No | Segment-only | Imprecise | 🏆 **Exact `[mm:ss]` clickable audio proof snippet** |
| **100K-Hour Monthly TCO** | `$195,000 / mo` ✕ | `$128,000 / mo` ✕ | `$420,000 / mo` ✕ | **`$24,500 / mo` 🏆 (-87.4% vs Tier 1, -94.2% vs Tier 3)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Meeting Intelligence, Diarization & CRM Sync Sizer](../reports/enterprise-meeting-intelligence-diarization-action-item-crm-sync-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-meeting-intelligence-diarization-action-item-crm-sync-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_meeting_intelligence_diarization_action_item_crm_sync_selection/run.py --check
  python -m unittest tests/test_enterprise_meeting_intelligence_diarization_action_item_crm_sync_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Ingest Per-Participant `RTP` Audio Streams (`Teams Graph / Zoom Raw Audio`) Instead of Downmixed Single-Channel MP3s**: Guarantee `100%` speaker identity for remote participants and isolate shared conference room mics.
- [ ] **Constrain Conference-Room Voiceprint Matching (`ECAPA-TDNN`) to Calendar Invitees**: Match shared-room speech against the enrolled voiceprints of invited attendees rather than an unconstrained company-wide search.
- [ ] **Auto-Inject Pre-Call Calendar & CRM Opportunity Vocabulary (`Phrase Boosting`) into the ASR Decoder**: Load attendee names, customer aliases, and active product SKUs before the call begins (`2.2%` term error rate).
- [ ] **Require Verbatim Quote + Clickable `[mm:ss]` Audio Span on Every Extracted Action Item**: Drop ungrounded or speculative tasks that lack an explicit owner and commitment verb.
- [ ] **Gate `Salesforce / Dynamics 365 / Jira` Writes Behind a 5-Second `1-Click Adaptive Card`**: Let the meeting owner approve or edit staged tasks in Teams/Slack before committing to the system of record (`0.4%` noise).
