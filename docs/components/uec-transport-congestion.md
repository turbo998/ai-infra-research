# Ultra Ethernet 1.1 (UEC / UET) Transport, Multi-Path Packet Spraying & Programmable Congestion Management (`spcl/pcm`)

## Executive Summary

Traditional RoCEv2 (`RDMA over Converged Ethernet v2`) transports InfiniBand verbs over UDP/IP (`UDP_PORT=4791`, `BTH + RETH`), relying on ordered per-QP delivery (`PSN` sequence checking) and lossless Priority Flow Control (`PFC`) + `DCQCN` rate control. Under large-scale AI training (`AllReduce`, `All-to-All` MoE expert dispatch) and disaggregated KV cache transfers, traditional RoCEv2 suffers from three structural failure modes:

1. **ECMP Hash Polarization**: Standard 5-tuple hashing pins elephant flows (e.g., 400 Gbps RDMA WRITEs) onto the same spine link while neighboring links sit idle.
2. **Out-of-Order (`OOO`) Go-Back-N Penalty**: Enabling naive packet spraying across ECMP paths triggers receiver PSN gap NACKs unless NICs implement deep hardware reorder buffers or out-of-order placement.
3. **PFC Pause Spreading & Victim-Flow HoL Blocking**: Hop-by-hop `PFC` XOFF frames stall entire ingress queues across switch tiers when incast bursts exceed shallow shared switch buffers.

The **Ultra Ethernet Consortium (UEC) Specification 1.1** (`2026-10-12`, [UEC 1.1 Press Release](https://ultraethernet.org/ultra-ethernet-consortium-releases-specification-v1-1-advancing-ai-and-hpc-networking-at-scale/)) and its official open-source reference provider **`ultraethernet/uet-ref-prov`** (`commit ae19b9776c42cd78f57118b9b7993ae913625e46`, `2026-10-13`) combined with the **Programmable Congestion Management (`spcl/pcm`)** architecture (`commit 356f67d890c16200f1bf8c31b50a05e76a2e16c7`, [arXiv:2610.13130](https://arxiv.org/abs/2610.13130)) replace single-path lossless assumptions with a layered **Semantic Sublayer (`SES`) + Packet Delivery Sublayer (`PDS`) + Transport Security Sublayer (`TSS`) + NSCC/REPS Congestion & Spraying loop**.

---

## 1. Evidence Classification & Primary Sources

| Evidence ID | Type | Source & Pin | Key Verified Scope |
| --- | --- | --- | --- |
| `ev-uec-1-1-announcement-2026` | `announcement` | [UEC v1.1 Release](https://ultraethernet.org/ultra-ethernet-consortium-releases-specification-v1-1-advancing-ai-and-hpc-networking-at-scale/) (`2026-03-26`) & [UE-Specification-1.1.pdf](https://ultraethernet.org/wp-content/uploads/sites/20/2026/10/UE-Specification-1.1.pdf) (`2026-10-12`) | UEC 1.1 adds Scale-Up **Unified Forwarding Header (`UFH`)** saving 20 B vs. IP and up to 11 B of Inter-Packet Gap (`IPG`), plus **In-Network Collectives (`INC`)** and hardware synchronization primitives. |
| `ev-uet-pkt-hdr-pds-ses-2026` | `implementation` | `ultraethernet/uet-ref-prov` @ `ae19b9776c42cd78f57118b9b7993ae913625e46` (`uet_pkt_hdr.h`, `uet_pds.h`, `uet_pds_rudi.c`) | Exact C byte layouts for `uet_entropy` (4 B), `uet_pds_req` (12 B), `uet_pds_rudi_req` (8 B), `uet_pds_ack_cc` (28 B with 64-bit `sack_bitmap`), `uet_ses_req_onm` (20 B), `uet_ses_req_smsg` (32 B), `uet_ses_req_std` (44 B), `uet_ses_sync_ext` (4 B), and RUDI replay/idempotency constraints. |
| `ev-uet-cc-smarttrack-spray-2026` | `implementation` | `ultraethernet/uet-ref-prov` @ `ae19b9776c42cd78f57118b9b7993ae913625e46` (`cc/uet_cc_st.h`, `cc/uet_cc_st.c`) | Reference sender congestion control (`SmaRTTrack`/NSCC) with `UET_CC_NO_SPRAY`, `UET_CC_OBLIVIOUS_SPRAY`, and `UET_CC_SELECTIVE_SPRAY` (`2-bit` per-EV state `GOOD`/`BAD`/`SKIP`), `quick_adapt` on packet trim/loss or high queuing delay, and 4-quadrant ECN+RTT control. |
| `ev-pcm-nscc-reps-smartt-swift-2026` | `implementation` | `spcl/pcm` @ `356f67d890c16200f1bf8c31b50a05e76a2e16c7` (`include/pcm.h`, `algorithms/nscc.c`, `algorithms/reps.c`, `algorithms/smartt.c`, `algorithms/swift.c`) & [arXiv:2610.13130](https://arxiv.org/abs/2610.13130) | Programmable Congestion Management event-driven pipeline (`Signal -> Accumulator -> Trigger -> Handler -> Control`), UEC `NSCC` phantom-ECN filter (`!ecn && q_delay > target`), `REPS` circular entropy cache (`VAR_EVC`), `SMaRTT`, and `Swift` flow-scaling target delay. |
| `ev-uec-fabric-congestion-model-2026` | `own-offline` | [`experiments/uec_fabric_congestion/run.py`](../../experiments/uec_fabric_congestion/run.py) & [`results.json`](../../experiments/uec_fabric_congestion/results.json) | Reproducible pure-Python verification of UET vs. RoCEv2 wire goodput, NSCC vs. SMaRTT vs. Swift cwnd transitions, and REPS vs. Selective/Oblivious Spraying across 16 parallel paths. |

---

## 2. UET Packet Architecture: `PDS`, `SES`, `TSS`, and Scale-Up `UFH`

### 2.1 Wire Encapsulation and Sublayer Separation

In `ultraethernet/uet-ref-prov/uet_pkt_hdr.h` (`lines 1-767`), UET cleanly separates **packet delivery (`PDS`)** from **memory semantics (`SES`)** and **cryptographic security (`TSS`)**:

```text
+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Ethernet L2 (14B) | IPv4 (20B) or UFH | uet_entropy (4B)  | PDS Header (4-12B)| SES Header(20-48B)| CRC32C(4B)/TSS(32)|
+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
                                        |<- UDP Port 49150 ->|
```

1. **Entropy Header (`struct uet_entropy`, 4 Bytes, `uet_pkt_hdr.h:22-25`)**:
   - Carries `uint16_t src_port` (populated with the multipathing entropy value `EV`) and `uint16_t dst_port = htons(49150)` (`UET_UDP_PORT = 49150`, `UET_IPPROTO = 253`).
   - Unlike UDP RoCEv2 (`port 4791`), UET uses a dedicated default DSCP policy (`UET_DEFAULT_MSG_DSCP = CS0 (0)` for data packets and `UET_DEFAULT_ACK_DSCP = EF (46)` for prioritized ACK/NACK feedback so congestion signals never queue behind bulk data).

2. **Packet Delivery Sublayer (`PDS`, `uet_pkt_hdr.h:59-340`, `uet_pds.h:35-41`)**:
   - Supports four explicit delivery modes (`uet_pds_mode_t`):
     - **`UET_PDS_MODE_ROD` (`0x03` / `0x0e`)**: Reliable Ordered Delivery via temporary Packet Delivery Contexts (`PDC`).
     - **`UET_PDS_MODE_RUD` (`0x02` / `0x0d`)**: Reliable Unordered Delivery via dynamic PDCs (`spdcid`, `dpdcid`) — the primary workhorse for sprayed bulk RDMA WRITE/SEND.
     - **`UET_PDS_MODE_RUDI` (`0x04` / `0x05`)**: Reliable Unordered Delivery for Idempotent operations (`uet_pds_rudi.c`). Uses an **8-byte stateless header** (`struct uet_pds_rudi_req`: 2 B prologue + 2 B reserved + 4 B `pkt_id`) with **zero PDC setup/teardown overhead**.
     - **`UET_PDS_MODE_UUD` (`0x06`)**: Unreliable Unordered Delivery (`struct uet_pds_uud_req`, **4 bytes**).
   - **Standard Request Header (`struct uet_pds_req`, 12 Bytes, `uet_pkt_hdr.h:136-148`)**:
     - `uint16_t prologue` (`type:5`, `next_hdr:4`, `flags:7` including `RETRANS`, `ACK_REQ`, `SYN`, `USE_RSV_PDC` for In-Network Collectives).
     - `uint16_t clear_psn_offset`, `uint32_t psn`, `uint16_t spdcid`, `uint16_t dpdcid`.
   - **Selective ACK with Congestion State (`struct uet_pds_ack_cc`, 28 Bytes, `uet_pkt_hdr.h:221-235`)**:
     - Embeds a **64-bit out-of-order bitmap (`uint64_t sack_bitmap`)** alongside `uint32_t ack_psn_offset`, `uint16_t sack_psn_offset`, `uint64_t ack_cc_state` (`service_time` & `rcvd_bytes`), and `uint8_t cack_psn_offset`. This 64-packet SACK window allows the receiver to absorb sprayed out-of-order arrivals without triggering Go-Back-N retransmission storms.
   - **Packet Trimming NACK Codes (`uet_pkt_hdr.h:269-297`)**:
     - Switches encountering microburst queue exhaustion trim the payload rather than dropping or asserting PFC pause, forwarding the header so the receiver immediately returns `UET_NACK_TRIMMED (0x01)` or `UET_NACK_TRIMMED_LAST_HOP (0x02)` with `struct uet_pds_nack` (12 B) or `struct uet_pds_nack_ccx` (28 B).

3. **Semantic Sublayer (`SES`, `uet_pkt_hdr.h:343-765`)**:
   - **Standard Header (`struct uet_ses_req_std`, 44 Bytes, `UET_HDR_REQ_STD = 0x1`)**: Full RMA/tagged matching header (`12 B cmn` + `8 B buf_off` + `4 B initiator` + `8 B mem_key_match_bits` + `8 B payload_len_msg_off` + `4 B req_len`).
   - **Medium Small-Message Header (`struct uet_ses_req_smsg`, 32 Bytes, `UET_HDR_REQ_MEDIUM = 0x2`)**: Compresses 64 B+ RMA/message headers by 12 B.
   - **Optimized Non-Matching Small Header (`struct uet_ses_req_onm`, 20 Bytes, `UET_HDR_REQ_SMALL = 0x3`)**: Only `12 B cmn` + `8 B buf_off` (saves **24 B** over standard SES headers for latency-critical 64 B scale-up accesses and atomics).
   - **AI Data-Type Atomics & Hardware Synchronization (`uet_pkt_hdr.h:593-684`)**:
     - `uet_ses_atomic_datatype_t` natively includes **`UET_TYPE_BF16 (0x10)`** and **`UET_TYPE_FP16 (0x11)`** alongside `FLOAT`/`DOUBLE`/integers, with control flags `UET_AMO_CTRL_CACHEABLE (0x1)` and `UET_AMO_CPU_COHERENT (0x2)`.
     - `struct uet_ses_sync_ext` (4 Bytes: `uint16_t group`, `uint16_t cnt`) attaches to `UET_SYNC_WRITE (0x10)` and `UET_SYNC_ATOMIC (0x11)`, enabling GPU kernels to trigger barrier/counter completions in NIC hardware without host CPU intervention.

4. **RUDI Replay & Idempotency Boundary (`uet_pds_rudi.c:302-322`)**:
   - Inspecting `uet_pds_rudi_rx_req()` in `uet_pds_rudi.c` reveals a critical architectural tradeoff: because RUDI avoids per-peer PDC state (`out_ht` uthash + `rto_list` on TX only), a duplicate packet arriving after ACK loss is passed up to SES again. Furthermore, source comments explicitly warn:
     > *"Since TSS works with RUDI (where there is no PDC), TSS has no replay protection. Therefore the target MR must be IDEMPOTENT_SAFE, so that duplicate requests have no side effects."*
   - **Architect Takeaway**: Use `RUDI` for idempotent KV cache block overwrites or fixed-slot status updates, never for non-idempotent atomic increments (`SUM`/`CSWAP`) or streaming FIFO appends.

### 2.2 Scale-Up Unified Forwarding Header (`UFH`) & Wire Goodput Comparison

In UEC 1.1 (`ev-uec-1-1-announcement-2026`), scale-up and converged fabrics gain the **Unified Forwarding Header (`UFH` — `UFH-16` and `UFH-32`)**, which saves **20 bytes over IPv4** and **up to 11 idle bytes in the Inter-Packet Gap (`IPG`)**. Our reproducible model (`experiments/uec_fabric_congestion/run.py`) quantifies the exact wire overhead and goodput across three representative AI workloads:

| Case ID (`own-offline`) | Payload | Mode & Headers | RoCEv2 Wire Overhead | UET Wire Overhead | Bytes Saved | RoCEv2 Goodput | UET Goodput | Goodput Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `scale_up_64b_atomic_rudi_onm_ufh` (800G Scale-Up) | `64 B` | `RUDI (8B) + ONM (20B) + UFH` | `98 B` | `63 B` | **`+35 B`** | `316.05 Gbps` | **`403.15 Gbps`** | **`+27.56%`** |
| `scale_out_256b_rudi_smsg_ipv4` (400G Scale-Out) | `256 B` | `RUDI (8B) + SMSG (32B) + IPv4` | `98 B` | `106 B` | `-8 B` | `289.27 Gbps` | `282.87 Gbps` | `-2.21%` |
| `scale_out_4096b_rod_std_tss_sec` (400G Encrypted Bulk) | `4096 B` | `ROD (12B) + STD (44B) + SYNC (4B) + TSS (32B)` | `98 B` (unencrypted) | `154 B` (encrypted) | `-56 B` | `390.65 Gbps` | `385.51 Gbps` | `-1.32%` |

- **Why this matters**: For 64-byte scale-up NVLink/UB-Mesh-style remote tensor/atomic accesses at 800 Gbps, standard RoCEv2 loses `60.5%` of line rate to L1/L2/IPv4/UDP/BTH/RETH framing (`316.05 Gbps` goodput). Combining UEC 1.1 `UFH` (`-20 B` L3 + `-11 B` IPG) with `RUDI` (`8 B`) and `UET_HDR_REQ_SMALL` (`20 B`) boosts 64 B goodput by **`+27.56%`** (`403.15 Gbps`). Conversely, on 4096 B scale-out bulk transfers, enabling full `ROD + STD + SYNC + TSS AES-GCM (32 B)` costs only **`1.32%`** goodput (`385.51 Gbps` vs. `390.65 Gbps`), making line-rate zero-trust RDMA encryption practical across multi-tenant GPU clouds.

---

## 3. Sender Congestion Control: UEC `NSCC` / `SmaRTTrack` vs. `SMaRTT` vs. `Swift`

### 3.1 Four-Quadrant ECN + Delay Control and Phantom-ECN Filtering

Both `ultraethernet/uet-ref-prov/cc/uet_cc_st.c` (`lines 154-537`) and `spcl/pcm/algorithms/nscc.c` (`lines 42-234`) implement UEC's hybrid **ECN + Delay sender congestion control (`NSCC` / `SmaRTTrack`)**. Unlike RoCEv2 `DCQCN` (which reacts solely to ECN via CNP packets) or Google `Swift` (which reacts solely to RTT delay), UEC NSCC fuses **switch ECN marks (`SIG_NUM_ECN`)**, **RTT queuing delay (`q_delay = rtt - base_rtt`)**, and **packet-trim NACKs (`UET_NACK_TRIMMED`)**:

1. **Initialization & Scaling (`uet_cc_st.c:154-174`)**:
   $$\text{BDP}_{\text{bytes}} = \frac{\text{BW}_{\text{Mbps}} \cdot \text{RTT}_{\text{base},\mu\text{s}}}{8}, \quad \text{cwnd}_{\max} = 1.25 \cdot \text{BDP}_{\text{bytes}}$$
   $$\tau_{\text{target}} = \begin{cases} 0.5 \cdot \text{RTT}_{\text{base}}, & \text{if Packet Trimming enabled} \\ 1.0 \cdot \text{RTT}_{\text{base}}, & \text{otherwise} \end{cases}, \quad \tau_{\text{QA}} = \begin{cases} 4 \cdot \tau_{\text{target}}, & \text{if Trimming enabled} \\ 2 \cdot \tau_{\text{target}}, & \text{otherwise} \end{cases}$$
   For a `400 Gbps` link with `12 us` base RTT and `4096 B` MTU (`trim_supp = true`):
   - $\text{BDP}_{\text{bytes}} = 600,000\text{ B}$ (`~146.5 MTUs`), $\text{cwnd}_{\max} = 750,000\text{ B}$, $\tau_{\text{target}} = 6.0\text{ }\mu\text{s}$, and Quick Adapt threshold $\tau_{\text{QA}} = 24.0\text{ }\mu\text{s}$.

2. **The Four-Quadrant Decision Matrix (`spcl/pcm/algorithms/nscc.c:88-165`, `uet_cc_st.c:480-508`)**:
   - **Branch 0 — `Quick Adapt` (`packet_trimmed || nack || q_delay > qa_thresh`)**:
     Immediately clamps $\text{cwnd} \leftarrow \max(\text{MTU}, \min(\text{cwnd}, \text{achieved\_bytes}))$ in a single RTT (`qa_ign_bytes = in_flight`), eliminating multi-RTT queue build-up during 16-to-1 or 64-to-1 AI incast.
   - **Quadrant 1 — `!ECN && q_delay < target_qdelay` (`fast_increase` / `proportional_increase`)**:
     When `q_delay == 0` for a full window, `cwnd` grows by `newly_acked_bytes` (exponential fast start); otherwise `cwnd` increases proportionally to $(\tau_{\text{target}} - q_{\text{delay}})$.
   - **Quadrant 2 — `!ECN && q_delay >= target_qdelay` (`fair_increase`)**:
     Delay exceeds target, but the switch has **not** set ECN (e.g., delay came from host/PCIe jitter or a transient cross-flow burst that already drained). Sender applies cautious additive `fair_increase` (`fi * mtu * acked_bytes / cwnd`).
   - **Quadrant 3 — `ECN && q_delay < target_qdelay` (`NOOP + Switch Entropy Path`)**:
     In `spcl/pcm/algorithms/nscc.c:110-118`:
     ```c
     } else if (get_signal(SIG_NUM_ECN) && get_var(VAR_Q_DELAY) < TARGET_Q_DELAY) {
         // Do nothing, just switch path
         set_var(VAR_NEW_EV, get_var(VAR_NEW_EV) + 1);
     ```
     This is a crucial multi-path insight: if a sprayed packet hits a micro-burst ECN mark on one spine switch, but end-to-end queuing delay remains below `target_qdelay`, cutting `cwnd` would throttle throughput across the entire 16-path fabric. Instead, NSCC **holds `cwnd` steady and rotates the entropy value (`EV`)** away from the congested spine.
   - **Quadrant 4 — `ECN && q_delay >= target_qdelay` (`multiplicative_decrease`)**:
     Confirmed fabric congestion (`md_factor = max(1 - gamma * (1 - target_qdelay / q_delay), 0.5)` plus fair decrease `fd * acked_bytes * cwnd / bdp`).

### 3.2 Quantitative Comparison (`NSCC` vs. `SMaRTT` vs. `Swift`)

From `experiments/uec_fabric_congestion/results.json` (`400 Gbps`, `12 us` base RTT, `4096 B` MTU, `BDP = 600,000 B`, `acked_mtus = 8` (`32,768 B`)):

| Scenario (`case_id`) | Initial `cwnd` | `q_delay` (`us`) | ECN / Trim | UEC NSCC Branch & Next `cwnd` | SMaRTT Next `cwnd` | Swift Next `cwnd` | Architectural Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `uncongested_zero_qdelay_fast_increase` | `300,000 B` | `0.0 us` | `False / False` | `fast_increase` -> **`332,768 B`** (`+10.9%`) | `300,291 B` | `300,447 B` | Reclaims idle 400G capacity after collective barrier in 1–2 RTTs. |
| `sub_target_qdelay_proportional_increase` | `480,000 B` | `3.0 us` | `False / False` | `proportional_increase` -> **`480,419 B`** | `480,098 B` | `480,280 B` | Smoothly converges toward `6.0 us` target delay without overshoot. |
| `ecn_marked_low_delay_noop_path_switch` | `600,000 B` | `3.0 us` | `True / False` | `ecn_low_delay_noop_switch_path` -> **`600,000 B`** | `595,904 B` | `600,224 B` | Avoids false cwnd cut on single-link microburst; steers traffic to healthy EV. |
| `ecn_marked_high_delay_multiplicative_decrease` | `600,000 B` | `10.0 us` | `True / False` | `multiplicative_decrease` -> **`390,174 B`** (`-35.0%`) | `578,850 B` | `454,484 B` | Fast proportional backoff when both ECN and delay confirm persistent queue buildup. |
| `incast_16to1_packet_trim_quick_adapt` | `600,000 B` | `30.0 us` | `True / Trim` | `quick_adapt` -> **`37,500 B`** (`= BDP/16`) | **`37,500 B`** | `300,000 B` (`-50%` cap) | Packet trimming + `Quick Adapt` snaps 16 concurrent senders directly to their exact fair share (`37.5 KB`) in 1 RTT, whereas Swift's `max_mdf=0.5` requires 4 consecutive RTTs of congestion to reach `37.5 KB`. |

---

## 4. Multi-Path Packet Spraying: `Oblivious` vs. `UET Selective Spray` vs. `PCM REPS`

### 4.1 Source-Code Mechanics of `UET_CC_SELECTIVE_SPRAY` and `REPS`

How should a 400G/800G AI NIC choose the 16-bit entropy value (`uet_entropy.src_port`) on every outgoing packet? We compared the two open-source reference implementations line by line:

1. **`UET_CC_SELECTIVE_SPRAY` (`ultraethernet/uet-ref-prov/cc/uet_cc_st.c:110-128, 225-287`)**:
   - Allocates a compact **2-bit per-path state array** `uint8_t ev_state[64]` (`256` entropy slots packed at 4 paths per byte):
     - `UET_CC_PATH_GOOD (0x0)`: Normal transmission path.
     - `UET_CC_PATH_SKIP (0x2)`: Set by `skip_path(cc, ev)` upon receiving an ECN-marked ACK or NACK (`uet_cc_st.c:225-238`).
     - `UET_CC_PATH_BAD (0x1)`: Set by `bad_path(cc, ev)` upon an RTO timeout (`uet_cc_st.c:239-256`).
   - During `select_entropy()` (`uet_cc_st.c:258-287`), the pointer scans forward:
     - If `state == GOOD`, it selects that `EV`.
     - If `state == SKIP`, it skips the path **once** and immediately flips `SKIP -> GOOD` (`uet_cc_set_ev_state(cc, ev, UET_CC_PATH_GOOD)`), halving the packet rate onto ECN-marked paths while continuing to probe them on the next cycle.
     - If `state == BAD`, it skips the path indefinitely until `bad_cnt >= num_paths` forces a reset.

2. **`REPS` — Recycled Entropy Packet Spraying (`spcl/pcm/algorithms/reps.c:1-60`, `arXiv:2610.13130`)**:
   - Instead of probing all paths and skipping bad ones once, `REPS` maintains a **circular cache `VAR_EVC[EVC_SIZE]`** of known-good entropy values (`EVC_SIZE = 16`, `EVC_MASK = 15`):
     - **On ACK reception (`reps.c:28-36`)**: Only if `!get_signal(SIG_ECN)` (the ACK arrived **without** ECN marking) does `REPS` insert `get_signal(SIG_ACK_EV)` into `VAR_EVC[head_idx]` and set `VAR_EVC_BITMAP[head_idx] = 1`. ECN-marked or timed-out packets are **never recycled into the cache**.
     - **On TX (`reps.c:39-58`)**: After an initial `explore_counter` warm-up of 16 packets, every outgoing packet pops a verified non-congested `EV` directly from `VAR_EVC`. It falls back to hash exploration (`_hash(ev_seed) & PATH_MASK`) only if `num_valid_evs == 0` (e.g., all cached paths became congested).

### 4.2 Quantitative Spraying Performance under Hotspot & Blackhole Faults

Using `experiments/uec_fabric_congestion/run.py` (`16` parallel `50 Gbps` paths = `800 Gbps` aggregate fabric, `1,024` packets of `4,096 B` = `4.194 MB` burst):

| Spraying Scenario (`case_id`) | Fabric Health (16 Paths) | Oblivious Spray (`UET_CC_OBLIVIOUS_SPRAY`) | UET Selective Spray (`uet_cc_st.c`) | PCM REPS (`spcl/pcm/algorithms/reps.c`) | REPS Speedup vs. Oblivious |
| --- | --- | --- | --- | --- | --- |
| `symmetric_16paths_no_congestion` | `16` healthy (`50G`), `0` congested | `41.94 us` (`800.0 Gbps`) | `41.94 us` (`800.0 Gbps`) | `41.94 us` (`800.0 Gbps`) | `1.00x` |
| `asymmetric_16paths_4_congested_hotspots` | `12` healthy (`50G`), `4` congested (`10G` hotspot) | `256` congested pkts -> `209.72 us` (`160.0 Gbps`) | `148` congested pkts -> `121.24 us` (`276.76 Gbps`) | **`4` congested pkts** -> **`55.71 us` (`602.35 Gbps`)** | **`3.76x`** (`+276.5%` bw) |
| `faulty_16paths_1_blackhole_3_congested` | `12` healthy (`50G`), `3` congested (`10G`), `1` link down (`1000 us` RTO) | `64` blackhole pkts + `192` cong pkts -> `1041.94 us` | `1` blackhole pkt + `114` cong pkts -> `1000.66 us` | **`1` blackhole pkt + `3` cong pkts** -> **`1000.66 us`** | `1.04x` (RTO-dominated) / **98.4% fewer tail retx** |

- **Key Architectural Insight**:
  1. **Hotspot Resilience (`asymmetric_16paths_4_congested_hotspots`)**: When 4 of 16 spine links are degraded to `10 Gbps` by competing jobs (leaving `12 x 50G = 600 Gbps` of healthy capacity), `Oblivious Spray` blindly sends `1/4` of packets (`256` packets) onto the `10 Gbps` links, throttling the entire collective completion to **`160.0 Gbps`** (`209.72 us`). `UET Selective Spray`'s 1-round `SKIP` toggle reduces congested packets to `148` (`276.76 Gbps`), while **`REPS` touches the 4 congested links only once during initial exploration (`4` packets total) and recycles the 12 healthy EVs for **`602.35 Gbps`** (`3.76x` faster than Oblivious Spray)**.
  2. **Blackhole Link Failure (`faulty_16paths_1_blackhole_3_congested`)**: Both `UET Selective Spray` (`UET_CC_PATH_BAD`) and `REPS` isolate a dark optical link after **a single lost packet (`1` packet vs. `64` packets under Oblivious Spray)**. However, unless switch hardware **Packet Trimming** or link-layer fast fail-over notifies the sender before `RTO` (`1000 us`), that single lost packet still gates collective tail latency at `~1000.66 us`. Hence **REPS + Switch Packet Trimming (`UET_NACK_TRIMMED`)** are complementary, not substitutes.

---

## 5. Programmable Congestion Management (`spcl/pcm`) Hardware Datapath Contract

 Why did UEC and ETH Zürich SPCL introduce **Programmable Congestion Management (`PCM`, `arXiv:2610.13130`)** instead of hard-coding a single congestion algorithm into NIC RTL?

Inspecting `spcl/pcm/include/pcm.h` (`lines 1-112`) shows that NIC vendors face diverging topology requirements (2-tier Rail-Only vs. 3-tier Clos vs. Scale-Up UFH vs. WAN cross-DC training). `PCM` decomposes congestion control into a deterministic 5-stage pipeline executable at line rate (`800 Gbps+`):

```text
[ Transport Events ] --> (1) PCM Signals --> (2) Hardware Accumulators --> (3) Triggers --> (4) Handler VM/C --> (5) Controls
  ACK / ECN / NACK        PCM_SIG_RTT         PCM_ACCUM_SUM / MIN          RAW / DELTA       nscc.c / reps.c     PCM_CTRL_CWND
  RTO / TX_BACKLOG        PCM_SIG_ACK_EV      PCM_ACCUM_MAX / LAST         MAGNITUDE         smartt.c / swift.c  PCM_CTRL_RATE
                                                                                                                 PCM_CTRL_EV
```

- **Hardware-Accumulated Signals (`pcm.h:26-48`)**: The NIC hardware datapath accumulates high-frequency per-packet events (`PCM_SIG_ACK`, `PCM_SIG_ECN`, `PCM_SIG_RTT`, `PCM_SIG_DATA_TX`, `PCM_SIG_IN_FLIGHT`, `PCM_SIG_ACK_EV`) using fixed-function accumulators (`SUM`, `MIN`, `MAX`, `LAST`) and fires the handler only when a programmable trigger (`PCM_TRIG_DELTA`, `PCM_TRIG_RAW`, `PCM_TRIG_MAGNITUDE`) is crossed (e.g., every `N` ACKed bytes or immediately upon `NACK`/`RTO`).
- **Unified Control Actuators (`pcm.h:85-98`)**: Every algorithm outputs standardized control updates (`PCM_CTRL_CWND`, `PCM_CTRL_RATE`, `PCM_CTRL_EV`), allowing an operator to hot-swap `nscc.c + reps.c` for scale-out AI collectives or `swift.c` for delay-sensitive storage traffic without respinning NIC silicon.

---

## 6. Cloud Solution Architect Decision Matrix & Falsifiable Questions

| Workload / Topology | Recommended Transport & CC Mode | Rationale & Quantitative Guardrail |
| --- | --- | --- |
| **Intra-Rack / Scale-Up Domain (`<= 64 GPUs`, 64B–256B fine-grained tensor/atomic ops)** | **UEC 1.1 `UFH` + `RUDI` + `UET_HDR_REQ_SMALL (ONM)`** | Cuts wire overhead from `98 B` (RoCEv2) to `63 B`, raising 64 B goodput at 800G from `316.05 Gbps` to **`403.15 Gbps` (`+27.56%`)**. Target MR must be `IDEMPOTENT_SAFE`. |
| **Scale-Out MoE `All-to-All` & Disaggregated KV Cache (`256–32,768 GPUs`, Clos/Rail-Optimized)** | **UET `RUD` + `NSCC` (`trim_supp=true`) + `REPS` + 64-pkt `sack_bitmap`** | `REPS` achieves **`3.76x`** higher throughput than Oblivious Spray under 25% spine congestion (`602.35 Gbps` vs. `160.0 Gbps`); `NSCC` + Packet Trimming clamps 16-to-1 incast to `37.5 KB` in **1 RTT**. |
| **Multi-Tenant / Cross-AZ Zero-Trust GPU Fabric (`4096 B` MTU)** | **UET `ROD/RUD` + `TSS` (`uet_sec_ssi` AES-GCM)** | Hardware session security sublayer (`16 B SSI + 16 B tag`) reduces 4096 B goodput by only **`1.32%`** (`385.51 Gbps` vs. `390.65 Gbps`). |

### Open Falsifiable Questions for Hardware Verification
1. **Switch Packet-Trimming Buffer Floor**: What minimum shared-buffer threshold (in KB per 800G port) on Broadcom Tomahawk 5 / Tomahawk 6 switches guarantees zero tail-drop timeouts (`RTO`) when 64 concurrent UET senders rely on `UET_NACK_TRIMMED` + `NSCC Quick Adapt`?
2. **REPS Cache Stale-EV Decay under Dynamic Traffic**: When a bursty MoE `All-to-All` phase pauses for `5–15 ms` of GEMM compute, do cached EVs in `VAR_EVC[16]` become stale and require re-seeding (`explore_counter = EVC_SIZE`) at the start of each step?
