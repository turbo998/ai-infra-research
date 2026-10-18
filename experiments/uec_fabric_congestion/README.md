# UEC 1.1 / UET / PCM Fabric Congestion & Multi-Path Spraying Model (`own-offline`)

This pure-Python standard-library experiment (`run.py`) evaluates three core architectural mechanisms from the Ultra Ethernet Consortium (UEC) 1.1 specification, the official libfabric reference provider (`ultraethernet/uet-ref-prov`), and the Programmable Congestion Management (`spcl/pcm`) SDK:

1. **Header Overhead & Wire Goodput Efficiency (`header_efficiency`)**:
   - Models exact byte layouts from `uet_pkt_hdr.h`:
     - `struct uet_entropy` (4 B), `struct uet_pds_req` (12 B for ROD/RUD), `struct uet_pds_rudi_req` (8 B for RUDI), `struct uet_pds_uud_req` (4 B for UUD).
     - `struct uet_ses_req_onm` (`UET_HDR_REQ_SMALL`, 20 B), `struct uet_ses_req_smsg` (`UET_HDR_REQ_MEDIUM`, 32 B), `struct uet_ses_req_std` (`UET_HDR_REQ_STD`, 44 B), plus optional `struct uet_ses_sync_ext` (4 B) and `struct uet_sec_ssi` (16 B header + 16 B AES-GCM authentication tag vs. 4 B CRC32C).
     - Compares scale-out IPv4/UDP against UEC 1.1 scale-up Unified Forwarding Header (`UFH`, saving 20 B over IPv4 and up to 11 B of Inter-Packet Gap) and baseline RoCEv2 (`IPv4 + UDP + BTH + RETH`).

2. **Four-Quadrant Sender Congestion Control (`congestion_control`)**:
   - Compares **UEC NSCC** (`uet_cc_st.c` & `spcl/pcm/algorithms/nscc.c`), **SMaRTT** (`spcl/pcm/algorithms/smartt.c`), and **Swift** (`spcl/pcm/algorithms/swift.c`) across:
     - `uncongested_zero_qdelay_fast_increase` (`!ecn && qdelay == 0`)
     - `sub_target_qdelay_proportional_increase` (`!ecn && 0 < qdelay < target_qdelay`)
     - `ecn_marked_low_delay_noop_path_switch` (`ecn && qdelay < target_qdelay` — NSCC holds `cwnd` and rotates entropy `EV`, avoiding phantom-ECN overreaction)
     - `ecn_marked_high_delay_multiplicative_decrease` (`ecn && qdelay >= target_qdelay`)
     - `incast_16to1_packet_trim_quick_adapt` (`UET_NACK_TRIMMED` or `qdelay > qa_thresh` — resets `cwnd` to `achieved_bytes` in 1 RTT)

3. **Multi-Path Load Balancing & Entropy Spraying (`multipath_spraying`)**:
   - Compares `UET_CC_OBLIVIOUS_SPRAY`, `UET_CC_SELECTIVE_SPRAY` (`uet_cc_st.c` 2-bit per-path `GOOD`/`SKIP`/`BAD` state table), and `REPS` (`spcl/pcm/algorithms/reps.c` circular ACK-EV cache recycling) across 16 parallel fabric paths under symmetric, hotspot-congested, and single-path blackhole conditions.

## Usage

```bash
python experiments/uec_fabric_congestion/run.py
python experiments/uec_fabric_congestion/run.py --check
```

## Boundary Note

This is an `own-offline` analytical and discrete-event model derived from published UEC 1.1 / `uet-ref-prov` / `spcl/pcm` source equations. It does not claim hardware ASIC switch telemetry, physical 800G transceiver bit-error measurements, or multi-rack production benchmark results.
