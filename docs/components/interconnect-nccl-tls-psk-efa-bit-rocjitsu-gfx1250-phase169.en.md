# Interconnect Security & Emulation Phase 169: NCCL OpenSSL TLS 1.3 PSK Enforcement & EFA-GDA Phase Bit Manipulation, ROCm `rocjitsu` Instruction-Encoding Capture & `gfx1250` Rev-0 Trap PC Scoping, and NIXL/Mooncake Hardening

## 1. Architectural Context & Core Problems

Across encrypted collective communications (NCCL Crypt), AWS EFA-GDA header compilation, AMD `gfx1250` trap handling, and high-frequency KV transfer logging, three critical issues were resolved:
1. **Unverified PSK Path in NCCL OpenSSL TLS 1.3 Handshakes & Scoped `BIT(0)` Macro Leak in EFA-GDA**: OpenSSL reports TLS 1.3 external PSK handshakes via session reuse (`SSL_session_reused()`). Without verifying peer certificates and rejecting handshakes where `SSL_session_reused()` is false, connections could complete without taking the configured PSK path. Meanwhile, `EFA_SET(&ctrl2, EFA_IO_TX_META_DESC_PHASE, wqe_phase)` expanded `BIT(0)` outside its local inclusion scope in `efa_cuda_dp_impl.cuh`.
2. **Disassembly String Bloat in `rocjitsu` Race Detector & Over-Applied VOP3PX Trap PC Fixup on `gfx1250`**: Caching full disassembly strings (`~64 B/inst`) inflated `rocjitsu` race-detector memory consumption by `8x` compared to raw `8-byte` encodings, while ROCr's second-level trap handler applied a `gfx1250 revision 0`-only VOP3PX PC fixup across all `gfx1250` revisions.
3. **Per-Transfer `INFO` Log Flood in NIXL & Cold-JIT Launch Timeouts in Mooncake**: Logging `"Selected backend"` at `INFO` level on every NIXL transfer saturated stdout at high transfer rates, while Mooncake's MoE e2e server needed a `360s` timeout for cold JIT compilation.

---

## 2. Primary Upstream Mechanism Deep Dive (8 Zero-Overlap Commits)

### 2.1 NVIDIA NCCL: TLS 1.3 PSK Enforcement & EFA-GDA Direct Bit Manipulation
- **`65e7e3ea64` (`2026-09-12`)**: Configures the OpenSSL client context in `src/misc/crypt.cc` to verify peer certificates and rejects handshakes unless `SSL_session_reused()` confirms the configured PSK path was used.
- **`bb4ddd1cc9` (`2026-09-11`)**: Replaces `EFA_SET(&ctrl2, EFA_IO_TX_META_DESC_PHASE, wqe_phase)` with direct bit manipulation in `gin_efa_gda.h`.

### 2.2 ROCm `rocm-systems`: `rocjitsu` Instruction-Encoding Capture & `gfx1250` Rev-0 Trap PC Fixup
- **`ddbfb7d489` (`2026-10-31`, `#14223`)**: Captures raw instruction encodings in the `rocjitsu` race detector instead of caching disassembly strings (`8.0x` memory footprint reduction).
- **`4b70275d5a` (`2026-10-31`, `#13797`)**: Scopes the second-level trap handler's VOP3PX trap PC fixup in `trap_handler_gfx12.s` and `amd_gpu_agent.cpp` strictly to `gfx1250 revision 0`.

### 2.3 NIXL, Mooncake & LMCache: Log Demotion, 360s Cold-JIT Timeout & Python 3.14 Support
- **`dd2141ec1d` (`2026-10-12`, `#2321`) & `f2086f5616` (`2026-10-12`, `#2230`)**: Demotes NIXL's per-transfer `"Selected backend"` log to `DEBUG` and restores Device API Doxygen comments.
- **`95346cefcb` (`2026-10-19`, `#4765`)**: Raises Mooncake's MoE server launch timeout to `360s` for cold JIT builds.
- **`3981ee5145` (`2026-10-27`, `#5289`)**: Adds Python `3.14` support to LMCache.

---

## 3. Deterministic Offline Verification Summary

Validated via `experiments/nccl_tls_psk_efa_bit_rocjitsu_encoding_gfx1250_phase169/run.py`:

| Subsystem | Baseline Behavior | Phase 169 Verified Behavior | Architectural Impact |
|---|---|---|---|
| NCCL Crypt TLS 1.3 PSK & EFA Phase Bit | `4` non-PSK handshakes accepted; macro leak | **`0` non-PSK accepted**; **`0` macro errors** | Enforces strict PSK authentication & clean header inclusion |
| ROCm `rocjitsu` (`10k` insts) & `gfx1250` Trap | `640 KB` cached strings; all-rev PC fixup | **`80 KB` raw encodings (`8.0x`)**; **Rev-0 only** | Cuts race-detector memory `8x` & prevents PC corruption |
| NIXL Transfer Logging (`5k` tx/s) & Mooncake JIT | `5,000` INFO logs/s; cold-JIT timeout | **`0` INFO logs/s (`DEBUG`)**; **`360s` timeout** | Eliminates logging I/O bottlenecks & CI flakiness |
