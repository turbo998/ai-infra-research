# FlashInfer Cake Grouped MXFP8 In-Tree Loader, vLLM DeepSeek-V4 EPLB Redundant Expert Slots & Kimi-K3 GB200 AgentX Topology

> **中文版**: [flashinfer-cake-grouped-mxfp8-quantize-vllm-dsv4-eplb-redundant-slots.md](flashinfer-cake-grouped-mxfp8-quantize-vllm-dsv4-eplb-redundant-slots.md)

## 1. Primary Source Evidence & Core Claims

| Repository & Commit | Evidence Mode | Source Paths | Mechanism & Quantitative Finding |
|---|---|---|---|
| `flashinfer-ai/flashinfer` `#6868` (`d8ddf1a367`, 2026-10-20) | `implementation` / `author-benchmark` | `flashinfer/quantization/fp8_quantization.py`, `csrc/cake_grouped_mxfp8_quantize/*_binding.cu` | Routes `cake_grouped_mxfp8_quantize` through the in-tree `CakeProgram` loader (`d8aa920df5a6252450bd` BF16, `f84d4eab2905894ac4c2` FP16) on `sm_100a` (B200) and `sm_103a` (GB300) with in-kernel 128-column padding and `[B, pm_tiles, pk_tiles, 32, 4, 4]` E8M0 scale layout. |
| `vllm-project/vllm` `#59611` (`9d4192aaa1`, 2026-10-20) | `implementation` | `vllm/models/deepseek_v4/nvidia/model.py`, `tests/evals/gsm8k/eplb_checks.py` | Passes `num_redundant_experts=first_layer.ffn.n_redundant_experts` in `get_expert_mapping()` and verifies byte-sum parity across all physical slots before traffic via `check_eplb_replica_slots` (reducing bad redundant slots from `8 -> 0`). |
| `vllm-project/vllm` `#62761` (`56e67b46a3`, 2026-10-20) | `implementation` | `recipes/models/moonshotai/Kimi-K3.yaml` | Configures Kimi-K3 GB200 AgentX across four 4-GPU trays (16 GPUs) as `TP8 x PP2 + DCP8` (`dcp_comm_backend=a2a`) instead of `TP4 x PP4`, paired with `TRTLLM_RAGGED` MLA prefill and `K=4` DSpark speculative decoding. |

## 2. Implementation Path, Formulas & Memory Layout

### 2.1 Cake Grouped MXFP8 In-Tree Launch & 128x4 Scale Layout
In `flashinfer/quantization/fp8_quantization.py`, `cake_grouped_mxfp8_quantize_launch(a, mask, quantized, scales)` launches a single allocation-free kernel for $a \in \mathbb{R}^{B \times M \times K}$:
1. **Shared 2-Byte Pointer ABI**: Both BF16 and FP16 generated programs read 2-byte elements through a `__nv_bfloat16*` pointer (`x = a if a.dtype == torch.bfloat16 else a.view(torch.bfloat16)`), with the FP16 kernel widening in registers.
2. **Padding & 6D Scale View**: Requires $K \bmod 32 = 0$ and $M, B \le 65535$. `quantized` has shape $[B, M, \text{padded\_K}]$ ($\text{padded\_K} = \lceil K / 128 \rceil \times 128$) and `scales` has shape $[B, \text{padded\_M}, \text{padded\_K}/32]$ ($\text{padded\_M} = \lceil M / 128 \rceil \times 128$), viewed as `[B, pm_tiles, pk_tiles, 32, 4, 4]`.
3. **3D Launch Grid**: With $\text{blocks\_per\_row} = \text{padded\_K} / 32$ and CTA width $T_{\text{CTA}}$:
   $$\text{grid} = \left(\left\lceil \frac{\text{blocks\_per\_row}}{T_{\text{CTA}}} \right\rceil,\; M,\; B\right),\quad \text{TOTAL\_TASKS} = B \times M \times \text{blocks\_per\_row}$$

### 2.2 Author-Reported Benchmark (`B=2, M=256, K=4096, T=512`)
- **B200 (`sm_100a`)**: BF16 `3.744 -> 3.808 us` (`0.983x`), FP16 `3.968 -> 4.000 us` (`0.992x`).
- **GB300 (`sm_103a`)**: BF16 `3.744 -> 3.776 us` (`0.992x`), FP16 `3.648 -> 3.648 us` (`1.000x`).

### 2.3 DeepSeek-V4 EPLB Redundant Slots & Kimi-K3 GB200 AgentX Profile
- **EPLB Redundant Slot Loading (`#59611`)**: Forwarding `num_redundant_experts` in `get_expert_mapping()` populates all $E_{\text{logical}} + 8$ physical slots at startup; `check_eplb_replica_slots` confirms `0` mismatched slots before serving traffic.
- **Kimi-K3 GB200 AgentX (`#62761`)**: Uses `TP8 x PP2 + DCP8` (`a2a`) across four 4-GPU GB200 trays, halving pipeline stages (`4 -> 2`) while keeping 1M context, FP8 KV cache, `TRTLLM_RAGGED` query quantization, and `K=4` DSpark speculation.

## 3. Deterministic Verification & Architect Exercises

- **Verifier**: `python experiments\flashinfer_cake_grouped_mxfp8_vllm_dsv4_eplb_kimi\run.py --check`
- **Exercise**: For $B=4, M=130, K=4128$, $\text{padded\_M}=256$ ($\text{pm\_tiles}=2$), $\text{padded\_K}=4224$ ($\text{pk\_tiles}=33$), $\text{blocks\_per\_row}=132$, and `scales` occupies $4 \times 256 \times 132 = 135,168\text{ B}$ (`[4, 2, 33, 32, 4, 4]`).
