# FlashInfer Cake MiniMax-H3 N-Split Tail, Kimi-K3 Fused Router Arm M Guard & SM120 Sage Attention (`Phase 101`)

Deterministic Python 3.12 standard-library model reproducing:
1. **MiniMax-H3 Attention Out-Projection N-Split Tail Schedule** (`bf82f4d797`, PR `#7363`): models half-width $256 \times 128$ tail-tile scheduling for `minimax_h3_out_proj` (BF16) and `minimax_h3_out_proj_mxfp8` (MXFP8 E4M3 + UE8M0 block-32 scales) on SM100 (`sm_100a`) and SM103 (`sm_103a`), yielding `1.0341x` BF16 geomean and `1.0323x` MXFP8 geomean (`1.057x` peak) on GB300.
2. **Kimi-K3 Fused MoE Router Arm M Co-Resident CTA Guard** (`e18eb1d2ee`, PR `#7365`): models `OWNER_CTAS = 896 // 7 = 128` plan-owner CTAs and verifies the 42 vs 43 SM boundary on CC 10.0 (`3 CTAs/SM`) and 31 vs 32 SM boundary on CC 10.3 (`4 CTAs/SM`), contrasting fixed 7-expert ownership in Arm M against strided ownership in Arms L/LP/GW.
3. **SM120 Block-Sparse Sage Attention First-Half $P$ Pre-Computation & TileLang Validation** (`967a8ed858`, PR `#7364`; `e84ce3de8a`, PR `#3350`; `f1a34efea6`, PR `#2860`): evaluates the 20-program metadata specialization, 8-shape cold-L2 speedups (`1.0031x` to `1.0100x`), and TileLang `cvt.rs` `(source dtype, destination dtype, round)` validation.

## Usage
```bash
python experiments/flashinfer_minimax_h3_tail_kimi_router_phase101/run.py --check
```
