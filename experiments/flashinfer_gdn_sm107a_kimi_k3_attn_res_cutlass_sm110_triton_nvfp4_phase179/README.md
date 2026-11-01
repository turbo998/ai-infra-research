# FlashInfer SM107a GDN Ring State-Fixup, B200/B300 Kimi-K3 Token-Queue Routing, CUTLASS Thor SM110 & Triton NVFP4 Scale Absorption (Phase 179)

Deterministic offline verification suite for Phase 179 (`priority: 188`, Track 1: `gpu-software-stack`).

## Audited Upstream Commits
- `flashinfer-ai/flashinfer` @ `44c7973afbf4` (#8206): `cake_gdn_cp` dedicated `sm_107a` ring state-fixup program (`1.05x` speedup)
- `flashinfer-ai/flashinfer` @ `1da0c4e558d3` (#8207): `cake_kimi_k3_attn_res` route $M \in [4096, 8192], K \in [6, 8]$ on B200/B300 to token-queue persistent programs (`1.18x` speedup)
- `flashinfer-ai/flashinfer` @ `150eef8ea280` (#8205): `cake_causal_conv1d_det_train` row-group backward for $264 \le C < 512$ on SM100/SM103 (`1.24x` speedup)
- `flashinfer-ai/flashinfer` @ `fe36017904b4` (#8203): `cake_selective_state_update` z-gated one-token ring on `sm_103a` (`1.14x` speedup)
- `Dao-AILab/flash-attention` @ `c46b8144f2d5` (#2559) & `c75d019dea9d` (#2746): CuTe SM100 Varlen Dynamic Persistent scheduler (`1.29x` speedup) & Hopper SM90 cleanup
- `NVIDIA/cutlass` @ `becfce08cdb5` (#3283) & `1fc71b3ed1ca` (#3299): Thor `SM110` `tcgen05` blockscaled ops & SM100 MMA descriptor offsetting
- `triton-lang/triton` @ `b41eba6a7aef` (#12285) & `78e80b9e9f7b` (#11478): NVFP4 row-scale absorption during MXFP8 conversion (`1.16x` speedup) & target-aware arithmetic vectorization
