# Warp-Specialized Kernel Engineering Simulator (`SM90` vs `SM100`)

Deterministic offline model verifying CUTLASS 3.x / FlashAttention-3 SM90 Hopper and SM100 Blackwell kernel mechanisms:
- Dynamic warpgroup register reallocation (`setmaxnreg.inc/dec`, `40/232` vs `24/240` regs/thread).
- Ping-Pong Consumer Warpgroups (`OrderedSequenceBarrier<2, 2>`) hiding Softmax/Epilogue latency inside peer MMA.
- Wave quantization on `132 SM` (H100) vs `148 SM` (B200) and `PipelineCLCFetchAsync` persistent scheduling.
- SM100 `256 KB/SM` Tensor Memory (`TMEM`) + `tcgen05` `2-CTA Cluster MMA` (`50%` L2 traffic reduction).
